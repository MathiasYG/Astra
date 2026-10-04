// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class DirectImageTableTests
{
    [Theory]
    [InlineData(0x80000000u, true)]
    [InlineData(0x10000000u, false)]
    public void ImageCandidatesDistinguishResourceLevelFromReservedBits(uint bit, bool valid)
    {
        var plan = ShaderResourcePlan.Extract(CreateWaveIndexedDescriptorProgram(), ShaderStage.Compute, Hash, 0, 2);
        bool Read(ulong address, out uint word)
        {
            if (!ReadWaveIndexedMemory(address, out word)) return false;
            if (address >= 0x1100 && (address - 0x1100) % 32 == 8) word |= bit;
            return true;
        }
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization));
        Assert.Equal(valid, snapshot.Images.Any(image => image[0] != 0));
        if (valid) Assert.All(snapshot.Images, image => Assert.Equal(bit, image[2] & bit));
    }

    internal static Gen5ShaderProgram WorkgroupImageProgram(bool unknownWrite = false) => Program([
        Sop2(0, "SLshrB32", 9, Gen5Operand.Scalar(8), Operand(1)),
        Sop2(4, "SMulI32", 9, Gen5Operand.Scalar(9), Operand(64)),
        ScalarBufferLoad(8, 0, 16, 8, dynamicOffsetRegister: 9),
        ScalarBufferLoad(16, 0, 24, 4, immediateOffset: 32, dynamicOffsetRegister: 9),
        Image(24, "ImageSampleLz", 16, samplerRegister: 24, dmask: 1),
        unknownWrite ? GlobalAccess(32, "FlatStoreDword", 0) : BufferStore(32, 4), EndProgram(40),
    ]);

    private static ShaderResourcePlan WorkgroupImagePlan(bool unknownWrite = false) =>
        ShaderResourcePlan.Extract(WorkgroupImageProgram(unknownWrite), ShaderStage.Compute, Hash, 0, 8,
            computeSystemRegisters: new Gen5ComputeSystemRegisters(8, null, null, null));

    private static ResourceRuntimeInputs WorkgroupImageInputs(uint groups = 4, bool alias = false,
        bool differentSampler = false, bool unreadable = false, bool provideClean = true, uint imageFormat = 20)
    {
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 || address >= 0x1080 || unreadable && address == 0x1040) return false;
            var record = (uint)(address - 0x1000) / 64;
            var component = (uint)(address - 0x1000) % 64 / 4;
            word = component switch
            {
                0 => record == 0 ? 0x2000u : 0x3000u,
                1 => imageFormat << 20,
                3 => 0xFACu | (9u << 28),
                8 => differentSampler && record != 0 ? 1u : 0u,
                9 => 0xFFF000,
                _ => 0,
            };
            return true;
        }
        return new ResourceRuntimeInputs
        {
            UserData = [0x1000, 64u << 16, 2, (20u << 12) | 0xFAC,
                alias ? 0x1040u : 0x4000u, 4u << 16, 16, (20u << 12) | 0xFAC],
            ReadMemory = Read, ReadCleanMemory = provideClean ? Read : null,
            ComputeState = new(64, 64, 1, 1, false, 0, 1, groups, 1, 1),
        };
    }

    public static (ResourceSnapshot Snapshot, ShaderCompileRequest Request, uint[] Registers, byte[] Table)
        PrepareWorkgroupImages()
    {
        var program = Program([.. WorkgroupImageProgram().Instructions.Where(instruction => instruction.Pc < 24),
            Vop1(24, "VMovB32", 0, Operand(0x3F000000)),
            Vop1(28, "VMovB32", 1, Operand(0x3F000000)),
            Vop1(32, "VMovB32", 2, Gen5Operand.Scalar(8)),
            Image(36, "ImageSampleLz", 16, samplerRegister: 24, dmask: 1),
            BufferAccess(44, "BufferStoreDword", 4, vectorData: 4, indexEnabled: true, vectorAddress: 2), EndProgram(52)]);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 8,
            computeSystemRegisters: new Gen5ComputeSystemRegisters(8, null, null, null));
        var inputs = WorkgroupImageInputs(imageFormat: 22);
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 8),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            LocalSizeX = 1, ThreadCountX = 4, ComputeSystemRegisters = new(8, null, null, null),
        };
        var table = new byte[128];
        for (uint offset = 0; offset < table.Length; offset += 4)
        {
            Assert.True(inputs.ReadCleanMemory!(0x1000 + offset, out var word));
            BitConverter.TryWriteBytes(table.AsSpan((int)offset), word);
        }
        return (snapshot, request, inputs.UserData.ToArray(), table);
    }

    [Fact]
    public void WorkgroupTablePreservesLoadsAndMapsRealOffsetsToDifferentImages()
    {
        var plan = WorkgroupImagePlan();
        Assert.NotNull(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!.Workgroup);
        Assert.All(plan.IndirectImages, access => Assert.True(access.KeyIsAddressOffset));
        Assert.All(plan.Memory.Entries.Where(memory => memory.Pc == 8), memory => Assert.False(memory.PlanningOnly));
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        Assert.True(ResourceMaterializer.Materialize(plan, WorkgroupImageInputs(), ref snapshot, ref specialization, out var failure), failure.ToString());
        Assert.True(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!.Workgroup!
            .TryEvaluate(plan, WorkgroupImageInputs(), out var keys, out var descriptors));
        Assert.Equal(new uint[] { 0, 64 }, keys);
        Assert.Equal(new uint[] { 0x2000, 0x3000 }, descriptors.Select(words => words[0]));
        Assert.Equal(new uint[] { 0x2000, 0x3000 }, snapshot.Images.Select(words => words[0]));
        Assert.Single(snapshot.Samplers);
        Assert.Equal(0xFFF000u, snapshot.Samplers[0][1]);
    }

    [Theory]
    [InlineData(0u, false, false, false, true)]
    [InlineData(65537u, false, false, false, true)]
    [InlineData(4u, true, false, false, true)]
    [InlineData(4u, false, true, false, true)]
    [InlineData(4u, false, false, true, true)]
    [InlineData(4u, false, false, false, false)]
    [InlineData(5u, false, false, false, true)]
    public void WorkgroupTableDeclinesMissingBoundsAliasDifferentSamplersOrUnreadableRecords(
        uint groups, bool alias, bool differentSampler, bool unreadable, bool clean)
    {
        var plan = WorkgroupImagePlan();
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        var original = snapshot;
        Assert.False(ResourceMaterializer.Materialize(plan,
            WorkgroupImageInputs(groups, alias, differentSampler, unreadable, clean), ref snapshot, ref specialization));
        Assert.Same(original, snapshot);
    }

    [Fact]
    public void WorkgroupTableDeclinesUnknownWriteAddresses()
    {
        var plan = WorkgroupImagePlan(unknownWrite: true);
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        Assert.False(ResourceMaterializer.Materialize(plan, WorkgroupImageInputs(), ref snapshot, ref specialization));
    }

    [Fact]
    public void WorkgroupTableChecksWritesAcrossDifferentGroups()
    {
        var program = Program([.. WorkgroupImageProgram().Instructions.Where(instruction => instruction.Pc < 24),
            ScalarBufferLoad(24, 0, 4, 4, immediateOffset: 48, dynamicOffsetRegister: 9),
            Image(32, "ImageSampleLz", 16, samplerRegister: 24, dmask: 1), BufferStore(40, 4), EndProgram(48)]);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 8,
            computeSystemRegisters: new Gen5ComputeSystemRegisters(8, null, null, null));
        var proof = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!.Workgroup!;
        ResourceRuntimeInputs Inputs(uint groups)
        {
            var original = WorkgroupImageInputs(groups);
            bool Read(ulong address, out uint word)
            {
                var offset = (uint)(address - 0x1000) % 64;
                if (address is >= 0x1000 and < 0x1080 && offset >= 48)
                {
                    word = offset switch { 48 => address < 0x1040 ? 0x4000u : 0x1000u,
                        52 => 4u << 16, 56 => 16, _ => (20u << 12) | 0xFAC };
                    return true;
                }
                return original.ReadCleanMemory!(address, out word);
            }
            return new() { UserData = original.UserData, ReadMemory = Read, ReadCleanMemory = Read,
                ComputeState = original.ComputeState };
        }
        Assert.True(proof.TryEvaluate(plan, Inputs(2), out _, out _));
        Assert.False(proof.TryEvaluate(plan, Inputs(4), out _, out _));
    }

    [Fact]
    public void WorkgroupTableCacheRebuildsWhenTheActualDispatchGrows()
    {
        var plan = WorkgroupImagePlan();
        var cache = new ResourceMaterializationCache();
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        foreach (var groups in new uint[] { 2, 2, 4 })
        {
            var inputs = WorkgroupImageInputs(groups);
            bool ReadResident(ulong address, Span<byte> bytes, bool clean)
            {
                for (var offset = 0; offset < bytes.Length; offset += 4)
                {
                    if (!inputs.ReadCleanMemory!(address + (uint)offset, out var word)) return false;
                    BitConverter.TryWriteBytes(bytes[offset..], word);
                }
                return true;
            }
            Assert.True(cache.Materialize(plan, inputs, ReadResident, ref snapshot, ref specialization, out _));
            Assert.Equal(groups == 2 ? 1 : 2, snapshot.Images.Length);
        }
        Assert.Equal((1, 2), (cache.Hits, cache.Misses));
    }

    [Fact]
    public void WorkgroupSamplerResourcesAreReusedForRepeatedInstructions()
    {
        var program = Program([.. WorkgroupImageProgram().Instructions.Where(instruction => instruction.Pc < 40),
            Image(40, "ImageSampleLz", 16, samplerRegister: 24, dmask: 1), EndProgram(48)]);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 8,
            computeSystemRegisters: new Gen5ComputeSystemRegisters(8, null, null, null));
        Assert.Single(plan.Info.Samplers);
        Assert.Single(plan.Info.Images);
    }

    [Fact]
    public void WorkgroupTableKeepsGdsWritesInTheirSeparateAddressSpace()
    {
        var program = Program([.. WorkgroupImageProgram().Instructions.Where(instruction => instruction.Pc < 40),
            DataShareWrite(40, gds: true), EndProgram(48)]);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 8,
            computeSystemRegisters: new Gen5ComputeSystemRegisters(8, null, null, null));
        Assert.Contains(plan.Memory.Entries, memory => memory.Kind == MemoryResourceKind.GlobalDataShare && !memory.PlanningOnly);
        ResourceSnapshot snapshot = new(); ResourceSpecialization specialization = new();
        Assert.True(ResourceMaterializer.Materialize(plan, WorkgroupImageInputs(), ref snapshot, ref specialization));
    }

    [Fact]
    public void WorkgroupTableDoesNotInventABoundForAnotherAxis()
    {
        var program = WorkgroupImageProgram() with
        {
            Instructions = WorkgroupImageProgram().Instructions.Select(instruction => instruction.Pc == 0
                ? Sop2(0, "SAddU32", 10, Gen5Operand.Scalar(8), Gen5Operand.Scalar(9))
                : instruction.Pc == 4 ? Sop2(4, "SMulI32", 9, Gen5Operand.Scalar(10), Operand(64)) : instruction).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 8,
            computeSystemRegisters: new Gen5ComputeSystemRegisters(8, 9, null, null));
        Assert.Null(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage);
    }

    private static Gen5ShaderProgram FiniteLaneReadProgram(bool unknownLane = false, bool partialWrite = false,
        bool loopWrite = false, bool wrongRestore = false, bool unknownPartial = false)
    {
        var prefix = FiniteBufferImageProgram().Instructions.Where(instruction => instruction.Pc < 20).ToList();
        prefix.AddRange([
            Sop1(20, "SMovB64", 106, Gen5Operand.Scalar(126)),
            Vop3(24, "VMadU32U24", 3, Operand(3), Gen5Operand.Vector(1), Gen5Operand.Vector(2)),
            Vopc(32, "VCmpxEqU32", Operand(0), 0),
            Sop1(36, "SMovB64", 126, Gen5Operand.Scalar(wrongRestore ? 104u : 106u)),
            Sop1(40, "SMovB64", 68, Gen5Operand.Scalar(106)),
            Sop1(44, "SFF1I32B64", 17, Gen5Operand.Scalar(68)),
            new(48, Gen5ShaderEncoding.Vop3, "VReadlaneB32", [0u, 0u],
                [Gen5Operand.Vector(3), Gen5Operand.Scalar(unknownLane ? 12u : 17u), Gen5Operand.Scalar(0)],
                [Gen5Operand.Scalar(17)], new Gen5Vop3Control(0, 0, 0, false, 0, null)),
            new(56, Gen5ShaderEncoding.Vop3, "VCmpEqU32", [0u, 0u],
                [Gen5Operand.Scalar(17), Gen5Operand.Vector(3)], [Gen5Operand.Scalar(70)], new Gen5Vop3Control(0, 0, 0, false, 0, 70)),
            Sop1(64, "SAndSaveexecB64", 64, Gen5Operand.Scalar(70)),
            Branch(68, "SCbranchExecz", 6),
            Sop2(72, "SMulI32", 106, Gen5Operand.Scalar(17), Operand(384)),
            ScalarBufferLoad(76, 0, 16, 8, dynamicOffsetRegister: 106),
            Image(84, "ImageLoad", 16, dmask: 1, vectorAddress: 4),
            loopWrite ? Vop1(92, "VMovB32", 3, Operand(999)) : Sop2(92, "SAndn2B64", 68, Gen5Operand.Scalar(68), Gen5Operand.Scalar(70)),
            Sop1(96, "SMovB64", 126, Gen5Operand.Scalar(64)), Branch(100, "SCbranchScc1", -15), EndProgram(104),
        ]);
        if (partialWrite)
        {
            prefix = prefix.Select(instruction => instruction.Pc >= 36 ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
            prefix.Add(Vop1(36, "VMovB32", 3, unknownPartial ? Gen5Operand.Vector(0) : Operand(999)));
            prefix.Sort((left, right) => left.Pc.CompareTo(right.Pc));
        }
        return Program([.. prefix]);
    }

    [Fact]
    public void FiniteLaneReadWaterfallKeepsTheSelectorOpaqueButProvesItsCandidates()
    {
        var plan = Extract(FiniteLaneReadProgram(), userDataCount: 4);
        var laneReads = plan.Graph.Values.Where(value => value.Kind == ScalarValueKind.FirstLane && value.Payload == 48).ToArray();
        Assert.NotEmpty(laneReads);
        Assert.All(laneReads, value => Assert.False(plan.ValidateRuntimeValue(value)));
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Null(source.ZeroExtentBufferSource);
        Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    [Theory]
    [InlineData(true, false, false, false)]
    [InlineData(false, true, false, false)]
    [InlineData(false, false, true, false)]
    [InlineData(false, false, false, true)]
    public void FiniteLaneReadWaterfallRejectsUnprovenMasksAndModifiedValues(bool unknownLane, bool partialWrite, bool loopWrite, bool wrongRestore)
    {
        var plan = Extract(FiniteLaneReadProgram(unknownLane, partialWrite, loopWrite, wrongRestore, unknownPartial: partialWrite), userDataCount: 4);
        Assert.Null(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [InlineData(true, true)]
    public void RestoredMaskRetainsBothUntouchedAndPartiallyWrittenSelectorValues(bool andSave, bool negated = false)
    {
        var program = FiniteLaneReadProgram(partialWrite: true);
        if (andSave) program = program with { Instructions = program.Instructions.Select(instruction => instruction.Pc == 32
            ? Sop1(32, negated ? "SAndn1SaveexecB64" : "SAndSaveexecB64", 102, Gen5Operand.Scalar(100)) : instruction).ToArray() };
        var plan = Extract(program, userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Equal(new uint[] { 0, 768, 1152, 1920, 999 * 384 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SavedMaskAliasRequiresAnUnchangedMatchingExecRestore(bool overwriteAlias)
    {
        var instructions = FiniteLaneReadProgram().Instructions.Select(instruction =>
            instruction with { Pc = instruction.Pc >= 32 ? instruction.Pc + 8 : instruction.Pc }).ToList();
        instructions[instructions.FindIndex(instruction => instruction.Pc == 20)] = Sop1(20, "SMovB64", 104, Gen5Operand.Scalar(126));
        instructions.Add(Sop1(32, "SMovB64", 126, Gen5Operand.Scalar(104)));
        instructions.Add(Sop1(36, "SMovB64", 106, Gen5Operand.Scalar(104)));
        if (overwriteAlias)
        {
            instructions = instructions.Select(instruction => instruction.Pc >= 36 ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
            instructions.Add(Sop1(36, "SMovB32", 104, Operand(0)));
        }
        var plan = Extract(Program(instructions.OrderBy(instruction => instruction.Pc).ToArray()), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        if (overwriteAlias) Assert.Null(source.IndirectImage);
        else Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    private static Gen5ShaderProgram CapturedMaskLaneProgram(bool wrongRestore = false, bool negated = false)
    {
        var instructions = FiniteLaneReadProgram().Instructions.Select(instruction =>
            instruction with { Pc = instruction.Pc >= 20 ? instruction.Pc + 12 : instruction.Pc }).ToList();
        instructions.AddRange([Sop1(20, negated ? "SAndn1SaveexecB64" : "SAndSaveexecB64", 104, Gen5Operand.Scalar(100)),
            Sop1(24, "SMovB64", 126, Gen5Operand.Scalar(104)), Nop(28)]);
        instructions[instructions.FindIndex(instruction => instruction.Pc == 32)] = Sop1(32, "SMovB64", 106, Gen5Operand.Scalar(104));
        instructions[instructions.FindIndex(instruction => instruction.Pc == 48)] = Sop1(48, "SMovB64", 126, Gen5Operand.Scalar(wrongRestore ? 102u : 104u));
        instructions[instructions.FindIndex(instruction => instruction.Pc == 52)] = Nop(52);
        instructions[instructions.FindIndex(instruction => instruction.Pc == 56)] = Sop1(56, "SFF1I32B64", 17, Gen5Operand.Scalar(104));
        instructions[instructions.FindIndex(instruction => instruction.Pc == 104)] = Sop2(104, "SAndn2B64", 104, Gen5Operand.Scalar(104), Gen5Operand.Scalar(70));
        return Program(instructions.OrderBy(instruction => instruction.Pc).ToArray());
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void InvariantSelectorSurvivesAWaterfallRestoreUnlessItsValueOrSavedMaskChanges(bool changeValue, bool changeMask)
    {
        var plan = Extract(InvariantAfterWaterfallProgram(changeValue, changeMask), userDataCount: 4);
        var source = plan.DescriptorSources[(int)Assert.Single(plan.Info.Images).Source];
        if (changeValue || changeMask) Assert.Null(source.IndirectImage);
        else Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    private static Gen5ShaderProgram InvariantAfterWaterfallProgram(bool changeValue = false, bool changeMask = false)
    {
        var instructions = FiniteLaneReadProgram().Instructions.Where(instruction => instruction.Pc < 104)
            .Select(instruction => instruction.Pc is 72 or 76 ? Nop(instruction.Pc) : instruction.Pc == 84
                ? changeValue ? Vop1(84, "VMovB32", 3, Gen5Operand.Vector(0))
                    : changeMask ? Sop1(84, "SMovB64", 64, Gen5Operand.Scalar(102)) : Nop(84)
                : instruction).ToList();
        instructions.AddRange([Sop1(104, "SMovB64", 126, Gen5Operand.Scalar(64)),
            ReadFirstLane(108, 106, 3), Sop2(112, "SMulI32", 106, Gen5Operand.Scalar(106), Operand(384)),
            ScalarBufferLoad(116, 0, 16, 8, dynamicOffsetRegister: 106),
            Image(124, "ImageLoad", 16, dmask: 1, vectorAddress: 4), EndProgram(132)]);
        return Program(instructions.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RemainingMaskCopyAllowsIndependentInstructionsButRejectsChangedMask(bool changeMask)
    {
        var instructions = FiniteLaneReadProgram().Instructions.Select(instruction => instruction.Pc >= 40
            ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
        instructions.Add(changeMask ? Vopc(40, "VCmpEqU32", Operand(0), 0) : Vop1(40, "VMovB32", 10, Operand(0)));
        var plan = Extract(Program(instructions.OrderBy(instruction => instruction.Pc).ToArray()), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        if (changeMask) Assert.Null(source.IndirectImage);
        else Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    [Fact]
    public void RestoredMaskUnionsValuesWrittenUnderAnArbitraryIntermediateMask()
    {
        var program = FiniteLaneReadProgram(partialWrite: true);
        program = program with { Instructions = program.Instructions.Select(instruction => instruction.Pc == 32
            ? Sop2(32, "SAndn2B64", 126, Gen5Operand.Scalar(100), Gen5Operand.Scalar(102)) : instruction).ToArray() };
        var plan = Extract(program, userDataCount: 4);
        Assert.Equal(new uint[] { 0, 768, 1152, 1920, 999 * 384 }, plan.DescriptorSources[(int)plan.Info.Images[0].Source]
            .IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    [Fact]
    public void SavedMaskProofAllowsAForwardExitThatSkipsTheDescriptorAccess()
    {
        var instructions = FiniteLaneReadProgram().Instructions.Select(instruction => instruction.Pc >= 32
            ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
        instructions.Add(Branch(32, "SCbranchVccz", 18));
        var plan = Extract(Program(instructions.OrderBy(instruction => instruction.Pc).ToArray()), userDataCount: 4);
        var selector = plan.Graph.Values.Last(value => value.Kind == ScalarValueKind.FirstLane && value.Payload == 52);
        Assert.True(ResourceTracker.TryGetStableLaneReadStart(plan.Graph.Program.Instructions,
            plan.Graph.Program.Instructions.ToList().FindIndex(instruction => instruction.Pc == 52), out _));
        Assert.True(IndirectSelectorValues.TryGetConstantValues(plan, selector, out var values));
        Assert.Equal(new uint[] { 0, 2, 3, 5 }, values.Order().ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void CapturedOldExecMaskRequiresAnExactRestoreBeforeTheLaneScan(bool wrongRestore, bool negated = false)
    {
        var plan = Extract(CapturedMaskLaneProgram(wrongRestore, negated), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        if (wrongRestore) Assert.Null(source.IndirectImage);
        else Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    public static Gen5ShaderProgram FiniteBufferImageProgram(bool unknown = false, bool expandExec = false,
        uint sourceSelect = 6, uint multiplier = 3, uint compare = 0, bool splitMultiply = false, bool packedWord = false)
    {
        var fullWord = new Gen5SdwaControl(6, 0, sourceSelect, 6, false, false, 0, 0, 0, false, null);
        return Program(
            Vopc(0, "VCmpEqU32", Operand(compare), 0),
            new(4, Gen5ShaderEncoding.Vop2, "VCndmaskB32", [0u, 0u],
                [Operand(0), unknown ? Gen5Operand.Vector(0) : Operand(packedWord ? 0x10000u : 1u)], [Gen5Operand.Vector(1)], fullWord),
            new(12, Gen5ShaderEncoding.Vop2, "VCndmaskB32", [0u, 0u],
                [Operand(0), Operand(packedWord ? 0x20000u : 2u)], [Gen5Operand.Vector(2)], fullWord),
            splitMultiply ? Vop3(20, "VMulU32U24", 3, Operand(multiplier), Gen5Operand.Vector(1)) :
                Vop3(20, "VMadU32U24", 3, Operand(multiplier), Gen5Operand.Vector(1), Gen5Operand.Vector(2)),
            packedWord ? new(28, Gen5ShaderEncoding.Vop1, "VMovB32", [0u, 0u],
                [Gen5Operand.Vector(3)], [Gen5Operand.Vector(3)],
                new Gen5SdwaControl(6, 0, 5, 6, false, false, 0, 0, 0, false, null)) :
            expandExec ? Sop1(28, "SMovB64", 126, Gen5Operand.Scalar(12)) : splitMultiply ?
                Vop2(28, "VAddI32", 3, Gen5Operand.Vector(3), Gen5Operand.Vector(2)) : Nop(28),
            ReadFirstLane(32, 106, 3),
            Sop2(36, "SMulI32", 106, Gen5Operand.Scalar(106), Operand(384)),
            ScalarBufferLoad(40, 0, 16, 8, dynamicOffsetRegister: 106),
            Image(48, "ImageLoad", 16, dmask: 1, vectorAddress: 4),
            EndProgram(56));
    }

    internal static (ResourceSnapshot Snapshot, ShaderCompileRequest Request, uint[] Registers) PrepareFiniteBufferImages(uint compare, bool laneRead = false, bool partialSelector = false, bool capturedMask = false, bool splitMultiply = false, bool afterWaterfall = false, bool negated = false, bool packedWord = false)
    {
        var program = FiniteBufferImageProgram(compare: compare) with
        {
            Instructions = FiniteBufferImageProgram(compare: compare, splitMultiply: splitMultiply, packedWord: packedWord).Instructions.Where(instruction => instruction.Pc < 48).Concat([
                Vop1(48, "VMovB32", 4, Operand(0)), Vop1(52, "VMovB32", 5, Operand(0)),
                Image(56, "ImageLoad", 16, dmask: 1, vectorAddress: 4),
                BufferAccess(64, "BufferStoreDword", 8, vectorData: 4), EndProgram(72),
            ]).ToArray(),
        };
        if (laneRead)
        {
            var laneProgram = afterWaterfall ? InvariantAfterWaterfallProgram() : capturedMask ? CapturedMaskLaneProgram(negated: negated) : FiniteLaneReadProgram();
            var imagePc = laneProgram.Instructions.Single(instruction => instruction.Opcode == "ImageLoad").Pc;
            var instructions = laneProgram.Instructions.Select(instruction =>
                instruction with { Pc = instruction.Pc + (instruction.Pc > imagePc ? 16u : 8u) }).ToList();
            instructions[instructions.FindIndex(instruction => instruction.Pc == 8)] = Vopc(8, "VCmpEqU32", Operand(compare), 0);
            var skipIndex = instructions.FindIndex(instruction => instruction.Opcode == "SCbranchExecz");
            instructions[skipIndex] = Branch(instructions[skipIndex].Pc, "SCbranchExecz", (short)(afterWaterfall ? 6 : 8));
            var backIndex = instructions.FindIndex(instruction => instruction.Opcode == "SCbranchScc1");
            instructions[backIndex] = Branch(instructions[backIndex].Pc, "SCbranchScc1", (short)(afterWaterfall ? -15 : -17));
            instructions.AddRange([Vop1(0, "VMovB32", 4, Operand(0)), Vop1(4, "VMovB32", 5, Operand(0)),
                BufferAccess(imagePc + 16, "BufferStoreDword", 8, vectorData: 4)]);
            program = program with { Instructions = instructions.OrderBy(instruction => instruction.Pc).ToArray() };
        }
        if (partialSelector)
        {
            var instructions = program.Instructions.Select(instruction => instruction with
                { Pc = instruction.Pc + (instruction.Pc >= 32 ? 12u : instruction.Pc >= 20 ? 4u : 0u) }).ToList();
            instructions[instructions.FindIndex(instruction => instruction.Pc == 32)] = Vopc(32, "VCmpxEqU32", Operand(compare), 0);
            instructions.AddRange([Sop1(20, "SMovB64", 106, Gen5Operand.Scalar(126)),
                Vop1(36, "VMovB32", 3, Operand(2)), Sop1(40, "SMovB64", 126, Gen5Operand.Scalar(106))]);
            program = program with { Instructions = instructions.OrderBy(instruction => instruction.Pc).ToArray() };
        }
        uint[] registers = [0x1000, 0, 2048, 0, 0, 0, 0, 0, 0, 0, 64, 0];
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 || address >= 0x1800) return false;
            var relative = address - 0x1000;
            word = (relative % 384 / 4) switch
            {
                0 => 0x2000u + (uint)(relative / 384) * 0x100,
                1 => 20u << 20,
                3 => 0xFACu | (9u << 28),
                _ => 0,
            };
            return true;
        }
        var plan = Extract(program, userDataCount: 12);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(registers, readCleanMemory: Read), ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 12),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        return (snapshot, new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 }, registers);
    }

    [Fact]
    public void PackedUpperWordPreservesTheFiniteSelectorDomain()
    {
        var plan = Extract(FiniteBufferImageProgram(packedWord: true), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    [Theory]
    [InlineData(true, 6u)]
    [InlineData(false, 4u)]
    public void ModifiedOrPartialPackedWordDeclinesTheProof(bool signed, uint destinationSelect)
    {
        var program = FiniteBufferImageProgram(packedWord: true);
        program = program with { Instructions = program.Instructions.Select(instruction => instruction.Pc == 28
            ? instruction with { Control = new Gen5SdwaControl(destinationSelect, 0, 5, 6, signed, false, 0, 0, 0, false, null) }
            : instruction).ToArray() };
        var plan = Extract(program, userDataCount: 4);
        Assert.Null(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage);
    }

    [Fact]
    public void FiniteScalarBufferImagesKeepEveryConditionalCandidate()
    {
        var plan = Extract(FiniteBufferImageProgram(), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Null(source.ZeroExtentBufferSource);
        var candidates = Assert.IsType<List<DirectImageCandidate>>(source.IndirectImage!.DirectCandidates);
        Assert.Equal(new uint[] { 0, 768, 1152, 1920 }, candidates.Select(candidate => candidate.Offset).Order().ToArray());
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 || address >= 0x1800) return false;
            var relative = address - 0x1000;
            word = (relative % 384 / 4) switch
            {
                0 => 0x2000u + (uint)(relative / 384) * 0x100u,
                1 => 20u << 20,
                3 => 0xFACu | (9u << 28),
                _ => 0,
            };
            return true;
        }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0, 2048, 0], readCleanMemory: Read),
            ref snapshot, ref specialization));
        Assert.Equal(4, snapshot.Images.Length);
        Assert.Equal(new uint[] { 0x2000, 0x2200, 0x2300, 0x2500 }, snapshot.Images.Select(item => item[0]).Order().ToArray());
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(plan.Graph.Program, 0, 4),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 64, ThreadCountX = 64 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);

        var prior = snapshot;
        Assert.False(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0, 32, 0], readCleanMemory: Read),
            ref snapshot, ref specialization));
        Assert.Same(prior, snapshot);
    }

    [Theory]
    [InlineData(true, false, 6u)]
    [InlineData(false, true, 6u)]
    [InlineData(false, false, 0u)]
    public void FiniteScalarBufferImagesRejectUnknownOrUnprovenLaneValues(bool unknown, bool expandExec, uint select)
    {
        var plan = Extract(FiniteBufferImageProgram(unknown, expandExec, select), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Null(source.IndirectImage);
        Assert.NotNull(source.ZeroExtentBufferSource);
    }

    [Fact]
    public void FiniteSelectorMadUsesOnlyLow24BitsOfMultiplicands()
    {
        var plan = Extract(FiniteBufferImageProgram(multiplier: 0x01000003), userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Equal(new uint[] { 0, 768, 1152, 1920 },
            source.IndirectImage!.DirectCandidates!.Select(candidate => candidate.Offset).Order().ToArray());
    }

    [Fact]
    public void FiniteSamplersPreserveDifferentReadableCandidatesAndRejectUnreadableOnes()
    {
        var program = FiniteBufferImageProgram() with
        {
            Instructions = FiniteBufferImageProgram().Instructions.Where(instruction => instruction.Pc < 48).Concat([
                ScalarBufferLoad(48, 0, 28, 4, immediateOffset: 256, dynamicOffsetRegister: 106),
                Image(56, "ImageSampleLz", 16, 28, dmask: 1, vectorAddress: 4), EndProgram(64),
            ]).ToArray(),
        };
        var plan = Extract(program, userDataCount: 4);
        var source = plan.DescriptorSources[(int)plan.Info.Samplers[0].Source];
        Assert.Null(source.ZeroExtentBufferSource);
        Assert.Equal(4, source.FiniteSamplerSources!.Count);
        var mismatch = false;
        var unreadable = false;
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 || address >= 0x2000) return false;
            var relative = address - 0x1000;
            var component = relative % 384 / 4;
            if (relative / 384 == 5 && component == 64 && unreadable) return false;
            word = component switch
            {
                0 => 0x2000u + (uint)(relative / 384) * 0x100,
                1 => 22u << 20,
                3 => 0xFACu | (9u << 28),
                64 => mismatch && relative / 384 == 5 ? 1u : 0u,
                66 => 0x10,
                _ => 0,
            };
            return true;
        }
        var dirtyReads = 0;
        bool Dirty(ulong address, out uint word) { dirtyReads++; word = 0; return false; }
        var inputs = Inputs([0x1000, 0, 4096, 0], readMemory: Dirty, readCleanMemory: Read);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        Assert.Equal(4, snapshot.Samplers.Length);
        Assert.All(snapshot.Samplers, sampler => Assert.Equal(new uint[] { 0, 0, 0x10, 0 }, sampler));
        Assert.Equal(0, dirtyReads);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 4),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(new ShaderCompileRequest(plan, resources, layout)
            { LocalSizeX = 64, ThreadCountX = 64 }, out _, out var error), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(new ShaderCompileRequest(plan, resources, layout)
            { LocalSizeX = 64, ThreadCountX = 64 }, out _, out var metalError), metalError);
        var prior = snapshot;
        mismatch = true;
        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        Assert.Equal(1u, snapshot.Samplers[^1][0]);
        Assert.Equal(0u, snapshot.Samplers[0][0]);
        prior = snapshot;
        mismatch = false;
        unreadable = true;
        Assert.False(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        Assert.Same(prior, snapshot);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MixedUintFloatCandidatesRequireLegalSamplersAndKeepTheirNativeTypes(bool linear)
    {
        var program = FiniteBufferImageProgram() with
        {
            Instructions = FiniteBufferImageProgram().Instructions.Where(instruction => instruction.Pc < 48).Concat([
                ScalarBufferLoad(48, 0, 28, 4, immediateOffset: 256, dynamicOffsetRegister: 106),
                Image(56, "ImageSampleLz", 16, 28, dmask: 1, vectorAddress: 4), EndProgram(64),
            ]).ToArray(),
        };
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 || address >= 0x2000) return false;
            var offset = address - 0x1000;
            word = (offset % 384 / 4) switch
            {
                0 => 0x2000u + (uint)(offset / 384) * 256,
                1 => (offset / 384 == 5 ? 22u : 20u) << 20,
                3 => 0xFAC | (9u << 28),
                66 => linear ? 0x05500000u : 0x05000000u,
                _ => 0,
            };
            return true;
        }
        var plan = Extract(program, userDataCount: 4);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var priorSnapshot = snapshot;
        var priorSpecialization = specialization;
        Assert.Equal(!linear, ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0, 4096, 0], readCleanMemory: Read),
            ref snapshot, ref specialization));
        if (linear)
        {
            Assert.Same(priorSnapshot, snapshot);
            Assert.Same(priorSpecialization, specialization);
            return;
        }
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        Assert.Contains(resources.Info.Images, image => image.NumericClass == ImageNumericClass.Uint);
        Assert.Contains(resources.Info.Images, image => image.NumericClass == ImageNumericClass.Float);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 4),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 64, ThreadCountX = 64 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out error), error);
    }

    internal static Gen5ShaderProgram CreateWaveIndexedDescriptorProgram()
    {
        return Program(
            ScalarLoad(0, 0, 16, immediateOffset: 0x80),
            Sop1(8, "SFF1I32B32", 18, Gen5Operand.Scalar(16)),
            Sop2(12, "SMulI32", 106, Gen5Operand.Scalar(18), new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x90)),
            MoveVectorFromScalar(16, 34, 18),
            Vop2(20, "VLshlrevB32", 15, Operand(4), Gen5Operand.Vector(34)),
            Vop3(24, "VLshlAddU32", 16, Gen5Operand.Vector(15), Operand(3), Gen5Operand.Vector(15)),
            Sop1(32, "SBitset0B32", 16, Gen5Operand.Scalar(18)),
            Vop2(36, "VAddI32", 15, new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x40), Gen5Operand.Vector(16)),
            GlobalMemory(40, "GlobalLoadDword", 0, 15, 22, 0),
            ReadFirstLane(48, 106, 22),
            Sop2(52, "SLshlB32", 106, Gen5Operand.Scalar(106), Operand(5)),
            ScalarLoad(56, 0, 4, 8, immediateOffset: 0x100, dynamicOffsetRegister: 106),
            Image(64, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            EndProgram(72));
    }

    internal static Gen5ShaderProgram CreateWaveIndexedReadLaneProgram(bool selfAddressed, bool restoreExec = true)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            ScalarLoad(0, 0, 16, immediateOffset: 0x80),
            Sop1(8, "SFF1I32B32", 18, Gen5Operand.Scalar(16)),
            Sop2(12, "SMulI32", 19, Gen5Operand.Scalar(18), new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x90)),
            MoveVectorFromScalar(16, 34, 18),
            Vop2(20, "VLshlrevB32", 15, Operand(4), Gen5Operand.Vector(34)),
            Vop3(24, "VLshlAddU32", selfAddressed ? 15u : 16u, Gen5Operand.Vector(15), Operand(3), Gen5Operand.Vector(15)),
            Sop2(32, "SLshlB32", 20, Operand(1), Gen5Operand.Scalar(18)),
            Sop2(36, "SXorB32", 16, Gen5Operand.Scalar(20), Gen5Operand.Scalar(16)),
            Vop2(40, "VAddI32", 15, new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x40), Gen5Operand.Vector(selfAddressed ? 15u : 16u)),
            GlobalMemory(44, "GlobalLoadDword", 0, 15, 22, 0),
            Sop1(52, "SMovB64", 12, Gen5Operand.Scalar(126)),
            Sop1(56, "SFF1I32B64", 24, Gen5Operand.Scalar(12)),
            new(60, Gen5ShaderEncoding.Vop3, "VReadlaneB32", [0u, 0u],
                [Gen5Operand.Vector(22), Gen5Operand.Scalar(24), Gen5Operand.Scalar(24)], [Gen5Operand.Scalar(106)], null),
            new(68, Gen5ShaderEncoding.Vop3, "VCmpEqU32", [0u, 0u],
                [Gen5Operand.Scalar(106), Gen5Operand.Vector(22)], [Gen5Operand.Scalar(14)], new Gen5Vop3Control(0, 0, 0, false, 0, 14)),
            Sop1(76, "SAndSaveexecB64", 28, Gen5Operand.Scalar(14)),
            Branch(80, "SCbranchExecz", 8),
            Sop2(84, "SLshlB32", 106, Gen5Operand.Scalar(106), Operand(5)),
            Sop2(88, "SAddI32", 107, Gen5Operand.Scalar(106), Operand(16)),
            ScalarLoad(92, 0, 4, 4, immediateOffset: 0x100, dynamicOffsetRegister: 106),
            ScalarLoad(100, 0, 8, 4, immediateOffset: 0x100, dynamicOffsetRegister: 107),
            Image(108, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            Sop2(116, "SAndn2B64", 12, Gen5Operand.Scalar(12), Gen5Operand.Scalar(14)),
            restoreExec ? Sop1(120, "SMovB64", 126, Gen5Operand.Scalar(28)) : Nop(120),
            Branch(124, "SCbranchScc1", -18),
            EndProgram(128),
        };
        return Program([.. instructions]);
    }

    public static Gen5ShaderProgram CreateProgram(uint mask = 1, bool split = true, bool bitScan = true)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            Vop2(0, "VAddU32", 12, Operand(mask), Gen5Operand.Vector(0)),
            ReadFirstLane(4, 18, 12),
            Sop1(8, bitScan ? "SFF1I32B32" : "SMovB32", 19, Gen5Operand.Scalar(18)),
            Sop2(12, "SLshlB32", 106, Gen5Operand.Scalar(19), Operand(5)),
            new(16, Gen5ShaderEncoding.Sopk, "SAddkI32", [0xB7EA0158], [new(Gen5OperandKind.LiteralConstant, 344)], [Gen5Operand.Scalar(106)], null),
            Sop2(20, "SAddI32", 107, Gen5Operand.Scalar(106), Operand(16)),
            ScalarLoad(24, 0, 4, split ? 4u : 8u, dynamicOffsetRegister: 106),
        };
        if (split) instructions.Add(ScalarLoad(32, 0, 8, 4, dynamicOffsetRegister: 107));
        instructions.AddRange([
            MoveScalar(40, 106, 999),
            MoveVector(44, 1, 0), MoveVector(48, 2, 0),
            Image(52, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            MoveScalar(60, 20, 0), MoveScalar(64, 21, 0), MoveScalar(68, 22, 64), MoveScalar(72, 23, 0),
            BufferAccess(76, "BufferStoreDword", 20, vectorData: 4), EndProgram(84),
        ]);
        return Program([.. instructions]);
    }

    public static bool ReadDescriptor(ulong address, out uint word)
    {
        word = 0;
        if (address < 0x1000 + 312 || address >= 0x1000 + 344 + 32 * 32) return false;
        var offset = address - 0x1000 - 312;
        var record = offset / 32;
        var component = offset % 32 / 4;
        word = component switch
        {
            0 => (record & 1) == 0 ? 0x2000u : 0x1000u,
            1 => 20u << 20,
            3 => 0xFACu | (9u << 28),
            _ => 0,
        };
        return true;
    }

    public static Gen5ShaderProgram CreateGuardedProgram(uint mask = 1, bool split = true)
    {
        var program = CreateProgram(mask, split);
        return program with
        {
            Instructions = program.Instructions.Where(instruction => instruction.Pc < 8)
                .Concat([
                    Sopc(8, "SCmpLgU32", Operand(0), Gen5Operand.Scalar(18)),
                    Branch(12, "SCbranchScc0", 19),
                ])
                .Concat(program.Instructions.Where(instruction => instruction.Pc >= 8)
                    .Select(instruction => instruction with { Pc = instruction.Pc + 8 })).ToArray(),
        };
    }

    public static (ShaderResourcePlan Plan, ResourceSnapshot Snapshot, ShaderCompileRequest Request) PrepareDirect(uint mask = 1, bool split = true, bool guarded = false)
    {
        var program = guarded ? CreateGuardedProgram(mask, split) : CreateProgram(mask, split);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadDescriptor), ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        return (plan, snapshot, new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 });
    }

    [Fact]
    public void WaveIndexedDescriptorTableMaterializesOnlyActiveMaskKeys()
    {
        var plan = ShaderResourcePlan.Extract(CreateWaveIndexedDescriptorProgram(), ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense);
        Assert.Equal(0u, selector.KeyBound);
        Assert.Equal(new WaveIndexedImageSelector(0x80, 0x40, 0x90), selector.WaveIndexed);

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadWaveIndexedMemory), ref snapshot, ref specialization));
        Assert.Equal(2, snapshot.Images.Length);
    }

    [Fact]
    public void CandidateWithAnUnsupportedFormatIsMaterializedAsNull()
    {
        var plan = ShaderResourcePlan.Extract(CreateWaveIndexedDescriptorProgram(), ShaderStage.Compute, Hash, 0, 2);
        bool Read(ulong address, out uint word)
        {
            var success = ReadWaveIndexedMemory(address, out word);
            if (address == 0x1000 + 0x100 + 5 * 32 + 4) word = 139u << 20;
            return success;
        }

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(2, snapshot.Images.Length);
        Assert.Single(snapshot.Images, image => image.All(word => word == 0));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReadLaneWaterfallSelectsWaveIndexedDescriptors(bool selfAddressed)
    {
        var program = CreateWaveIndexedReadLaneProgram(selfAddressed);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense, "dense");
        Assert.Equal(0x100u, selector.TableOffset);
        Assert.Equal(new WaveIndexedImageSelector(0x80, 0x40, 0x90), selector.WaveIndexed);
        Assert.True(Assert.Single(plan.IndirectImages).KeyIsAddressOffset, "address-offset key");

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadWaveIndexedMemory), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(2, snapshot.Images.Length);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 64, ThreadCountX = 64 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);
    }

    [Fact]
    public void ReadLaneWithoutRestoredExecutionIsNotWaveIndexed()
    {
        Assert.Throws<ResourcePlanException>(() =>
            ShaderResourcePlan.Extract(CreateWaveIndexedReadLaneProgram(selfAddressed: true, restoreExec: false), ShaderStage.Compute, Hash, 0, 2));
    }

    private static bool ReadWaveIndexedMemory(ulong address, out uint word)
    {
        word = 0;
        if (address == 0x1000 + 0x80)
        {
            word = (1u << 1) | (1u << 4);
            return true;
        }

        if (address == 0x1000 + 0x40 + 0x90 || address == 0x1000 + 0x40 + 4 * 0x90)
        {
            word = address == 0x1000 + 0x40 + 0x90 ? 2u : 5u;
            return true;
        }

        if (address < 0x1000 + 0x100 || address >= 0x1000 + 0x100 + 6 * 32)
            return false;
        var relative = address - 0x1000 - 0x100;
        var record = relative / 32;
        if (record is not (2 or 5))
            return false;
        word = (relative % 32 / 4) switch
        {
            0 => (record & 1) == 0 ? 0x2000u : 0x1000u,
            1 => 20u << 20,
            3 => 0xFACu | (9u << 28),
            _ => 0,
        };
        return true;
    }

    public static (ResourceSnapshot Snapshot, ShaderCompileRequest Request) PrepareMixedDimensions(uint mask, bool arrayFirst)
    {
        var program = CreateGuardedProgram(mask);
        program = program with
        {
            Instructions = program.Instructions.Select(instruction => instruction.Pc switch
            {
                48 => MoveVector(48, 3, 1),
                60 => Image(60, "ImageLoad", 4, dimension: 5, dmask: 1, vectorAddress: 1),
                _ => instruction,
            }).ToArray(),
        };
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 + 344 || address >= 0x1000 + 344 + 32 * 32) return false;
            var offset = address - 0x1000 - 344;
            var record = offset / 32;
            if (record is not (0 or 31)) return true;
            var array = (record == 0) == arrayFirst;
            word = (offset % 32 / 4) switch
            {
                0 => array ? 0x1000u : 0x2000u,
                1 => 20u << 20,
                3 => 0xFACu | ((array ? 13u : 9u) << 28),
                4 => array ? 1u : 0u,
                _ => 0,
            };
            return true;
        }
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        return (snapshot, new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MixedDimensionsRetainEveryCandidateAndUseSeparateBindings(bool arrayFirst)
    {
        var (snapshot, request) = PrepareMixedDimensions(1, arrayFirst);
        Assert.Equal(3, snapshot.Images.Length);
        var images = request.Resources.Info.Images;
        Assert.Equal(3, images[0].IndirectResources.Count);
        Assert.Contains(images, image => image.Dimension == ImageDimension.Dim2D);
        Assert.Contains(images, image => image.Dimension == ImageDimension.Dim2DArray);
        var mapping = (int)images[0].IndirectMappingOffset;
        Assert.Equal(32u, snapshot.FlattenedResourceTable[mapping]);
        Assert.Equal(344u + 31 * 32, snapshot.FlattenedResourceTable[mapping + 63]);
        Assert.Equal(2u, snapshot.FlattenedResourceTable[mapping + 64]);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out error));
        Assert.Contains("not supported on Metal", error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DirectTableRetainsAllBitScanResultsAndCompiles(bool split)
    {
        var (plan, snapshot, request) = PrepareDirect(split: split);
        Assert.Single(plan.IndirectImages);
        Assert.True(plan.IndirectImages[0].KeyIsAddressOffset);
        Assert.False(plan.Info.UsesDeviceAddresses);
        Assert.Empty(plan.DynamicReads);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.Equal(33, selector.DirectCandidates!.Count);
        Assert.Contains(selector.DirectCandidates, candidate => candidate.Offset == 312);
        Assert.Contains(selector.DirectCandidates, candidate => candidate.Offset == 344 + 31 * 32);
        Assert.Equal(2, snapshot.Images.Length);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    public void DirectImageCapacityPreservesTheLimitAndPublishedState(int distinctCount)
    {
        var (plan, snapshot, _) = PrepareDirect();
        var specialization = new ResourceSpecialization();
        var previousSnapshot = snapshot;
        var previousSpecialization = specialization;
        bool Read(ulong address, out uint word)
        {
            var success = ReadDescriptor(address, out word);
            if (success && (address - 0x1000 - 312) % 32 == 0)
                word = 0x2000 + (uint)(((address - 0x1000 - 312) / 32) % (ulong)distinctCount);
            return success;
        }

        var success = ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read),
            ref snapshot, ref specialization, out var failure);
        Assert.Equal(distinctCount <= ShaderResourceInfo.MaxImages, success);
        if (success)
        {
            Assert.Equal(ResourceMaterializationFailure.None, failure);
            Assert.Equal(distinctCount, snapshot.Images.Length);
        }
        else
        {
            Assert.Equal(ResourceMaterializationFailure.ImageCapacityExceeded, failure);
            Assert.Same(previousSnapshot, snapshot);
            Assert.Same(previousSpecialization, specialization);
        }
    }

    [Fact]
    public void UnboundedDirectTableIsStillRejected()
    {
        Assert.Throws<ResourcePlanException>(() => ShaderResourcePlan.Extract(CreateProgram(bitScan: false), ShaderStage.Compute, Hash, 0, 2));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void NonZeroGuardExcludesTheUnreachableDescriptor(bool split, bool equality)
    {
        var program = CreateGuardedProgram(split: split);
        if (equality)
            program = program with
            {
                Instructions = program.Instructions.Select(instruction => instruction.Pc switch
                {
                    8 => Sopc(8, "SCmpEqI32", Gen5Operand.Scalar(18), Operand(0)),
                    12 => Branch(12, "SCbranchScc1", 19),
                    _ => instruction,
                }).ToArray(),
            };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense);
        Assert.Equal(32u, selector.KeyBound);
        Assert.Equal(344u, selector.TableOffset);
        Assert.True(Assert.Single(plan.IndirectImages).KeyIsAddressOffset);
        bool Read(ulong address, out uint word)
        {
            Assert.True(address >= 0x1000 + 344);
            return ReadDescriptor(address, out word);
        }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization));
        var (_, _, request) = PrepareDirect(split: split, guarded: true);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    [Theory]
    [InlineData(600u, 8u)]
    [InlineData(444u, 3u)]
    [InlineData(312u, 32u)]
    [InlineData(1400u, 32u)]
    public void DenseTableEndsBeforeASamplerLoadedFromTheSameBase(uint samplerOffset, uint expectedBound)
    {
        var program = CreateGuardedProgram();
        program = program with
        {
            Instructions = program.Instructions.Where(instruction => instruction.Pc != 52).Select(instruction => instruction.Pc switch
            {
                48 => ScalarLoad(48, 0, 12, 4, immediateOffset: (int)samplerOffset),
                60 => Image(60, "ImageSample", 4, samplerRegister: 12, dmask: 1, vectorAddress: 1),
                _ => instruction,
            }).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense);
        Assert.Equal(344u, selector.TableOffset);
        Assert.Equal(expectedBound, selector.KeyBound);

        var highest = 0ul;
        bool Read(ulong address, out uint word)
        {
            if (address >= 0x1000 + samplerOffset && address < 0x1000 + samplerOffset + 16)
            {
                word = 0;
                return true;
            }

            if (address >= 0x1000 + 344) highest = Math.Max(highest, address);
            return ReadDescriptor(address, out word);
        }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], Read, Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.True(highest < 0x1000 + 344 + expectedBound * 32, $"read 0x{highest:X} beyond the table");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoopMustRecheckTheBitScanInput(bool bypassGuard)
    {
        var program = CreateGuardedProgram();
        program = program with
        {
            Instructions = program.Instructions.Where(instruction => instruction.Pc < 92)
                .Select(instruction => instruction.Pc == 12 ? Branch(12, "SCbranchScc0", 21) : instruction)
                .Concat([
                    ReadFirstLane(92, 18, 13),
                    Branch(96, "SBranch", bypassGuard ? (short)-21 : (short)-23),
                    EndProgram(100),
                ]).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        if (bypassGuard)
        {
            Assert.Equal(33, selector.DirectCandidates!.Count);
            Assert.Contains(selector.DirectCandidates, candidate => candidate.Offset == 312);
        }
        else
        {
            Assert.True(selector.Dense);
            Assert.Equal(32u, selector.KeyBound);
            Assert.Equal(344u, selector.TableOffset);
        }
    }

    [Theory]
    [InlineData("opposite-branch")]
    [InlineData("different-register")]
    [InlineData("guard-bypass")]
    [InlineData("condition-write")]
    [InlineData("input-write")]
    public void UnprovenGuardRetainsTheZeroInputDescriptor(string variation)
    {
        var program = CreateGuardedProgram();
        var instructions = program.Instructions.ToList();
        switch (variation)
        {
            case "opposite-branch":
                instructions[instructions.FindIndex(instruction => instruction.Pc == 12)] = Branch(12, "SCbranchScc1", 19);
                break;
            case "different-register":
                instructions[instructions.FindIndex(instruction => instruction.Pc == 8)] =
                    Sopc(8, "SCmpLgU32", Operand(0), Gen5Operand.Scalar(17));
                break;
            case "guard-bypass":
                instructions[0] = Branch(0, "SCbranchScc1", 3);
                break;
            case "condition-write":
                instructions = instructions.Select(instruction => instruction.Pc >= 12
                    ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
                instructions.Add(Sopc(12, "SCmpEqU32", Operand(0), Operand(0)));
                break;
            case "input-write":
                instructions = instructions.Select(instruction => instruction.Pc >= 16
                    ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
                instructions[instructions.FindIndex(instruction => instruction.Pc == 12)] = Branch(12, "SCbranchScc0", 20);
                instructions.Add(ReadFirstLane(16, 18, 13));
                break;
        }
        program = program with { Instructions = instructions.OrderBy(instruction => instruction.Pc).ToArray() };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var candidates = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!.DirectCandidates!;
        Assert.Equal(33, candidates.Count);
        Assert.Contains(candidates, candidate => candidate.Offset == 312);
    }

    [Fact]
    public void DescriptorWordsUsedByArithmeticAreNotRemoved()
    {
        var program = CreateProgram();
        program = program with
        {
            Instructions = program.Instructions.Select(instruction => instruction.Pc == 40
                ? Vop1(40, "VMovB32", 15, Gen5Operand.Scalar(4)) : instruction).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        Assert.Single(plan.IndirectImages);
        Assert.True(plan.Info.UsesDeviceAddresses);
        Assert.True(plan.Memory.TryGetIndex(24, 0, out var memoryIndex));
        Assert.False(plan.Memory[memoryIndex].PlanningOnly);
    }

    [Fact]
    public void DirectTableSlotWithAnUnsupportedFormatIsMaterializedAsNull()
    {
        var program = CreateProgram();
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var unsupported = true;
        bool Read(ulong address, out uint word)
        {
            var success = ReadDescriptor(address, out word);
            if (unsupported && address == 0x1000 + 344 + 3 * 32 + 4) word = 139u << 20;
            return success;
        }

        var clean = new ResourceSnapshot();
        var cleanSpecialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadDescriptor), ref clean, ref cleanSpecialization));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(clean.Images.Length + 1, snapshot.Images.Length);
        Assert.Single(snapshot.Images, image => image.All(word => word == 0));
        unsupported = false;
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out failure), $"materialize {failure}");
        Assert.Equal(clean.Images.Length, snapshot.Images.Length);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void FailedReadOrIncompatibleCandidateDoesNotPublish(bool incompatible, bool guarded)
    {
        var (plan, original, _) = PrepareDirect(guarded: guarded);
        var snapshot = original;
        var specialization = new ResourceSpecialization();
        var originalSpecialization = specialization;
        bool Read(ulong address, out uint word)
        {
            var success = ReadDescriptor(address, out word);
            if (address == 0x1000 + 344 + 12)
            {
                if (!incompatible) return false;
                word = (word & 0x0FFFFFFF) | (11u << 28);
            }
            return success;
        }
        Assert.False(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization,
            out var failure));
        Assert.Equal(incompatible ? ResourceMaterializationFailure.IncompatibleImageCandidates : ResourceMaterializationFailure.Other, failure);
        Assert.Same(original, snapshot);
        Assert.Same(originalSpecialization, specialization);
    }
}
