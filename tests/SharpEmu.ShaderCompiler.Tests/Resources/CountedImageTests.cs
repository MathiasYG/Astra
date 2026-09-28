// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class CountedImageTests
{
    private const uint Stride = 0x178;

    private static Gen5ShaderProgram CountedProgram(bool clearExec = true, bool expandExec = false)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            MoveScalar(0, 8, 0),
            Sop2(4, "SMulI32", 10, Gen5Operand.Scalar(8), Operand(Stride)),
            Sop2(8, "SAddU32", 11, Gen5Operand.Scalar(10), Operand(0x1B8)),
            Sopc(12, "SCmpLtU32", Gen5Operand.Scalar(8), Gen5Operand.Scalar(9)),
            clearExec ? Sop2(16, "SCselectB64", 36, Gen5Operand.Scalar(126), Operand(0)) : Nop(16),
            clearExec ? Sop1(20, "SMovB64", 126, Gen5Operand.Scalar(36)) : Nop(20),
            expandExec ? Sop1(24, "SMovB64", 126, Operand(1)) : Nop(24),
            Branch(28, "SCbranchExecz", 8),
            ScalarLoad(32, 0, 40, count: 4, immediateOffset: 0x2B8, dynamicOffsetRegister: 10),
            ScalarLoad(40, 0, 32, count: 8, dynamicOffsetRegister: 11),
            Image(48, "ImageSampleLz", 32, 40),
            Sop2(56, "SAddU32", 8, Gen5Operand.Scalar(8), Operand(1)),
            Branch(60, "SBranch", -15),
            EndProgram(64),
        };
        return Program([.. instructions]);
    }

    private static uint[] UserData(uint count)
    {
        var data = new uint[16];
        data[0] = 0x1000;
        data[8] = 0;
        data[9] = count;
        return data;
    }

    private static Gen5ShaderProgram ScalarBranchTableProgram(bool skipLoadsOnFalse = true, bool incrementFromZero = true)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            MoveScalar(0, 8, incrementFromZero ? 0u : 0xFFFFFFFFu),
            Sop2(4, "SLshlB32", 10, Gen5Operand.Scalar(8), Operand(5)),
            Sopc(8, "SCmpLtI32", Gen5Operand.Scalar(8), Gen5Operand.Scalar(9)),
            Branch(12, skipLoadsOnFalse ? "SCbranchScc0" : "SCbranchScc1", 8),
            ScalarLoad(16, 0, 40, count: 4, immediateOffset: 0x2B8),
            ScalarLoad(24, 0, 32, count: 8, dynamicOffsetRegister: 10),
            Image(32, "ImageSampleLz", 32, 40),
            Sop2(40, "SAddI32", 8, Gen5Operand.Scalar(8), Operand(1)),
            Branch(44, "SBranch", -11),
            EndProgram(48),
        };
        return Program([.. instructions]);
    }

    [Fact]
    public void SignedScalarBranchGuardsShiftedDescriptorTable()
    {
        var plan = Extract(ScalarBranchTableProgram());
        var image = Assert.Single(plan.Info.Images);
        var selector = Assert.IsType<IndirectImageSelector>(plan.DescriptorSources[(int)image.Source].IndirectImage);
        Assert.Equal(32u, selector.EntryStride);
        Assert.NotNull(selector.RuntimeKeyBound);

        var memory = ResourceTrackerTests.LinearMemory();
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        ResourceTrackerTests.WriteImage(memory, 0x1000, descriptor);
        ResourceTrackerTests.WriteImage(memory, 0x1000 + 32, descriptor);
        for (uint dword = 0; dword < 4; dword++)
            memory.At(0x1000 + 0x2B8 + dword * 4) = 0;
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(UserData(2), readMemory: memory.Read,
                readCleanMemory: memory.Read),
            ref snapshot, ref specialization));
        Assert.Equal(descriptor, Assert.Single(snapshot.Images));
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void ScalarBranchMustSkipReadsAndStartAtZero(bool skipsOnFalse, bool startsAtZero)
    {
        var error = Assert.Throws<ResourcePlanException>(() =>
            Extract(ScalarBranchTableProgram(skipsOnFalse, startsAtZero)));
        Assert.Contains("not a valid runtime value", error.Message);
    }

    private static TestWordMemory Table(uint count, bool varyingSampler = false)
    {
        var memory = ResourceTrackerTests.LinearMemory();
        var image = ResourceTrackerTests.ImageDescriptor();
        for (uint entry = 0; entry < count; entry++)
        {
            ResourceTrackerTests.WriteImage(memory, 0x1000 + 0x1B8 + entry * Stride, image);
            memory.At(0x1000 + 0x2B8 + entry * Stride) = varyingSampler && entry == 1 ? 1u : 0u;
        }
        return memory;
    }

    [Fact]
    public void GuardedStridedTablesUseTheRuntimeCount()
    {
        var plan = Extract(CountedProgram());
        var image = Assert.Single(plan.Info.Images);
        var sampler = Assert.Single(plan.Info.Samplers);
        var selector = Assert.IsType<IndirectImageSelector>(plan.DescriptorSources[(int)image.Source].IndirectImage);
        Assert.Equal(Stride, selector.EntryStride);
        Assert.NotNull(selector.RuntimeKeyBound);
        Assert.NotNull(plan.DescriptorSources[(int)sampler.Source].CountedSampler);
        Assert.Single(plan.IndirectImages);

        var memory = Table(3);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(UserData(3), readCleanMemory: memory.Read),
            ref snapshot, ref specialization));
        Assert.Equal(ResourceTrackerTests.ImageDescriptor(), Assert.Single(snapshot.Images));
        Assert.Equal([0u, 0u, 0u, 0u], Assert.Single(snapshot.Samplers));
    }

    [Fact]
    public void ZeroCountNeedsNoDescriptorReads()
    {
        var plan = Extract(CountedProgram());
        var reads = 0;
        bool Reject(ulong address, out uint word) { reads++; word = 0; return false; }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(UserData(0), readCleanMemory: Reject),
            ref snapshot, ref specialization));
        Assert.Equal(0, reads);
        Assert.All(Assert.Single(snapshot.Images), word => Assert.Equal(0u, word));
    }

    [Fact]
    public void DifferentReachableSamplersFailWithoutPublishing()
    {
        var plan = Extract(CountedProgram());
        var memory = Table(2, varyingSampler: true);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.False(ResourceMaterializer.Materialize(plan, Inputs(UserData(2), readCleanMemory: memory.Read),
            ref snapshot, ref specialization));
        Assert.Empty(snapshot.Images);
    }

    [Fact]
    public void DistinctReachableImagesCompileWithTheIndexMapping()
    {
        var plan = Extract(CountedProgram());
        var memory = Table(2);
        memory.At(0x1000 + 0x1B8 + Stride) += 0x100;
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(UserData(2), readCleanMemory: memory.Read),
            ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        Assert.Equal(2, resources.Info.Images.Count);
        Assert.True(resources.Info.Images[0].IndirectSearchIterations > 0);
        var layout = BindingLayout.Allocate(
            resources.Info,
            BindingLayout.CollectUserDataRegisters(plan.Graph.Program, 0, 64),
            false,
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources),
            false);
        var request = new ShaderCompileRequest(plan, resources, layout);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void MissingGuardOrExpandedExecRejectsTheDescriptorPlan(bool clearExec, bool expandExec)
    {
        var error = Assert.Throws<ResourcePlanException>(() => Extract(CountedProgram(clearExec, expandExec)));
        Assert.Contains("not a valid runtime value", error.Message);
    }
}
