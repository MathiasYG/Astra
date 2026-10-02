// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

using System.Buffers.Binary;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ResourceMaterializerTests
{
    private const uint Format32Float = 22;
    private const uint Format32Sint = 21;
    private const uint Format11x2x10Uint = 34;
    private const uint Format8x2Uscaled = 16;
    private const uint ImageType2D = 9;

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(64)]
    [InlineData(65)]
    public void MaterialImageCapacityPreservesTheLimitAndPublishedState(int distinctCount)
    {
        var plan = Extract(ResourceTrackerTests.IndirectImageProgram(false));
        uint[] userData = [0x1000, 224 << 16, (uint)distinctCount, 0, 0x10000, 16 << 16, (uint)distinctCount * 2, 0, 7];
        var memory = new TestWordMemory { Words = new uint[0x11000 / 4] };
        for (var index = 0; index < distinctCount; index++)
        {
            memory.At(0x1000 + (ulong)index * 224 + 4) = (uint)index;
            var descriptor = ResourceTrackerTests.ImageDescriptor();
            descriptor[0] += (uint)index;
            ResourceTrackerTests.WriteImage(memory, 0x10000 + (ulong)index * 32, descriptor);
        }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var previousSnapshot = snapshot;
        var previousSpecialization = specialization;
        var success = ResourceMaterializer.Materialize(plan, Inputs(userData, readCleanMemory: memory.Read),
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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void IncompatibleImageCapturePreservesFailureAndPublishedState(bool captureEnabled)
    {
        var plan = Extract(ResourceTrackerTests.IndirectImageProgram(false));
        uint[] userData = [0x1000, 224 << 16, 2, 0, 0x2000, 16 << 16, 4, 0, 7];
        var memory = ResourceTrackerTests.LinearMemory();
        var first = ResourceTrackerTests.ImageDescriptor();
        var second = first.ToArray();
        second[3] = (second[3] & 0x0FFFFFFF) | (10u << 28);
        ResourceTrackerTests.WriteImage(memory, 0x2000, first);
        ResourceTrackerTests.WriteImage(memory, 0x2020, second);
        memory.At(0x1000 + 36) = 1;
        var inputs = Inputs(userData, readCleanMemory: memory.Read);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var previousSnapshot = snapshot;
        var previousSpecialization = specialization;
        var captures = new List<IndirectImageFailure>();

        Assert.False(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization,
            out var failureKind, captureEnabled ? captures.Add : null));
        Assert.Equal(ResourceMaterializationFailure.IncompatibleImageCandidates, failureKind);
        Assert.Same(previousSnapshot, snapshot);
        Assert.Same(previousSpecialization, specialization);
        if (captureEnabled)
        {
            var failure = Assert.Single(captures);
            Assert.Equal(plan.Info.Images[0].FirstUsePc, failure.InstructionAddress);
            Assert.Equal(0u, failure.RootResource);
            Assert.Equal(0u, failure.ExemplarResource);
            Assert.Equal(1u, failure.IncompatibleResource);
            Assert.Equal(userData[..4], failure.MaterialDescriptor);
            Assert.Equal(userData[4..8], failure.HeapDescriptor);
            Assert.Equal(new uint[] { 0, 1 }, failure.Keys);
            Assert.Equal(new uint[] { 0, 1 }, failure.CandidateIndices);
            Assert.Equal(first, failure.TableDescriptors[0]);
            Assert.Equal(second, failure.TableDescriptors[1]);
            Assert.Equal(second, failure.ImageDescriptors[1]);
            Assert.NotEqual(failure.ImageSpecializations[0].Dimension, failure.ImageSpecializations[1].Dimension);
            Assert.Equal(userData, failure.UserData);
            var diagnostic = Assert.IsType<IndirectSelectorDiagnostic>(failure.SelectorDiagnostic);
            Assert.Equal("full_domain_no_proof", diagnostic.SelectionMode);
            Assert.Null(diagnostic.EvaluationFailure);
            Assert.Empty(diagnostic.MemoryReads);
            Assert.Equal(new SelectorKeyProbe(36, 1), Assert.Single(diagnostic.KeyProbes));
        }
        else
        {
            Assert.Empty(captures);
        }

        ResourceTrackerTests.WriteImage(memory, 0x2020, first);
        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization, captures.Add));
        Assert.Equal(captureEnabled ? 1 : 0, captures.Count);
    }

    [Fact]
    public void FailureCaptureIncludesBothScalarLoadComponentsWithoutExtraReads()
    {
        var original = ResourceTrackerTests.IndirectImageProgram(false);
        var plan = Extract(Program([ScalarLoad(0xF00, 40, 52, count: 2), .. original.Instructions]));
        var registers = new uint[64];
        new uint[] { 0x1000, 224 << 16, 2, 0, 0x2000, 16 << 16, 4, 0 }.CopyTo(registers, 0);
        registers[40] = 0x3000;
        var memory = ResourceTrackerTests.LinearMemory();
        memory.At(0x3000) = 0;
        memory.At(0x3004) = 1;
        memory.At(0x1000 + 36) = 1;
        var first = ResourceTrackerTests.ImageDescriptor();
        var second = first.ToArray();
        second[3] = (second[3] & 0x0FFFFFFF) | (10u << 28);
        ResourceTrackerTests.WriteImage(memory, 0x2000, first);
        ResourceTrackerTests.WriteImage(memory, 0x2020, second);
        var addresses = new List<ulong>();
        bool ReadWord(ulong address, out uint word)
        {
            addresses.Add(address);
            return memory.Read(address, out word);
        }
        var inputs = Inputs(registers, readMemory: ReadWord, readCleanMemory: ReadWord);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.False(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        var baselineAddresses = addresses.ToArray();
        addresses.Clear();
        var captures = new List<IndirectImageFailure>();
        Assert.False(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization, captures.Add));
        Assert.Equal(baselineAddresses, addresses);
        var reads = Assert.Single(captures).ScalarReads.Where(read => read.InstructionAddress == 0xF00).ToArray();
        Assert.Equal(2, reads.Length);
        Assert.Equal(0u, Assert.Single(reads, read => read.ComponentIndex == 0).Value);
        Assert.Equal(1u, Assert.Single(reads, read => read.ComponentIndex == 1).Value);
    }

    private static IEnumerable<Gen5ShaderInstruction> ImageWords(ref uint pc, uint register, uint address, uint format)
    {
        uint[] words = [address, format << 20, 3 | (3 << 14), 0xFAC | (ImageType2D << 28), 0, 0, 0, 0];
        var instructions = new List<Gen5ShaderInstruction>();
        for (uint dword = 0; dword < 8; dword++)
        {
            instructions.Add(MoveScalar(pc, register + dword, words[dword]));
            pc += 8;
        }

        return instructions;
    }

    private static IEnumerable<Gen5ShaderInstruction> SamplerWords(ref uint pc, uint register, uint dword0)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        for (uint dword = 0; dword < 4; dword++)
        {
            instructions.Add(MoveScalar(pc, register + dword, dword == 0 ? dword0 : 0));
            pc += 8;
        }

        return instructions;
    }

    [Fact]
    public void MappedTable_UsesTheDirectReaderByDefault()
    {
        var plan = Extract(Program(
            MoveScalar(0, 4, 0x1000),
            MoveScalar(4, 5, 0),
            ScalarLoad(8, 4, destination: 6),
            EndProgram(16)));
        var cleanReads = 0;
        var inputs = Inputs(
            [],
            readMemory: (ulong address, out uint word) => { word = 0x12345678; return address == 0x1000; },
            readCleanMemory: (ulong address, out uint word) => { cleanReads++; word = 0; return false; });
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        Assert.Equal(0, cleanReads);
        Assert.Equal([0x12345678u], snapshot.FlattenedResourceTable);
    }

    [Fact]
    public void UnbasedFlatCacheHit_Materializes()
    {
        var plan = Extract(Program(GlobalAccess(0, "FlatStoreDword", 0, vectorAddress: 2), EndProgram(8)));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([]), ref snapshot, ref specialization));
        Assert.Empty(snapshot.Buffers);
        Assert.Empty(snapshot.Images);
    }

    [Fact]
    public void FailedMaterialization_LeavesSnapshotAndSpecializationUnchanged()
    {
        var plan = Extract(Program(BufferLoad(0, 0), EndProgram(8)));
        var snapshot = new ResourceSnapshot { UserData = [0xFEEDBEEF] };
        var specialization = new ResourceSpecialization { Buffers = [new BufferSpecialization(7, 0, 0)] };
        var priorSnapshot = snapshot;
        var priorSpecialization = specialization;

        Assert.False(ResourceMaterializer.Materialize(plan, Inputs([]), ref snapshot, ref specialization));
        Assert.Same(priorSnapshot, snapshot);
        Assert.Same(priorSpecialization, specialization);
        Assert.Equal([0xFEEDBEEFu], snapshot.UserData);
        Assert.Equal(7u, specialization.Buffers[0].PackedStride);
    }

    [Fact]
    public void MixedSampler_DuplicatesTheCorrectSnapshot()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.AddRange(ImageWords(ref pc, 16, 0x1000, Format32Float));
        instructions.AddRange(ImageWords(ref pc, 24, 0x2000, Format11x2x10Uint));
        instructions.AddRange(SamplerWords(ref pc, 32, 0x11111111));
        instructions.AddRange(SamplerWords(ref pc, 36, 0x22222222));
        instructions.Add(Image(pc, "ImageSample", 16, 32));
        instructions.Add(Image(pc + 8, "ImageSample", 16, 36));
        instructions.Add(Image(pc + 16, "ImageSample", 24, 36));
        instructions.Add(EndProgram(pc + 24));
        var plan = Extract(Program([.. instructions]));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([]), ref snapshot, ref specialization));
        Assert.Equal(3, snapshot.Samplers.Length);
        Assert.Equal(snapshot.Samplers[1], snapshot.Samplers[2]);
        Assert.NotEqual(snapshot.Samplers[0], snapshot.Samplers[2]);
    }

    [Fact]
    public void ScaledImage_MaterializesAsAFloatSampledImage()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.AddRange(ImageWords(ref pc, 16, 0x1000, Format8x2Uscaled));
        instructions.AddRange(SamplerWords(ref pc, 24, 0));
        instructions.Add(Image(pc, "ImageSample", 16, 24));
        instructions.Add(EndProgram(pc + 8));
        var plan = Extract(Program([.. instructions]));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([]), ref snapshot, ref specialization));
        var applied = ResourceMaterializer.ApplyTo(plan, specialization);
        var image = Assert.Single(applied.Info.Images);
        Assert.Equal(ImageNumericClass.Float, image.NumericClass);
    }

    [Theory]
    [InlineData(GuestImageFormat.Format8Srgb)]
    [InlineData(GuestImageFormat.Format8x2Srgb)]
    public void NarrowSrgbStorage_RetainsFormatWithoutChangingSampledSpecialization(uint format)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.AddRange(ImageWords(ref pc, 16, 0x1000, format));
        instructions.AddRange(SamplerWords(ref pc, 24, 0));
        instructions.Add(Image(pc, "ImageSample", 16, 24)); pc += 8;
        instructions.Add(Image(pc, "ImageStore", 16)); pc += 8;
        instructions.Add(EndProgram(pc));
        var plan = Extract(Program([.. instructions]));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([]), ref snapshot, ref specialization));
        var applied = ResourceMaterializer.ApplyTo(plan, specialization);
        var storage = Assert.Single(applied.Info.Images, image => image.ResourceClass == ImageResourceClass.Storage);
        var sampled = Assert.Single(applied.Info.Images, image => image.ResourceClass == ImageResourceClass.Sampled);
        Assert.Equal(format, storage.ConversionFormat);
        Assert.Equal(GuestImageFormat.Invalid, sampled.ConversionFormat);
        Assert.False(Assert.Single(applied.Info.Samplers).ForcePointFiltering);
    }

    // Three images share one sampler; the packed and signed ones need point filtering,
    // so the sampler splits and their accesses sample through the duplicate.
    [Fact]
    public void SignedImage_SplitsTheSharedSamplerIntoPointAndNativeVariants()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.AddRange(ImageWords(ref pc, 16, 0x1000, Format32Float));
        instructions.AddRange(ImageWords(ref pc, 24, 0x2000, Format11x2x10Uint));
        instructions.AddRange(ImageWords(ref pc, 32, 0x3000, Format32Sint));
        instructions.AddRange(SamplerWords(ref pc, 40, 0));
        var floatPc = pc;
        var packedPc = pc + 8;
        var signedPc = pc + 16;
        instructions.Add(Image(floatPc, "ImageSample", 16, 40));
        instructions.Add(Image(packedPc, "ImageSample", 24, 40));
        instructions.Add(Image(signedPc, "ImageSample", 32, 40));
        instructions.Add(EndProgram(pc + 24));
        var plan = Extract(Program([.. instructions]));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([]), ref snapshot, ref specialization));
        var applied = ResourceMaterializer.ApplyTo(plan, specialization);
        Assert.Equal(2, applied.Info.Samplers.Count);
        Assert.False(applied.Info.Samplers[0].ForcePointFiltering);
        Assert.True(applied.Info.Samplers[1].ForcePointFiltering);
        Assert.Equal(0u, applied.Info.SampledPairs[0].Sampler);
        Assert.Equal(1u, applied.Info.SampledPairs[1].Sampler);
        Assert.Equal(1u, applied.Info.SampledPairs[2].Sampler);
        Assert.True(plan.Memory.TryGetIndex(signedPc, 0, out var signedIndex));
        Assert.Equal(1u, applied.SamplerByMemoryIndex[signedIndex]);
        Assert.True(plan.Memory.TryGetIndex(floatPc, 0, out var floatIndex));
        Assert.False(applied.SamplerByMemoryIndex.ContainsKey(floatIndex));
        Assert.Equal(2, snapshot.Samplers.Length);
    }
}

public sealed class ScalarBufferRangeReadTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BatchedOrRefusedReadsPreserveDescriptorsAndCacheInvalidation(bool allowRange)
    {
        var plan = Extract(ResourceTrackerTests.IndirectImageProgram(false));
        var memory = ResourceTrackerTests.LinearMemory();
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        ResourceTrackerTests.WriteImage(memory, 0x2000, descriptor);
        var ranges = 0;
        bool Range(ulong address, Span<uint> words)
        {
            ranges++;
            words.Fill(0xDEADBEEF);
            if (!allowRange)
                return false;
            for (var index = 0; index < words.Length; index++)
                if (!memory.Read(address + (ulong)index * 4, out words[index]))
                    return false;
            return true;
        }
        bool Resident(ulong address, Span<byte> bytes, bool clean)
        {
            for (var offset = 0; offset < bytes.Length; offset += 4)
            {
                if (!memory.Read(address + (ulong)offset, out var word))
                    return false;
                BinaryPrimitives.WriteUInt32LittleEndian(bytes[offset..], word);
            }
            return true;
        }

        var inputs = Inputs([0x1000, 224u << 16, 2, 0, 0x2000, 16u << 16, 4, 0, 7],
            readCleanMemory: memory.Read) with { ReadCleanWords = Range };
        var reference = new ResourceSnapshot();
        var referenceSpecialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, inputs with { ReadCleanWords = null },
            ref reference, ref referenceSpecialization));

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var cache = new ResourceMaterializationCache();
        Assert.True(cache.Materialize(plan, inputs, Resident, ref snapshot, ref specialization, out _));
        Assert.True(ranges > 0);
        Assert.Equal(reference.FlattenedResourceTable, snapshot.FlattenedResourceTable);
        Assert.Equal(reference.Images[0], snapshot.Images[0]);
        Assert.Equal(referenceSpecialization, specialization);
        var readsAfterFirst = ranges;
        Assert.True(cache.Materialize(plan, inputs, Resident, ref snapshot, ref specialization, out _));
        Assert.Equal(readsAfterFirst, ranges);
        Assert.Equal(1, cache.Hits);

        memory.At(0x2000) += 1;
        Assert.True(cache.Materialize(plan, inputs, Resident, ref snapshot, ref specialization, out _));
        Assert.True(ranges > readsAfterFirst);
        Assert.Equal(descriptor[0] + 1, snapshot.Images[0][0]);
    }

    [Fact]
    public void PartialBoundsAndAddressOverflowRetainIndividualWordReads()
    {
        var calls = 0;
        bool Range(ulong address, Span<uint> words)
        {
            calls++;
            Assert.Equal(0x1004ul, address);
            words.Fill(7);
            return true;
        }
        var inputs = Inputs([]) with { ReadCleanWords = Range };
        Span<uint> words = stackalloc uint[8];
        Assert.False(ScalarBufferRangeRead.TryRead([0x1000, 0, 31, 0], 0, 0, inputs, words));
        Assert.False(ScalarBufferRangeRead.TryRead([0x1000, 0, uint.MaxValue, 0], 0,
            uint.MaxValue - 4, inputs, words));
        Assert.False(ScalarBufferRangeRead.TryRead([0xFFFFFFF0, 0xFFFF, 64, 0], 0, 0, inputs, words));
        Assert.Equal(0, calls);
        Assert.True(ScalarBufferRangeRead.TryRead([0x1003, 0, 36, 0], 5, 0, inputs, words));
        Assert.Equal(1, calls);
        Assert.All(words.ToArray(), word => Assert.Equal(7u, word));
    }
}

public sealed class ShiftIndexedMaterialTests
{
    private static Gen5ShaderProgram MaterialProgram(uint shift, bool multiply = false) => Program(
        MoveVectorFromScalar(0, 1, 8),
        ReadFirstLane(4, 9, 1),
        Sop2(8, multiply ? "SMulI32" : "SLshlB32", 10, Gen5Operand.Scalar(9),
            Operand(multiply ? 1u << (int)(shift & 31) : shift)),
        Sop2(12, "SAddU32", 11, Gen5Operand.Scalar(10), Operand(4)),
        ScalarBufferLoad(16, 0, 12, dynamicOffsetRegister: 11),
        Sop2(24, "SLshlB32", 13, Gen5Operand.Scalar(12), Operand(5)),
        ScalarBufferLoad(28, 4, 16, count: 8, dynamicOffsetRegister: 13),
        MoveScalar(36, 24, 0), MoveScalar(40, 25, 0),
        MoveScalar(44, 26, 0), MoveScalar(48, 27, 0),
        Image(52, "ImageSample", 16, 24), EndProgram(60));

    [Theory]
    [InlineData(0u, 1u)]
    [InlineData(5u, 32u)]
    [InlineData(37u, 32u)]
    [InlineData(31u, 0x80000000u)]
    public void ConstantShiftPreservesTheMaterialStrideAndImmediate(uint count, uint expectedStride)
    {
        var plan = Extract(MaterialProgram(count));
        var selector = Assert.Single(plan.DescriptorSources, source => source.IndirectImage is not null).IndirectImage!;
        Assert.Equal(expectedStride, selector.SelectorStride);
        Assert.Equal(4u, selector.SelectorOffset);
        Assert.Single(plan.IndirectImages);
    }

    [Theory]
    [InlineData(5u)]
    [InlineData(37u)]
    public void ShiftAndMultiplyProduceTheSameNonemptyDescriptorTable(uint count)
    {
        var memory = new TestWordMemory { Words = new uint[0x1040 / 4], RequireAlignment = true };
        memory.At(0x1004) = 0;
        memory.At(0x1024) = 1;
        // RGBA32 float, identity component mapping, 2D image.
        uint[] first = [0x20, 77u << 20, 3 | (3 << 14), 0xFAC | (9u << 28), 0, 0, 0, 0];
        var second = (uint[])first.Clone();
        second[0] = 0x40;
        for (var component = 0; component < first.Length; component++)
        {
            memory.At(0x2000 + (ulong)component * 4) = first[component];
            memory.At(0x2020 + (ulong)component * 4) = second[component];
        }
        uint[] userData = [0x1000, 32u << 16, 2, 0, 0x2000, 16u << 16, 4, 0, 0];
        var results = new List<ResourceSnapshot>();
        foreach (var multiply in new[] { false, true })
        {
            var plan = Extract(MaterialProgram(count, multiply));
            var snapshot = new ResourceSnapshot();
            var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readCleanMemory: memory.Read),
                ref snapshot, ref specialization));
            results.Add(snapshot);
        }
        Assert.Equal(2, results[0].Images.Length);
        Assert.Equal(first, results[0].Images[0]);
        Assert.Equal(second, results[0].Images[1]);
        Assert.Equal(results[1].FlattenedResourceTable, results[0].FlattenedResourceTable);
        Assert.Equal(results[1].Images.SelectMany(words => words), results[0].Images.SelectMany(words => words));
    }
}
