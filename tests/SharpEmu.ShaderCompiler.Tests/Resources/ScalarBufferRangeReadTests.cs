// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

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
