// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ResourceMaterializationCacheReadRecordingTests
{
    [Fact]
    public void ConflictingObservationsOfOneAddressDoNotPublishCacheEntry()
    {
        var plan = Extract(ResourceTrackerTests.IndirectImageProgram(false));
        var memory = ResourceTrackerTests.LinearMemory();
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        ResourceTrackerTests.WriteImage(memory, 0x1000, descriptor);
        memory.At(0x1024) = 0;
        var observations = 0;
        bool Read(ulong address, out uint word)
        {
            if (address == 0x1004 && observations++ == 0)
            {
                word = 0;
                return true;
            }
            return memory.Read(address, out word);
        }
        var inputs = Inputs([0x1000, 224u << 16, 2, 0, 0x1000, 16u << 16, 4, 0, 7],
            readMemory: Read, readCleanMemory: Read);
        bool Resident(ulong address, Span<byte> bytes, bool clean)
        {
            for (var index = 0; index < bytes.Length; index += 4)
            {
                if (!memory.Read(address + (ulong)index, out var word))
                    return false;
                BinaryPrimitives.WriteUInt32LittleEndian(bytes[index..], word);
            }
            return true;
        }
        var cache = new ResourceMaterializationCache();
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(cache.Materialize(plan, inputs, Resident, ref snapshot, ref specialization, out _));
        Assert.True(observations > 1);
        Assert.Contains(snapshot.Images, words => words.SequenceEqual(descriptor));
        Assert.Equal(1, cache.Uncacheable);
        observations = 0;
        Assert.True(cache.Materialize(plan, inputs, Resident, ref snapshot, ref specialization, out _));
        Assert.Equal(2, cache.Uncacheable);
        Assert.Equal(0, cache.Hits);
    }
}
