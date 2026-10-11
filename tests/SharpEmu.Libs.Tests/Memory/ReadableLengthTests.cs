// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Memory;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Libs.Tests.Memory;

public sealed class ReadableLengthTests
{
    private const ulong GuestBase = 0x2_0000_0000;

    // Only CanRead, so ReadableLength falls back to the interface's search over it.
    private sealed class SearchOnly(ICpuMemory inner) : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => throw new NotSupportedException();
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => throw new NotSupportedException();
        public bool CanRead(ulong address, ulong size) => inner.CanRead(address, size);
    }

    // The one-query answer of guest memory equals the longest prefix CanRead accepts, across backed
    // views, a gap between them, a private region and unmapped memory.
    [Fact]
    public void ReadableLength_EqualsTheLongestPrefixCanReadAccepts()
    {
        var views = HostViewMemory.Create();
        using var memory = new PhysicalVirtualMemory(viewHost: views, backingBytes: 32UL * 1024 * 1024);
        var granule = views.Granularity;
        Assert.True(memory.TryHoldRangeAtOrAbove(GuestBase, 3 * granule, granule, out var first));
        Assert.True(memory.TryMapBacked(first, granule, 0, GuestPageProtection.Read | GuestPageProtection.Write, out _));
        Assert.True(memory.TryMapBacked(first + 2 * granule, granule, granule, GuestPageProtection.Read | GuestPageProtection.Write, out _));
        Assert.True(memory.TryAllocateAtOrAbove(GuestBase + 0x1000_0000, 0x10000, executable: false, 0x4000, out var region));

        var reference = new SearchOnly(memory);
        ulong[] addresses =
        [
            first, first + 0x10, first + granule - 1, first + granule, first + 2 * granule, first + 3 * granule - 4,
            first + 3 * granule, region, region + 0x8000, region + 0xFFFF, region + 0x10000,
        ];
        ulong[] sizes = [1, 4, 0x1000, granule, granule + 1, 2 * granule, 3 * granule, 0x20000, 1UL << 40, ulong.MaxValue];
        foreach (var address in addresses)
        {
            foreach (var size in sizes)
            {
                var expected = ((ICpuMemory)reference).ReadableLength(address, size);
                Assert.Equal(expected, ((ICpuMemory)memory).ReadableLength(address, size));
            }
        }

        Assert.Equal(granule, ((ICpuMemory)memory).ReadableLength(first, 3 * granule));
        Assert.Equal(0x10000UL, ((ICpuMemory)memory).ReadableLength(region, 1UL << 40));
        Assert.Equal(0UL, ((ICpuMemory)memory).ReadableLength(first + granule, 4));
    }
}
