// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Scheduling;

[Collection(SchedulingStateCollection.Name)]
public sealed class GpuWritebackEpochTests
{
    [Fact]
    public void StoresSinceACountAreCollectedWithTheirRanges()
    {
        var since = GpuWritebackEpoch.Value;
        GpuWritebackEpoch.Advance(0x1000, 0x40);
        GpuWritebackEpoch.Advance(0x8000, 0x100);
        var ranges = new List<(ulong Address, ulong Size)>();

        Assert.True(GpuWritebackEpoch.TryCollectSince(since, ranges));
        Assert.Equal([(0x1000UL, 0x40UL), (0x8000UL, 0x100UL)], ranges);
    }

    [Fact]
    public void AStoreWithoutARangeCannotBeExcluded()
    {
        var since = GpuWritebackEpoch.Value;
        GpuWritebackEpoch.Advance(0x1000, 0x40);
        GpuWritebackEpoch.Advance();

        Assert.False(GpuWritebackEpoch.TryCollectSince(since, new List<(ulong, ulong)>()));
    }

    [Fact]
    public void StoresNoLongerRecordedCannotBeExcluded()
    {
        var since = GpuWritebackEpoch.Value;
        for (var store = 0; store < 2048; store++)
        {
            GpuWritebackEpoch.Advance(0x1000 + (ulong)store * 0x40, 0x40);
        }

        Assert.False(GpuWritebackEpoch.TryCollectSince(since, new List<(ulong, ulong)>()));
    }
}
