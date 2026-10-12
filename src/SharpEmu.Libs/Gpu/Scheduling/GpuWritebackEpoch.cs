// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Scheduling;

// Counts the stores of GPU data into guest memory: buffer and image downloads, and buffer transfers
// the caches complete there. Work done ahead of the render thread from guest memory is still
// current while the count has not moved, or while none of the stores since touched what it read.
internal static class GpuWritebackEpoch
{
    // The ranges of the latest stores, by count. A store with no range (size 0) may have written
    // anything.
    private const int RangeCapacity = 1024;
    private static readonly (ulong Address, ulong Size)[] _ranges = new (ulong, ulong)[RangeCapacity];
    private static readonly object _lock = new();
    private static long _value;

    public static long Value => Volatile.Read(ref _value);

    public static void Advance() => Advance(0, 0);

    public static void Advance(ulong address, ulong size)
    {
        lock (_lock)
        {
            var value = _value + 1;
            _ranges[value & (RangeCapacity - 1)] = (address, size);
            Volatile.Write(ref _value, value);
        }
    }

    // Adds the ranges stored since the count was since; false when one of them is unknown or they
    // are no longer all recorded.
    public static bool TryCollectSince(long since, List<(ulong Address, ulong Size)> ranges)
    {
        lock (_lock)
        {
            if (_value - since > RangeCapacity)
            {
                return false;
            }

            for (var value = since + 1; value <= _value; value++)
            {
                var range = _ranges[value & (RangeCapacity - 1)];
                if (range.Size == 0)
                {
                    return false;
                }

                ranges.Add(range);
            }

            return true;
        }
    }
}
