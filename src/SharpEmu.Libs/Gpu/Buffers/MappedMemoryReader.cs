// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpEmu.Libs.Gpu.Buffers;

// Copies out of a host mapping of device-local memory. Such mappings are uncached or
// write-combining: an ordinary copy reads them at a few tens of MB/s, while streaming loads,
// which bypass the cache hierarchy, read the same memory more than ten times faster.
internal static unsafe class MappedMemoryReader
{
    public static void Copy(byte* source, Span<byte> destination)
    {
        var length = destination.Length;
        if (!Sse41.IsSupported || length < 64)
        {
            new ReadOnlySpan<byte>(source, length).CopyTo(destination);
            return;
        }

        // Streaming loads need 16-byte alignment: copy the unaligned head normally.
        var head = (int)((16 - ((nuint)source & 15)) & 15);
        new ReadOnlySpan<byte>(source, head).CopyTo(destination);
        fixed (byte* target = destination)
        {
            var offset = head;
            for (; offset + 64 <= length; offset += 64)
            {
                var a = Sse41.LoadAlignedVector128NonTemporal(source + offset);
                var b = Sse41.LoadAlignedVector128NonTemporal(source + offset + 16);
                var c = Sse41.LoadAlignedVector128NonTemporal(source + offset + 32);
                var d = Sse41.LoadAlignedVector128NonTemporal(source + offset + 48);
                Vector128.Store(a, target + offset);
                Vector128.Store(b, target + offset + 16);
                Vector128.Store(c, target + offset + 32);
                Vector128.Store(d, target + offset + 48);
            }

            for (; offset + 16 <= length; offset += 16)
            {
                Vector128.Store(Sse41.LoadAlignedVector128NonTemporal(source + offset), target + offset);
            }

            new ReadOnlySpan<byte>(source + offset, length - offset).CopyTo(destination[offset..]);
        }
    }
}
