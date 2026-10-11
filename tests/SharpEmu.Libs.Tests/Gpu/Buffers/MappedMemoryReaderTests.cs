// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Buffers;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Buffers;

public sealed unsafe class MappedMemoryReaderTests
{
    // Every source alignment and length around the 16- and 64-byte steps copies exactly the
    // requested bytes and nothing past the destination.
    [Fact]
    public void Copy_MatchesTheSourceForEveryAlignmentAndLength()
    {
        var random = new System.Random(77);
        var source = new byte[8192 + 64];
        random.NextBytes(source);
        fixed (byte* start = source)
        {
            var aligned = (byte*)(((nuint)start + 63) & ~(nuint)63);
            for (var misalignment = 0; misalignment < 16; misalignment++)
            {
                foreach (var length in new[] { 0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 4095, 4096, 4097, 8000 })
                {
                    var destination = new byte[length + 8];
                    Array.Fill(destination, (byte)0xCD);
                    MappedMemoryReader.Copy(aligned + misalignment, destination.AsSpan(0, length));
                    Assert.True(new ReadOnlySpan<byte>(aligned + misalignment, length).SequenceEqual(destination.AsSpan(0, length)),
                        $"misalignment={misalignment} length={length}");
                    Assert.All(destination.AsSpan(length).ToArray(), value => Assert.Equal(0xCD, value));
                }
            }
        }
    }
}
