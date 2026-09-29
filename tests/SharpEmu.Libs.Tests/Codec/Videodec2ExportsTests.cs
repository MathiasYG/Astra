// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Codec;
using Xunit;

namespace SharpEmu.Libs.Tests;

public sealed class Videodec2ExportsTests
{
    private const ulong MemoryBase = 0x2_0000_0000;
    private const ulong OutputInfoAddress = MemoryBase + 0x100;

    private readonly FakeCpuMemory _memory = new(MemoryBase, 0x1000);

    [Fact]
    public void OutputInfo_NoPictureClearsReadyAtOffsetEightOnly()
    {
        var ctx = new CpuContext(_memory, Generation.Gen5);
        WriteSentinelOutputInfo();

        Assert.True(Videodec2Exports.TryWriteOutputInfo(ctx, OutputInfoAddress, false, 0, 0));

        var bytes = ReadOutputInfo();
        Assert.Equal(0xCC, bytes[0]);
        Assert.Equal(0x00, bytes[8]);
        Assert.Equal(0xCC, bytes[16]);
        Assert.Equal(0xCC, bytes[24]);
    }

    [Fact]
    public void OutputInfo_PictureWritesReadyAndDimensionsAtGuestOffsets()
    {
        var ctx = new CpuContext(_memory, Generation.Gen5);
        WriteSentinelOutputInfo();

        Assert.True(Videodec2Exports.TryWriteOutputInfo(ctx, OutputInfoAddress, true, 1920, 1080));

        var bytes = ReadOutputInfo();
        Assert.Equal(0xCC, bytes[0]);
        Assert.Equal(0x01, bytes[8]);
        Assert.Equal(1920UL, BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(16, sizeof(ulong))));
        Assert.Equal(1080UL, BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(24, sizeof(ulong))));
    }

    [Fact]
    public void Decode_NoDecoderClearsReadyFlagAtGuestOffset()
    {
        var ctx = new CpuContext(_memory, Generation.Gen5);
        ctx[CpuRegister.Rdi] = 0xDEAD_0001;
        ctx[CpuRegister.Rcx] = OutputInfoAddress;
        WriteSentinelOutputInfo();

        Assert.Equal(0, Videodec2Exports.Videodec2Decode(ctx));

        var bytes = ReadOutputInfo();
        Assert.Equal(0xCC, bytes[0]);
        Assert.Equal(0x00, bytes[8]);
    }

    [Fact]
    public void Flush_NoDecoderClearsReadyFlagAtGuestOffset()
    {
        var ctx = new CpuContext(_memory, Generation.Gen5);
        ctx[CpuRegister.Rdi] = 0xDEAD_0002;
        ctx[CpuRegister.Rdx] = OutputInfoAddress;
        WriteSentinelOutputInfo();

        Assert.Equal(0, Videodec2Exports.Videodec2Flush(ctx));

        var bytes = ReadOutputInfo();
        Assert.Equal(0xCC, bytes[0]);
        Assert.Equal(0x00, bytes[8]);
    }

    private void WriteSentinelOutputInfo()
    {
        var sentinel = new byte[0x20];
        Array.Fill(sentinel, (byte)0xCC);
        Assert.True(_memory.TryWrite(OutputInfoAddress, sentinel));
    }

    private byte[] ReadOutputInfo()
    {
        var bytes = new byte[0x20];
        Assert.True(_memory.TryRead(OutputInfoAddress, bytes));
        return bytes;
    }
}
