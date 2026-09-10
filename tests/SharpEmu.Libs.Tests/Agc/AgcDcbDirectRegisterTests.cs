// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class AgcDcbDirectRegisterTests
{
    private const ulong BaseAddress = 0x4_0000_0000;
    private const ulong CommandBufferAddress = BaseAddress + 0x1000;
    private const ulong CommandsStart = BaseAddress + 0x2000;
    private const ulong CommandsEnd = BaseAddress + 0x4000;

    [Fact]
    public void DcbSetShRegisterDirect_WritesPacketAndAdvancesCursor()
    {
        var memory = new FakeCpuMemory(BaseAddress, 0x10000);
        var context = new CpuContext(memory, Generation.Gen5);
        WriteUInt64(memory, CommandBufferAddress + 0x10, CommandsStart);
        WriteUInt64(memory, CommandBufferAddress + 0x18, CommandsEnd);
        WriteUInt64(memory, CommandBufferAddress + 0x20, 0);
        WriteUInt64(memory, CommandBufferAddress + 0x28, 0);
        WriteUInt32(memory, CommandBufferAddress + 0x30, 0);

        const uint registerOffset = 0x1234;
        const uint registerValue = 0xDEADBEEF;
        context[CpuRegister.Rdi] = CommandBufferAddress;
        context[CpuRegister.Rsi] = registerOffset | ((ulong)registerValue << 32);

        var result = AgcExports.DcbSetShRegisterDirect(context);

        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, result);
        Assert.Equal(CommandsStart, context[CpuRegister.Rax]);
        Assert.Equal(CommandsStart + 12, ReadUInt64(memory, CommandBufferAddress + 0x10));
        Assert.Equal(0xC0017600u, ReadUInt32(memory, CommandsStart));
        Assert.Equal(registerOffset, ReadUInt32(memory, CommandsStart + 4));
        Assert.Equal(registerValue, ReadUInt32(memory, CommandsStart + 8));
    }

    [Fact]
    public void DcbSetShRegisterDirectGetSize_ReturnsPacketSize()
    {
        var context = new CpuContext(
            new FakeCpuMemory(BaseAddress, 0x1000),
            Generation.Gen5);

        Assert.Equal(12, AgcExports.DcbSetShRegisterDirectGetSize(context));
        Assert.Equal(12UL, context[CpuRegister.Rax]);
    }

    private static uint ReadUInt32(FakeCpuMemory memory, ulong address)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        Assert.True(memory.TryRead(address, bytes));
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static ulong ReadUInt64(FakeCpuMemory memory, ulong address)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        Assert.True(memory.TryRead(address, bytes));
        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }

    private static void WriteUInt32(FakeCpuMemory memory, ulong address, uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        Assert.True(memory.TryWrite(address, bytes));
    }

    private static void WriteUInt64(FakeCpuMemory memory, ulong address, ulong value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        Assert.True(memory.TryWrite(address, bytes));
    }
}
