// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class AgcCreateShaderTests
{
    private const ulong BaseAddress = 0x3_0000_0000;
    private const int MemorySize = 0x4000;

    private const ulong DestinationAddress = BaseAddress + 0x0000;
    private const ulong HeaderAddress = BaseAddress + 0x0100;
    private const ulong CodeAddress = BaseAddress + 0x1000;
    private const ulong RegistersAddress = BaseAddress + 0x0500;

    private const uint ShaderFileHeader = 0x34333231;
    private const uint ShaderVersion = 0x18;

    private const ulong ShaderCodeOffset = 0x10;
    private const ulong ShaderShRegistersOffset = 0x20;
    private const ulong ShaderTypeOffset = 0x5A;
    private const ulong ShaderNumShRegistersOffset = 0x5C;

    private const uint SpiShaderPgmLoPs = 0x8;
    private const uint SpiShaderPgmHiPs = 0x9;

    [Fact]
    public void CreateShader_StandardPsShader_PatchesPgmLoHi()
    {
        var (memory, ctx) = SetupContext();
        SetupHeader(memory, shaderType: 1, registerCount: 2);

        // Put PS PGM_LO and PGM_HI registers in the table
        WriteUInt32(memory, RegistersAddress + 0, SpiShaderPgmLoPs);
        WriteUInt32(memory, RegistersAddress + 4, 0);
        WriteUInt32(memory, RegistersAddress + 8, SpiShaderPgmHiPs);
        WriteUInt32(memory, RegistersAddress + 12, 0);

        ctx[CpuRegister.Rdi] = DestinationAddress;
        ctx[CpuRegister.Rsi] = HeaderAddress;
        ctx[CpuRegister.Rdx] = CodeAddress;

        var result = AgcExports.CreateShader(ctx);

        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, result);
        Assert.Equal(HeaderAddress, ReadUInt64(memory, DestinationAddress));
        Assert.Equal(CodeAddress, ReadUInt64(memory, HeaderAddress + ShaderCodeOffset));

        var expectedLo = (uint)((CodeAddress >> 8) & 0xFFFF_FFFFUL);
        var expectedHi = (uint)((CodeAddress >> 40) & 0xFFUL);
        Assert.Equal(expectedLo, ReadUInt32(memory, RegistersAddress + 4));
        Assert.Equal(expectedHi, ReadUInt32(memory, RegistersAddress + 12));
    }

    [Fact]
    public void CreateShader_ShaderType8WithoutPgmRegisters_SucceedsAndSkipsPatch()
    {
        var (memory, ctx) = SetupContext();
        SetupHeader(memory, shaderType: 8, registerCount: 2);

        // Registers are user data registers (e.g. 0x0 and 0x1), not PGM_LO
        WriteUInt32(memory, RegistersAddress + 0, 0x0);
        WriteUInt32(memory, RegistersAddress + 4, 0x1111);
        WriteUInt32(memory, RegistersAddress + 8, 0x1);
        WriteUInt32(memory, RegistersAddress + 12, 0x2222);

        ctx[CpuRegister.Rdi] = DestinationAddress;
        ctx[CpuRegister.Rsi] = HeaderAddress;
        ctx[CpuRegister.Rdx] = CodeAddress;

        var result = AgcExports.CreateShader(ctx);

        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, result);
        Assert.Equal(HeaderAddress, ReadUInt64(memory, DestinationAddress));
        Assert.Equal(CodeAddress, ReadUInt64(memory, HeaderAddress + ShaderCodeOffset));
        // Registers table is unchanged
        Assert.Equal(0x1111u, ReadUInt32(memory, RegistersAddress + 4));
        Assert.Equal(0x2222u, ReadUInt32(memory, RegistersAddress + 12));
    }

    [Fact]
    public void CreateShader_ZeroRegisters_Succeeds()
    {
        var (memory, ctx) = SetupContext();
        SetupHeader(memory, shaderType: 8, registerCount: 0);

        ctx[CpuRegister.Rdi] = DestinationAddress;
        ctx[CpuRegister.Rsi] = HeaderAddress;
        ctx[CpuRegister.Rdx] = CodeAddress;

        var result = AgcExports.CreateShader(ctx);

        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_OK, result);
        Assert.Equal(HeaderAddress, ReadUInt64(memory, DestinationAddress));
        Assert.Equal(CodeAddress, ReadUInt64(memory, HeaderAddress + ShaderCodeOffset));
    }

    private static (FakeCpuMemory Memory, CpuContext Ctx) SetupContext()
    {
        var memory = new FakeCpuMemory(BaseAddress, MemorySize);
        var ctx = new CpuContext(memory, Generation.Gen5);
        return (memory, ctx);
    }

    private static void SetupHeader(FakeCpuMemory memory, byte shaderType, byte registerCount)
    {
        WriteUInt32(memory, HeaderAddress + 0, ShaderFileHeader);
        WriteUInt32(memory, HeaderAddress + 4, ShaderVersion);

        // Pointers are relative to their field offset
        if (registerCount > 0)
        {
            var relativeRegisters = RegistersAddress - (HeaderAddress + ShaderShRegistersOffset);
            WriteUInt64(memory, HeaderAddress + ShaderShRegistersOffset, relativeRegisters);
        }
        else
        {
            WriteUInt64(memory, HeaderAddress + ShaderShRegistersOffset, 0);
        }

        WriteByte(memory, HeaderAddress + ShaderTypeOffset, shaderType);
        WriteByte(memory, HeaderAddress + ShaderNumShRegistersOffset, registerCount);
    }

    private static byte ReadByte(FakeCpuMemory memory, ulong address)
    {
        Span<byte> buf = stackalloc byte[1];
        Assert.True(memory.TryRead(address, buf));
        return buf[0];
    }

    private static uint ReadUInt32(FakeCpuMemory memory, ulong address)
    {
        Span<byte> buf = stackalloc byte[sizeof(uint)];
        Assert.True(memory.TryRead(address, buf));
        return BinaryPrimitives.ReadUInt32LittleEndian(buf);
    }

    private static ulong ReadUInt64(FakeCpuMemory memory, ulong address)
    {
        Span<byte> buf = stackalloc byte[sizeof(ulong)];
        Assert.True(memory.TryRead(address, buf));
        return BinaryPrimitives.ReadUInt64LittleEndian(buf);
    }

    private static void WriteByte(FakeCpuMemory memory, ulong address, byte value) =>
        Assert.True(memory.TryWrite(address, [value]));

    private static void WriteUInt32(FakeCpuMemory memory, ulong address, uint value)
    {
        Span<byte> buf = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(buf, value);
        Assert.True(memory.TryWrite(address, buf));
    }

    private static void WriteUInt64(FakeCpuMemory memory, ulong address, ulong value)
    {
        Span<byte> buf = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(buf, value);
        Assert.True(memory.TryWrite(address, buf));
    }
}
