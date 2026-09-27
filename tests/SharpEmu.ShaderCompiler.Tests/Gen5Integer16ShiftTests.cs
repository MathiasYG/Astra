// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5Integer16ShiftTests
{
    [Fact]
    public void Vop3LshlrevB16SelectsAndPreservesDestinationHalf()
    {
        var shift = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Vop3,
            "VLshlrevB16",
            [0xD7144000, 0x00021A81],
            [Gen5Operand.Source(129), Gen5Operand.Vector(13), Gen5Operand.Scalar(0)],
            [Gen5Operand.Vector(0)],
            new Gen5Vop3Control(0, 0, 0, false, 8, null));
        var program = Program(
            MoveVector(0, 0, 0x12345678),
            MoveVector(4, 13, 0x00000021),
            shift,
            EndProgram(16));
        var request = Request(program, userDataCount: 0);

        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error),
            error);

        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains((ushort)SpirvOp.ShiftLeftLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.BitwiseAnd, opcodes);
        Assert.Contains((ushort)SpirvOp.BitwiseOr, opcodes);
    }

    private static List<ushort> ReadOpcodes(byte[] spirv)
    {
        var opcodes = new List<ushort>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var length = (int)(header >> 16);
            Assert.True(length > 0);
            opcodes.Add((ushort)(header & 0xFFFF));
            offset += length * sizeof(uint);
        }

        return opcodes;
    }
}
