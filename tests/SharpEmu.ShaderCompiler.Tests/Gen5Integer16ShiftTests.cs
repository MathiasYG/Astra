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

    [Fact]
    public void Vop3SignedSubwordAndHalfFloatOperationsCompile()
    {
        var program = Program(
            MoveVector(0, 0, 0x12345678),
            MoveVector(4, 1, 0x4321FEDC),
            MoveVector(8, 2, 0x3C004000),
            Vop3Instruction(12, "VSubNcI16", 0, clamp: true, operandSelect: 8),
            Vop3Instruction(20, "VLshrrevB16", 1, operandSelect: 8),
            Vop3Instruction(28, "VMin3F16", 2),
            Vop3Instruction(36, "VMed3F16", 3),
            EndProgram(44));
        var request = Request(program, userDataCount: 0);

        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error),
            error);

        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains((ushort)SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ISub, opcodes);
        Assert.Contains((ushort)SpirvOp.SLessThan, opcodes);
        Assert.Contains((ushort)SpirvOp.SGreaterThan, opcodes);
        Assert.Contains((ushort)SpirvOp.IsNan, opcodes);
        Assert.Contains((ushort)SpirvOp.Select, opcodes);
        Assert.Contains((ushort)SpirvOp.ExtInst, opcodes);
        Assert.DoesNotContain((ushort)SpirvCapability.Float16, ReadCapabilities(shader.Spirv));
    }

    private static Gen5ShaderInstruction Vop3Instruction(
        uint pc,
        string opcode,
        uint destination,
        bool clamp = false,
        uint operandSelect = 0) =>
        new(
            pc,
            Gen5ShaderEncoding.Vop3,
            opcode,
            [0u, 0u],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(destination)],
            new Gen5Vop3Control(0, 0, 0, clamp, operandSelect, null));

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

    private static HashSet<uint> ReadCapabilities(byte[] spirv)
    {
        var capabilities = new HashSet<uint>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var length = (int)(header >> 16);
            Assert.True(length > 0);
            if ((header & 0xFFFF) == (ushort)SpirvOp.Capability)
            {
                capabilities.Add(BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset + sizeof(uint))));
            }

            offset += length * sizeof(uint);
        }

        return capabilities;
    }
}
