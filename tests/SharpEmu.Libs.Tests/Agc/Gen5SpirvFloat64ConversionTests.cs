// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class Gen5SpirvFloat64ConversionTests
{
    private const ulong ShaderAddress = 0x1_0000_0000;

    [Fact]
    public void VCvtF64I32_EmitsBothWordsWithoutOptionalFloat64Feature()
    {
        var memory = new FakeCpuMemory(ShaderAddress, 0x1000);
        Gen5ShaderAtomicDecodeTests.WriteProgram(memory, ShaderAddress, [0x7E060907]);
        var context = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                context,
                ShaderAddress,
                out var program,
                out var decodeError),
            decodeError);

        var request = ResourceTestProgram.Request(program, userDataCount: 0);
        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var compileError),
            compileError);

        var instructions = ReadInstructions(shader.Spirv);
        var opcodes = instructions.Select(instruction => instruction.Opcode).ToArray();
        Assert.Contains((ushort)SpirvOp.ExtInst, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftLeftLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ULessThan, opcodes);
        Assert.True(opcodes.Count(opcode => opcode == (ushort)SpirvOp.Store) >= 2);
        Assert.DoesNotContain(
            instructions
                .Where(instruction => instruction.Opcode == (ushort)SpirvOp.Capability)
                .Select(instruction => instruction.FirstOperand),
            capability => capability == (uint)SpirvCapability.Float64);
    }

    [Fact]
    public void VRcpF64_UsesIntegerSignificandDivisionAndPreservesInt64Portability()
    {
        var memory = new FakeCpuMemory(ShaderAddress, 0x1000);
        Gen5ShaderAtomicDecodeTests.WriteProgram(memory, ShaderAddress, [0x7E065F03]);
        var context = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                context,
                ShaderAddress,
                out var program,
                out var decodeError),
            decodeError);

        var request = ResourceTestProgram.Request(program, userDataCount: 0);
        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var compileError),
            compileError);

        var instructions = ReadInstructions(shader.Spirv);
        var opcodes = instructions.Select(instruction => instruction.Opcode).ToArray();
        Assert.Contains((ushort)SpirvOp.UDiv, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftLeftLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ExtInst, opcodes);
        var capabilities = instructions
            .Where(instruction => instruction.Opcode == (ushort)SpirvOp.Capability)
            .Select(instruction => instruction.FirstOperand);
        Assert.Contains((uint)SpirvCapability.Int64, capabilities);
        Assert.DoesNotContain((uint)SpirvCapability.Float64, capabilities);
    }

    [Fact]
    public void VCvtF32F64_UsesIntegerRoundingWithoutOptionalFloat64Feature()
    {
        var memory = new FakeCpuMemory(ShaderAddress, 0x1000);
        Gen5ShaderAtomicDecodeTests.WriteProgram(memory, ShaderAddress, [0x7E021F01]);
        var context = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                context,
                ShaderAddress,
                out var program,
                out var decodeError),
            decodeError);

        var request = ResourceTestProgram.Request(program, userDataCount: 0);
        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var compileError),
            compileError);

        var instructions = ReadInstructions(shader.Spirv);
        var opcodes = instructions.Select(instruction => instruction.Opcode).ToArray();
        Assert.Contains((ushort)SpirvOp.ULessThan, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ShiftLeftLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.ExtInst, opcodes);
        var capabilities = instructions
            .Where(instruction => instruction.Opcode == (ushort)SpirvOp.Capability)
            .Select(instruction => instruction.FirstOperand);
        Assert.Contains((uint)SpirvCapability.Int64, capabilities);
        Assert.DoesNotContain((uint)SpirvCapability.Float64, capabilities);
    }

    private static IReadOnlyList<(ushort Opcode, uint FirstOperand)> ReadInstructions(byte[] spirv)
    {
        Assert.Equal(0x07230203u, BinaryPrimitives.ReadUInt32LittleEndian(spirv));
        var instructions = new List<(ushort Opcode, uint FirstOperand)>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var wordCount = checked((int)(header >> 16));
            Assert.InRange(wordCount, 1, (spirv.Length - offset) / sizeof(uint));
            var firstOperand = wordCount > 1
                ? BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset + sizeof(uint)))
                : 0;
            instructions.Add(((ushort)header, firstOperand));
            offset += wordCount * sizeof(uint);
        }

        return instructions;
    }
}
