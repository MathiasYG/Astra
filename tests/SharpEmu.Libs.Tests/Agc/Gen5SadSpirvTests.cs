// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class Gen5SadSpirvTests
{
    private const ulong ShaderAddress = 0x1_0000_0000;

    [Fact]
    public void SadU32_EmitsUnsignedAbsoluteDifferenceAndAdd()
    {
        // V_SAD_U32 v3, v0, v1, v2
        var spirv = Compile([0xD15D0003u, 0x040A0300u]);
        var instructions = EnumerateInstructions(spirv).ToArray();

        Assert.Contains(instructions, instruction =>
            instruction.Op == (ushort)SpirvOp.ExtInst &&
            ReadWord(spirv, instruction.Offset + 16) == 41); // UMax
        Assert.Contains(instructions, instruction =>
            instruction.Op == (ushort)SpirvOp.ExtInst &&
            ReadWord(spirv, instruction.Offset + 16) == 38); // UMin
        Assert.Contains(instructions, instruction => instruction.Op == (ushort)SpirvOp.ISub);
        Assert.Contains(instructions, instruction => instruction.Op == (ushort)SpirvOp.IAdd);
    }

    private static byte[] Compile(uint[] programWords)
    {
        var memory = new FakeCpuMemory(ShaderAddress, 0x2000);
        var context = new CpuContext(memory, Generation.Gen5);
        Gen5ShaderAtomicDecodeTests.WriteProgram(memory, ShaderAddress, programWords);
        var shaderRegisters = new Dictionary<uint, uint>
        {
            [Gen5ShaderAtomicDecodeTests.ComputePgmRsrc2Register] = 16u << 1,
        };

        Assert.True(
            Gen5ShaderTranslator.TryCreateState(
                context,
                ShaderAddress,
                0,
                shaderRegisters,
                Gen5ShaderAtomicDecodeTests.ComputeUserDataRegister,
                out var state,
                out var error),
            error);
        Assert.True(
            Gen5ShaderScalarEvaluator.TryEvaluate(context, state, out var evaluation, out error),
            error);
        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state, evaluation, 1, 1, 1, out var shader, out error),
            error);
        return shader.Spirv;
    }

    private static IEnumerable<(ushort Op, int Offset)> EnumerateInstructions(byte[] spirv)
    {
        for (var offset = 5 * sizeof(uint); offset + sizeof(uint) <= spirv.Length;)
        {
            var word = ReadWord(spirv, offset);
            var wordCount = (int)(word >> 16);
            if (wordCount <= 0)
            {
                yield break;
            }

            yield return ((ushort)word, offset);
            offset += wordCount * sizeof(uint);
        }
    }

    private static uint ReadWord(byte[] spirv, int offset) =>
        BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset, sizeof(uint)));
}
