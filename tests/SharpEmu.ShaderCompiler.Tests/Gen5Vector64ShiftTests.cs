// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5Vector64ShiftTests
{
    [Theory]
    [InlineData(0x2FFu, "VLshlrevB64")]
    [InlineData(0x300u, "VLshrrevB64")]
    public void Vector64ShiftDecodesAsA64BitVgprPairOperation(uint opcode, string expectedOpcode)
    {
        var program = Decode(
        [
            // v[4:5] = v[2:3] << or >> s0[5:0].
            0xD2000004u | (opcode << 16),
            (0u << 0) | (258u << 9) | (258u << 18),
            0xBF810000u,
        ]);

        var instruction = program.Instructions[0];
        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(4), Gen5Operand.Vector(5)],
            instruction.Destinations);
        Assert.Equal(Gen5Operand.Scalar(0), instruction.Sources[0]);
        Assert.Equal(Gen5Operand.Vector(2), instruction.Sources[1]);
    }

    [Theory]
    [InlineData("VLshlrevB64")]
    [InlineData("VLshrrevB64")]
    public void Vector64ShiftLowersToWideShiftAndPairedStores(string opcode)
    {
        var shift = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            opcode,
            [0, 0],
            [Gen5Operand.Scalar(0), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(4), Gen5Operand.Vector(5)],
            new Gen5Vop3Control(0, 0, 0, false, 0, null));
        var request = ResourceTestProgram.Request(
            new Gen5ShaderProgram(0, [shift, ResourceTestProgram.EndProgram(8)]),
            userDataCount: 0);

        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error),
            error);

        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains((ushort)SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains((ushort)SpirvOp.UConvert, opcodes);
        if (opcode == "VLshlrevB64")
        {
            Assert.Contains((ushort)SpirvOp.ShiftLeftLogical, opcodes);
        }
    }

    private static Gen5ShaderProgram Decode(IReadOnlyList<uint> words)
    {
        const ulong address = 0x1_0000_0000;
        var memory = new TestCpuMemory(address, words.Count * sizeof(uint));
        var bytes = new byte[words.Count * sizeof(uint)];
        for (var index = 0; index < words.Count; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                bytes.AsSpan(index * sizeof(uint)),
                words[index]);
        }

        Assert.True(memory.TryWrite(address, bytes));
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                new CpuContext(memory, Generation.Gen5),
                address,
                out var program,
                out var error),
            error);
        return program;
    }

    private static IReadOnlyList<ushort> ReadOpcodes(byte[] spirv)
    {
        var opcodes = new List<ushort>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var wordCount = checked((int)(header >> 16));
            Assert.InRange(wordCount, 1, (spirv.Length - offset) / sizeof(uint));
            opcodes.Add((ushort)header);
            offset += wordCount * sizeof(uint);
        }

        return opcodes;
    }

    private sealed class TestCpuMemory(ulong baseAddress, int size) : ICpuMemory
    {
        private readonly byte[] _storage = new byte[size];

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!TryResolve(virtualAddress, destination.Length, out var offset))
            {
                return false;
            }

            _storage.AsSpan(offset, destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!TryResolve(virtualAddress, source.Length, out var offset))
            {
                return false;
            }

            source.CopyTo(_storage.AsSpan(offset, source.Length));
            return true;
        }

        private bool TryResolve(ulong virtualAddress, int length, out int offset)
        {
            offset = 0;
            if (virtualAddress < baseAddress || virtualAddress - baseAddress > int.MaxValue)
            {
                return false;
            }

            offset = (int)(virtualAddress - baseAddress);
            return offset <= _storage.Length - length;
        }
    }
}
