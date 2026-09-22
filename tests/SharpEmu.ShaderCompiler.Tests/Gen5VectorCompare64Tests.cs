// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5VectorCompare64Tests
{
    [Fact]
    public void DecodeVop3Signed64Compare()
    {
        // V_CMPX_NE_I64 with SRC0=0, SRC1=v7:v8, and the unused VOPC SRC2=s0.
        var instruction = Decode(0xD4B5007E, 0x00020E80);

        Assert.Equal("VCmpxNeI64", instruction.Opcode);
        Assert.Empty(instruction.Destinations);
        Assert.Equal(
            new[] { Gen5Operand.Source(128), Gen5Operand.Vector(7), Gen5Operand.Scalar(0) },
            instruction.Sources);
    }

    [Fact]
    public void DecodeVop3Unsigned64Compare()
    {
        var instruction = Decode(0xD4F5007E, 0x00020E80);

        Assert.Equal("VCmpxNeU64", instruction.Opcode);
        Assert.Empty(instruction.Destinations);
        Assert.Equal(
            new[] { Gen5Operand.Source(128), Gen5Operand.Vector(7), Gen5Operand.Scalar(0) },
            instruction.Sources);
    }

    [Fact]
    public void CompilesOnBothBackends()
    {
        var request = Request(
            Program(
                Vop3(
                    0,
                    "VCmpxNeI64",
                    0,
                    Gen5Operand.Source(128),
                    Gen5Operand.Vector(7),
                    Gen5Operand.Scalar(0)),
                EndProgram(8)));

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    [Fact]
    public void UnsignedCompilesOnBothBackends()
    {
        var request = Request(
            Program(
                Vop3(
                    0,
                    "VCmpxNeU64",
                    0,
                    Gen5Operand.Source(128),
                    Gen5Operand.Vector(7),
                    Gen5Operand.Scalar(0)),
                EndProgram(8)));

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    private static Gen5ShaderInstruction Decode(params uint[] instructionWords)
    {
        var bytes = new byte[(instructionWords.Length + 1) * sizeof(uint)];
        for (var index = 0; index < instructionWords.Length; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)), instructionWords[index]);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(instructionWords.Length * sizeof(uint)), 0xBF810000);
        var context = new CpuContext(new InstructionMemory(bytes), Generation.Gen5);
        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(context, 0x1000, out var program, out var error), error);
        return program.Instructions[0];
    }

    private sealed class InstructionMemory(byte[] bytes) : ICpuMemory
    {
        public bool TryRead(ulong address, Span<byte> destination)
        {
            if (address < 0x1000 || destination.Length > bytes.Length ||
                address - 0x1000 > (ulong)(bytes.Length - destination.Length))
                return false;
            bytes.AsSpan((int)(address - 0x1000), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong address, ReadOnlySpan<byte> source) => false;
    }
}
