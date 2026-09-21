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

public sealed class Gen5AlignByteTests
{
    [Fact]
    public void DecodeAlignByte()
    {
        var instruction = Decode(0xD54F0003, 0x0401FE80, 0x04030301);

        Assert.Equal("VAlignbyteB32", instruction.Opcode);
        Assert.Equal(Gen5Operand.Vector(3), Assert.Single(instruction.Destinations));
        Assert.Equal(Gen5Operand.Source(128), instruction.Sources[0]);
        Assert.Equal(new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x04030301), instruction.Sources[1]);
        Assert.Equal(Gen5Operand.Vector(0), instruction.Sources[2]);
    }

    [Fact]
    public void CompilesOnBothBackends()
    {
        var program = Program(
            Vop3(
                0,
                "VAlignbyteB32",
                3,
                Gen5Operand.Source(128),
                new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x04030301),
                Gen5Operand.Vector(0)),
            EndProgram(8));
        var plan = SharpEmu.ShaderCompiler.Resources.ShaderResourcePlan.Extract(
            program,
            SharpEmu.ShaderCompiler.Resources.ShaderStage.Compute,
            Hash,
            0,
            64);
        var resources = SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.ApplyTo(
            plan,
            SharpEmu.ShaderCompiler.Resources.ResourceSpecialization.Default(plan.Info));
        var layout = SharpEmu.ShaderCompiler.Resources.BindingLayout.Allocate(
            resources.Info,
            SharpEmu.ShaderCompiler.Resources.BindingLayout.CollectUserDataRegisters(program, 0, 64),
            false,
            false,
            false,
            0);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            LocalSizeX = 1,
            ThreadCountX = 1,
        };

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
