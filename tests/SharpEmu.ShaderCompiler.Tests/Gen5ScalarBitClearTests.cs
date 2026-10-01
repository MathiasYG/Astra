// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5ScalarBitClearTests
{
    public static TheoryData<uint, uint, uint> Values => new()
    {
        { 0xFFFFFFFF, 0, 0xFFFFFFFE },
        { 0xFFFFFFFF, 31, 0x7FFFFFFF },
        { 0xFFFFFFFF, 32, 0xFFFFFFFE },
        { 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF },
        { 0, 31, 0 },
        { 0xA5A5A5A5, 2, 0xA5A5A5A1 },
        { 0xA5A5A5A5, 1, 0xA5A5A5A5 },
    };

    [Fact]
    public void DecodeReportedWord()
    {
        var instruction = Decode(0xBEEB1B9F);
        Assert.Equal("SBitset0B32", instruction.Opcode);
        Assert.Equal(Gen5Operand.Scalar(107), Assert.Single(instruction.Destinations));
        Assert.Equal(Gen5Operand.Source(0x9F), Assert.Single(instruction.Sources));
    }

    [Fact]
    public void DecodeReported64BitWord()
    {
        var instruction = Decode(0xBEA61C33);
        Assert.Equal("SBitset0B64", instruction.Opcode);
        Assert.Equal(Gen5Operand.Scalar(38), Assert.Single(instruction.Destinations));
        Assert.Equal(Gen5Operand.Source(0x33), Assert.Single(instruction.Sources));
    }

    [Theory]
    [InlineData(0xBE881B09u)]
    [InlineData(0xBE881D09u)]
    public void BitUpdatePreservesTheImplicitDestinationInput(uint word)
    {
        var registers = BindingLayout.CollectUserDataRegisters(Program(Decode(word), EndProgram(4)), 0, 16);
        Assert.Contains(8u, registers);
        Assert.Contains(9u, registers);
    }

    [Fact]
    public void BitUpdate64ReadsTheWholeImplicitDestinationPairAndOneBitIndexRegister()
    {
        var registers = BindingLayout.CollectUserDataRegisters(Program(Decode(0xBE881C0A), EndProgram(4)), 0, 16);
        Assert.Contains(8u, registers);
        Assert.Contains(9u, registers);
        Assert.Contains(10u, registers);
        Assert.DoesNotContain(11u, registers);
    }

    [Theory]
    [MemberData(nameof(Values))]
    public void ResourceEvaluationClearsOnlySelectedBit(uint initial, uint selector, uint expected)
    {
        var program = Program(
            MoveScalar(0, 4, initial),
            Decode(0xBE841B08) with { Pc = 4 },
            MoveScalar(8, 5, 0), MoveScalar(12, 6, 16), MoveScalar(16, 7, 0),
            BufferLoad(20, 4), EndProgram(28));
        var plan = Extract(program);
        var registers = new uint[16];
        registers[8] = selector;
        Assert.True(RuntimeValueEvaluator.EvaluateDescriptorSource(plan, plan.Info.Buffers[0].Source,
            Inputs(registers), out var result));
        Assert.Equal(expected, result.Dwords[0]);
    }

    [Theory]
    [InlineData(0u, 0xA5A5A5A4u, 0x80000001u)]
    [InlineData(31u, 0x25A5A5A5u, 0x80000001u)]
    [InlineData(32u, 0xA5A5A5A5u, 0x80000000u)]
    [InlineData(63u, 0xA5A5A5A5u, 0x00000001u)]
    [InlineData(64u, 0xA5A5A5A4u, 0x80000001u)]
    [InlineData(0xFFFFFFFFu, 0xA5A5A5A5u, 0x00000001u)]
    public void ResourceEvaluationClearsOnlySelectedBitIn64BitPair(uint selector, uint expectedLow, uint expectedHigh)
    {
        var program = Program(
            MoveScalar(0, 4, 0xA5A5A5A5),
            MoveScalar(4, 5, 0x80000001),
            MoveScalar(8, 6, 16),
            MoveScalar(12, 7, 0),
            MoveScalar(16, 8, selector),
            Decode(0xBE841C08) with { Pc = 20 },
            BufferLoad(24, 4),
            EndProgram(32));
        var plan = Extract(program);
        Assert.True(RuntimeValueEvaluator.EvaluateDescriptorSource(plan, plan.Info.Buffers[0].Source,
            Inputs([]), out var result));
        Assert.Equal(expectedLow, result.Dwords[0]);
        Assert.Equal(expectedHigh, result.Dwords[1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompilesOnBothBackends(bool overlapDestination)
    {
        var (plan, resources, layout) = Prepare(CreateReadbackProgram(false, overlapDestination, true));
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    [Fact]
    public void BitClear64CompilesOnBothBackends()
    {
        var program = Program(
            MoveScalar(0, 8, 0xFFFFFFFF),
            MoveScalar(4, 9, 0xFFFFFFFF),
            MoveScalar(8, 10, 63),
            Decode(0xBE881C0A) with { Pc = 12 },
            EndProgram(16));
        var (plan, resources, layout) = Prepare(program);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    public static Gen5ShaderProgram CreateReadbackProgram(bool emptyExecutionMask, bool overlapDestination, bool condition)
    {
        return Program(
            Decode(condition ? 0xBF008080u : 0xBF008180u),
            MoveScalar(4, 126, emptyExecutionMask ? 0u : 1u),
            Decode(overlapDestination ? 0xBE881B08u : 0xBE881B09u) with { Pc = 8 },
            Decode(0x850A8180) with { Pc = 12 },
            MoveScalar(16, 126, 1),
            MoveVectorFromScalar(20, 6, 8),
            MoveVectorFromScalar(24, 7, 10),
            BufferAccess(28, "BufferStoreDwordx2", 4, dwords: 2, vectorData: 6), EndProgram(36));
    }

    private static Gen5ShaderInstruction Decode(uint word)
    {
        var bytes = new byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, word);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), 0xBF810000);
        var context = new CpuContext(new InstructionMemory(bytes), Generation.Gen5);
        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(context, 0x1000, out var program, out var error), error);
        return program.Instructions[0];
    }

    private sealed class InstructionMemory(byte[] bytes) : ICpuMemory
    {
        public bool TryRead(ulong address, Span<byte> destination)
        {
            if (address < 0x1000 || destination.Length > bytes.Length ||
                address - 0x1000 > (ulong)(bytes.Length - destination.Length)) return false;
            bytes.AsSpan((int)(address - 0x1000), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong address, ReadOnlySpan<byte> source) => false;
    }
}

public sealed class Gen5ScalarBitReplicateTests
{
    [Fact]
    public void DecodeReportedLiteralInstruction()
    {
        var bytes = new byte[12];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, 0xBEFE3BFF);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), 0x8000_0001);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8), 0xBF81_0000);
        var context = new CpuContext(new InstructionMemory(bytes), Generation.Gen5);

        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(context, 0x1000, out var program, out var error), error);
        var instruction = program.Instructions[0];
        Assert.Equal("SBitreplicateB64B32", instruction.Opcode);
        Assert.Equal(2, instruction.Words.Count);
        Assert.Equal(Gen5Operand.Scalar(126), Assert.Single(instruction.Destinations));
        Assert.Equal(Gen5Operand.Source(0xFF, 0x8000_0001), Assert.Single(instruction.Sources));
    }

    [Theory]
    [InlineData(0u, 0u, 0u)]
    [InlineData(1u, 3u, 0u)]
    [InlineData(0x8000_0000u, 0u, 0xC000_0000u)]
    [InlineData(0xA5A5_0001u, 3u, 0xCC33_CC33u)]
    public void ResourceGraphReplicatesEachSourceBit(uint source, uint expectedLow, uint expectedHigh)
    {
        var program = Program(
            MoveScalar(0, 8, source),
            Sop1(4, "SBitreplicateB64B32", 4, Gen5Operand.Scalar(8)),
            MoveScalar(8, 6, 16), MoveScalar(12, 7, 0),
            BufferLoad(16, 4), EndProgram(24));
        var plan = Extract(program);

        Assert.True(RuntimeValueEvaluator.EvaluateDescriptorSource(
            plan, plan.Info.Buffers[0].Source, Inputs([]), out var result));
        Assert.Equal(expectedLow, result.Dwords[0]);
        Assert.Equal(expectedHigh, result.Dwords[1]);
    }

    [Fact]
    public void CompilesOnBothBackends()
    {
        var program = Program(
            MoveScalar(0, 8, 0xA5A5_0001),
            Sop1(4, "SBitreplicateB64B32", 4, Gen5Operand.Scalar(8)),
            MoveScalar(8, 6, 16), MoveScalar(12, 7, 0),
            BufferLoad(16, 4), EndProgram(24));
        var (plan, resources, layout) = Prepare(program);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    private sealed class InstructionMemory(byte[] bytes) : ICpuMemory
    {
        public bool TryRead(ulong address, Span<byte> destination)
        {
            if (address < 0x1000 || destination.Length > bytes.Length ||
                address - 0x1000 > (ulong)(bytes.Length - destination.Length)) return false;
            bytes.AsSpan((int)(address - 0x1000), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong address, ReadOnlySpan<byte> source) => false;
    }
}
