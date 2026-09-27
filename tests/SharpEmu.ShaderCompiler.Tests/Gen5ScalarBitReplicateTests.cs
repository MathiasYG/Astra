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
