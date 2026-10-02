// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

using System.Buffers.Binary;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Tests.Agc;

// Typed loads use the instruction format; untyped loads specialize on the descriptor format.
public sealed class Gen5TypedBufferLoadSpirvTests
{
    private const string DescriptorFormatTableName = "gfx10BufferFormats";

    [Fact]
    public void TypedLoad_DoesNotDecodeTheDescriptorFormat()
    {
        var shader = CompileBufferProgram(typed: true, typedFormat: 22);

        Assert.DoesNotContain(DescriptorFormatTableName, ModuleNames(shader.Spirv));
    }

    [Fact]
    public void FormattedUntypedLoad_UsesSpecializedFormatAndConversionTable()
    {
        var request = CreateCompileRequest("BufferLoadFormatXyzw", typed: false, typedFormat: 0, dwordCount: 4);
        Assert.Equal(77u, Assert.Single(request.Resources.Info.Buffers).DescriptorFormat);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);

        Assert.Contains(DescriptorFormatTableName, ModuleNames(shader.Spirv));
    }

    [Fact]
    public void TypedLoad_WithAnUnknownInstructionFormat_ReadsRawDwords()
    {
        const uint reservedFormat = 30;
        var shader = CompileBufferProgram(typed: true, typedFormat: reservedFormat);

        Assert.DoesNotContain(DescriptorFormatTableName, ModuleNames(shader.Spirv));
    }

    [Fact]
    public void TypedD16Load_IsRejected()
    {
        var request = CreateCompileRequest("TBufferLoadFormatD16Xy", typed: true, typedFormat: 64, dwordCount: 1);

        Assert.False(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("TBufferLoadFormatD16Xy", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TypedSubwordStore_MergesThroughAnAtomicCompareExchange()
    {
        // Format 14 is 8_8: the two bytes merge into their dword without touching the other two.
        var shader = CompileBufferProgram("TBufferStoreFormatXy", typed: true, typedFormat: 14, dwordCount: 2);

        Assert.Contains(OpAtomicCompareExchange, ModuleOpcodes(shader.Spirv));
    }

    [Fact]
    public void TypedWholeDwordStore_StoresWithoutAnAtomicUpdate()
    {
        // Format 77 is 32x4 float: whole aligned dwords need no read-modify-write.
        var shader = CompileBufferProgram("TBufferStoreFormatXyzw", typed: true, typedFormat: 77, dwordCount: 4);

        Assert.DoesNotContain(OpAtomicCompareExchange, ModuleOpcodes(shader.Spirv));
    }

    private const uint OpAtomicCompareExchange = 230;

    private static Gen5SpirvShader CompileBufferProgram(string opcode, bool typed, uint typedFormat, uint dwordCount)
    {
        var request = CreateCompileRequest(opcode, typed, typedFormat, dwordCount);
        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error),
            error);
        return shader;
    }

    private static IReadOnlyList<uint> ModuleOpcodes(byte[] spirv)
    {
        var opcodes = new List<uint>();
        var words = new uint[spirv.Length / sizeof(uint)];
        Buffer.BlockCopy(spirv, 0, words, 0, spirv.Length);
        for (var offset = 5; offset < words.Length; offset += (int)(words[offset] >> 16))
        {
            opcodes.Add(words[offset] & 0xFFFF);
        }

        return opcodes;
    }

    private static Gen5SpirvShader CompileBufferProgram(bool typed, uint typedFormat)
    {
        var request = CreateCompileRequest(
            typed ? "TBufferLoadFormatXyzw" : "BufferLoadFormatXyzw",
            typed,
            typedFormat,
            dwordCount: 4);
        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error),
            error);
        return shader;
    }

    private static ShaderCompileRequest CreateCompileRequest(
        string opcode,
        bool typed,
        uint typedFormat,
        uint dwordCount)
    {
        var load = new Gen5ShaderInstruction(
            0,
            typed ? Gen5ShaderEncoding.Mtbuf : Gen5ShaderEncoding.Mubuf,
            opcode,
            [0, 0],
            [Gen5Operand.Vector(0), Gen5Operand.Scalar(8), Gen5Operand.Source(128, null)],
            [Gen5Operand.Vector(4)],
            new Gen5BufferMemoryControl(
                dwordCount,
                0,
                4,
                8,
                0,
                IndexEnabled: false,
                OffsetEnabled: false,
                Glc: false,
                Slc: false,
                Typed: typed,
                TypedFormat: typedFormat));
        var end = new Gen5ShaderInstruction(8, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null);
        var program = new Gen5ShaderProgram(0, [load, end]);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 1, 0, 12);
        uint[] userData = [0, 0, 0, 0, 0, 0, 0, 0, 0x2000, 0, 64, 77u << 12];
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, ResourceTestProgram.Inputs(userData),
            ref snapshot, ref specialization, out var failure), failure.ToString());
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, 12), false, false, false);
        return new ShaderCompileRequest(plan, resources, layout);
    }

    private static IReadOnlyList<string> ModuleNames(byte[] spirv)
    {
        const uint opName = 5;
        var names = new List<string>();
        var words = new uint[spirv.Length / sizeof(uint)];
        Buffer.BlockCopy(spirv, 0, words, 0, spirv.Length);
        for (var offset = 5; offset < words.Length;)
        {
            var wordCount = (int)(words[offset] >> 16);
            if ((words[offset] & 0xFFFF) == opName)
            {
                var bytes = new byte[(wordCount - 2) * sizeof(uint)];
                Buffer.BlockCopy(words, (offset + 2) * sizeof(uint), bytes, 0, bytes.Length);
                names.Add(Encoding.UTF8.GetString(bytes).TrimEnd('\0'));
            }

            offset += wordCount;
        }

        return names;
    }
}

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
