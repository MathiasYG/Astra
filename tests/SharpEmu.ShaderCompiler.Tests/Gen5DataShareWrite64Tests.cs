// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

using System.Diagnostics;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5DataShareWrite64Tests
{
    public const uint InitialLowerWord = 0xDEADBEEF;
    public const uint InitialUpperWord = 0xFEEDABCD;

    public static IEnumerable<object[]> PairedWriteCases()
    {
        foreach (var largeStride in new[] { false, true })
        foreach (var global in new[] { false, true })
        {
            uint highestOffset = largeStride ? 63u : 255u;
            (uint First, uint Second)[] offsets = [(0, 8), (8, 0), (3, 3),
                (highestOffset, 1), (1, highestOffset), (highestOffset, highestOffset)];
            foreach (var writeEnabled in new[] { false, true })
            foreach (var (first, second) in offsets)
                yield return [global, first, second, writeEnabled, 20u, largeStride];
            yield return [global, 0u, 8u, true, 10u, largeStride];
            yield return [global, 0u, 8u, true, 11u, largeStride];
        }
    }

    public static uint SourceWord(uint register) => register switch
    {
        10 => 0xA1234567,
        11 => 0xB2345678,
        12 => 0xC3456789,
        20 => 0x11223344,
        21 => 0x55667788,
        _ => throw new ArgumentOutOfRangeException(nameof(register)),
    };

    public static uint[] ExpectedWords(uint firstOffset, uint secondOffset, bool writeEnabled, uint secondSource) => !writeEnabled
        ? [InitialLowerWord, InitialUpperWord, InitialLowerWord, InitialUpperWord]
        : [SourceWord(10), SourceWord(11), SourceWord(firstOffset == secondOffset ? 10 : secondSource),
            SourceWord(firstOffset == secondOffset ? 11 : secondSource + 1)];

    [Theory]
    [MemberData(nameof(PairedWriteCases))]
    public void PairedWriteDecodesIndependentSourcePairsAndCompilesOnBothBackends(
        bool global, uint firstOffset, uint secondOffset, bool writeEnabled, uint secondSource, bool largeStride)
    {
        var program = CreateReadbackProgram(global, firstOffset, secondOffset, writeEnabled, secondSource, largeStride);
        var opcode = largeStride ? "DsWrite2St64B64" : "DsWrite2B64";
        var write = Assert.Single(program.Instructions, instruction => instruction.Opcode == opcode);
        Assert.Equal([Gen5Operand.Vector(3), Gen5Operand.Vector(10), Gen5Operand.Vector(11),
            Gen5Operand.Vector(secondSource), Gen5Operand.Vector(secondSource + 1)], write.Sources);
        Assert.Empty(write.Destinations);
        var control = Assert.IsType<Gen5DataShareControl>(write.Control);
        Assert.Equal(firstOffset, control.Offset0);
        Assert.Equal(secondOffset, control.Offset1);
        Assert.Equal(global, control.Gds);
        var (plan, resources, layout) = Prepare(program);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    public static Gen5ShaderProgram CreateReadbackProgram(bool global, uint firstOffset, uint secondOffset, bool writeEnabled,
        uint secondSource, bool largeStride)
    {
        uint[] words = [(largeStride ? 0xD93C0000u : 0xD9380000u) | (global ? 1u << 17 : 0) | firstOffset | (secondOffset << 8),
            3u | (10u << 8) | (secondSource << 16), 0xBF810000];
        var bytes = new byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)), words[index]);
        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(new CpuContext(new InstructionMemory(bytes), Generation.Gen5), 0x1000,
            out var decoded, out var error), error);

        var offsetStride = largeStride ? 512u : 8u;
        var firstByteOffset = firstOffset * offsetStride;
        var secondByteOffset = secondOffset * offsetStride;
        return Program(
            MoveVector(0, 3, 8), MoveVector(4, 30, InitialLowerWord), MoveVector(8, 31, InitialUpperWord),
            DataShare(12, "DsWriteB64", global, [Gen5Operand.Vector(3), Gen5Operand.Vector(30), Gen5Operand.Vector(31)], [],
                firstByteOffset & 0xFF, firstByteOffset >> 8),
            DataShare(20, "DsWriteB64", global, [Gen5Operand.Vector(3), Gen5Operand.Vector(30), Gen5Operand.Vector(31)], [],
                secondByteOffset & 0xFF, secondByteOffset >> 8),
            MoveVector(28, 10, SourceWord(10)), MoveVector(32, 11, SourceWord(11)), MoveVector(36, 12, SourceWord(12)),
            MoveVector(40, 20, SourceWord(20)), MoveVector(44, 21, SourceWord(21)),
            MoveScalar(48, 126, writeEnabled ? 1u : 0u), decoded.Instructions[0] with { Pc = 52 }, MoveScalar(60, 126, 1),
            DataShare(64, "DsReadB64", global, [Gen5Operand.Vector(3)], [4, 5], firstByteOffset & 0xFF, firstByteOffset >> 8),
            DataShare(72, "DsReadB64", global, [Gen5Operand.Vector(3)], [6, 7], secondByteOffset & 0xFF, secondByteOffset >> 8),
            BufferAccess(80, "BufferStoreDwordx4", 4, dwords: 4, vectorData: 4), EndProgram(88));
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

public sealed class Gen5DataShareAtomic64Tests
{
    [Theory]
    [InlineData(32u)]
    [InlineData(64u)]
    public void MixedWidthLdsAccessesProduceValidSpirv(uint waveSize)
    {
        var program = Program(
            MoveVector(0, 0, 0),
            MoveVector(4, 1, 0xFFFF_FFFF),
            MoveVector(8, 2, 1),
            MoveVector(12, 3, 1),
            DataShare(16, "DsWriteB64", false,
                [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)], []),
            DataShare(24, "DsAddU64", false,
                [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)], []),
            DataShare(32, "DsAddU32", false,
                [Gen5Operand.Vector(0), Gen5Operand.Vector(3)], [], offset0: 8, offset1: 2),
            DataShare(40, "DsReadB64", false, [Gen5Operand.Vector(0)], [4, 5]),
            EndProgram(48));
        var (plan, resources, layout) = Prepare(program);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            LocalSizeX = waveSize,
            ThreadCountX = waveSize,
            WaveSize = waveSize,
        };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);

        var integerWidths = new Dictionary<uint, uint>();
        var atomicResultTypes = new List<uint>();
        for (var offset = 5 * sizeof(uint); offset < shader.Spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(shader.Spirv.AsSpan(offset));
            var length = (int)(header >> 16);
            Assert.True(length > 0);
            uint Operand(int index) => BinaryPrimitives.ReadUInt32LittleEndian(
                shader.Spirv.AsSpan(offset + index * sizeof(uint)));
            if ((header & 0xFFFF) == (uint)SpirvOp.TypeInt)
                integerWidths.Add(Operand(1), Operand(2));
            if ((header & 0xFFFF) == (uint)SpirvOp.AtomicIAdd)
                atomicResultTypes.Add(Operand(1));
            offset += length * sizeof(uint);
        }
        Assert.Contains(atomicResultTypes, type => integerWidths[type] == 64);
        Assert.Contains(atomicResultTypes, type => integerWidths[type] == 32);

        var sdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
        if (string.IsNullOrWhiteSpace(sdk)) return;
        var validator = Path.Combine(sdk, OperatingSystem.IsWindows() ? "Bin/spirv-val.exe" : "bin/spirv-val");
        if (!File.Exists(validator)) return;

        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, shader.Spirv);
            var start = new ProcessStartInfo(validator)
            {
                UseShellExecute = false,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            start.ArgumentList.Add("--target-env");
            start.ArgumentList.Add("vulkan1.3");
            start.ArgumentList.Add(path);
            using var process = Process.Start(start)!;
            var validationError = process.StandardError.ReadToEnd();
            process.WaitForExit();
            Assert.True(process.ExitCode == 0, validationError);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
