// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Buffers.Binary;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5DataShareAtomic64Tests
{
    [Fact]
    public void MixedWidthLdsAccessesProduceValidSpirv()
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
            LocalSizeX = 1,
            ThreadCountX = 1,
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
