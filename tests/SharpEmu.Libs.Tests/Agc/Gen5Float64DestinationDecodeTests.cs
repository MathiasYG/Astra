// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class Gen5Float64DestinationDecodeTests
{
    [Theory]
    [InlineData(0x7E060903u, 0u, "VCvtF64I32")]
    [InlineData(0x7E065F03u, 0u, "VRcpF64")]
    [InlineData(0xD5650003u, 0x000E0B03u, "VMulF64")]
    [InlineData(0xD54C0003u, 0x042E0B03u, "VFmaF64")]
    public void DoubleResultDecodesBothDestinationWords(uint first, uint second, string opcode)
    {
        const ulong address = 0x1000;
        var memory = new FakeCpuMemory(address, 0x1000);
        Gen5ShaderAtomicDecodeTests.WriteProgram(memory, address,
            second == 0 ? [first] : [first, second]);
        var context = new CpuContext(memory, Generation.Gen5);
        Assert.True(Gen5ShaderTranslator.TryDecodeProgram(context, address, out var program, out var error), error);
        var instruction = program.Instructions[0];
        Assert.Equal(opcode, instruction.Opcode);
        Assert.Equal(new[] { Gen5Operand.Vector(3), Gen5Operand.Vector(4) }, instruction.Destinations);
    }
}