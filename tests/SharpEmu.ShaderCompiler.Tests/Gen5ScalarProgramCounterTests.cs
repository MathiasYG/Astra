// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5ScalarProgramCounterTests
{
    [Fact]
    public void SwappcB64CompilesOnBothBackends()
    {
        var program = Program(
            Sop1(0, "SSwappcB64", 8, Gen5Operand.Scalar(4)),
            EndProgram(4));
        var request = Request(program);

        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError),
            spirvError);
        Assert.True(
            Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError),
            metalError);
    }
}
