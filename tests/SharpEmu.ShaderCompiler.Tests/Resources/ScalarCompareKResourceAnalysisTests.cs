// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ScalarCompareKResourceAnalysisTests
{
    [Theory]
    [InlineData("SCmpkGeU32", 0xFFFF, ScalarOperation.UGreaterThanEqual32, 0xFFFFu)]
    [InlineData("SCmpkGeI32", 0xFFFF, ScalarOperation.SGreaterThanEqual32, 0xFFFF_FFFFu)]
    public void ScalarCompareKUsesItsEncodedSourceAndImmediate(
        string opcode,
        ushort immediate,
        ScalarOperation expectedComparison,
        uint expectedImmediate)
    {
        var program = Program(
            ScalarCompareK(0, opcode, source: 10, immediate),
            Sop2(4, "SMulI32", 12, Gen5Operand.Scalar(10), Operand(384)),
            EndProgram(8));
        var graph = ScalarValueGraph.Build(program, userDataBase: 10, userDataCount: 1);

        var source = Assert.Single(graph.Values, value =>
            value.Kind == ScalarValueKind.UserData && value.UserDataRegister == 10);
        var comparison = Assert.Single(graph.Values, value =>
            value.Kind == ScalarValueKind.Operation && value.Operation == expectedComparison);
        var product = Assert.Single(graph.Values, value =>
            value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.IMul32);

        Assert.Same(source, comparison.Operands[0]);
        Assert.Equal(expectedImmediate, comparison.Operands[1].ConstantU32);
        Assert.Same(source, product.Operands[0]);
        Assert.Equal(384u, product.Operands[1].ConstantU32);
    }

    private static Gen5ShaderInstruction ScalarCompareK(uint pc, string opcode, uint source, ushort immediate)
    {
        var word = ((source & 0x7Fu) << 16) | immediate;
        return new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopk, opcode, [word],
            [new Gen5Operand(Gen5OperandKind.EncodedConstant, immediate)],
            [Gen5Operand.Scalar(source)], null);
    }
}
