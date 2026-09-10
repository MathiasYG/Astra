// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class Gen5VertexOffsetTests
{
    [Fact]
    public void NggSadProlog_ResolvesVertexOffsetFromUserData()
    {
        var state = CreateState(
            new Gen5ShaderInstruction(
                0,
                Gen5ShaderEncoding.Vop3,
                "VSadU32",
                [],
                [Gen5Operand.Scalar(18), Gen5Operand.Source(128), Gen5Operand.Vector(5)],
                [Gen5Operand.Vector(5)],
                null),
            userData: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0]);

        Assert.True(Gen5ShaderTranslator.TryGetEmbeddedFetchVertexOffset(state, out var offset));
        Assert.Equal(8, offset);
    }

    [Fact]
    public void NggSadProlog_WithNonzeroMiddleOperand_IsNotClassified()
    {
        var state = CreateState(
            new Gen5ShaderInstruction(
                0,
                Gen5ShaderEncoding.Vop3,
                "VSadU32",
                [],
                [Gen5Operand.Scalar(18), Gen5Operand.Source(129), Gen5Operand.Vector(5)],
                [Gen5Operand.Vector(5)],
                null),
            userData: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0]);

        Assert.False(Gen5ShaderTranslator.TryGetEmbeddedFetchVertexOffset(state, out _));
    }

    [Fact]
    public void VertexOffsetAfterFirstFetch_IsNotClassified()
    {
        var fetch = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Mubuf,
            "BufferLoadFormatX",
            [],
            [],
            [],
            new Gen5BufferMemoryControl(1, 0, 0, 0, 0, true, false, false, false));
        var state = CreateState(
            fetch,
            userData: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
            new Gen5ShaderInstruction(
                4,
                Gen5ShaderEncoding.Vop3,
                "VSadU32",
                [],
                [Gen5Operand.Scalar(18), Gen5Operand.Source(128), Gen5Operand.Vector(5)],
                [Gen5Operand.Vector(5)],
                null));

        Assert.False(Gen5ShaderTranslator.TryGetEmbeddedFetchVertexOffset(state, out _));
    }

    private static Gen5ShaderState CreateState(
        Gen5ShaderInstruction first,
        IReadOnlyList<uint> userData,
        params Gen5ShaderInstruction[] rest) =>
        new(
            new Gen5ShaderProgram(0, [first, .. rest]),
            userData,
            Metadata: null,
            UserDataScalarRegisterBase: 8);
}
