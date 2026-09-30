// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ScalarImageConsumerTests
{
    private static ShaderResourcePlan Plan(Gen5ShaderInstruction consumer)
    {
        var program = DirectImageTableTests.CreateProgram();
        program = program with
        {
            Instructions = program.Instructions.Select(instruction =>
                instruction.Pc == 40 ? consumer : instruction).ToArray(),
        };
        return ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
    }

    private static void AssertSuppressed(ShaderResourcePlan plan, bool suppressed)
    {
        Assert.Single(plan.IndirectImages);
        Assert.True(plan.Memory.TryGetIndex(24, 0, out var index));
        Assert.Equal(suppressed, plan.Memory[index].PlanningOnly);
    }

    [Theory]
    [InlineData("VMovrelsB32")]
    [InlineData("VMovreldB32")]
    [InlineData("VMovrelsdB32")]
    [InlineData("VMovrelsd2B32")]
    public void VectorRelativeMoveDoesNotReadScalarImageDescriptor(string opcode)
    {
        var consumer = new Gen5ShaderInstruction(40, Gen5ShaderEncoding.Vop1,
            opcode, [0u], [Gen5Operand.Vector(1), Gen5Operand.Scalar(124)],
            [Gen5Operand.Vector(12)], null);
        var plan = Plan(consumer);
        AssertSuppressed(plan, true);
    }

    [Theory]
    [InlineData("SMovrelsB32")]
    [InlineData("SMovreldB32")]
    public void ScalarRelativeMoveStillPreventsDescriptorLoadSuppression(string opcode)
    {
        var plan = Plan(Sop1(40, opcode, 20, Gen5Operand.Scalar(1)));
        AssertSuppressed(plan, false);
    }

    [Fact]
    public void VectorRelativeDestinationStillReadsItsExplicitScalarSource()
    {
        var plan = Plan(Vop1(40, "VMovreldB32", 12, Gen5Operand.Scalar(4)));
        AssertSuppressed(plan, false);
    }
}
