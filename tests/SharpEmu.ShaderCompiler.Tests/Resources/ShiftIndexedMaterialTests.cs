// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ShiftIndexedMaterialTests
{
    private static Gen5ShaderProgram MaterialProgram(uint shift, bool multiply = false) => Program(
        MoveVectorFromScalar(0, 1, 8),
        ReadFirstLane(4, 9, 1),
        Sop2(8, multiply ? "SMulI32" : "SLshlB32", 10, Gen5Operand.Scalar(9),
            Operand(multiply ? 1u << (int)(shift & 31) : shift)),
        Sop2(12, "SAddU32", 11, Gen5Operand.Scalar(10), Operand(4)),
        ScalarBufferLoad(16, 0, 12, dynamicOffsetRegister: 11),
        Sop2(24, "SLshlB32", 13, Gen5Operand.Scalar(12), Operand(5)),
        ScalarBufferLoad(28, 4, 16, count: 8, dynamicOffsetRegister: 13),
        MoveScalar(36, 24, 0), MoveScalar(40, 25, 0),
        MoveScalar(44, 26, 0), MoveScalar(48, 27, 0),
        Image(52, "ImageSample", 16, 24), EndProgram(60));

    [Theory]
    [InlineData(0u, 1u)]
    [InlineData(5u, 32u)]
    [InlineData(37u, 32u)]
    [InlineData(31u, 0x80000000u)]
    public void ConstantShiftPreservesTheMaterialStrideAndImmediate(uint count, uint expectedStride)
    {
        var plan = Extract(MaterialProgram(count));
        var selector = Assert.Single(plan.DescriptorSources, source => source.IndirectImage is not null).IndirectImage!;
        Assert.Equal(expectedStride, selector.SelectorStride);
        Assert.Equal(4u, selector.SelectorOffset);
        Assert.Single(plan.IndirectImages);
    }

    [Theory]
    [InlineData(5u)]
    [InlineData(37u)]
    public void ShiftAndMultiplyProduceTheSameNonemptyDescriptorTable(uint count)
    {
        var memory = new TestWordMemory { Words = new uint[0x1040 / 4], RequireAlignment = true };
        memory.At(0x1004) = 0;
        memory.At(0x1024) = 1;
        // RGBA32 float, identity component mapping, 2D image.
        uint[] first = [0x20, 77u << 20, 3 | (3 << 14), 0xFAC | (9u << 28), 0, 0, 0, 0];
        var second = (uint[])first.Clone();
        second[0] = 0x40;
        for (var component = 0; component < first.Length; component++)
        {
            memory.At(0x2000 + (ulong)component * 4) = first[component];
            memory.At(0x2020 + (ulong)component * 4) = second[component];
        }
        uint[] userData = [0x1000, 32u << 16, 2, 0, 0x2000, 16u << 16, 4, 0, 0];
        var results = new List<ResourceSnapshot>();
        foreach (var multiply in new[] { false, true })
        {
            var plan = Extract(MaterialProgram(count, multiply));
            var snapshot = new ResourceSnapshot();
            var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readCleanMemory: memory.Read),
                ref snapshot, ref specialization));
            results.Add(snapshot);
        }
        Assert.Equal(2, results[0].Images.Length);
        Assert.Equal(first, results[0].Images[0]);
        Assert.Equal(second, results[0].Images[1]);
        Assert.Equal(results[1].FlattenedResourceTable, results[0].FlattenedResourceTable);
        Assert.Equal(results[1].Images.SelectMany(words => words), results[0].Images.SelectMany(words => words));
    }
}
