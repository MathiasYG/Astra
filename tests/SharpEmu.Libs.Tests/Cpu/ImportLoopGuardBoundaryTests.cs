// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Cpu.Native;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

public sealed class ImportLoopGuardBoundaryTests
{
    [Theory]
    [InlineData("n88vx3C5nW8")] // gettimeofday
    [InlineData("-2IRUCO--PM")] // clock_gettime
    [InlineData("0V5nU-Z6t4U")] // sceKernelGetProcessTime
    [InlineData("aI6lQW5v57k")] // sceKernelGetProcessTimeCounter
    public void TimeObservationResetsRepeatingImportPattern(string nid)
    {
        Assert.True(DirectExecutionBackend.IsImportLoopGuardBoundary(nid));
    }

    [Fact]
    public void OrdinaryImportDoesNotResetRepeatingImportPattern()
    {
        Assert.False(DirectExecutionBackend.IsImportLoopGuardBoundary("ordinary-import"));
    }
}
