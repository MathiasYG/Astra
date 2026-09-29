// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Codec;
using Xunit;

namespace SharpEmu.Libs.Tests;

public sealed class Videodec2DecoderTests
{
    [Fact]
    public void WaitUntilDeadline_ReturnsWhenDeadlinePasses()
    {
        var stopwatch = Stopwatch.StartNew();

        var cancelled = Videodec2Decoder.WaitUntilDeadline(
            CancellationToken.None,
            DateTime.UtcNow.AddMilliseconds(20));

        Assert.False(cancelled);
        Assert.InRange(stopwatch.Elapsed, TimeSpan.FromMilliseconds(10), TimeSpan.FromSeconds(2));
    }

    [Fact]
    public void WaitUntilDeadline_WakesWhenCancellationIsRequested()
    {
        using var cancellation = new CancellationTokenSource();
        var cancelThread = new Thread(() =>
        {
            Thread.Sleep(20);
            cancellation.Cancel();
        })
        {
            IsBackground = true,
        };
        cancelThread.Start();

        var cancelled = Videodec2Decoder.WaitUntilDeadline(
            cancellation.Token,
            DateTime.UtcNow.AddSeconds(5));

        Assert.True(cancelled);
        Assert.True(cancelThread.Join(TimeSpan.FromSeconds(1)));
    }
}
