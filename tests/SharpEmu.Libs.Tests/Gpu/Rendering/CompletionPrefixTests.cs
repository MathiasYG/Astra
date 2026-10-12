// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

public sealed class CompletionPrefixTests
{
    [Fact]
    public void ItemsFinishedInOrderAdvanceTheCount()
    {
        var prefix = new CompletionPrefix();
        prefix.Complete(0);
        prefix.Complete(1);

        Assert.Equal(2, prefix.Count);
    }

    [Fact]
    public void AnItemFinishedEarlyCountsOnlyOnceTheGapCloses()
    {
        var prefix = new CompletionPrefix();
        prefix.Complete(1);
        prefix.Complete(3);
        Assert.Equal(0, prefix.Count);

        prefix.Complete(0);
        Assert.Equal(2, prefix.Count);

        prefix.Complete(2);
        Assert.Equal(4, prefix.Count);
    }
}
