// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Pthread;

public sealed class PthreadHandleIdentityTests
{
    [Fact]
    public void CreatedPthreadObjectExposesItsNumericIdentity()
    {
        var handle = KernelPthreadState.CreateThreadHandle("identity-test");

        Assert.True(KernelPthreadState.TryGetThreadIdentity(handle, out var identity));
        Assert.NotEqual(0UL, identity.UniqueId);
        Assert.Equal(unchecked((int)identity.UniqueId),
            Marshal.ReadInt32(unchecked((nint)handle)));
    }
}
