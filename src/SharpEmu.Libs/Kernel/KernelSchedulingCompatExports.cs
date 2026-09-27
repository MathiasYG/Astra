// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;

namespace SharpEmu.Libs.Kernel;

public static class KernelSchedulingCompatExports
{
    // FreeBSD's POSIX scheduling policies: OTHER=0, FIFO=1, RR=2.
    [SysAbiExport(ExportName = "sched_get_priority_max",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int GetPriorityMax(CpuContext ctx)
    {
        var policy = unchecked((int)ctx[CpuRegister.Rdi]);
        return ctx.SetReturn(policy switch
        {
            0 => 0,
            1 or 2 => 31,
            _ => -1,
        });
    }

    [SysAbiExport(ExportName = "sched_get_priority_min",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libKernel")]
    public static int GetPriorityMin(CpuContext ctx)
    {
        var policy = unchecked((int)ctx[CpuRegister.Rdi]);
        return ctx.SetReturn(policy switch
        {
            0 => 0,
            1 or 2 => 1,
            _ => -1,
        });
    }
}
