// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.Libs.Tests.Ampr;

public sealed class AmprVirtualAddressRangesTests
{
    private const string Nid = "wkQR9+xTFKY";
    private const ulong MemoryBase = 0x1_0000_0000;

    [Fact]
    public void AmmGetVirtualAddressRanges_ReportsConfiguredGuestMemoryPools()
    {
        var memory = new FakeCpuMemory(MemoryBase, 0x1000);
        var context = new CpuContext(memory, Generation.Gen5);
        var directBaseOut = MemoryBase + 0x100;
        var directSizeOut = MemoryBase + 0x108;
        var flexibleBaseOut = MemoryBase + 0x110;
        var flexibleSizeOut = MemoryBase + 0x118;

        context[CpuRegister.Rdi] = directBaseOut;
        context[CpuRegister.Rsi] = directSizeOut;
        context[CpuRegister.Rdx] = flexibleBaseOut;
        context[CpuRegister.Rcx] = flexibleSizeOut;

        var manager = new ModuleManager();
        manager.RegisterExports(
            SharpEmu.Generated.SysAbiExportRegistry.CreateExports(Generation.Gen5));

        Assert.True(manager.TryGetExport(Nid, out var export));
        Assert.Equal("sceAmprAmmGetVirtualAddressRanges", export.Name);
        Assert.Equal("libSceAmpr", export.LibraryName);
        Assert.Equal(Generation.Gen5, export.Target);
        Assert.Equal(OrbisGen2Result.ORBIS_GEN2_OK, manager.Dispatch(Nid, context));

        Assert.Equal(0UL, ReadUInt64(memory, directBaseOut));
        Assert.Equal(GuestMemoryLayout.DirectBytes, ReadUInt64(memory, directSizeOut));
        Assert.Equal(GuestMemoryLayout.FlexibleOffset, ReadUInt64(memory, flexibleBaseOut));
        Assert.Equal(GuestMemoryLayout.FlexibleBytes, ReadUInt64(memory, flexibleSizeOut));
    }

    private static ulong ReadUInt64(FakeCpuMemory memory, ulong address)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        Assert.True(memory.TryRead(address, bytes));
        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }
}
