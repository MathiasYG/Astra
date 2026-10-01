// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Ampr;
using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.Libs.Tests.Ampr;

[Collection("AmprFileRegistry")]
public sealed class AmprWriteAddressTests
{
    [Fact]
    public void MeasureCommandSizeWriteAddress0400_MatchesOnCompletionVariant()
    {
        const string nid = "4fgtGfXDrFc";
        const ulong memoryBase = 0x1_0000_0000;
        var memory = new FakeCpuMemory(memoryBase, 0x1000);
        var context = new CpuContext(memory, Generation.Gen5);
        var manager = CreateManagerWithExport(
            nid,
            "sceAmprMeasureCommandSizeWriteAddress_04_00");

        Assert.Equal(OrbisGen2Result.ORBIS_GEN2_OK, manager.Dispatch(nid, context));
        var measured = context[CpuRegister.Rax];

        Assert.Equal(0, AmprExports.MeasureCommandSizeWriteAddressOnCompletion(context));
        Assert.Equal(context[CpuRegister.Rax], measured);
    }

    [Fact]
    public void CommandBufferWriteAddress0400_WritesValueOnCompletion()
    {
        const string nid = "j0+3uJMxYJY";
        const ulong memoryBase = 0x1_0000_0000;
        const ulong commandBufferAddress = memoryBase + 0x100;
        const ulong recordBufferAddress = memoryBase + 0x200;
        const ulong watcherAddress = memoryBase + 0x800;
        const ulong watcherValue = 1;
        var memory = new FakeCpuMemory(memoryBase, 0x1000);
        var context = new CpuContext(memory, Generation.Gen5);
        var manager = CreateManagerWithExport(
            nid,
            "sceAmprCommandBufferWriteAddress_04_00");

        context[CpuRegister.Rdi] = commandBufferAddress;
        context[CpuRegister.Rsi] = recordBufferAddress;
        context[CpuRegister.Rdx] = 0x100;

        Assert.Equal(0, AmprExports.CommandBufferConstructor(context));
        Assert.Equal(0, AmprExports.CommandBufferSetBuffer(context));

        context[CpuRegister.Rdi] = commandBufferAddress;
        context[CpuRegister.Rsi] = watcherAddress;
        context[CpuRegister.Rdx] = watcherValue;

        Assert.Equal(OrbisGen2Result.ORBIS_GEN2_OK, manager.Dispatch(nid, context));

        Span<byte> watcher = stackalloc byte[sizeof(ulong)];
        Assert.True(memory.TryRead(watcherAddress, watcher));
        Assert.Equal(0UL, BinaryPrimitives.ReadUInt64LittleEndian(watcher));

        Assert.Equal(0, AmprExports.CompleteCommandBuffer(context, commandBufferAddress));

        Assert.True(memory.TryRead(watcherAddress, watcher));
        Assert.Equal(watcherValue, BinaryPrimitives.ReadUInt64LittleEndian(watcher));
    }

    private static ModuleManager CreateManagerWithExport(string nid, string exportName)
    {
        var manager = new ModuleManager();
        manager.RegisterExports(
            SharpEmu.Generated.SysAbiExportRegistry.CreateExports(Generation.Gen5));

        Assert.True(manager.TryGetExport(nid, out var export), $"NID {nid} did not register.");
        Assert.Equal(exportName, export.Name);
        Assert.Equal("libSceAmpr", export.LibraryName);
        Assert.Equal(Generation.Gen5, export.Target);
        return manager;
    }
}

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
