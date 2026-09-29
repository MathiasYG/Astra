// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;

namespace SharpEmu.HLE.Host.Windows;

internal sealed unsafe partial class WindowsHostViews : IHostViewMemory
{
    internal static bool FailAliasMapForTests;
    internal static bool FailProtectForTests;

    private const uint MEM_COMMIT = 0x1000;
    private const uint MEM_RESERVE = 0x2000;
    private const uint MEM_RELEASE = 0x8000;
    private const uint MEM_FREE = 0x10000;
    private const uint MEM_COALESCE_PLACEHOLDERS = 0x1;
    private const uint MEM_PRESERVE_PLACEHOLDER = 0x2;
    private const uint MEM_REPLACE_PLACEHOLDER = 0x4000;
    private const uint MEM_RESERVE_PLACEHOLDER = 0x40000;
    private const uint PAGE_NOACCESS = 0x01;
    private const uint PAGE_READONLY = 0x02;
    private const uint PAGE_READWRITE = 0x04;
    private const uint PAGE_EXECUTE = 0x10;
    private const uint PAGE_EXECUTE_READ = 0x20;
    private const uint PAGE_EXECUTE_READWRITE = 0x40;
    private const uint PAGE_EXECUTE_WRITECOPY = 0x80;
    private const uint SEC_RESERVE = 0x4000000;
    private const uint FILE_MAP_READ_WRITE = 0x6;
    private static readonly nint InvalidHandle = -1;

    public WindowsHostViews()
    {
        GetSystemInfo(out var info);
        PageSize = info.PageSize;
        Granularity = info.AllocationGranularity;
    }

    public ulong PageSize { get; }

    public ulong Granularity { get; }

    public bool TryCreateBacking(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
    {
        backing = null;
        failure = HostViewFailure.BackingUnavailable;
        if (size == 0)
        {
            return false;
        }

        var handle = CreateFileMappingW(InvalidHandle, null, PAGE_EXECUTE_READWRITE | SEC_RESERVE, (uint)(size >> 32), (uint)size, null);
        if (handle == 0)
        {
            return false;
        }

        var alias = FailAliasMapForTests ? null : MapViewOfFile(handle, FILE_MAP_READ_WRITE, 0, 0, (nuint)size);
        if (alias == null)
        {
            CloseHandle(handle);
            return false;
        }

        var pageCount = size / PageSize + (size % PageSize == 0 ? 0UL : 1UL);
        var bitmapBytes = (pageCount + 7) / 8;
        byte[]? committedPageBitmap = null;
        if (bitmapBytes <= int.MaxValue)
        {
            try
            {
                committedPageBitmap = new byte[(int)bitmapBytes];
            }
            catch (OutOfMemoryException)
            {
                // Keep the backing usable; it will fall back to committing each requested range.
            }
        }

        backing = new HostBackingObject((ulong)alias, size, ReleaseBackingObject)
        {
            Handle = handle,
            CommittedPageBitmap = committedPageBitmap,
        };
        failure = HostViewFailure.None;
        return true;
    }

    public HostViewFailure CommitBacking(HostBackingObject backing, ulong offset, ulong size)
    {
        lock (backing.Gate)
        {
            return CommitBackingLocked(backing, offset, size);
        }
    }

    public ulong ReserveHole(ulong address, ulong size)
    {
        if (!HostViewMemory.IsValidRange(address, size) || address % Granularity != 0)
        {
            return 0;
        }

        var ptr = VirtualAlloc2(GetCurrentProcess(), (void*)address, (nuint)size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, null, 0);
        if (ptr == null)
        {
            return 0;
        }

        if ((ulong)ptr != address)
        {
            VirtualFree(ptr, 0, MEM_RELEASE);
            return 0;
        }

        return address;
    }

    public IReadOnlyList<HostAddressRange> ReserveFreeAddressRanges(ulong start, ulong end, ulong minimumSize)
    {
        var reserved = new List<HostAddressRange>();
        if (start >= end || minimumSize == 0)
        {
            return reserved;
        }

        var current = start;
        while (current < end)
        {
            if (VirtualQuery((void*)current, out var info, (nuint)sizeof(MemoryBasicInformation)) == 0)
            {
                break;
            }

            var regionEnd = Math.Min(end, info.BaseAddress + info.RegionSize);
            if (info.State == MEM_FREE)
            {
                var reserveStart = AlignUp(Math.Max(start, info.BaseAddress), Granularity);
                var reserveEnd = AlignDown(regionEnd, Granularity);
                if (reserveEnd > reserveStart && reserveEnd - reserveStart >= minimumSize)
                {
                    var size = reserveEnd - reserveStart;
                    if (ReserveHole(reserveStart, size) == reserveStart)
                    {
                        reserved.Add(new HostAddressRange(reserveStart, size));
                    }
                }
            }

            if (regionEnd <= current)
            {
                break;
            }
            current = regionEnd;
        }

        return reserved;
    }

    public bool SplitHole(ulong address, ulong size) =>
        VirtualFree((void*)address, (nuint)size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);

    public bool JoinHoles(ulong address, ulong size) =>
        VirtualFree((void*)address, (nuint)size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);

    public bool FreeHole(ulong address, ulong size) =>
        VirtualFree((void*)address, 0, MEM_RELEASE);

    public bool TryMapView(HostBackingObject backing, ulong address, ulong offset, ulong size, HostPageProtection protection, out HostViewFailure failure)
    {
        if (address == 0 || address % PageSize != 0)
        {
            failure = HostViewFailure.WrongHostAddress;
            return false;
        }

        lock (backing.Gate)
        {
            if (backing.IsDisposed)
            {
                failure = HostViewFailure.BackingUnavailable;
                return false;
            }

            if (!HostViewMemory.IsValidOffset(backing, offset, size, PageSize))
            {
                failure = HostViewFailure.OffsetOutOfBounds;
                return false;
            }

            var commitFailure = CommitBackingLocked(backing, offset, size);
            if (commitFailure != HostViewFailure.None)
            {
                failure = commitFailure;
                return false;
            }

            if (size > ulong.MaxValue - address)
            {
                failure = HostViewFailure.WrongHostAddress;
                return false;
            }

            // A view cannot map as no-access. Map writable, then apply the protection.
            var mapProtection = protection == HostPageProtection.NoAccess ? PAGE_READWRITE : GetNativeProtection(protection);
            var process = GetCurrentProcess();
            var ptr = MapViewOfFile3(backing.Handle, process, (void*)address, offset, (nuint)size, MEM_REPLACE_PLACEHOLDER, mapProtection, null, 0);
            if (ptr == null)
            {
                failure = HostViewFailure.PlaceholderMapFailed;
                return false;
            }

            if ((ulong)ptr != address)
            {
                UnmapViewOfFile2(process, ptr, MEM_PRESERVE_PLACEHOLDER);
                failure = HostViewFailure.WrongHostAddress;
                return false;
            }

            if (FailProtectForTests || !VirtualProtect(ptr, (nuint)size, GetNativeProtection(protection), out _))
            {
                UnmapViewOfFile2(process, ptr, MEM_PRESERVE_PLACEHOLDER);
                failure = HostViewFailure.ProtectFailed;
                return false;
            }

            failure = HostViewFailure.None;
            return true;
        }
    }

    public bool UnmapView(ulong address, ulong size) =>
        UnmapViewOfFile2(GetCurrentProcess(), (void*)address, MEM_PRESERVE_PLACEHOLDER);

    public bool CommitPrivate(ulong address, ulong size, HostPageProtection protection)
    {
        var ptr = VirtualAlloc2(GetCurrentProcess(), (void*)address, (nuint)size, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, GetNativeProtection(protection), null, 0);
        if (ptr == null)
        {
            return false;
        }

        if ((ulong)ptr != address)
        {
            VirtualFree(ptr, 0, MEM_RELEASE);
            return false;
        }

        return true;
    }

    public bool ReleasePrivate(ulong address, ulong size) =>
        VirtualFree((void*)address, (nuint)size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);

    public bool ChangeAccess(ulong address, ulong size, HostPageProtection protection) =>
        VirtualProtect((void*)address, (nuint)size, GetNativeProtection(protection), out _);

    public bool FreeOwnedRange(ulong address, ulong size)
    {
        var end = address + size;
        var current = address;
        while (current < end)
        {
            if (VirtualQuery((void*)current, out var info, (nuint)sizeof(MemoryBasicInformation)) == 0)
            {
                return false;
            }

            if (info.State == MEM_COMMIT)
            {
                if (!VirtualFree((void*)info.BaseAddress, (nuint)info.RegionSize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
                {
                    return false;
                }

                continue;
            }

            if (info.State == MEM_RESERVE && !VirtualFree((void*)info.AllocationBase, 0, MEM_RELEASE))
            {
                return false;
            }

            current = info.BaseAddress + info.RegionSize;
        }

        return true;
    }

    private static ulong AlignUp(ulong value, ulong alignment)
    {
        var remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private static ulong AlignDown(ulong value, ulong alignment) => value - value % alignment;

    private HostViewFailure CommitBackingLocked(HostBackingObject backing, ulong offset, ulong size)
    {
        if (backing.IsDisposed)
        {
            return HostViewFailure.BackingUnavailable;
        }

        if (size == 0)
        {
            return offset <= backing.Size ? HostViewFailure.None : HostViewFailure.OffsetOutOfBounds;
        }

        if (offset >= backing.Size || size > backing.Size - offset)
        {
            return HostViewFailure.OffsetOutOfBounds;
        }

        var end = offset + size;
        var pageStart = AlignDown(offset, PageSize);
        if (end > ulong.MaxValue - (PageSize - 1))
        {
            return HostViewFailure.OffsetOutOfBounds;
        }

        var pageEnd = AlignUp(end, PageSize);
        var backingPageEnd = backing.Size > ulong.MaxValue - (PageSize - 1)
            ? ulong.MaxValue - ulong.MaxValue % PageSize
            : AlignUp(backing.Size, PageSize);
        if (pageEnd > backingPageEnd || backing.AliasBase > ulong.MaxValue - pageStart)
        {
            return HostViewFailure.OffsetOutOfBounds;
        }

        var bitmap = backing.CommittedPageBitmap;
        var firstPage = pageStart / PageSize;
        var endPage = pageEnd / PageSize;
        var currentPage = firstPage;
        while (currentPage < endPage)
        {
            if (bitmap is not null && IsPageCommitted(bitmap, currentPage))
            {
                currentPage++;
                continue;
            }

            var runStart = currentPage;
            currentPage++;
            while (currentPage < endPage &&
                   (bitmap is null || !IsPageCommitted(bitmap, currentPage)))
            {
                currentPage++;
            }

            var runOffset = runStart * PageSize;
            var address = backing.AliasBase + runOffset;
            var commitSize = (currentPage - runStart) * PageSize;
            var committed = VirtualAlloc((void*)address, (nuint)commitSize, MEM_COMMIT, PAGE_READWRITE);
            if ((ulong)committed != address)
            {
                return HostViewFailure.BackingCommitFailed;
            }

            if (bitmap is not null)
            {
                for (var page = runStart; page < currentPage; page++)
                {
                    SetPageCommitted(bitmap, page);
                }
            }
        }

        return HostViewFailure.None;
    }

    private static bool IsPageCommitted(byte[] bitmap, ulong pageIndex) =>
        (bitmap[(int)(pageIndex / 8)] & (1 << (int)(pageIndex % 8))) != 0;

    private static void SetPageCommitted(byte[] bitmap, ulong pageIndex) =>
        bitmap[(int)(pageIndex / 8)] |= (byte)(1 << (int)(pageIndex % 8));

    private static void ReleaseBackingObject(HostBackingObject backing)
    {
        UnmapViewOfFile((void*)backing.AliasBase);
        CloseHandle(backing.Handle);
    }

    private static uint GetNativeProtection(HostPageProtection protection) => protection switch
    {
        HostPageProtection.NoAccess => PAGE_NOACCESS,
        HostPageProtection.ReadOnly => PAGE_READONLY,
        HostPageProtection.ReadWrite => PAGE_READWRITE,
        HostPageProtection.Execute => PAGE_EXECUTE,
        HostPageProtection.ReadExecute => PAGE_EXECUTE_READ,
        HostPageProtection.ReadWriteExecute => PAGE_EXECUTE_READWRITE,
        HostPageProtection.ExecuteWriteCopy => PAGE_EXECUTE_WRITECOPY,
        _ => throw new ArgumentOutOfRangeException(nameof(protection), protection, null),
    };

    [StructLayout(LayoutKind.Sequential)]
    private struct SystemInfo
    {
        public ushort ProcessorArchitecture;
        public ushort Reserved;
        public uint PageSize;
        public void* MinimumApplicationAddress;
        public void* MaximumApplicationAddress;
        public nuint ActiveProcessorMask;
        public uint NumberOfProcessors;
        public uint ProcessorType;
        public uint AllocationGranularity;
        public ushort ProcessorLevel;
        public ushort ProcessorRevision;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryBasicInformation
    {
        public ulong BaseAddress;
        public ulong AllocationBase;
        public uint AllocationProtect;
        public uint Alignment1;
        public ulong RegionSize;
        public uint State;
        public uint Protect;
        public uint Type;
        public uint Alignment2;
    }

    [LibraryImport("kernel32.dll")]
    private static partial nuint VirtualQuery(void* address, out MemoryBasicInformation info, nuint length);

    [LibraryImport("kernelbase.dll", SetLastError = true)]
    private static partial void* VirtualAlloc2(void* process, void* baseAddress, nuint size, uint allocationType, uint pageProtection, void* extendedParameters, uint parameterCount);

    [LibraryImport("kernelbase.dll", SetLastError = true)]
    private static partial void* MapViewOfFile3(nint fileMapping, void* process, void* baseAddress, ulong offset, nuint viewSize, uint allocationType, uint pageProtection, void* extendedParameters, uint parameterCount);

    [LibraryImport("kernelbase.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool UnmapViewOfFile2(void* process, void* baseAddress, uint unmapFlags);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial nint CreateFileMappingW(nint file, void* attributes, uint protect, uint maximumSizeHigh, uint maximumSizeLow, ushort* name);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial void* MapViewOfFile(nint fileMapping, uint desiredAccess, uint offsetHigh, uint offsetLow, nuint bytesToMap);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    private static partial void* VirtualAlloc(void* address, nuint size, uint allocationType, uint protect);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool UnmapViewOfFile(void* baseAddress);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CloseHandle(nint handle);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool VirtualFree(void* address, nuint size, uint freeType);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool VirtualProtect(void* address, nuint size, uint newProtect, out uint oldProtect);

    [LibraryImport("kernel32.dll")]
    private static partial void GetSystemInfo(out SystemInfo info);

    [LibraryImport("kernel32.dll")]
    private static partial void* GetCurrentProcess();
}
