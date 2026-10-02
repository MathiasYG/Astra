// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Images;

// PendingDcc holds a metadata fill seen before its color target was bound; it stays invisible
// to the metadata queries until the target classifies the address.
public enum SurfaceMetadataKind : byte
{
    PendingDcc,
    CMask,
    FMask,
    HTile,
    Dcc,
}

public sealed class SurfaceMetadata
{
    public SurfaceMetadataKind Kind;
    private uint _clearMask;
    // The 32-bit mask covers the first slices. Higher slices share the same
    // default clear state, with sparse exceptions after individual writes.
    private bool _extraSlicesClearByDefault;
    private readonly HashSet<uint> _extraSliceExceptions = [];

    public uint ClearMask
    {
        get => _clearMask;
        set
        {
            _clearMask = value;
            _extraSlicesClearByDefault = value == uint.MaxValue;
            _extraSliceExceptions.Clear();
        }
    }

    public bool IsSliceClear(uint slice) => slice < 32
        ? (_clearMask & (1u << (int)slice)) != 0
        : _extraSlicesClearByDefault != _extraSliceExceptions.Contains(slice);

    public void SetSliceClear(uint slice, bool isClear)
    {
        if (slice < 32)
        {
            if (isClear)
                _clearMask |= 1u << (int)slice;
            else
                _clearMask &= ~(1u << (int)slice);
            return;
        }

        if (isClear == _extraSlicesClearByDefault)
            _extraSliceExceptions.Remove(slice);
        else
            _extraSliceExceptions.Add(slice);
    }

    public uint FillValue = 0xffffffff;
    public ulong FillSize;
}
