// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

internal static class ScalarBufferRangeRead
{
    // Use one clean read only when every scalar load is in bounds and contiguous.
    // A refusal leaves the caller's original word reads and OOB handling intact.
    internal static bool TryRead(ReadOnlySpan<uint> descriptor, uint dynamicOffset,
        uint immediateOffset, ResourceRuntimeInputs inputs, Span<uint> words)
    {
        if (inputs.ReadCleanWords is null || descriptor.Length != 4 || words.IsEmpty)
            return false;

        var lastImmediate = (ulong)immediateOffset + (ulong)(words.Length - 1) * sizeof(uint);
        if (lastImmediate > uint.MaxValue)
            return false;

        const ulong addressMask = 0x0000_FFFF_FFFF_FFFFul;
        var stride = (descriptor[1] >> 16) & 0x3FFF;
        var size = stride == 0 ? descriptor[2] : (ulong)stride * descriptor[2];
        var offset = ((ulong)dynamicOffset + immediateOffset) & ~3ul;
        var length = (ulong)words.Length * sizeof(uint);
        if (offset > size || length > size - offset)
            return false;

        var baseAddress = ((descriptor[0] | ((ulong)descriptor[1] << 32)) & addressMask) & ~3ul;
        if (offset > addressMask - baseAddress || length - 1 > addressMask - baseAddress - offset)
            return false;

        return inputs.ReadCleanWords(baseAddress + offset, words);
    }
}
