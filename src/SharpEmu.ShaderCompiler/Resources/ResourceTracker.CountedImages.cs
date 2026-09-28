// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

public sealed partial class ResourceTracker
{
    private bool TryMatchCountedTable(
        ScalarValue handle,
        int width,
        uint pc,
        out ScalarValue[] reads,
        out int[] memoryIndices,
        out uint heapSource,
        out uint tableOffset,
        out uint dynamicBase,
        out uint stride,
        out ScalarValue bound)
    {
        reads = [];
        memoryIndices = [];
        heapSource = tableOffset = dynamicBase = stride = 0;
        bound = null!;
        if (handle.Operands.Length != width) return false;

        reads = new ScalarValue[width];
        memoryIndices = new int[width];
        ScalarValue? address = null;
        ScalarValue? dynamicOffset = null;
        uint immediate = 0;
        for (var dword = 0; dword < width; dword++)
        {
            // A descriptor loaded before an inner loop is carried through a
            // self-referential phi. The value is still the same scalar load.
            var read = _graph.ResolveInvariantPhi(handle.Operands[dword]);
            if (read is null) return false;
            if (read.Kind != ScalarValueKind.ScalarAddressWord || read.Operands.Length != 2 ||
                read.MemoryIndex < 0 || read.MemoryIndex >= _plan.Memory.Count ||
                !MemoryIndexBelongsTo(read.MemoryIndex, read)) return false;
            var memory = _plan.Memory[read.MemoryIndex];
            if (memory.Kind != MemoryResourceKind.ScalarAddress ||
                memory.DataBits != 32 || memory.DataDwords != 1 ||
                memory.Offset < (uint)dword * sizeof(uint)) return false;
            var componentBase = memory.Offset - (uint)dword * sizeof(uint);
            if (dword == 0) immediate = componentBase;
            else if (componentBase != immediate) return false;
            if (address is not null && !_graph.Equivalent(address, read.Operands[0])) return false;
            if (dynamicOffset is not null && !_graph.Equivalent(dynamicOffset, read.Operands[1])) return false;
            address = read.Operands[0];
            dynamicOffset = read.Operands[1];
            reads[dword] = read;
            memoryIndices[dword] = read.MemoryIndex;
        }

        if (address is null || dynamicOffset is null || address.Kind != ScalarValueKind.AddressHandle)
            return false;
        if (dynamicOffset.Kind == ScalarValueKind.Operation &&
            dynamicOffset.Operation == ScalarOperation.IAdd32 && dynamicOffset.Operands.Length == 2)
        {
            if (dynamicOffset.Operands[0].IsConstant)
            {
                dynamicBase = dynamicOffset.Operands[0].ConstantU32;
                dynamicOffset = dynamicOffset.Operands[1];
            }
            else if (dynamicOffset.Operands[1].IsConstant)
            {
                dynamicBase = dynamicOffset.Operands[1].ConstantU32;
                dynamicOffset = dynamicOffset.Operands[0];
            }
            else return false;
        }

        if (dynamicOffset.Kind != ScalarValueKind.Operation || dynamicOffset.Operands.Length != 2)
            return false;
        ScalarValue key;
        if (dynamicOffset.Operation == ScalarOperation.ShiftLeft32 &&
            dynamicOffset.Operands[1].IsConstant &&
            dynamicOffset.Operands[1].ConstantU32 < 32)
        {
            stride = 1u << (int)dynamicOffset.Operands[1].ConstantU32;
            key = dynamicOffset.Operands[0];
        }
        else if (dynamicOffset.Operation == ScalarOperation.IMul32 && dynamicOffset.Operands[0].IsConstant)
        {
            stride = dynamicOffset.Operands[0].ConstantU32;
            key = dynamicOffset.Operands[1];
        }
        else if (dynamicOffset.Operation == ScalarOperation.IMul32 && dynamicOffset.Operands[1].IsConstant)
        {
            stride = dynamicOffset.Operands[1].ConstantU32;
            key = dynamicOffset.Operands[0];
        }
        else return false;
        if (stride < (uint)width * sizeof(uint) ||
            (ulong)immediate + dynamicBase > uint.MaxValue ||
            !CountedImageBound.TryGet(_plan, key, pc,
                memoryIndices.Select(index => _plan.Memory[index].Pc).Min(), out bound) ||
            !MakeRuntimeAddressSource(address, pc, out heapSource, out _))
            return false;
        tableOffset = immediate + dynamicBase;
        return true;
    }

    private bool TryMakeCountedImage(ScalarValue handle, uint pc, out IndirectImagePlan plan)
    {
        plan = null!;
        if (handle.Kind != ScalarValueKind.ImageHandle) return false;
        if (!TryMatchCountedTable(handle, 8, pc, out var reads, out var memoryIndices,
                out var heapSource, out var tableOffset, out var dynamicBase, out var stride, out var bound))
            return false;
        if (reads.Where((read, index) => !ReferenceEquals(read, handle.Operands[index]))
                .Any(read => !_uses.TryGetValue(read, out var users) ||
                    users.Any(user => user.Kind != ScalarValueKind.Phi ||
                        _graph.ResolveInvariantPhi(user) is not { } invariant ||
                        !_graph.Equivalent(invariant, read))) ||
            reads.Where((read, index) => ReferenceEquals(read, handle.Operands[index]))
                .Any(read => !UsesOnly(read, [handle]))) return false;

        var heap = _sources[(int)heapSource];
        var dwords = Enumerable.Repeat(reads[0], 8).ToArray();
        dwords[0] = heap.Dwords[0];
        dwords[1] = heap.Dwords[1];
        var source = InternSource(new DescriptorSource
        {
            Dwords = dwords,
            IndirectImage = new IndirectImageSelector(0, heapSource, 0, 0, 0)
            {
                Dense = true,
                TableOffset = tableOffset,
                DynamicOffsetBase = dynamicBase,
                RuntimeKeyBound = bound,
                EntryStride = stride,
            },
        });
        var suppress = memoryIndices.All(index => HasOnlyImageConsumers(_plan.Memory[index], handle));
        plan = new IndirectImagePlan
        {
            Handle = handle,
            Source = source,
            Key = reads[0],
            KeyIsAddressOffset = true,
            HeapSource = heapSource,
            SuppressMemoryReads = suppress,
            Memory = memoryIndices,
            Reads = reads,
        };
        return true;
    }

    private bool TryMakeCountedSampler(ScalarValue handle, uint pc, out uint source)
    {
        source = 0;
        if (handle.Kind != ScalarValueKind.SamplerHandle ||
            !TryMatchCountedTable(handle, 4, pc, out _, out _, out var heapSource,
                out var tableOffset, out var dynamicBase, out var stride, out var bound)) return false;

        source = InternSource(new DescriptorSource
        {
            Dwords = Enumerable.Repeat(_graph.Constant(0u), 4).ToArray(),
            CountedSampler = new CountedSamplerSelector(heapSource, tableOffset, dynamicBase, stride, bound),
        });
        return true;
    }
}
