// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;

namespace SharpEmu.ShaderCompiler.Resources;

// Interns descriptor sources, builds the dense buffer, image, sampler and pair tables
// and patches each access with its index. A failure names hash, stage and pc.
public sealed partial class ResourceTracker
{
    private const uint SamplerBorderClampMask = (1u << 2) | (1u << 5) | (1u << 8);
    private const uint SamplerDword3ReservedMask = 0x3FFF_F000u;

    private readonly ShaderResourcePlan _plan;
    private readonly ScalarValueGraph _graph;
    private readonly ShaderResourceInfo _info = new();
    private readonly List<DescriptorSource> _sources = [];
    private readonly List<(int Index, uint Resource, uint Sampler, bool HasSampler)> _memoryPatches = [];
    private readonly List<IndirectImagePlan> _indirectImages = [];
    private readonly List<IndirectSamplerPlan> _indirectSamplers = [];
    private readonly Dictionary<ScalarValue, List<ScalarValue>> _uses;
    private readonly Dictionary<int, List<ScalarValue>> _readsByMemory = [];

    private sealed class IndirectImagePlan
    {
        public ScalarValue Handle = null!;
        public uint Source;
        public ScalarValue Key = null!;
        public uint HeapSource;
        public bool KeyIsAddressOffset;
        public bool SuppressMemoryReads = true;
        public int[] Memory = new int[8];
        public ScalarValue[] Reads = new ScalarValue[8];
    }

    private sealed class IndirectSamplerPlan
    {
        public ScalarValue Handle = null!;
        public uint Source;
        public ScalarValue Key = null!;
        public uint HeapSource;
        public int[] Memory = new int[4];
        public ScalarValue[] Reads = new ScalarValue[4];
    }

    public sealed record Result(
        IReadOnlyList<DescriptorSource> Sources,
        ShaderResourceInfo Info,
        IReadOnlyList<IndirectImageAccess> IndirectImages,
        IReadOnlyList<IndirectSamplerAccess> IndirectSamplers,
        IReadOnlySet<ScalarValue> IndirectReads);

    private ResourceTracker(ShaderResourcePlan plan)
    {
        _plan = plan;
        _graph = plan.Graph;
        var roots = new List<ScalarValue>();
        foreach (var access in plan.Accesses)
        {
            if (access?.Handle is { } handle)
            {
                roots.Add(handle);
            }

            if (access?.SamplerHandle is { } sampler)
            {
                roots.Add(sampler);
            }

            if (access?.Read is { } read)
            {
                roots.Add(read);
            }
        }

        roots.AddRange(plan.TableReads.Select(read => read.Value));
        roots.AddRange(plan.DynamicReads);
        _uses = _graph.CollectUses(roots);
        foreach (var value in _uses.Keys.Concat(roots))
        {
            if (value.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord)
            {
                if (!_readsByMemory.TryGetValue(value.MemoryIndex, out var list))
                {
                    list = [];
                    _readsByMemory[value.MemoryIndex] = list;
                }

                if (!list.Contains(value))
                {
                    list.Add(value);
                }
            }
        }
    }

    public static Result Track(ShaderResourcePlan plan) => new ResourceTracker(plan).Run();

    private Result Run()
    {
        PlanIndirectImages();
        PlanIndirectSamplers();
        for (var index = 0; index < _plan.Memory.Count; index++)
        {
            Collect(index);
        }

        LinkImageAliases();
        foreach (var (index, resource, sampler, hasSampler) in _memoryPatches)
        {
            var memory = _plan.Memory[index];
            memory.Resource = resource;
            if (hasSampler)
            {
                memory.Sampler = sampler;
            }
        }

        var indirectReads = new HashSet<ScalarValue>();
        var indirectAccesses = new List<IndirectImageAccess>();
        var indirectSamplerAccesses = new List<IndirectSamplerAccess>();
        foreach (var plan in _indirectImages)
        {
            if (plan.SuppressMemoryReads)
            {
                foreach (var index in plan.Memory)
                {
                    _plan.Memory[index].PlanningOnly = true;
                }

                foreach (var read in plan.Reads)
                {
                    indirectReads.Add(read);
                }
            }

            for (var index = 0; index < _plan.Accesses.Length; index++)
            {
                if (_plan.Accesses[index]?.Handle is { } handle && ReferenceEquals(handle, plan.Handle))
                {
                    indirectAccesses.Add(new IndirectImageAccess(index, plan.Key, plan.HeapSource)
                    {
                        KeyIsAddressOffset = plan.KeyIsAddressOffset,
                    });
                }
            }
        }

        foreach (var plan in _indirectSamplers)
        {
            foreach (var index in plan.Memory)
            {
                _plan.Memory[index].PlanningOnly = true;
            }

            foreach (var read in plan.Reads)
            {
                indirectReads.Add(read);
            }

            for (var index = 0; index < _plan.Accesses.Length; index++)
            {
                if (_plan.Accesses[index]?.SamplerHandle is { } handle && ReferenceEquals(handle, plan.Handle))
                {
                    indirectSamplerAccesses.Add(new IndirectSamplerAccess(index, plan.Key, plan.HeapSource));
                }
            }
        }

        return new Result(_sources, _info, indirectAccesses, indirectSamplerAccesses, indirectReads);
    }

    private ResourcePlanException Failure(uint pc, string reason) =>
        new($"shader resource tracking: hash=0x{_plan.Hash:X16} stage={_plan.Stage} pc=0x{pc:X8} {reason}");

    // ---- descriptor sources ----

    // Copies a handle's dwords into a source. A sampler that no clamp axis sets to
    // border mode drops its border colour, which is unused then.
    private DescriptorSource MakeSource(
        ScalarValue handle,
        uint width,
        bool sampler,
        bool sampleAdjust,
        uint pc,
        bool imageR128 = false)
    {
        if (handle.Operands.Length != width)
        {
            throw Failure(pc, $"{handle.Kind} has {handle.Operands.Length} descriptor dwords, expected {width}");
        }

        var dwords = (ScalarValue[])handle.Operands.Clone();
        if (sampleAdjust)
        {
            dwords[3] = CanonicalizeSampleAdjustDword3(dwords[3]);
        }

        // R128 image instructions consume a compact 128-bit T# (the first four
        // words).  The remaining SGPRs are free for ordinary shader temporaries,
        // so represent them as defined zero padding for host descriptor users.
        if (imageR128 && !sampler && dwords.Length >= 8)
        {
            Array.Fill(dwords, _graph.Constant(0u), 4, 4);
        }

        var dword0 = dwords[0];
        if (sampler && dword0.IsConstant && (dword0.ConstantU32 & SamplerBorderClampMask) == 0)
        {
            dwords[3] = _graph.Constant(0u);
        }

        return new DescriptorSource { Dwords = dwords };
    }

    private static uint PossibleBits(ScalarValue value)
    {
        if (value.IsConstant)
        {
            return value.Type == ScalarValueType.U32 ? value.ConstantU32 : uint.MaxValue;
        }

        if (value.Kind != ScalarValueKind.Operation)
        {
            return uint.MaxValue;
        }

        return value.Operation switch
        {
            ScalarOperation.And32 => PossibleBits(value.Operands[0]) & PossibleBits(value.Operands[1]),
            ScalarOperation.Or32 => PossibleBits(value.Operands[0]) | PossibleBits(value.Operands[1]),
            ScalarOperation.ShiftLeft32 when value.Operands[1].IsConstant =>
                PossibleBits(value.Operands[0]) << (int)(value.Operands[1].ConstantU32 & 31),
            _ => uint.MaxValue,
        };
    }

    private ScalarValue CanonicalizeSampleAdjustDword3(ScalarValue value)
    {
        for (;;)
        {
            if (value.Kind != ScalarValueKind.Operation || value.Operation != ScalarOperation.Or32)
            {
                return value;
            }

            var left = value.Operands[0];
            var right = value.Operands[1];
            var leftReserved = (PossibleBits(left) & ~SamplerDword3ReservedMask) == 0;
            var rightReserved = (PossibleBits(right) & ~SamplerDword3ReservedMask) == 0;
            if (leftReserved && rightReserved)
            {
                return _graph.Constant(0u);
            }

            if (leftReserved)
            {
                value = right;
            }
            else if (rightReserved)
            {
                value = left;
            }
            else
            {
                return value;
            }
        }
    }

    private bool ValidateSource(DescriptorSource source, out uint badDword)
    {
        for (badDword = 0; badDword < source.DwordCount; badDword++)
        {
            var dword = source.Dwords[badDword];
            if (dword.Type != ScalarValueType.U32 || !_plan.ValidateRuntimeValue(dword))
            {
                return false;
            }
        }

        return true;
    }

    // Some shaders assemble a uniform image descriptor from a scalar-buffer load and
    // carry the remaining dwords through loop state.  Those words are still runtime
    // values: the materializer can evaluate them against the dispatch's buffer
    // descriptor.  Keep the proof narrow so a descriptor made from unrelated loads,
    // or an unproven fully-indirect heap record, is still rejected below.
    private bool CanMaterializeMixedDescriptorSource(ScalarValue handle, DescriptorSource source)
    {
        var reads = source.Dwords
            .Where(value => value.Kind == ScalarValueKind.ScalarBufferWord)
            .ToArray();
        if (reads.Length == 0)
        {
            return false;
        }

        var first = reads[0];
        if (first.MemoryIndex < 0 || first.MemoryIndex >= _plan.Memory.Count)
        {
            return false;
        }

        var firstMemory = _plan.Memory[first.MemoryIndex];
        if (firstMemory.Kind != MemoryResourceKind.ScalarBuffer || firstMemory.DataBits != 32 ||
            firstMemory.DataDwords != 1)
        {
            return false;
        }

        // A complete scalar load can also be the heap half of the legacy
        // material/heap image pattern.  Leave that path to the indirect proof;
        // accepting it here would make malformed selector layouts look direct.
        if (reads.Length == source.DwordCount && first.Operands.Length >= 2 &&
            MatchHeapOffset(first.Operands[1], out var heapKey, out _) &&
            TryUnwrapLoopDescriptorRead(heapKey, out var materialKey) &&
            materialKey.Kind == ScalarValueKind.ScalarBufferWord)
        {
            return false;
        }

        var instruction = _graph.Program.Instructions.FirstOrDefault(candidate => candidate.Pc == firstMemory.Pc);
        if (instruction?.Control is not Gen5ScalarMemoryControl control ||
            !instruction.Opcode.StartsWith("SBufferLoadDword", StringComparison.Ordinal) ||
            control.DestinationCount < 4)
        {
            return false;
        }

        foreach (var read in reads)
        {
            if (read.MemoryIndex < 0 || read.MemoryIndex >= _plan.Memory.Count)
            {
                return false;
            }

            var memory = _plan.Memory[read.MemoryIndex];
            if (memory.Pc != firstMemory.Pc || memory.Kind != MemoryResourceKind.ScalarBuffer ||
                memory.DataBits != 32 || memory.DataDwords != 1 || memory.ComponentIndex >= control.DestinationCount ||
                memory.Offset != unchecked((uint)(control.ImmediateOffsetBytes + (int)memory.ComponentIndex * sizeof(uint))))
            {
                return false;
            }
        }

        return true;
    }

    private uint InternSource(DescriptorSource source)
    {
        for (var candidate = 0; candidate < _sources.Count; candidate++)
        {
            var current = _sources[candidate];
            if (current.DwordCount != source.DwordCount || current.DynamicBuffer != source.DynamicBuffer ||
                !EquivalentIndirectSelector(current.IndirectImage, source.IndirectImage) ||
                !EquivalentIndirectSelector(current.IndirectSampler, source.IndirectSampler))
            {
                continue;
            }

            var same = true;
            for (var index = 0; index < source.DwordCount && same; index++)
            {
                same = _graph.Equivalent(current.Dwords[index], source.Dwords[index]);
            }

            if (same)
            {
                return (uint)candidate;
            }
        }

        _sources.Add(source);
        return (uint)(_sources.Count - 1);
    }

    private static bool EquivalentIndirectSelector(IndirectImageSelector? left, IndirectImageSelector? right)
    {
        if (left is null || right is null)
        {
            return left is null && right is null;
        }

        return left.MaterialSource == right.MaterialSource && left.HeapSource == right.HeapSource &&
               left.SelectorStride == right.SelectorStride && left.SelectorOffset == right.SelectorOffset &&
               left.HeapIsAddress == right.HeapIsAddress &&
               left.KeyArgument == right.KeyArgument && left.HeapStride == right.HeapStride &&
               left.HeapOffset == right.HeapOffset &&
               left.DescriptorArray == right.DescriptorArray &&
               (left.DescriptorHeapOffsets ?? []).SequenceEqual(right.DescriptorHeapOffsets ?? []) &&
               (left.DescriptorHeapMasks ?? []).SequenceEqual(right.DescriptorHeapMasks ?? []) &&
               EquivalentDescriptorHeapAlternates(left.DescriptorHeapAlternates, right.DescriptorHeapAlternates) &&
               (left.DescriptorTableSlots ?? []).SequenceEqual(right.DescriptorTableSlots ?? []) &&
               (left.DescriptorStaticValues ?? []).SequenceEqual(right.DescriptorStaticValues ?? []);
    }

    private static bool EquivalentDescriptorHeapAlternates(
        IReadOnlyList<uint[]>? left,
        IReadOnlyList<uint[]>? right)
    {
        if (left is null || right is null)
        {
            return left is null && right is null;
        }

        return left.Count == right.Count &&
               left.Zip(right).All(pair => pair.First.AsSpan().SequenceEqual(pair.Second));
    }

    private uint GetHandleSource(
        ScalarValue? handle,
        ScalarValueKind expected,
        uint width,
        uint pc,
        bool sampler = false,
        bool sampleAdjust = false,
        bool imageR128 = false)
    {
        if (handle is null || handle.Kind != expected)
        {
            throw Failure(pc, $"memory operation requires {expected}");
        }

        var source = MakeSource(handle, width, sampler, sampleAdjust, pc, imageR128);
        if (expected == ScalarValueKind.BufferHandle && TryMakeDynamicBufferSource(handle, pc, out var dynamicSource))
        {
            _info.UsesDeviceAddresses = true;
            return InternSource(dynamicSource);
        }
        if (expected is ScalarValueKind.ImageHandle or ScalarValueKind.BufferHandle)
        {
            if (source.Dwords.Any(value => value.Kind == ScalarValueKind.ScalarBufferWord) &&
                !CanMaterializeMixedDescriptorSource(handle, source))
            {
                var dword = Array.FindIndex(source.Dwords, value => value.Kind == ScalarValueKind.ScalarBufferWord);
                throw Failure(pc, $"{expected} dword {dword} is not a valid runtime value");
            }
        }

        if (!ValidateSource(source, out var badDword))
        {
            throw Failure(pc, $"{expected} dword {badDword} is not a valid runtime value");
        }

        return InternSource(source);
    }

    // A buffer descriptor can be loaded as four scalar words through a runtime
    // descriptor table. The dynamic offset cannot be evaluated by the host at
    // compile time, so retain the descriptor in scalar registers and lower the
    // actual buffer access through the device-address page table.
    private bool TryMakeDynamicBufferSource(ScalarValue handle, uint pc, out DescriptorSource source)
    {
        source = null!;
        if (handle.Kind != ScalarValueKind.BufferHandle || handle.Operands.Length != 4 ||
            handle.Operands.Any(value => value.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord) ||
                value.MemoryIndex < 0 || value.MemoryIndex >= _plan.Memory.Count))
        {
            return false;
        }

        var first = handle.Operands[0];
        var firstMemory = _plan.Memory[first.MemoryIndex];
        var instruction = _graph.Program.Instructions.FirstOrDefault(candidate => candidate.Pc == firstMemory.Pc);
        var isScalarAddress = first.Kind == ScalarValueKind.ScalarAddressWord;
        if (instruction?.Control is not Gen5ScalarMemoryControl control ||
            (isScalarAddress
                ? !instruction.Opcode.StartsWith("SLoadDwordx4", StringComparison.Ordinal)
                : !instruction.Opcode.StartsWith("SBufferLoadDwordx4", StringComparison.Ordinal)) ||
            control.DestinationCount < 4 || control.DynamicOffsetRegister is null || first.Operands.Length < 2)
        {
            return false;
        }

        var addressHandle = first.Operands[0];
        var offset = first.Operands[1];
        for (var component = 0; component < handle.Operands.Length; component++)
        {
            var read = handle.Operands[component];
            var memory = _plan.Memory[read.MemoryIndex];
            if ((isScalarAddress && memory.Kind != MemoryResourceKind.ScalarAddress) ||
                (!isScalarAddress && memory.Kind != MemoryResourceKind.ScalarBuffer) ||
                memory.Pc != firstMemory.Pc || memory.ComponentIndex != (uint)component ||
                memory.Offset != firstMemory.Offset + (uint)component * sizeof(uint) ||
                !_graph.Equivalent(addressHandle, read.Operands[0]) ||
                !_graph.Equivalent(offset, read.Operands[1]) ||
                !MemoryIndexBelongsTo(read.MemoryIndex, read))
            {
                return false;
            }
        }

        source = new DescriptorSource
        {
            Dwords = Enumerable.Repeat(_graph.Constant(0u), 4).ToArray(),
            DynamicBuffer = true,
        };
        return true;
    }

    // ---- dense tables ----

    private static uint ByteExtent(MemoryAccessInfo memory)
    {
        var bytes = Math.Max((memory.DataBits + 7) / 8, 1u);
        var count = Math.Max(memory.DataDwords, 1u);
        var end = (ulong)memory.Offset + (ulong)bytes * count;
        return end > uint.MaxValue ? uint.MaxValue : (uint)end;
    }

    private uint AddBuffer(uint source, MemoryAccessInfo memory, uint pc)
    {
        for (var index = 0; index < _info.Buffers.Count; index++)
        {
            if (_info.Buffers[index].Source == source)
            {
                Merge(_info.Buffers[index], memory, pc);
                return (uint)index;
            }
        }

        if (_info.Buffers.Count >= ShaderResourceInfo.MaxBuffers)
        {
            return DescriptorConstants.NoIndex;
        }

        var resource = new BufferResource
        {
            Source = source,
            FirstUsePc = pc,
            DynamicDescriptor = _sources[(int)source].DynamicBuffer,
        };
        Merge(resource, memory, pc);
        _info.Buffers.Add(resource);
        return (uint)(_info.Buffers.Count - 1);
    }

    private static void Merge(BufferResource resource, MemoryAccessInfo memory, uint pc)
    {
        var atomic = memory.Access == MemoryAccess.Atomic;
        var write = memory.Access == MemoryAccess.Write || atomic;
        resource.FirstUsePc = Math.Min(resource.FirstUsePc, pc);
        resource.MaxByteExtent = Math.Max(resource.MaxByteExtent, ByteExtent(memory));
        resource.Read |= !write || atomic;
        resource.Written |= write;
        resource.Atomic |= atomic;
        resource.Formatted |= memory.Formatted;
        resource.Scalar |= memory.Kind == MemoryResourceKind.ScalarBuffer;
    }

    private uint AddImage(uint source, MemoryAccessInfo memory, uint pc)
    {
        var resourceClass = memory.ImageClass;
        var mip = resourceClass == ImageResourceClass.Storage && memory.ImageHasMip ? ImageMipMode.DynamicStorage : ImageMipMode.None;
        var depth = (memory.ImageSampleFlags & ImageSampleFlags.Compare) != 0;
        for (var index = 0; index < _info.Images.Count; index++)
        {
            var image = _info.Images[index];
            if (image.Source == source && image.ResourceClass == resourceClass && image.Dimension == memory.ImageDimension &&
                image.MipMode == mip && image.DepthCompare == depth && image.R128 == memory.ImageR128)
            {
                Merge(image, memory, pc);
                return (uint)index;
            }
        }

        if (_info.Images.Count >= ShaderResourceInfo.MaxImages)
        {
            return DescriptorConstants.NoIndex;
        }

        var added = new ImageResource
        {
            Source = source,
            FirstUsePc = pc,
            ResourceClass = resourceClass,
            Dimension = memory.ImageDimension,
            MipMode = mip,
            DepthCompare = depth,
            R128 = memory.ImageR128,
        };
        Merge(added, memory, pc);
        _info.Images.Add(added);
        return (uint)(_info.Images.Count - 1);
    }

    private static void Merge(ImageResource image, MemoryAccessInfo memory, uint pc)
    {
        var atomic = memory.Access == MemoryAccess.Atomic;
        var write = memory.Access == MemoryAccess.Write || atomic;
        image.FirstUsePc = Math.Min(image.FirstUsePc, pc);
        image.Read |= !write || atomic;
        image.Written |= write;
        image.Atomic |= atomic;
    }

    private uint AddSampler(uint source, uint pc)
    {
        for (var index = 0; index < _info.Samplers.Count; index++)
        {
            if (_info.Samplers[index].Source == source)
            {
                _info.Samplers[index].FirstUsePc = Math.Min(_info.Samplers[index].FirstUsePc, pc);
                return (uint)index;
            }
        }

        if (_info.Samplers.Count >= ShaderResourceInfo.MaxSamplers)
        {
            return DescriptorConstants.NoIndex;
        }

        _info.Samplers.Add(new SamplerResource { Source = source, FirstUsePc = pc });
        return (uint)(_info.Samplers.Count - 1);
    }

    private void AddSampledPair(uint image, uint sampler, uint pc)
    {
        foreach (var pair in _info.SampledPairs)
        {
            if (pair.Image == image && pair.Sampler == sampler)
            {
                pair.FirstUsePc = Math.Min(pair.FirstUsePc, pc);
                return;
            }
        }

        if (_info.SampledPairs.Count >= ShaderResourceInfo.MaxSampledPairs)
        {
            throw Failure(pc, "sampled image/sampler pair limit exceeded");
        }

        _info.SampledPairs.Add(new SampledImagePair { Image = image, Sampler = sampler, FirstUsePc = pc });
    }

    private void AddMemoryPatch(int index, uint resource, uint sampler, bool hasSampler, uint pc)
    {
        for (var patch = 0; patch < _memoryPatches.Count; patch++)
        {
            var existing = _memoryPatches[patch];
            if (existing.Index != index)
            {
                continue;
            }

            if (existing.Resource != resource || (hasSampler && existing.HasSampler && existing.Sampler != sampler))
            {
                throw Failure(pc, "memory metadata is reused with incompatible resources");
            }

            if (hasSampler)
            {
                _memoryPatches[patch] = (index, resource, sampler, true);
            }

            return;
        }

        _memoryPatches.Add((index, resource, sampler, hasSampler));
    }

    // ---- collection ----

    private void Collect(int index)
    {
        var memory = _plan.Memory[index];
        var access = _plan.Accesses[index];
        var isBuffer = memory.Kind is MemoryResourceKind.Buffer or MemoryResourceKind.ScalarBuffer;
        var isAddress = memory.Kind is MemoryResourceKind.ScalarAddress or MemoryResourceKind.Flat or MemoryResourceKind.Global or MemoryResourceKind.Scratch;
        var isImage = memory.Kind == MemoryResourceKind.Image;
        if (!isBuffer && !isAddress && !isImage)
        {
            return;
        }

        if (access is null)
        {
            throw Failure(memory.Pc, "memory operation has no resource handle");
        }

        if (memory.PlanningOnly || IsIndirectPlanningMemory(index))
        {
            return;
        }

        if (isBuffer)
        {
            var source = GetHandleSource(access.Handle, ScalarValueKind.BufferHandle, 4, memory.Pc);
            var resource = AddBuffer(source, memory, memory.Pc);
            if (resource == DescriptorConstants.NoIndex)
            {
                throw Failure(memory.Pc, "buffer resource limit exceeded");
            }

            AddMemoryPatch(index, resource, 0, false, memory.Pc);
            return;
        }

        if (isAddress)
        {
            if (memory.Kind == MemoryResourceKind.Scratch)
            {
                // Scratch addresses are private to the current invocation. They
                // are lowered to private scratch storage, so they have no host
                // resource handle or descriptor-table binding to materialize.
                return;
            }

            if (access.Handle is null || access.Handle.Kind != ScalarValueKind.AddressHandle)
            {
                throw Failure(memory.Pc, "address operation requires an address handle");
            }

            if (access.Handle.Operands.Length != 2)
            {
                throw Failure(memory.Pc, "an address handle must have two address dwords");
            }

            _info.UsesDeviceAddresses = true;
            return;
        }

        if (memory.ImageClass == ImageResourceClass.None)
        {
            throw Failure(memory.Pc, "image operation has invalid resource kind");
        }

        uint imageSource;
        var indirect = _indirectImages.FirstOrDefault(plan => ReferenceEquals(plan.Handle, access.Handle));
        if (indirect is not null)
        {
            imageSource = indirect.Source;
        }
        else
        {
            imageSource = GetHandleSource(access.Handle, ScalarValueKind.ImageHandle, 8, memory.Pc, imageR128: memory.ImageR128);
        }

        var image = AddImage(imageSource, memory, memory.Pc);
        if (image == DescriptorConstants.NoIndex)
        {
            throw Failure(memory.Pc, "image resource limit exceeded");
        }

        uint sampler = 0;
        if (memory.NeedsSampler)
        {
            if (access.SamplerHandle is null)
            {
                throw Failure(memory.Pc, "sampled image operation has no sampler handle");
            }

            var sampleAdjust = (memory.ImageSampleFlags & ImageSampleFlags.Adjust) != 0;
            var indirectSampler = _indirectSamplers.FirstOrDefault(plan => ReferenceEquals(plan.Handle, access.SamplerHandle));
            var samplerSource = indirectSampler is null
                ? GetHandleSource(access.SamplerHandle, ScalarValueKind.SamplerHandle, 4, memory.Pc, sampler: true, sampleAdjust)
                : indirectSampler.Source;
            sampler = AddSampler(samplerSource, memory.Pc);
            if (sampler == DescriptorConstants.NoIndex)
            {
                throw Failure(memory.Pc, "sampler resource limit exceeded");
            }

            AddSampledPair(image, sampler, memory.Pc);
        }

        AddMemoryPatch(index, image, sampler, memory.NeedsSampler, memory.Pc);
    }

    private void LinkImageAliases()
    {
        foreach (var buffer in _info.Buffers)
        {
            var bufferSource = _sources[(int)buffer.Source];
            if (bufferSource.DwordCount != 4)
            {
                continue;
            }

            for (var image = 0; image < _info.Images.Count; image++)
            {
                var imageSource = _sources[(int)_info.Images[image].Source];
                if (imageSource.DwordCount != 8 || imageSource.IndirectImage is not null)
                {
                    continue;
                }

                var alias = true;
                for (var dword = 0; dword < 4 && alias; dword++)
                {
                    alias = _graph.Equivalent(bufferSource.Dwords[dword], imageSource.Dwords[dword]);
                }

                if (alias)
                {
                    buffer.ImageAlias = (uint)image;
                    break;
                }
            }
        }
    }

    // ---- indirect images ----

    private bool IsIndirectPlanningMemory(int index) =>
        _indirectImages.Any(plan => plan.SuppressMemoryReads && plan.Memory.Contains(index)) ||
        _indirectSamplers.Any(plan => plan.Memory.Contains(index));

    private void PlanIndirectImages()
    {
        for (var index = 0; index < _plan.Memory.Count; index++)
        {
            var memory = _plan.Memory[index];
            if (memory.Kind != MemoryResourceKind.Image || _plan.Accesses[index]?.Handle is not { } handle ||
                _indirectImages.Any(plan => ReferenceEquals(plan.Handle, handle)))
            {
                continue;
            }

            if (TryMakeStridedIndirectImage(handle, memory.Pc, memory.ImageR128, out var plan) ||
                TryMakeIndirectImage(handle, memory.Pc, out plan) ||
                TryMakeDenseIndirectImage(handle, memory.Pc, out plan) ||
                TryMakeDirectImage(handle, out plan))
            {
                _indirectImages.Add(plan);
            }
        }
    }

    private void PlanIndirectSamplers()
    {
        for (var index = 0; index < _plan.Memory.Count; index++)
        {
            var memory = _plan.Memory[index];
            if (memory.Kind != MemoryResourceKind.Image || _plan.Accesses[index]?.SamplerHandle is not { } handle ||
                _indirectSamplers.Any(plan => ReferenceEquals(plan.Handle, handle)))
            {
                continue;
            }

            if (TryMakeStridedIndirectSampler(handle, memory.Pc, out var plan))
            {
                _indirectSamplers.Add(plan);
            }
        }
    }

    private MemoryAccessInfo? ScalarReadMemory(ScalarValue read, out int index)
    {
        index = read.MemoryIndex;
        if (read.Kind != ScalarValueKind.ScalarBufferWord || index >= _plan.Memory.Count)
        {
            return null;
        }

        var memory = _plan.Memory[index];
        return memory.Kind == MemoryResourceKind.ScalarBuffer && memory.DataBits == 32 && memory.DataDwords == 1 ? memory : null;
    }

    private MemoryAccessInfo? ScalarAddressReadMemory(ScalarValue read, out int index)
    {
        index = read.MemoryIndex;
        if (read.Kind != ScalarValueKind.ScalarAddressWord || index < 0 || index >= _plan.Memory.Count)
        {
            return null;
        }

        var memory = _plan.Memory[index];
        return memory.Kind == MemoryResourceKind.ScalarAddress && memory.DataBits == 32 && memory.DataDwords == 1 ? memory : null;
    }

    private MemoryAccessInfo? ScalarDescriptorReadMemory(ScalarValue read, out int index, out bool address)
    {
        address = false;
        var buffer = ScalarReadMemory(read, out index);
        if (buffer is not null)
        {
            return buffer;
        }

        var scalarAddress = ScalarAddressReadMemory(read, out index);
        address = scalarAddress is not null;
        return scalarAddress;
    }

    private bool MemoryIndexBelongsTo(int index, ScalarValue owner) =>
        !_readsByMemory.TryGetValue(index, out var readers) || readers.All(reader => ReferenceEquals(reader, owner));

    private bool UsesOnly(ScalarValue value, IReadOnlyList<ScalarValue> users) =>
        _uses.TryGetValue(value, out var uses) && uses.Count != 0 && uses.All(user => users.Any(candidate => ReferenceEquals(candidate, user)));

    private bool MakeRuntimeBufferSource(ScalarValue handle, uint pc, out uint sourceIndex, out DescriptorSource source)
    {
        sourceIndex = 0;
        source = null!;
        if (handle.Kind != ScalarValueKind.BufferHandle)
        {
            return false;
        }

        source = MakeSource(handle, 4, false, false, pc);
        if (!ValidateSource(source, out _))
        {
            return false;
        }

        sourceIndex = InternSource(source);
        return true;
    }

    private bool MakeRuntimeAddressSource(ScalarValue handle, uint pc, out uint sourceIndex, out DescriptorSource source)
    {
        sourceIndex = 0;
        source = null!;
        if (handle.Kind != ScalarValueKind.AddressHandle)
        {
            return false;
        }

        source = MakeSource(handle, 2, false, false, pc);
        if (!ValidateSource(source, out _))
        {
            return false;
        }

        sourceIndex = InternSource(source);
        return true;
    }

    private static bool MatchMaterialOffset(ScalarValue value, out ScalarValue selector, out uint stride, out uint offset)
    {
        selector = null!;
        stride = 0;
        offset = 0;
        if (value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.IAdd32)
        {
            if (value.Operands[0].IsConstant)
            {
                offset = value.Operands[0].ConstantU32;
                value = value.Operands[1];
            }
            else if (value.Operands[1].IsConstant)
            {
                offset = value.Operands[1].ConstantU32;
                value = value.Operands[0];
            }
            else
            {
                return false;
            }
        }

        if (value.Kind != ScalarValueKind.Operation || value.Operation != ScalarOperation.IMul32)
        {
            return false;
        }

        if (value.Operands[0].IsConstant)
        {
            stride = value.Operands[0].ConstantU32;
            selector = value.Operands[1];
        }
        else if (value.Operands[1].IsConstant)
        {
            stride = value.Operands[1].ConstantU32;
            selector = value.Operands[0];
        }
        else
        {
            return false;
        }

        return stride != 0 && selector.Kind == ScalarValueKind.FirstLane;
    }

    // A material-table key selects a heap record whose eight dwords are the image
    // descriptor. The key read, the heap reads and their shift must feed nothing else.
    private bool TryMakeIndirectImage(ScalarValue handle, uint pc, out IndirectImagePlan plan)
    {
        plan = null!;
        if (handle.Kind != ScalarValueKind.ImageHandle || handle.Operands.Length != 8)
        {
            return false;
        }

        var heapReads = new ScalarValue[8];
        ScalarValue? heapHandle = null;
        ScalarValue? heapOffset = null;
        var memoryIndices = new int[8];
        for (var dword = 0; dword < 8; dword++)
        {
            var read = handle.Operands[dword];
            heapReads[dword] = read;
            var memory = ScalarReadMemory(read, out var memoryIndex);
            if (memory is null || memory.Offset != (uint)dword * sizeof(uint) || !MemoryIndexBelongsTo(memoryIndex, read))
            {
                return false;
            }

            var currentHandle = read.Operands[0];
            if (heapHandle is not null && !ReferenceEquals(currentHandle, heapHandle))
            {
                return false;
            }

            heapHandle = currentHandle;
            if (dword == 0)
            {
                heapOffset = read.Operands[1];
            }
            else if (!_graph.Equivalent(heapOffset!, read.Operands[1]))
            {
                return false;
            }

            memoryIndices[dword] = memoryIndex;
        }

        if (heapOffset!.Kind != ScalarValueKind.Operation || heapOffset.Operation != ScalarOperation.ShiftLeft32 ||
            !heapOffset.Operands[1].IsConstant || heapOffset.Operands[1].ConstantU32 != 5)
        {
            return false;
        }

        var materialRead = heapOffset.Operands[0];
        var materialMemory = ScalarReadMemory(materialRead, out var materialMemoryIndex);
        if (materialMemory is null || materialMemory.Offset != 0 || !MemoryIndexBelongsTo(materialMemoryIndex, materialRead))
        {
            return false;
        }

        var materialHandle = materialRead.Operands[0];
        if (!MatchMaterialOffset(materialRead.Operands[1], out var selector, out var selectorStride, out var selectorOffset))
        {
            return false;
        }

        if (!UsesOnly(materialRead, [heapOffset]) || !UsesOnly(heapOffset, heapReads))
        {
            return false;
        }

        foreach (var read in heapReads)
        {
            if (!UsesOnly(read, [handle]))
            {
                return false;
            }
        }

        if (!MakeRuntimeBufferSource(materialHandle, pc, out var materialSourceIndex, out var materialSource) ||
            !MakeRuntimeBufferSource(heapHandle!, pc, out var heapSourceIndex, out var heapSource))
        {
            return false;
        }

        var imageDwords = new ScalarValue[8];
        Array.Copy(materialSource.Dwords, 0, imageDwords, 0, 4);
        Array.Copy(heapSource.Dwords, 0, imageDwords, 4, 4);
        var imageSource = new DescriptorSource
        {
            Dwords = imageDwords,
            IndirectImage = new IndirectImageSelector(materialSourceIndex, heapSourceIndex, selectorStride, selectorOffset, 0)
            {
                SelectorValues = IndirectSelectorValues.Create(_plan, selector),
            },
        };

        plan = new IndirectImagePlan
        {
            Handle = handle,
            Source = InternSource(imageSource),
            Key = materialRead,
            HeapSource = heapSourceIndex,
            Memory = memoryIndices,
            Reads = heapReads,
        };
        return true;
    }

    // A structured descriptor record can be selected by a key table whose entry
    // stride differs from the record stride.  Unlike the legacy material/heap form,
    // the image handle may carry its upper dwords through loop state; the host reads
    // the complete descriptor record from the heap once the key is known.
    private bool TryMakeStridedIndirectImage(ScalarValue handle, uint pc, bool imageR128, out IndirectImagePlan plan)
    {
        plan = null!;
        if (handle.Kind != ScalarValueKind.ImageHandle || handle.Operands.Length != 8)
        {
            return false;
        }

        var reads = new List<ScalarValue>();
        var memoryIndices = new List<int>();
        ScalarValue? heapHandle = null;
        ScalarValue? heapOffset = null;
        bool? heapIsAddress = null;
        uint descriptorOffset = 0;
        for (var dword = 0; dword < 4; dword++)
        {
            if (!TryUnwrapLoopDescriptorRead(handle.Operands[dword], out var read))
            {
                return false;
            }

            var memory = ScalarDescriptorReadMemory(read, out var memoryIndex, out var addressHeap);
            if (memory is null || (dword == 0 ? false : memory.Offset != descriptorOffset + (uint)dword * sizeof(uint)) ||
                !MemoryIndexBelongsTo(memoryIndex, read))
            {
                return false;
            }

            if (heapIsAddress is not null && heapIsAddress.Value != addressHeap)
            {
                return false;
            }

            heapIsAddress = addressHeap;

            var currentHandle = read.Operands[0];
            if (heapHandle is not null && !ReferenceEquals(currentHandle, heapHandle))
            {
                return false;
            }

            heapHandle = currentHandle;
            if (dword == 0)
            {
                descriptorOffset = memory.Offset;
                heapOffset = read.Operands[1];
            }
            else if (!_graph.Equivalent(heapOffset!, read.Operands[1]))
            {
                return false;
            }

            reads.Add(read);
            memoryIndices.Add(memoryIndex);
        }

        if (heapOffset is null || !MatchHeapOffset(heapOffset, out var rawKey, out var heapStride))
        {
            return false;
        }

        // Some shaders walk a descriptor array with a monotonic loop index:
        //   offset = index * recordStride
        // The descriptor words are runtime scalar-buffer reads, but the index is
        // not itself a memory read.  Track this as an offset-keyed indirect image
        // so the host can enumerate the records without weakening validation for
        // arbitrary dynamic arithmetic.
        var descriptorArray = false;
        ScalarValue key;
        var rawIsMaterial = TryUnwrapLoopDescriptorRead(rawKey, out var materialKey);
        var rawIsArray = heapIsAddress != true && MatchDescriptorArrayIndex(heapOffset, out _);
        if (rawIsArray && rawKey.Kind == ScalarValueKind.Phi && rawKey.Operands.Length > 2)
        {
            descriptorArray = true;
            key = reads[0];
        }
        else if (rawIsMaterial &&
            materialKey.Kind == ScalarValueKind.ScalarBufferWord)
        {
            key = materialKey;
        }
        else if (rawIsArray)
        {
            descriptorArray = true;
            key = reads[0];
        }
        else
        {
            return false;
        }

        var keyMemory = ScalarReadMemory(key, out var keyMemoryIndex);
        if (!descriptorArray && (keyMemory is null || !MemoryIndexBelongsTo(keyMemoryIndex, key)))
        {
            return false;
        }

        var materialHandle = keyMemory is null ? null : key.Operands[0];
        ScalarValue? selector = null;
        uint selectorStride = 0;
        uint selectorOffset = 0;
        if (!descriptorArray &&
            !MatchKeyOffset(key.Operands[1], out selector, out selectorStride, out selectorOffset))
        {
            return false;
        }

        // The legacy scalar-buffer proof owns the ordinary IMul32 material
        // offset form.  Keep this structured path's broader multiplication
        // matching scoped to raw address heaps, where the descriptor records
        // cannot be handled by the legacy path.
        var keyOffset = key.Operands[1];
        if (keyOffset.Kind == ScalarValueKind.Operation && keyOffset.Operation == ScalarOperation.IAdd32 &&
            keyOffset.Operands.Length == 2)
        {
            keyOffset = keyOffset.Operands[0].IsConstant ? keyOffset.Operands[1] : keyOffset.Operands[0];
        }

        if (!descriptorArray && heapIsAddress != true && keyOffset.Kind == ScalarValueKind.Operation &&
            keyOffset.Operation == ScalarOperation.IMul32)
        {
            return false;
        }

        var keyMemoryOffset = keyMemory?.Offset ?? 0;
        if (!descriptorArray && keyMemoryOffset > uint.MaxValue - selectorOffset)
        {
            return false;
        }

        if (!descriptorArray)
        {
            selectorOffset += keyMemoryOffset;
        }

        if (descriptorArray)
        {
            var arraySourceValid = MakeRuntimeBufferSource(heapHandle!, pc, out var arrayHeapSourceIndex, out _);
            if (!arraySourceValid)
            {
                return false;
            }

            var arrayDwords = Enumerable.Repeat(_graph.Constant(0u), 8).ToArray();
            var arraySource = new DescriptorSource
            {
                Dwords = arrayDwords,
                IndirectImage = new IndirectImageSelector(arrayHeapSourceIndex, arrayHeapSourceIndex, 0, 0, 0)
                {
                    DescriptorArray = true,
                    HeapStride = heapStride,
                    HeapOffset = descriptorOffset,
                    HeapIsAddress = false,
                },
            };

            plan = new IndirectImagePlan
            {
                Handle = handle,
                Source = InternSource(arraySource),
                Key = key,
                HeapSource = arrayHeapSourceIndex,
                KeyIsAddressOffset = true,
                Memory = [.. memoryIndices],
                Reads = [.. reads],
            };
            return true;
        }

        if (!MakeRuntimeBufferSource(materialHandle!, pc, out var materialSourceIndex, out var materialSource))
        {
            return false;
        }

        uint heapSourceIndex;
        DescriptorSource heapSource;
        var heapSourceValid = heapIsAddress == true
            ? MakeRuntimeAddressSource(heapHandle!, pc, out heapSourceIndex, out heapSource)
            : MakeRuntimeBufferSource(heapHandle!, pc, out heapSourceIndex, out heapSource);
        if (!heapSourceValid)
        {
            return false;
        }

        var imageDwords = Enumerable.Repeat(_graph.Constant(0u), 8).ToArray();
        if (heapIsAddress != true)
        {
            Array.Copy(heapSource.Dwords, imageDwords, 4);
            if (!imageR128)
            {
                Array.Copy(heapSource.Dwords, 0, imageDwords, 4, 4);
            }
        }
        var descriptorHeapOffsets = Enumerable.Repeat(uint.MaxValue, 8).ToArray();
        var descriptorHeapMasks = Enumerable.Repeat(uint.MaxValue, 8).ToArray();
        var descriptorHeapAlternates = Enumerable.Range(0, 8).Select(_ => Array.Empty<uint>()).ToArray();
        var descriptorTableSlots = Enumerable.Repeat(uint.MaxValue, 8).ToArray();
        var descriptorStaticValues = new uint[8];
        for (var dword = 0; dword < 4; dword++)
        {
            descriptorHeapOffsets[dword] = descriptorOffset + (uint)dword * sizeof(uint);
        }

        // Structured records can assemble the extended image words from other
        // offsets in the same record. Preserve those offsets instead of assuming
        // that all eight words are contiguous.
        for (var dword = imageR128 ? 8 : 4; dword < 8; dword++)
        {
            var value = handle.Operands[dword];
            if (ScalarDescriptorReadMemory(value, out var fieldMemoryIndex, out _) is not null &&
                value.Operands.Length >= 2 && ReferenceEquals(value.Operands[0], heapHandle) &&
                _graph.Equivalent(value.Operands[1], heapOffset) &&
                MemoryIndexBelongsTo(fieldMemoryIndex, value))
            {
                reads.Add(value);
                memoryIndices.Add(fieldMemoryIndex);
            }

            if (TryMatchDescriptorField(value, heapHandle!, heapOffset, out var fieldOffset, out var fieldMask, out var alternateFieldOffsets, out var tableSlot))
            {
                if (fieldOffset != uint.MaxValue)
                {
                    descriptorHeapOffsets[dword] = fieldOffset;
                }

                descriptorHeapMasks[dword] = fieldMask;
                descriptorHeapAlternates[dword] = alternateFieldOffsets.ToArray();

                if (tableSlot != uint.MaxValue)
                {
                    // A loop-carried descriptor can retain a flattened table word
                    // on one edge and reload the corresponding heap field on another.
                    // Keep both sources; materialization will require them to agree.
                    descriptorTableSlots[dword] = tableSlot;
                }
            }
            else if (!TryGetStaticDescriptorValue(value, out descriptorStaticValues[dword]))
            {
                return false;
            }
        }

        var imageSource = new DescriptorSource
        {
            Dwords = imageDwords,
            IndirectImage = new IndirectImageSelector(materialSourceIndex, heapSourceIndex, selectorStride, selectorOffset, 0)
            {
                HeapStride = heapStride,
                HeapOffset = descriptorOffset,
                HeapIsAddress = heapIsAddress == true,
                DescriptorHeapOffsets = descriptorHeapOffsets,
                DescriptorHeapMasks = descriptorHeapMasks,
                DescriptorHeapAlternates = descriptorHeapAlternates,
                DescriptorTableSlots = descriptorTableSlots,
                DescriptorStaticValues = descriptorStaticValues,
            },
        };

        plan = new IndirectImagePlan
        {
            Handle = handle,
            Source = InternSource(imageSource),
            Key = key,
            HeapSource = heapSourceIndex,
            Memory = [.. memoryIndices],
            Reads = [.. reads],
        };
        return true;
    }

    // Descriptor words can be carried around a loop using a self-referential phi.
    // This is still a runtime load when every non-phi incoming edge resolves to the
    // same scalar descriptor read.  Keep the proof strict: divergent reads, other
    // arithmetic, and undefined incoming values remain rejected by the normal path.
    private bool TryUnwrapLoopDescriptorRead(ScalarValue value, out ScalarValue read)
    {
        read = null!;
        var pending = new Stack<ScalarValue>();
        var visited = new HashSet<ScalarValue>();
        pending.Push(value);
        while (pending.TryPop(out var current))
        {
            if (!visited.Add(current))
            {
                continue;
            }

            if (ScalarDescriptorReadMemory(current, out _, out var currentAddressHeap) is not null)
            {
                if (read is null)
                {
                    read = current;
                }
                else
                {
                    var readMemory = ScalarDescriptorReadMemory(read, out _, out var readAddressHeap);
                    if (readMemory is null || readAddressHeap != currentAddressHeap || !_graph.Equivalent(read, current))
                    {
                        read = null!;
                        return false;
                    }
                }

                continue;
            }

            if (current.Kind != ScalarValueKind.Phi)
            {
                read = null!;
                return false;
            }

            foreach (var operand in current.Operands)
            {
                if (!ReferenceEquals(operand, current))
                {
                    pending.Push(operand);
                }
            }
        }

        return read is not null;
    }

    private bool TryMatchDescriptorField(
        ScalarValue value,
        ScalarValue heapHandle,
        ScalarValue heapOffset,
        out uint fieldOffset,
        out uint fieldMask,
        out IReadOnlyList<uint> alternateFieldOffsets,
        out uint tableSlot)
    {
        fieldOffset = uint.MaxValue;
        fieldMask = uint.MaxValue;
        alternateFieldOffsets = [];
        tableSlot = uint.MaxValue;
        var foundOffsets = new HashSet<uint>();
        uint? foundTableSlot = null;
        uint? foundMask = null;
        var pending = new Stack<(ScalarValue Value, uint Mask)>();
        var visited = new HashSet<ScalarValue>();
        pending.Push((value, uint.MaxValue));
        while (pending.TryPop(out var current))
        {
            if (!visited.Add(current.Value))
            {
                continue;
            }

            var memory = ScalarDescriptorReadMemory(current.Value, out _, out _);
            if (memory is not null && current.Value.Operands.Length >= 2 &&
                ReferenceEquals(current.Value.Operands[0], heapHandle) &&
                _graph.Equivalent(current.Value.Operands[1], heapOffset) &&
                MemoryIndexBelongsTo(current.Value.MemoryIndex, current.Value))
            {
                foundOffsets.Add(memory.Offset);
                if (foundMask is not null && foundMask.Value != current.Mask)
                {
                    return false;
                }

                foundMask = current.Mask;
                continue;
            }

            if (current.Value.Kind == ScalarValueKind.ResourceTableWord)
            {
                if (current.Value.Payload > uint.MaxValue)
                {
                    return false;
                }

                var currentSlot = (uint)current.Value.Payload;
                if (foundTableSlot is not null && foundTableSlot.Value != currentSlot)
                {
                    return false;
                }

                if (foundMask is not null && foundMask.Value != current.Mask)
                {
                    return false;
                }

                foundMask = current.Mask;
                foundTableSlot = currentSlot;
                continue;
            }

            if (current.Value.Kind == ScalarValueKind.Phi)
            {
                foreach (var operand in current.Value.Operands)
                {
                    if (!ReferenceEquals(operand, current.Value))
                    {
                        pending.Push((operand, current.Mask));
                    }
                }

                continue;
            }

            // Structured descriptors sometimes mask a runtime field before
            // placing it in the image handle.  Preserve the operation rather
            // than treating the source as an unrelated descriptor word.  The
            // materializer applies this constant mask to both heap and table
            // candidates before it checks that alternate paths agree.
            if (current.Value.Kind == ScalarValueKind.Operation &&
                current.Value.Operation == ScalarOperation.And32 && current.Value.Operands.Length == 2)
            {
                var left = current.Value.Operands[0];
                var right = current.Value.Operands[1];
                if (left.IsConstant && !right.IsConstant)
                {
                    pending.Push((right, current.Mask & left.ConstantU32));
                    continue;
                }

                if (right.IsConstant && !left.IsConstant)
                {
                    pending.Push((left, current.Mask & right.ConstantU32));
                    continue;
                }
            }

            // A loop's initial descriptor word can be a neutral cselect while
            // the steady-state value comes from the structured record.
            if (!TryGetStaticDescriptorValue(current.Value, out _))
            {
                return false;
            }
        }

        if (foundOffsets.Count == 0 && foundTableSlot is null)
        {
            return false;
        }

        if (foundOffsets.Count > 0)
        {
            var selectedOffset = foundOffsets.Min();
            fieldOffset = selectedOffset;
            alternateFieldOffsets = foundOffsets.Where(offset => offset != selectedOffset).Order().ToArray();
        }

        fieldMask = foundMask ?? uint.MaxValue;
        tableSlot = foundTableSlot ?? uint.MaxValue;
        return true;
    }

    private static bool TryGetStaticDescriptorValue(ScalarValue value, out uint result)
    {
        uint? found = null;
        var pending = new Stack<ScalarValue>();
        var visited = new HashSet<ScalarValue>();
        pending.Push(value);
        while (pending.TryPop(out var current))
        {
            if (!visited.Add(current))
            {
                continue;
            }

            uint currentValue;
            if (current.IsConstant)
            {
                currentValue = current.ConstantU32;
            }
            else if (current.Kind == ScalarValueKind.Select && current.Operands.Length == 3 &&
                     current.Operands[0].Kind == ScalarValueKind.Undefined &&
                     current.Operands[1].IsConstant && current.Operands[2].IsConstant)
            {
                // An unmaterialised cselect condition can occur in descriptor
                // padding; the false arm is the neutral/default encoding.
                currentValue = current.Operands[2].ConstantU32;
            }
            else if (current.Kind == ScalarValueKind.Phi)
            {
                foreach (var operand in current.Operands)
                {
                    if (!ReferenceEquals(operand, current))
                    {
                        pending.Push(operand);
                    }
                }

                continue;
            }
            else
            {
                result = 0;
                return false;
            }

            if (found is not null && found.Value != currentValue)
            {
                result = 0;
                return false;
            }

            found = currentValue;
        }

        result = found ?? 0;
        return found is not null;
    }

    private static bool TryGetDescriptorTableSlot(ScalarValue value, out uint slot)
    {
        slot = 0;
        uint? found = null;
        var pending = new Stack<ScalarValue>();
        var visited = new HashSet<ScalarValue>();
        pending.Push(value);
        while (pending.TryPop(out var current))
        {
            if (!visited.Add(current))
            {
                continue;
            }

            if (current.Kind == ScalarValueKind.ResourceTableWord)
            {
                if (current.Payload > uint.MaxValue)
                {
                    return false;
                }

                var currentSlot = (uint)current.Payload;
                if (found is not null && found.Value != currentSlot)
                {
                    return false;
                }

                found = currentSlot;
                continue;
            }

            if (current.Kind != ScalarValueKind.Phi)
            {
                return false;
            }

            foreach (var operand in current.Operands)
            {
                if (!ReferenceEquals(operand, current))
                {
                    pending.Push(operand);
                }
            }
        }

        if (found is null)
        {
            return false;
        }

        slot = found.Value;
        return true;
    }

    private bool TryMakeStridedIndirectSampler(ScalarValue handle, uint pc, out IndirectSamplerPlan plan)
    {
        plan = null!;
        if (handle.Kind != ScalarValueKind.SamplerHandle || handle.Operands.Length != 4)
        {
            return false;
        }

        var reads = new ScalarValue[4];
        var memoryIndices = new int[4];
        ScalarValue? heapHandle = null;
        ScalarValue? heapOffset = null;
        uint descriptorOffset = 0;
        for (var dword = 0; dword < reads.Length; dword++)
        {
            if (!TryUnwrapLoopDescriptorRead(handle.Operands[dword], out var read))
            {
                return false;
            }

            var memory = ScalarReadMemory(read, out var memoryIndex);
            if (memory is null || (dword != 0 && memory.Offset != descriptorOffset + (uint)dword * sizeof(uint)) ||
                !MemoryIndexBelongsTo(memoryIndex, read))
            {
                return false;
            }

            var currentHandle = read.Operands[0];
            if (heapHandle is not null && !ReferenceEquals(currentHandle, heapHandle))
            {
                return false;
            }

            heapHandle = currentHandle;
            if (dword == 0)
            {
                descriptorOffset = memory.Offset;
                heapOffset = read.Operands[1];
            }
            else if (!_graph.Equivalent(heapOffset!, read.Operands[1]))
            {
                return false;
            }

            reads[dword] = read;
            memoryIndices[dword] = memoryIndex;
        }

        if (heapOffset is null || !MatchHeapOffset(heapOffset, out var key, out var heapStride) ||
            key.Kind != ScalarValueKind.ScalarBufferWord)
        {
            return false;
        }

        var keyMemory = ScalarReadMemory(key, out var keyMemoryIndex);
        if (keyMemory is null || keyMemory.Offset != 0 || !MemoryIndexBelongsTo(keyMemoryIndex, key) ||
            !MatchKeyOffset(key.Operands[1], out _, out var selectorStride, out var selectorOffset))
        {
            return false;
        }

        if (!MakeRuntimeBufferSource(key.Operands[0], pc, out var materialSourceIndex, out _) ||
            !MakeRuntimeBufferSource(heapHandle!, pc, out var heapSourceIndex, out var heapSource))
        {
            return false;
        }

        var samplerSource = new DescriptorSource
        {
            Dwords = [.. heapSource.Dwords],
            IndirectSampler = new IndirectImageSelector(materialSourceIndex, heapSourceIndex, selectorStride, selectorOffset, 0)
            {
                HeapStride = heapStride,
                HeapOffset = descriptorOffset,
            },
        };

        plan = new IndirectSamplerPlan
        {
            Handle = handle,
            Source = InternSource(samplerSource),
            Key = key,
            HeapSource = heapSourceIndex,
            Memory = memoryIndices,
            Reads = reads,
        };
        return true;
    }

    private static bool MatchHeapOffset(ScalarValue value, out ScalarValue key, out uint stride)
    {
        key = null!;
        stride = 0;
        if (value.Kind != ScalarValueKind.Operation || value.Operands.Length != 2)
        {
            return false;
        }

        if (value.Operation == ScalarOperation.ShiftLeft32 && value.Operands[1].IsConstant &&
            value.Operands[1].ConstantU32 < 32)
        {
            stride = 1u << (int)value.Operands[1].ConstantU32;
            key = value.Operands[0];
            if (key.Kind == ScalarValueKind.Operation && key.Operation == ScalarOperation.And32 &&
                key.Operands.Length == 2)
            {
                if (key.Operands[0].IsConstant)
                {
                    key = key.Operands[1];
                }
                else if (key.Operands[1].IsConstant)
                {
                    key = key.Operands[0];
                }
            }

            return stride != 0;
        }

        if (value.Operation != ScalarOperation.IMul32)
        {
            return false;
        }

        if (value.Operands[0].IsConstant)
        {
            stride = value.Operands[0].ConstantU32;
            key = value.Operands[1];
        }
        else if (value.Operands[1].IsConstant)
        {
            stride = value.Operands[1].ConstantU32;
            key = value.Operands[0];
        }

        return stride != 0;
    }

    private static bool MatchKeyOffset(ScalarValue value, out ScalarValue selector, out uint stride, out uint offset)
    {
        selector = null!;
        stride = 0;
        offset = 0;
        if (value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.IAdd32 &&
            value.Operands.Length == 2)
        {
            if (value.Operands[0].IsConstant)
            {
                offset = value.Operands[0].ConstantU32;
                value = value.Operands[1];
            }
            else if (value.Operands[1].IsConstant)
            {
                offset = value.Operands[1].ConstantU32;
                value = value.Operands[0];
            }
            else
            {
                return false;
            }
        }

        if (value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.ShiftLeft32 &&
            value.Operands.Length == 2 && value.Operands[1].IsConstant && value.Operands[1].ConstantU32 < 32)
        {
            stride = 1u << (int)value.Operands[1].ConstantU32;
            selector = value.Operands[0];
            return true;
        }

        if (value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.IMul32 &&
            value.Operands.Length == 2)
        {
            if (value.Operands[0].IsConstant)
            {
                stride = value.Operands[0].ConstantU32;
                selector = value.Operands[1];
            }
            else if (value.Operands[1].IsConstant)
            {
                stride = value.Operands[1].ConstantU32;
                selector = value.Operands[0];
            }

            return stride != 0;
        }

        return false;
    }

    private static bool MatchDescriptorArrayIndex(ScalarValue value, out uint stride)
    {
        stride = 0;
        if (!MatchHeapOffset(value, out var index, out stride) || index.Kind != ScalarValueKind.Phi ||
            index.Operands.Length < 2 || index.Operands.Any(operand => operand.Type != ScalarValueType.U32 ||
                operand.Kind == ScalarValueKind.Undefined))
        {
            return false;
        }

        // A descriptor-array selector is often carried through several control
        // flow joins.  In that form the graph contains an entry value plus one
        // or more loop/back-edge values rather than the simple two-input
        // induction phi.  The fixed multiply above is the important proof: the
        // selector addresses records of one known stride, while the descriptor
        // words themselves are separately proven scalar-buffer reads.
        if (index.Operands.Length != 2)
        {
            return true;
        }

        var initial = index.Operands.FirstOrDefault(operand => operand.IsConstant);
        var update = index.Operands.FirstOrDefault(operand => !operand.IsConstant);
        if (initial is null || update is null || update.Kind != ScalarValueKind.Operation ||
            update.Operation != ScalarOperation.IAdd32 || update.Operands.Length != 2)
        {
            return false;
        }

        var carriesIndex = ReferenceEquals(update.Operands[0], index) ? update.Operands[1] :
            ReferenceEquals(update.Operands[1], index) ? update.Operands[0] : null;
        return carriesIndex is { IsConstant: true } && carriesIndex.ConstantU32 != 0;
    }

    private static bool IsLoopDescriptorValue(ScalarValue value)
    {
        var pending = new Stack<ScalarValue>();
        var visited = new HashSet<ScalarValue>();
        pending.Push(value);
        while (pending.TryPop(out var current))
        {
            if (!visited.Add(current))
            {
                continue;
            }

            if (current.Kind is ScalarValueKind.ResourceTableWord or ScalarValueKind.ScalarBufferWord or ScalarValueKind.Constant)
            {
                continue;
            }

            if (current.Kind != ScalarValueKind.Phi)
            {
                return false;
            }

            foreach (var operand in current.Operands)
            {
                pending.Push(operand);
            }
        }

        return true;
    }
}
