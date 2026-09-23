// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpEmu.ShaderCompiler.Resources;

// Resolves a plan against one draw: descriptor sources, indirect image tables and the
// specialization. Both outputs change only when the whole materialisation succeeds.
public static class ResourceMaterializer
{
    private const ulong AddressMask = 0x0000_FFFF_FFFF_FFFFul;
    private const ulong MaxIndirectImageProbes = 65536;

    // Written to standard error like every specialization refusal; the host turns it
    // into its fatal.
    public static Action<string> SpecializationFailed { get; set; } = message => Console.Error.WriteLine($"shader resource specialization failed: {message}");

    // Descriptor arrays can be written by an earlier GPU dispatch. The clean
    // reader deliberately refuses such a range, while the normal reader performs
    // the required CPU synchronization before returning the guest word. Use that
    // synchronized path only after the clean read declines the address.
    private static bool TryReadStableWord(ResourceRuntimeInputs inputs, ulong address, out uint word)
    {
        word = 0;
        if (inputs.ReadCleanMemory is { } clean && clean(address, out word))
        {
            return true;
        }

        if (inputs.ReadMemory is { } reader && !reader.Equals(inputs.ReadCleanMemory) && reader(address, out word))
        {
            return true;
        }

        return false;
    }

    private sealed class IndirectImageTable
    {
        public uint Resource;
        public List<uint> Keys = [];
        public List<uint> Candidates = [];
        public List<DescriptorWords> Descriptors = [];
        public uint[] MaterialDescriptor = [];
        public uint[] HeapDescriptor = [];
        public uint SelectorStride;
        public uint SelectorOffset;
        public IndirectSelectorDiagnostic? SelectorDiagnostic;
    }

    private sealed class IndirectSamplerTable
    {
        public uint Resource;
        public List<uint> Keys = [];
        public List<uint> Candidates = [];
        public List<DescriptorWords> Descriptors = [];
        public uint[] MaterialDescriptor = [];
        public uint[] HeapDescriptor = [];
        public uint SelectorStride;
        public uint SelectorOffset;
    }

    private sealed class MaterializedSnapshot
    {
        public uint[][] Buffers = [];
        public uint[][] Images = [];
        public uint[][] Samplers = [];
        public uint[] FlattenedTable = [];
        public uint[] UserData = [];
        public List<IndirectImageTable> IndirectImages = [];
        public List<IndirectSamplerTable> IndirectSamplers = [];
    }

    public static bool Materialize(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        Action<IndirectImageFailure>? captureIndirectImageFailure = null)
        => Materialize(plan, inputs, ref snapshot, ref specialization, out _, captureIndirectImageFailure);

    public static bool Materialize(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure,
        Action<IndirectImageFailure>? captureIndirectImageFailure = null)
    {
        using var totalProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.Total);
        if (!MaterializeSnapshot(plan, inputs, captureIndirectImageFailure is not null, out var materialized, out failure))
        {
            return false;
        }

        // The written ranges follow the table reads so every store can check its own.
        DeviceAddressRange[] ranges;
        using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.DeviceAddressRanges))
            ranges = DeviceAddressRangePlanner.Evaluate(plan, inputs);
        foreach (var range in ranges)
        {
            if (!plan.WrittenRangeSlotByHandle.TryGetValue(range.Handle, out var slot))
            {
                continue;
            }

            var offset = (int)slot;
            materialized.FlattenedTable[offset] = (uint)range.Base;
            materialized.FlattenedTable[offset + 1] = (uint)(range.Base >> 32);
            materialized.FlattenedTable[offset + 2] = (uint)Math.Min(range.Size, uint.MaxValue);
        }

        if (!BuildSpecialization(plan, materialized, out var nextSnapshot, out var nextSpecialization, out failure, captureIndirectImageFailure))
        {
            return false;
        }

        using var outputProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.OutputAssembly);
        snapshot = new ResourceSnapshot
        {
            Buffers = nextSnapshot.Buffers,
            Images = nextSnapshot.Images,
            Samplers = nextSnapshot.Samplers,
            FlattenedResourceTable = nextSnapshot.FlattenedTable,
            UserData = nextSnapshot.UserData,
            DeviceAddressRanges = ranges,
        };
        specialization = nextSpecialization;
        failure = ResourceMaterializationFailure.None;
        return true;
    }

    // ---- snapshot ----

    private static bool MaterializeSnapshot(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        bool captureSelectorDiagnostic, out MaterializedSnapshot snapshot, out ResourceMaterializationFailure failure)
    {
        using var snapshotProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.Snapshot);
        failure = ResourceMaterializationFailure.Other;
        snapshot = new MaterializedSnapshot();
        if (plan.RequiresSpecializationMemory && inputs.ReadCleanMemory is null)
        {
            return false;
        }

        if (!RuntimeValueEvaluator.EvaluateSources(plan, plan.MaterializationSources, inputs, plan.CleanFlatSlots,
            evaluateTable: true, out var values, out var table, out var activeSources,
            additionalTableWords: checked(plan.WrittenRangeCount * ShaderResourcePlan.WrittenRangeDwordCount)))
        {
            SpecializationFailed(DiagnoseSnapshotEvaluationFailure(plan, inputs, activeSources));
            return false;
        }

        var cursor = 0;
        snapshot.Buffers = new uint[plan.Info.Buffers.Count][];
        for (var index = 0; index < snapshot.Buffers.Length; index++)
            snapshot.Buffers[index] = values[cursor++].Dwords;
        snapshot.FlattenedTable = table;
        snapshot.Images = new uint[plan.Info.Images.Count][];
        for (var imageIndex = 0; imageIndex < plan.Info.Images.Count; imageIndex++)
        {
            var image = plan.Info.Images[imageIndex];
            var source = plan.DescriptorSources[(int)image.Source];
            if (source.IndirectImage is { } indirect)
            {
                if (activeSources.Length != 0 && !activeSources[image.Source])
                {
                    snapshot.Images[imageIndex] = new uint[8];
                    continue;
                }
                var cleanInputs = inputs.WithReader(inputs.ReadCleanMemory);
                if (indirect.DirectCandidates is { } directCandidates)
                {
                    if (!RuntimeValueEvaluator.EvaluateSources(plan, directCandidates.Select(candidate => candidate.Source).ToArray(),
                        cleanInputs, [], evaluateTable: false, out var descriptors, out _)) return false;
                    var directTable = new IndirectImageTable { Resource = (uint)imageIndex };
                    for (var candidateIndex = 0; candidateIndex < descriptors.Count; candidateIndex++)
                    {
                        var descriptor = descriptors[candidateIndex];
                        if (NullImageDescriptor(descriptor.Dwords) || !ValidImageDescriptor(descriptor.Dwords, image.R128) ||
                            !ReservedImageBitsClear(descriptor.Dwords))
                            descriptor = DescriptorWords.Empty(8);
                        var existing = directTable.Descriptors.FindIndex(candidate => candidate.SameAs(descriptor));
                        if (existing < 0)
                        {
                            existing = directTable.Descriptors.Count;
                            directTable.Descriptors.Add(descriptor);
                        }
                        directTable.Keys.Add(directCandidates[candidateIndex].Offset);
                        directTable.Candidates.Add((uint)existing);
                    }
                    snapshot.Images[imageIndex] = directTable.Descriptors[(int)directTable.Candidates[0]].Dwords;
                    if (directTable.Descriptors.Count > 1) snapshot.IndirectImages.Add(directTable);
                    continue;
                }
        if (indirect.Dense)
                {
                    if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.HeapSource], cleanInputs, [], evaluateTable: false,
                        out var denseSources, out _))
                        return false;
                    if (!MaterializeDenseIndirectImage(indirect, denseSources[0], image, image.R128, inputs, out var denseTable, out failure))
                        return false;
                    snapshot.Images[imageIndex] = denseTable.Descriptors[(int)denseTable.Candidates[0]].Dwords;
                    if (denseTable.Descriptors.Count > 1)
                    {
                        denseTable.Resource = (uint)imageIndex;
                        snapshot.IndirectImages.Add(denseTable);
                    }
                    continue;
                }
        if (indirect.DescriptorArray)
                {
                    if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.HeapSource], cleanInputs, [], evaluateTable: false, out var arrayTables, out _))
                    {
                        return false;
                    }

                    if (!MaterializeDescriptorArray(indirect, arrayTables[0], image.R128, inputs,
                        out var arrayTable, out failure))
                    {
                        return false;
                    }

                    snapshot.Images[imageIndex] = arrayTable.Descriptors[(int)arrayTable.Candidates[0]].Dwords;
                    if (arrayTable.Descriptors.Count > 1)
                    {
                        arrayTable.Resource = (uint)imageIndex;
                        snapshot.IndirectImages.Add(arrayTable);
                    }

                    continue;
                }

                if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.MaterialSource, indirect.HeapSource], cleanInputs, [], evaluateTable: false, out var tables, out _))
                {
                    return false;
                }

                if (!MaterializeIndirectImage(plan, indirect, tables[0], tables[1], snapshot.FlattenedTable, image.R128, inputs,
                    captureSelectorDiagnostic, out var indirectTable, out failure))
                {
                    return false;
                }

                snapshot.Images[imageIndex] = indirectTable.Descriptors[(int)indirectTable.Candidates[0]].Dwords;
                if (indirectTable.Descriptors.Count > 1)
                {
                    indirectTable.Resource = (uint)imageIndex;
                    snapshot.IndirectImages.Add(indirectTable);
                }
            }
            else
            {
                var descriptor = values[cursor++];
                if (!ValidImageDescriptor(descriptor.Dwords, image.R128))
                {
                    descriptor = DescriptorWords.Empty(descriptor.DwordCount);
                }

                snapshot.Images[imageIndex] = descriptor.Dwords;
            }
        }

        snapshot.Samplers = new uint[plan.Info.Samplers.Count][];
        for (var index = 0; index < snapshot.Samplers.Length; index++)
        {
            var source = plan.DescriptorSources[(int)plan.Info.Samplers[index].Source];
            if (source.IndirectSampler is not { } indirect)
            {
                snapshot.Samplers[index] = values[cursor++].Dwords;
                continue;
            }

            var cleanInputs = inputs.WithReader(inputs.ReadCleanMemory);
            if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.MaterialSource, indirect.HeapSource], cleanInputs, [],
                evaluateTable: false, out var tables, out _))
            {
                return false;
            }

            if (!MaterializeIndirectSampler(plan, indirect, tables[0], tables[1], inputs, out var samplerTable))
            {
                return false;
            }

            snapshot.Samplers[index] = samplerTable.Descriptors[(int)samplerTable.Candidates[0]].Dwords;
            if (samplerTable.Descriptors.Count > 1)
            {
                samplerTable.Resource = (uint)index;
                snapshot.IndirectSamplers.Add(samplerTable);
            }
        }
        snapshot.UserData = inputs.UserData.ToArray();
        return true;
    }

    private static string DiagnoseSnapshotEvaluationFailure(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        IReadOnlyList<bool> activeSources)
    {
        var cleanEvaluator = new RuntimeValueEvaluator(
            plan,
            inputs.WithReader(inputs.ReadCleanMemory));
        var evaluator = new RuntimeValueEvaluator(
            plan,
            inputs,
            plan.CleanFlatSlots,
            cleanEvaluator);

        foreach (var sourceIndex in plan.MaterializationSources)
        {
            if (sourceIndex >= plan.DescriptorSources.Count)
            {
                return $"descriptor source {sourceIndex} is outside the source table";
            }

            if (activeSources.Count != 0 && !activeSources[(int)sourceIndex])
            {
                continue;
            }

            var source = plan.DescriptorSources[(int)sourceIndex];
            for (var dword = 0; dword < source.Dwords.Length; dword++)
            {
                if (!evaluator.Evaluate(source.Dwords[dword], out _))
                {
                    var dynamicBufferNote = string.IsNullOrEmpty(source.DynamicBufferRejectionReason)
                        ? string.Empty
                        : $" Dynamic-buffer recognition rejected it: {source.DynamicBufferRejectionReason}.";
                    return $"descriptor source {sourceIndex} dword {dword} cannot be evaluated:{dynamicBufferNote} " +
                        DescribeEvaluationValue(plan, inputs, evaluator, source.Dwords[dword]);
                }
            }
        }

        foreach (var read in plan.TableReads)
        {
            var clean = read.FlatOffset < plan.CleanFlatSlots.Count &&
                plan.CleanFlatSlots[(int)read.FlatOffset] != 0;
            var selected = clean ? cleanEvaluator : evaluator;
            if (read.FlatOffset >= plan.TableReads.Count || !selected.Evaluate(read.Value, out _))
            {
                return $"resource table read {read.FlatOffset} cannot be evaluated: " +
                    DescribeEvaluationValue(plan, inputs, selected, read.Value);
            }
        }

        return "descriptor snapshot evaluation failed without an isolated source";
    }

    private static string DescribeEvaluationValue(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        RuntimeValueEvaluator evaluator,
        ScalarValue value,
        int depth = 0)
    {
        if (depth >= 6)
        {
            return value.ToString();
        }

        if (value.Kind == ScalarValueKind.ResourceTableWord)
        {
            var slot = (int)value.Payload;
            if ((uint)slot < plan.TableReads.Count)
            {
                var read = plan.TableReads[slot];
                return $"table-slot={slot} flat={read.FlatOffset} value=" +
                    DescribeEvaluationValue(plan, inputs, evaluator, read.Value, depth + 1);
            }

            return $"table-slot={slot} (outside table)";
        }

        if (value.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord &&
            value.MemoryIndex >= 0 && value.MemoryIndex < plan.Memory.Count)
        {
            var memory = plan.Memory[value.MemoryIndex];
            var detail = $"{value.Kind} memory={value.MemoryIndex} pc=0x{memory.Pc:X8} " +
                $"opcode={memory.Opcode} offset={memory.Offset} operands=[{string.Join(", ", value.Operands.Select(operand =>
                    DescribeEvaluationValue(plan, inputs, evaluator, operand, depth + 1)))}]";
            if (value.Operands.Length >= 2 && value.Operands[0].Operands.Length >= 2 &&
                evaluator.EvaluateWide(value.Operands[0].Operands[0], out var low) &&
                evaluator.EvaluateWide(value.Operands[0].Operands[1], out var high) &&
                evaluator.EvaluateWide(value.Operands[1], out var dynamicOffset))
            {
                var baseAddress = ((high << 32) | (uint)low) & AddressMask;
                var relative = (long)(int)memory.Offset + (uint)dynamicOffset;
                if (relative >= 0 && baseAddress <= AddressMask - (ulong)relative)
                {
                    var address = (baseAddress + (ulong)relative) & ~3ul;
                    var regular = inputs.ReadMemory is not null && inputs.ReadMemory(address, out _);
                    var clean = inputs.ReadCleanMemory is not null && inputs.ReadCleanMemory(address, out _);
                    detail += $" base=0x{baseAddress:X16} dynamic=0x{dynamicOffset:X} " +
                        $"address=0x{address:X16} readable={regular} clean={clean}";
                }
            }

            return detail;
        }

        if (value.Kind is ScalarValueKind.AddressHandle or ScalarValueKind.BufferHandle or
            ScalarValueKind.ImageHandle or ScalarValueKind.SamplerHandle)
        {
            return $"{value.Kind}[{string.Join(", ", value.Operands.Select(operand =>
                DescribeEvaluationValue(plan, inputs, evaluator, operand, depth + 1)))}]";
        }

        if (value.Kind is ScalarValueKind.Phi or ScalarValueKind.Select or ScalarValueKind.Operation)
        {
            return $"{value} operands=[{string.Join(", ", value.Operands.Select(operand =>
                DescribeEvaluationValue(plan, inputs, evaluator, operand, depth + 1)))}]";
        }

        return value.ToString();
    }

    private static bool MaterializeDescriptorArray(
        IndirectImageSelector indirect,
        DescriptorWords heap,
        bool r128,
        ResourceRuntimeInputs inputs,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        if (heap.DwordCount != 4 || indirect.HeapIsAddress || indirect.HeapStride == 0)
        {
            return false;
        }

        var size = ScalarBufferSize(heap.Dwords);
        // A zero-sized (or truncated) scalar buffer is a valid null resource.
        // Keep one all-zero candidate so the image is materialized as null; the
        // scalar-buffer reader already returns zero for words outside its size.
        var recordCount = indirect.HeapOffset > size || size - indirect.HeapOffset < 8u * sizeof(uint)
            ? 1ul
            : (size - indirect.HeapOffset - 8u * sizeof(uint)) / indirect.HeapStride + 1;
        if (recordCount == 0 || recordCount > MaxIndirectImageProbes)
        {
            return false;
        }


        var table = new IndirectImageTable
        {
            MaterialDescriptor = heap.Dwords,
            HeapDescriptor = heap.Dwords,
            SelectorStride = indirect.HeapStride,
            SelectorOffset = indirect.HeapOffset,
        };

        for (ulong index = 0; index < recordCount; index++)
        {
            var key = index * indirect.HeapStride;
            if (key > uint.MaxValue)
            {
                return false;
            }

            var candidate = new uint[8];
            for (uint dword = 0; dword < candidate.Length; dword++)
            {
                if (!TryReadIndirectImageWord(indirect, heap.Dwords, [], (uint)key, dword, inputs, out candidate[dword]))
                {
                    return false;
                }
            }

            if (NullImageDescriptor(candidate) || !ValidImageDescriptor(candidate, r128))
            {
                Array.Clear(candidate);
            }

            var words = new DescriptorWords(candidate);
            var found = table.Descriptors.FindIndex(existing => existing.SameAs(words));
            if (found < 0)
            {
                if (table.Descriptors.Count >= ShaderResourceInfo.MaxImages)
                {
                    failure = ResourceMaterializationFailure.ImageCapacityExceeded;
                    return Fail("descriptor-array candidates exceed the dense image resource limit");
                }

                table.Descriptors.Add(words);
                found = table.Descriptors.Count - 1;
            }

            table.Keys.Add((uint)key);
            table.Candidates.Add((uint)found);
        }

        result = table;
        return table.Descriptors.Count != 0;
    }

    private static bool NullImageDescriptor(ReadOnlySpan<uint> descriptor) =>
        descriptor[0] == 0 && (descriptor[1] & 0xFF) == 0;

    private static bool ReservedImageBitsClear(ReadOnlySpan<uint> descriptor)
    {
        ReadOnlySpan<uint> reserved =
        [
            0x00000000u, 0x20000000u, 0xf0003000u, 0x00000000u,
            0xe000e000u, 0xf9000000u, 0x00007b00u, 0x00000000u,
        ];
        for (var dword = 0; dword < reserved.Length; dword++)
        {
            if ((descriptor[dword] & reserved[dword]) != 0)
                return false;
        }
        return true;
    }

    private static bool ValidImageDescriptor(ReadOnlySpan<uint> descriptor, bool r128)
    {
        var type = GuestImageFormat.ImageTypeOf(descriptor);
        var format = GuestImageFormat.FormatOf(descriptor);
        if (type < GuestImageFormat.ImageType1D || format == GuestImageFormat.Invalid || format > GuestImageFormat.MaxFormat)
        {
            return false;
        }

        if (r128 && type is not (GuestImageFormat.ImageType1D or GuestImageFormat.ImageType2D or GuestImageFormat.ImageType2DMsaa))
        {
            return false;
        }

        if (type is GuestImageFormat.ImageTypeCube or
            GuestImageFormat.ImageType1DArray or
            GuestImageFormat.ImageType2DArray or
            GuestImageFormat.ImageType2DMsaaArray)
        {
            var lastSlice = descriptor[4] & 0x1FFF;
            var baseSlice = (descriptor[4] >> 16) & 0x1FFF;
            if (baseSlice > lastSlice)
            {
                return false;
            }

        }

        if (type is GuestImageFormat.ImageType2DMsaa or GuestImageFormat.ImageType2DMsaaArray)
        {
            var baseLevel = (descriptor[3] >> 12) & 0xF;
            var fragments = (descriptor[3] >> 16) & 0xF;
            var maxMip = (descriptor[5] >> 4) & 0xF;
            return baseLevel == 0 && fragments is >= 1 and <= 3 && (r128 || maxMip == fragments);
        }

        return true;
    }

    private static ulong ScalarBufferSize(ReadOnlySpan<uint> descriptor)
    {
        var stride = (descriptor[1] >> 16) & 0x3FFF;
        return stride == 0 ? descriptor[2] : (ulong)stride * descriptor[2];
    }

    private static bool ReadScalarBufferWord(ReadOnlySpan<uint> descriptor, uint dynamicOffset, uint immediateOffset, ResourceRuntimeInputs inputs, out uint word)
    {
        word = 0;
        var byteOffset = (ulong)dynamicOffset + immediateOffset;
        var aligned = byteOffset & ~3ul;
        var size = ScalarBufferSize(descriptor);
        if (aligned > size || size - aligned < sizeof(uint))
        {
            return true;
        }

        var baseAddress = ((descriptor[0] | ((ulong)descriptor[1] << 32)) & AddressMask) & ~3ul;
        if (aligned > AddressMask - baseAddress)
        {
            return false;
        }

        if (inputs.ReadCleanMemory is null && inputs.ReadMemory is null)
        {
            return false;
        }

        return TryReadStableWord(inputs, baseAddress + aligned, out word);
    }

    private static bool ReadAddressWord(
        ReadOnlySpan<uint> descriptor,
        uint dynamicOffset,
        uint immediateOffset,
        ResourceRuntimeInputs inputs,
        out uint word)
    {
        word = 0;
        if (descriptor.Length < 2)
        {
            return false;
        }

        var baseAddress = ((ulong)descriptor[0] | ((ulong)descriptor[1] << 32)) & AddressMask;
        var byteOffset = (ulong)dynamicOffset + immediateOffset;
        var aligned = byteOffset & ~3ul;
        if (aligned > AddressMask - baseAddress)
        {
            return false;
        }

        if (inputs.ReadCleanMemory is null && inputs.ReadMemory is null)
        {
            return false;
        }

        return TryReadStableWord(inputs, baseAddress + aligned, out word);
    }

    private static bool ReadIndirectHeapWord(
        IndirectImageSelector indirect,
        ReadOnlySpan<uint> descriptor,
        uint dynamicOffset,
        uint immediateOffset,
        ResourceRuntimeInputs inputs,
        out uint word)
        => indirect.HeapIsAddress
            ? ReadAddressWord(descriptor, dynamicOffset, immediateOffset, inputs, out word)
            : ReadScalarBufferWord(descriptor, dynamicOffset, immediateOffset, inputs, out word);

    // Enumerates every material key that can pass the table's bounds and reads the
    // heap descriptor each selects; stale or invalid descriptors become null.
    private static bool MaterializeIndirectImage(
        ShaderResourcePlan plan,
        IndirectImageSelector indirect,
        DescriptorWords material,
        DescriptorWords heap,
        ReadOnlySpan<uint> flattenedTable,
        bool r128,
        ResourceRuntimeInputs inputs,
        bool captureSelectorDiagnostic,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        var expectedHeapDwords = indirect.HeapIsAddress ? 2u : 4u;
        if (material.DwordCount != 4 || heap.DwordCount != expectedHeapDwords)
        {
            return false;
        }

        // Scalar-buffer offsets are byte offsets. The descriptor stride only
        // bounds the record footprint; a key table may intentionally pack keys
        // more tightly than the descriptor's declared structured stride.

        var period = 1ul << 32;
        var step = (ulong)BigInteger.GreatestCommonDivisor(indirect.SelectorStride, period);
        var residue = indirect.SelectorOffset % step;
        var size = ScalarBufferSize(material.Dwords);
        var limit = Math.Min(uint.MaxValue, size + 3);
        var probeCount = residue <= limit ? (limit - residue) / step + 1 : 0;
        uint[]? provenOffsets = null;
        var diagnostic = captureSelectorDiagnostic ? new IndirectSelectorDiagnostic() : null;
        if (indirect.SelectorValues is { } selectorValues)
        {
            var evaluated = selectorValues.TryEvaluate(plan, inputs, out var selectors, diagnostic);
            if (evaluated)
                provenOffsets = selectors.Select(selector => unchecked(selector * indirect.SelectorStride + indirect.SelectorOffset)).Distinct().ToArray();
            if (diagnostic is not null)
            {
                diagnostic.SelectionMode = evaluated ? "bounded" : "full_domain_evaluation_failed";
                diagnostic.SelectorIndices = selectors;
                diagnostic.ProvenOffsets = provenOffsets ?? [];
            }
        }
        if (provenOffsets is null && probeCount > MaxIndirectImageProbes)
        {
            return false;
        }

        var keys = new List<uint> { 0 };
        var seen = new HashSet<uint> { 0 };
        var offsets = provenOffsets is not null ? provenOffsets.Select(offset => (ulong)offset) :
            Enumerable.Range(0, (int)probeCount).Select(index => residue + (ulong)index * step);
        foreach (var offset in offsets)
        {
            if (!ReadScalarBufferWord(material.Dwords, (uint)offset, 0, inputs, out var key))
            {
                return false;
            }

            if (seen.Add(key))
            {
                keys.Add(key);
                diagnostic?.KeyProbes.Add(new((uint)offset, key));
            }

        }

        var probed = new List<uint[]>(keys.Count);
        foreach (var key in keys)
        {
            var candidate = new uint[8];
            for (uint dword = 0; dword < 8; dword++)
            {
                if (!TryReadIndirectImageWord(indirect, heap.Dwords, flattenedTable, key, dword, inputs, out candidate[dword]))
                {
                    return false;
                }
            }

            if (NullImageDescriptor(candidate) || !ValidImageDescriptor(candidate, r128) ||
                !ReservedImageBitsClear(candidate))
            {
                Array.Clear(candidate);
            }
            probed.Add(candidate);
        }

        if (!FinishIndirectImage(
                probed,
                keys,
                out var table,
                out failure))
        {
            return false;
        }

        table.MaterialDescriptor = material.Dwords;
        table.HeapDescriptor = heap.Dwords;
        table.SelectorStride = indirect.SelectorStride;
        table.SelectorOffset = indirect.SelectorOffset;
        table.SelectorDiagnostic = diagnostic;
        result = table;
        return true;
    }

    private static bool MaterializeDenseIndirectImage(
        IndirectImageSelector indirect,
        DescriptorWords heap,
        ImageResource image,
        bool r128,
        ResourceRuntimeInputs inputs,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        if (heap.DwordCount != 2 || indirect.KeyBound == 0 || indirect.KeyBound > MaxIndirectImageProbes ||
            (inputs.ReadCleanMemory is null && inputs.ReadMemory is null))
            return false;

        var baseAddress = (((ulong)heap.Dwords[1] << 32) | heap.Dwords[0]) & AddressMask;
        var probed = new List<uint[]>((int)indirect.KeyBound);
        for (uint key = 0; key < indirect.KeyBound; key++)
        {
            var candidate = new uint[8];
            var entry = (ulong)indirect.TableOffset + ((ulong)key << 5);
            for (uint dword = 0; dword < 8; dword++)
            {
                var relative = entry + dword * sizeof(uint);
                if (relative > AddressMask || baseAddress > AddressMask - relative ||
                    !TryReadStableWord(inputs, baseAddress + relative, out candidate[dword]))
                return false;
            }

            if (NullImageDescriptor(candidate) || !ValidImageDescriptor(candidate, r128) ||
                !ReservedImageBitsClear(candidate))
                Array.Clear(candidate);
            probed.Add(candidate);
        }

        return FinishIndirectImage(
            probed,
            Enumerable.Range(0, (int)indirect.KeyBound)
                .Select(index => unchecked(indirect.DynamicOffsetBase + ((uint)index << 5))),
            out result,
            out failure);
    }

    private static bool FinishIndirectImage(
        IReadOnlyList<uint[]> probed,
        IEnumerable<uint> keys,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        result.Keys.AddRange(keys);
        foreach (var candidate in probed)
        {
            var words = new DescriptorWords(candidate);
            var found = result.Descriptors.FindIndex(existing => existing.SameAs(words));
            if (found < 0)
            {
                if (result.Descriptors.Count >= ShaderResourceInfo.MaxImages)
                {
                    failure = ResourceMaterializationFailure.ImageCapacityExceeded;
                    return false;
                }
                found = result.Descriptors.Count;
                result.Descriptors.Add(words);
            }
            result.Candidates.Add((uint)found);
        }
        return true;
    }

    private static bool TryReadIndirectImageWord(
        IndirectImageSelector indirect,
        ReadOnlySpan<uint> heapDescriptor,
        ReadOnlySpan<uint> flattenedTable,
        uint key,
        uint dword,
        ResourceRuntimeInputs inputs,
        out uint word)
    {
        word = 0;
        if (indirect.DescriptorHeapOffsets is { Count: >= 8 } offsets && dword < offsets.Count)
        {
            var mask = indirect.DescriptorHeapMasks is { Count: >= 8 }
                ? indirect.DescriptorHeapMasks[(int)dword]
                : uint.MaxValue;
            var fieldOffset = offsets[(int)dword];
            var hasHeapField = fieldOffset != uint.MaxValue;
            var hasTableSlot = indirect.DescriptorTableSlots is { Count: >= 8 } tableSlots &&
                tableSlots[(int)dword] != uint.MaxValue;
            var hasWord = false;
            var expectedWord = 0u;
            if (hasTableSlot)
            {
                var slot = indirect.DescriptorTableSlots![(int)dword];
                if (slot >= flattenedTable.Length)
                {
                    return false;
                }

                expectedWord = flattenedTable[(int)slot];
                expectedWord &= mask;
                hasWord = true;
            }

            if (hasHeapField)
            {
                var recordOffset = (ulong)key * indirect.HeapStride;
                if (recordOffset > uint.MaxValue - fieldOffset)
                {
                    return false;
                }

                if (!ReadIndirectHeapWord(indirect, heapDescriptor, (uint)(recordOffset + fieldOffset), 0, inputs, out var heapWord))
                {
                    return false;
                }

                var maskedHeapWord = heapWord & mask;
                if (hasWord && expectedWord != maskedHeapWord)
                {
                    return false;
                }

                expectedWord = maskedHeapWord;
                hasWord = true;
            }

            if (indirect.DescriptorHeapAlternates is { Count: >= 8 } alternates)
            {
                foreach (var alternateOffset in alternates[(int)dword])
                {
                    var recordOffset = (ulong)key * indirect.HeapStride;
                    if (recordOffset > uint.MaxValue - alternateOffset ||
                        !ReadIndirectHeapWord(indirect, heapDescriptor, (uint)(recordOffset + alternateOffset), 0, inputs, out var alternateWord))
                    {
                        return false;
                    }

                    var maskedAlternateWord = alternateWord & mask;
                    if (hasWord && expectedWord != maskedAlternateWord)
                    {
                        return false;
                    }

                    expectedWord = maskedAlternateWord;
                    hasWord = true;
                }
            }

            if (hasWord)
            {
                word = expectedWord;
                return true;
            }

            if (indirect.DescriptorStaticValues is not { Count: >= 8 } staticValues)
            {
                return false;
            }

            word = staticValues[(int)dword];
            return true;
        }

        var heapOffset = indirect.DescriptorArray
            ? (ulong)key + indirect.HeapOffset
            : (ulong)key * indirect.HeapStride + indirect.HeapOffset;
        var immediateOffset = (ulong)dword * sizeof(uint);
        if (heapOffset > uint.MaxValue || immediateOffset > uint.MaxValue - heapOffset)
        {
            return false;
        }

        return ReadIndirectHeapWord(indirect, heapDescriptor, (uint)heapOffset, (uint)immediateOffset, inputs, out word);
    }

    private static bool MaterializeIndirectSampler(
        ShaderResourcePlan plan,
        IndirectImageSelector indirect,
        DescriptorWords material,
        DescriptorWords heap,
        ResourceRuntimeInputs inputs,
        out IndirectSamplerTable result)
    {
        result = new IndirectSamplerTable();
        if (material.DwordCount != 4 || heap.DwordCount != 4)
        {
            return false;
        }

        var period = 1ul << 32;
        var step = (ulong)BigInteger.GreatestCommonDivisor(indirect.SelectorStride, period);
        var residue = indirect.SelectorOffset % step;
        var size = ScalarBufferSize(material.Dwords);
        var limit = Math.Min(uint.MaxValue, size + 3);
        var probeCount = residue <= limit ? (limit - residue) / step + 1 : 0;
        if (probeCount > MaxIndirectImageProbes)
        {
            return false;
        }

        var keys = new List<uint> { 0 };
        var seen = new HashSet<uint> { 0 };
        for (ulong index = 0, offset = residue; index < probeCount; index++, offset += step)
        {
            if (!ReadScalarBufferWord(material.Dwords, (uint)offset, 0, inputs, out var key))
            {
                return false;
            }

            if (seen.Add(key))
            {
                keys.Add(key);
            }
        }

        var table = new IndirectSamplerTable
        {
            Keys = keys,
            MaterialDescriptor = material.Dwords,
            HeapDescriptor = heap.Dwords,
            SelectorStride = indirect.SelectorStride,
            SelectorOffset = indirect.SelectorOffset,
        };
        foreach (var key in keys)
        {
            var candidate = new uint[4];
            var heapOffset = (ulong)key * indirect.HeapStride + indirect.HeapOffset;
            for (uint dword = 0; dword < 4; dword++)
            {
                if (heapOffset > uint.MaxValue ||
                    !ReadScalarBufferWord(heap.Dwords, (uint)heapOffset, dword * sizeof(uint), inputs, out candidate[dword]))
                {
                    return false;
                }
            }

            var words = new DescriptorWords(candidate);
            var found = table.Descriptors.FindIndex(existing => existing.SameAs(words));
            if (found < 0)
            {
                if (table.Descriptors.Count >= ShaderResourceInfo.MaxSamplers)
                {
                    return false;
                }

                table.Descriptors.Add(words);
                table.Candidates.Add((uint)table.Descriptors.Count - 1);
            }
            else
            {
                table.Candidates.Add((uint)found);
            }
        }

        result = table;
        return true;
    }

    // ---- specialization ----

    private static ImageDimension DescriptorDimension(ReadOnlySpan<uint> descriptor, ImageDimension requested)
    {
        var isArray = requested is ImageDimension.Dim1DArray or ImageDimension.Dim2DArray or ImageDimension.Dim2DMsaaArray;
        return GuestImageFormat.ImageTypeOf(descriptor) switch
        {
            GuestImageFormat.ImageType1D => ImageDimension.Dim1D,
            GuestImageFormat.ImageType1DArray => isArray ? ImageDimension.Dim1DArray : ImageDimension.Dim1D,
            GuestImageFormat.ImageType3D => ImageDimension.Dim3D,
            GuestImageFormat.ImageTypeCube => ImageDimension.Dim2DArray,
            GuestImageFormat.ImageType2DArray => isArray ? ImageDimension.Dim2DArray : ImageDimension.Dim2D,
            GuestImageFormat.ImageType2DMsaaArray => isArray ? ImageDimension.Dim2DMsaaArray : ImageDimension.Dim2DMsaa,
            GuestImageFormat.ImageType2D => ImageDimension.Dim2D,
            GuestImageFormat.ImageType2DMsaa => ImageDimension.Dim2DMsaa,
            _ => ImageDimension.Unknown,
        };
    }

    private static uint ImageConversionFormat(uint format) =>
        GuestImageFormat.Remap(format) != format ? format : GuestImageFormat.Invalid;

    private static bool RequiresPointSampler(ImageNumericClass numericClass, uint conversionFormat) =>
        numericClass == ImageNumericClass.Sint || conversionFormat != GuestImageFormat.Invalid;

    private static uint StorageMipCount(ImageResource image, ReadOnlySpan<uint> descriptor)
    {
        if (image.MipMode != ImageMipMode.DynamicStorage || NullImageDescriptor(descriptor))
        {
            return 1;
        }

        var baseLevel = (descriptor[3] >> 12) & 0xF;
        var last = (descriptor[3] >> 16) & 0xF;
        return baseLevel <= last ? last - baseLevel + 1 : 0;
    }

    private static bool Fail(string message)
    {
        SpecializationFailed(message);
        return false;
    }

    private static bool BuildSpecialization(
        ShaderResourcePlan plan,
        MaterializedSnapshot snapshot,
        out MaterializedSnapshot specializedSnapshot,
        out ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure,
        Action<IndirectImageFailure>? captureIndirectImageFailure)
    {
        using var specializationProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.Specialization);
        failure = ResourceMaterializationFailure.Other;
        specializedSnapshot = snapshot;
        specialization = new ResourceSpecialization();
        var info = plan.Info;
        var imageCount = info.Images.Count;
        var samplerCount = info.Samplers.Count;
        var mappingWordCount = 0;
        foreach (var table in snapshot.IndirectImages)
        {
            if (table.Resource >= info.Images.Count || table.Descriptors.Count < 2)
            {
                return Fail("indirect image table has an invalid root or candidate count");
            }

            if (imageCount + table.Descriptors.Count - 1 > ShaderResourceInfo.MaxImages)
            {
                failure = ResourceMaterializationFailure.ImageCapacityExceeded;
                return Fail("indirect image candidates exceed the dense image resource limit");
            }

            imageCount += table.Descriptors.Count - 1;
            mappingWordCount = checked(mappingWordCount + 1 + table.Keys.Count * 2);
        }

        foreach (var table in snapshot.IndirectSamplers)
        {
            if (table.Resource >= info.Samplers.Count || table.Descriptors.Count < 2)
            {
                return Fail("indirect sampler table has an invalid root or candidate count");
            }

            if (samplerCount + table.Descriptors.Count - 1 > ShaderResourceInfo.MaxSamplers)
            {
                return Fail("indirect sampler candidates exceed the dense sampler resource limit");
            }

            samplerCount += table.Descriptors.Count - 1;
            mappingWordCount = checked(mappingWordCount + 1 + table.Keys.Count * 2);
        }

        // Each draw owns these arrays. Only indirect candidates require a larger table.
        Array.Resize(ref snapshot.Images, imageCount);
        Array.Resize(ref snapshot.Samplers, samplerCount);
        var mappingCursor = snapshot.FlattenedTable.Length;
        Array.Resize(ref snapshot.FlattenedTable, checked(mappingCursor + mappingWordCount));
        var imageCursor = info.Images.Count;
        var images = new List<ImageSpecialization>(imageCount);
        foreach (var image in info.Images)
        {
            images.Add(new ImageSpecialization(
                image.NumericClass, image.Dimension, image.MipCount, image.ConversionFormat, image.ShaderSwizzle,
                image.IndirectRoot, image.IndirectMappingOffset, image.IndirectSearchIterations, image.Cube));
        }

        foreach (var table in snapshot.IndirectImages)
        {
            var rootImage = images[(int)table.Resource];
            for (var candidate = 1; candidate < table.Descriptors.Count; candidate++)
            {
                images.Add(rootImage with { IndirectRoot = table.Resource });
                snapshot.Images[imageCursor++] = table.Descriptors[candidate].Dwords;
            }

            var mappingOffset = (uint)mappingCursor;
            images[(int)table.Resource] = rootImage with
            {
                IndirectRoot = table.Resource,
                IndirectMappingOffset = mappingOffset,
                IndirectSearchIterations = (uint)BitOperations.Log2((uint)table.Keys.Count) + 1,
            };
            mappingCursor += 1 + table.Keys.Count * 2;
            var order = Enumerable.Range(0, table.Keys.Count).OrderBy(index => table.Keys[index]).ToArray();
            snapshot.FlattenedTable[(int)mappingOffset] = (uint)table.Keys.Count;
            for (var entry = 0; entry < order.Length; entry++)
            {
                var source = order[entry];
                var offset = (int)mappingOffset + 1 + entry * 2;
                snapshot.FlattenedTable[offset] = table.Keys[source];
                snapshot.FlattenedTable[offset + 1] = table.Candidates[source];
            }

            snapshot.Images[(int)table.Resource] = table.Descriptors[0].Dwords;
        }

        var samplers = new List<SamplerSpecialization>(samplerCount);
        foreach (var sampler in info.Samplers)
        {
            samplers.Add(new SamplerSpecialization(
                sampler.IndirectRoot, sampler.IndirectMappingOffset, sampler.IndirectSearchIterations));
        }

        var samplerCursor = info.Samplers.Count;
        foreach (var table in snapshot.IndirectSamplers)
        {
            var rootSampler = samplers[(int)table.Resource];
            for (var candidate = 1; candidate < table.Descriptors.Count; candidate++)
            {
                samplers.Add(rootSampler with { IndirectRoot = table.Resource });
                snapshot.Samplers[samplerCursor++] = table.Descriptors[candidate].Dwords;
            }

            var mappingOffset = (uint)mappingCursor;
            samplers[(int)table.Resource] = rootSampler with
            {
                IndirectRoot = table.Resource,
                IndirectMappingOffset = mappingOffset,
                IndirectSearchIterations = (uint)BitOperations.Log2((uint)table.Keys.Count) + 1,
            };
            mappingCursor += 1 + table.Keys.Count * 2;
            var order = Enumerable.Range(0, table.Keys.Count).OrderBy(index => table.Keys[index]).ToArray();
            snapshot.FlattenedTable[(int)mappingOffset] = (uint)table.Keys.Count;
            for (var entry = 0; entry < order.Length; entry++)
            {
                var source = order[entry];
                var offset = (int)mappingOffset + 1 + entry * 2;
                snapshot.FlattenedTable[offset] = table.Keys[source];
                snapshot.FlattenedTable[offset + 1] = table.Candidates[source];
            }

            snapshot.Samplers[(int)table.Resource] = table.Descriptors[0].Dwords;
        }

        var buffers = new List<BufferSpecialization>(info.Buffers.Count);
        for (var index = 0; index < info.Buffers.Count; index++)
        {
            var descriptor = snapshot.Buffers[index];
            if (descriptor.Length != 4)
            {
                return Fail($"buffer descriptor {index} has invalid width");
            }

            var words = descriptor;
            if ((words[3] >> 30) != 0)
            {
                Array.Clear(words);
            }

            var stride = (words[1] >> 16) & 0x3FFF;
            var swizzleEnabled = (words[1] >> 31) != 0;
            var indexStride = (words[3] >> 21) & 0x3;
            var addThreadId = (words[3] >> 23) & 0x1;
            var packedStride = stride | ((swizzleEnabled ? 1u : 0u) << 14) | (indexStride << 16) | (addThreadId << 20);
            var swizzle = stride != 0 && ((packedStride >> 14) & 1) != 0;
            if (stride == 0)
            {
                packedStride &= ~((1u << 14) | (3u << 16));
            }
            else if (!swizzle)
            {
                packedStride &= ~(3u << 16);
            }

            var formatted = info.Buffers[index].Formatted;
            buffers.Add(new BufferSpecialization(
                packedStride,
                formatted ? (words[3] >> 12) & 0x7F : DescriptorConstants.InvalidFormat,
                formatted ? words[3] & 0xFFF : DescriptorConstants.IdentityDestinationSelect));
        }

        for (var index = 0; index < images.Count; index++)
        {
            var descriptor = snapshot.Images[index];
            var image = images[index];
            var baseIndex = index < info.Images.Count ? (uint)index : image.IndirectRoot;
            if (baseIndex >= info.Images.Count)
            {
                return Fail($"image resource {index} has an invalid root");
            }

            var baseImage = info.Images[(int)baseIndex];
            if (baseImage.ResourceClass == ImageResourceClass.None || (baseImage.Atomic && baseImage.ResourceClass != ImageResourceClass.Storage))
            {
                return Fail($"image resource {index} has an invalid class");
            }

            var mipCount = StorageMipCount(baseImage, descriptor);
            if (mipCount == 0)
            {
                return Fail($"storage image descriptor {index} has an invalid mip range");
            }

            image = image with { MipCount = mipCount };
            if (NullImageDescriptor(descriptor))
            {
                images[index] = image with
                {
                    NumericClass = baseImage.Atomic ? ImageNumericClass.Uint : ImageNumericClass.Float,
                    Dimension = ImageDimension.Dim2D,
                    Cube = false,
                };
                continue;
            }

            var dimension = DescriptorDimension(descriptor, baseImage.Dimension);
            if (dimension == ImageDimension.Unknown)
            {
                return Fail(
                    $"image descriptor {index} has unsupported type {GuestImageFormat.ImageTypeOf(descriptor)}: " +
                    string.Join(",", descriptor.Select(word => $"{word:x8}")));
            }

            var format = GuestImageFormat.FormatOf(descriptor);
            if (baseImage.Atomic && format is not (GuestImageFormat.Format32Uint or GuestImageFormat.Format32Sint or GuestImageFormat.Format32Float))
            {
                return Fail($"atomic image descriptor {index} uses unsupported format {format}");
            }

            var storage = baseImage.ResourceClass == ImageResourceClass.Storage;
            var conversionFormat = ImageConversionFormat(format);
            var shaderSwizzle = storage || conversionFormat != GuestImageFormat.Invalid ? descriptor[3] & 0xFFF : image.ShaderSwizzle;
            var rawSintStorage = storage && format == GuestImageFormat.Format32Sint && baseImage.Written && !baseImage.Read && !baseImage.Atomic;
            var numericClass = GuestImageFormat.SampledNumericClass(format);
            if (storage)
            {
                if ((!rawSintStorage && numericClass == ImageNumericClass.Sint) || numericClass == ImageNumericClass.Unsupported)
                {
                    return Fail($"storage image descriptor {index} uses unsupported format {format}");
                }

                if (rawSintStorage)
                {
                    numericClass = ImageNumericClass.Uint;
                }
            }
            else if (numericClass == ImageNumericClass.Unsupported || (baseImage.DepthCompare && numericClass != ImageNumericClass.Float))
            {
                return Fail($"sampled image descriptor {index} uses unsupported format {format}");
            }

            images[index] = image with
            {
                Dimension = dimension,
                Cube = GuestImageFormat.ImageTypeOf(descriptor) == GuestImageFormat.ImageTypeCube,
                ConversionFormat = conversionFormat,
                ShaderSwizzle = shaderSwizzle,
                NumericClass = baseImage.Atomic ? ImageNumericClass.Uint : numericClass,
            };
        }

        for (var rootIndex = 0; rootIndex < images.Count; rootIndex++)
        {
            var root = images[rootIndex];
            if (root.IndirectRoot != rootIndex)
            {
                continue;
            }

            var keyCount = root.IndirectMappingOffset < snapshot.FlattenedTable.Length ? snapshot.FlattenedTable[(int)root.IndirectMappingOffset] : 0;
            if (root.IndirectSearchIterations == 0 || keyCount < 2 ||
                (ulong)root.IndirectMappingOffset + 1 + (ulong)keyCount * 2 > (ulong)snapshot.FlattenedTable.Length)
            {
                return Fail("indirect image specialization has an invalid key mapping");
            }

            var exemplar = DescriptorConstants.NoIndex;
            var resourceCount = 0;
            for (var resource = 0; resource < images.Count; resource++)
            {
                if (images[resource].IndirectRoot != rootIndex)
                {
                    continue;
                }

                resourceCount++;
                if (exemplar == DescriptorConstants.NoIndex && !NullImageDescriptor(snapshot.Images[resource]))
                {
                    exemplar = (uint)resource;
                }
            }

            if (resourceCount < 2 || exemplar == DescriptorConstants.NoIndex)
            {
                return Fail("indirect image specialization has no typed candidate");
            }

            var imageClass = images[(int)exemplar];
            var separateSampledDimensions = info.Images[rootIndex].ResourceClass == ImageResourceClass.Sampled;
            for (var candidate = 0; candidate < images.Count; candidate++)
            {
                var image = images[candidate];
                if (image.IndirectRoot != rootIndex)
                {
                    continue;
                }

                if (NullImageDescriptor(snapshot.Images[candidate]))
                {
                    image = image with
                    {
                        NumericClass = imageClass.NumericClass,
                        Dimension = imageClass.Dimension,
                        MipCount = imageClass.MipCount,
                        ConversionFormat = imageClass.ConversionFormat,
                        ShaderSwizzle = imageClass.ShaderSwizzle,
                        Cube = imageClass.Cube,
                    };
                    images[candidate] = image;
                }

                // Each sampled candidate has its own binding and coordinate width.
                var compatibleDimensions = image.Dimension == imageClass.Dimension ||
                    separateSampledDimensions && !image.Cube && !imageClass.Cube &&
                    image.Dimension is ImageDimension.Dim2D or ImageDimension.Dim2DArray &&
                    imageClass.Dimension is ImageDimension.Dim2D or ImageDimension.Dim2DArray;
                if (image.NumericClass != imageClass.NumericClass || !compatibleDimensions ||
                    image.MipCount != imageClass.MipCount || image.ConversionFormat != imageClass.ConversionFormat ||
                    image.ShaderSwizzle != imageClass.ShaderSwizzle || image.Cube != imageClass.Cube)
                {
                    failure = ResourceMaterializationFailure.IncompatibleImageCandidates;
                    if (captureIndirectImageFailure is not null)
                    {
                        var table = snapshot.IndirectImages.Single(table => table.Resource == rootIndex);
                        captureIndirectImageFailure(new IndirectImageFailure(
                            info.Images[rootIndex].FirstUsePc, (uint)rootIndex, exemplar, (uint)candidate,
                            table.SelectorStride, table.SelectorOffset,
                            [.. table.MaterialDescriptor], [.. table.HeapDescriptor],
                            [.. table.Keys], [.. table.Candidates],
                            table.Descriptors.Select(words => words.Dwords.ToArray()).ToArray(),
                            snapshot.Images.Select(words => words.ToArray()).ToArray(),
                            [.. images], [.. snapshot.UserData])
                        {
                            SelectorDiagnostic = table.SelectorDiagnostic,
                            ScalarReads = plan.TableReads.Select(read =>
                            {
                                var access = plan.Memory[read.Value.MemoryIndex];
                                return new MaterializedScalarRead(access.Pc, access.ComponentIndex, read.FlatOffset,
                                    snapshot.FlattenedTable[(int)read.FlatOffset]);
                            }).ToArray(),
                        });
                    }
                    return Fail($"indirect image table at pc 0x{info.Images[rootIndex].FirstUsePc:x8} has incompatible candidates: exemplar={exemplar} candidate={candidate}");
                }
            }
        }

        if (!BuildSamplerPlan(info, images, out var samplerPlan))
        {
            return Fail("specialized sampler layout exceeds its resource limit");
        }

        Array.Resize(ref snapshot.Samplers, checked((int)samplerPlan.SamplerCount));
        for (var index = 0; index < info.Samplers.Count; index++)
        {
            var target = samplerPlan.PointSampler[index];
            if (target != DescriptorConstants.NoIndex && target >= info.Samplers.Count)
            {
                snapshot.Samplers[target] = snapshot.Samplers[index];
            }
        }

        specialization = new ResourceSpecialization { Buffers = buffers, Images = images, Samplers = samplers };
        specializedSnapshot = snapshot;
        return true;
    }

    private sealed class SamplerPlan
    {
        public uint[] PointSampler = new uint[ShaderResourceInfo.MaxSamplers];
        public uint SamplerCount;
    }

    // A sampler that some pair uses with a point-only image needs a point-filtering
    // copy; when every pair does, the sampler itself switches.
    private static bool BuildSamplerPlan(ShaderResourceInfo info, IReadOnlyList<ImageSpecialization> images, out SamplerPlan plan)
    {
        plan = new SamplerPlan();
        if (info.Samplers.Count > plan.PointSampler.Length)
        {
            return false;
        }

        Array.Fill(plan.PointSampler, DescriptorConstants.NoIndex);
        plan.SamplerCount = (uint)info.Samplers.Count;
        var usage = new byte[ShaderResourceInfo.MaxSamplers];
        foreach (var pair in info.SampledPairs)
        {
            if (pair.Image >= images.Count || pair.Sampler >= info.Samplers.Count)
            {
                return false;
            }

            var image = images[(int)pair.Image];
            usage[pair.Sampler] |= RequiresPointSampler(image.NumericClass, image.ConversionFormat) ? (byte)2 : (byte)1;
        }

        for (var index = 0; index < info.Samplers.Count; index++)
        {
            if ((usage[index] & 2) == 0)
            {
                continue;
            }

            if ((usage[index] & 1) == 0)
            {
                plan.PointSampler[index] = (uint)index;
            }
            else
            {
                if (plan.SamplerCount >= ShaderResourceInfo.MaxSamplers)
                {
                    return false;
                }

                plan.PointSampler[index] = plan.SamplerCount++;
            }
        }

        return true;
    }

    // Applies a specialization to the plan's tables: buffer strides and formats, image
    // classes and indirect candidates, point samplers and the sampler each access uses.
    public static SpecializedResourceInfo ApplyTo(ShaderResourcePlan plan, ResourceSpecialization specialization)
    {
        var source = plan.Info;
        if (source.Buffers.Count != specialization.Buffers.Count || source.Images.Count > specialization.Images.Count ||
            (specialization.Samplers.Count != 0 && source.Samplers.Count > specialization.Samplers.Count))
        {
            throw new ResourcePlanException(
                $"shader resource specialization does not match the plan: hash=0x{plan.Hash:X16} stage={plan.Stage} " +
                $"buffers={source.Buffers.Count}/{specialization.Buffers.Count} images={source.Images.Count}/{specialization.Images.Count}");
        }

        var info = source.Clone();
        for (var index = 0; index < info.Buffers.Count; index++)
        {
            info.Buffers[index].PackedStride = specialization.Buffers[index].PackedStride;
            info.Buffers[index].DescriptorFormat = specialization.Buffers[index].DescriptorFormat;
            info.Buffers[index].DescriptorSwizzle = specialization.Buffers[index].DescriptorSwizzle;
        }

        for (var index = 0; index < specialization.Images.Count; index++)
        {
            var specialized = specialization.Images[index];
            if (index >= info.Images.Count)
            {
                if (specialized.IndirectRoot >= source.Images.Count)
                {
                    throw new ResourcePlanException($"shader resource specialization names an invalid indirect root: hash=0x{plan.Hash:X16} image={index}");
                }

                info.Images.Add(source.Images[(int)specialized.IndirectRoot].Clone());
            }

            var image = info.Images[index];
            image.NumericClass = specialized.NumericClass;
            image.Dimension = specialized.Dimension;
            image.MipCount = specialized.MipCount;
            image.ConversionFormat = specialized.ConversionFormat;
            image.ShaderSwizzle = specialized.ShaderSwizzle;
            image.IndirectRoot = specialized.IndirectRoot;
            image.IndirectMappingOffset = specialized.IndirectMappingOffset;
            image.IndirectSearchIterations = specialized.IndirectSearchIterations;
            image.Cube = specialized.Cube;
            image.IndirectResources = [];
        }

        for (var index = 0; index < info.Images.Count; index++)
        {
            var root = info.Images[index].IndirectRoot;
            if (root != DescriptorConstants.NoIndex)
            {
                info.Images[(int)root].IndirectResources.Add((uint)index);
            }
        }

        if (specialization.Samplers.Count != 0)
        {
            for (var index = 0; index < specialization.Samplers.Count; index++)
            {
                var specialized = specialization.Samplers[index];
                if (index >= info.Samplers.Count)
                {
                    if (specialized.IndirectRoot >= source.Samplers.Count)
                    {
                        throw new ResourcePlanException($"shader resource specialization names an invalid indirect root: hash=0x{plan.Hash:X16} sampler={index}");
                    }

                    info.Samplers.Add(source.Samplers[(int)specialized.IndirectRoot].Clone());
                }

                var sampler = info.Samplers[index];
                sampler.IndirectRoot = specialized.IndirectRoot;
                sampler.IndirectMappingOffset = specialized.IndirectMappingOffset;
                sampler.IndirectSearchIterations = specialized.IndirectSearchIterations;
                sampler.IndirectResources = [];
            }

            for (var index = 0; index < info.Samplers.Count; index++)
            {
                var root = info.Samplers[index].IndirectRoot;
                if (root != DescriptorConstants.NoIndex)
                {
                    info.Samplers[(int)root].IndirectResources.Add((uint)index);
                }
            }
        }

        if (!BuildSamplerPlan(source, specialization.Images, out var samplerPlan))
        {
            throw new ResourcePlanException($"shader resource specialization exceeds the sampler limit: hash=0x{plan.Hash:X16}");
        }

        for (var index = 0; index < source.Samplers.Count; index++)
        {
            var target = samplerPlan.PointSampler[index];
            if (target == DescriptorConstants.NoIndex)
            {
                continue;
            }

            if (target == index)
            {
                info.Samplers[index].ForcePointFiltering = true;
            }
            else
            {
                var sampler = source.Samplers[index].Clone();
                sampler.ForcePointFiltering = true;
                info.Samplers.Add(sampler);
            }
        }

        foreach (var pair in info.SampledPairs)
        {
            var image = info.Images[(int)pair.Image];
            if (RequiresPointSampler(image.NumericClass, image.ConversionFormat))
            {
                pair.Sampler = samplerPlan.PointSampler[pair.Sampler];
            }
        }

        // A guest sampler can be shared by ordinary sampling and depth-reference
        // sampling. Vulkan bakes compareEnable into VkSampler, so those uses cannot
        // share one host sampler. Split only the mixed cases; compare-only samplers
        // can use their existing slot.
        var compareUsage = new byte[ShaderResourceInfo.MaxSamplers];
        foreach (var pair in info.SampledPairs)
        {
            var image = info.Images[(int)pair.Image];
            compareUsage[pair.Sampler] |= image.DepthCompare ? (byte)2 : (byte)1;
        }

        var compareSampler = new uint[ShaderResourceInfo.MaxSamplers];
        Array.Fill(compareSampler, DescriptorConstants.NoIndex);
        for (var index = 0; index < info.Samplers.Count; index++)
        {
            if ((compareUsage[index] & 2) == 0)
            {
                continue;
            }

            if ((compareUsage[index] & 1) == 0)
            {
                info.Samplers[index].DepthCompare = true;
                compareSampler[index] = (uint)index;
                continue;
            }

            if (info.Samplers.Count >= ShaderResourceInfo.MaxSamplers)
            {
                throw new ResourcePlanException($"shader resource specialization exceeds the sampler limit: hash=0x{plan.Hash:X16}");
            }

            var sampler = info.Samplers[index].Clone();
            sampler.DepthCompare = true;
            compareSampler[index] = (uint)info.Samplers.Count;
            info.Samplers.Add(sampler);
        }

        foreach (var pair in info.SampledPairs)
        {
            if (info.Images[(int)pair.Image].DepthCompare)
            {
                pair.Sampler = compareSampler[pair.Sampler];
            }
        }

        var samplerByMemory = new Dictionary<int, uint>();
        for (var index = 0; index < plan.Memory.Count; index++)
        {
            var memory = plan.Memory[index];
            if (memory.Kind != MemoryResourceKind.Image || memory.Resource >= info.Images.Count)
            {
                continue;
            }

            var image = info.Images[(int)memory.Resource];
            if (!memory.NeedsSampler || memory.Sampler >= source.Samplers.Count)
            {
                continue;
            }

            var sampler = memory.Sampler;
            if (RequiresPointSampler(image.NumericClass, image.ConversionFormat))
            {
                sampler = samplerPlan.PointSampler[sampler];
            }

            if (image.DepthCompare)
            {
                sampler = compareSampler[sampler];
            }

            if (sampler != memory.Sampler)
            {
                samplerByMemory[index] = sampler;
            }
        }

        return new SpecializedResourceInfo { Info = info, SamplerByMemoryIndex = samplerByMemory };
    }
}
