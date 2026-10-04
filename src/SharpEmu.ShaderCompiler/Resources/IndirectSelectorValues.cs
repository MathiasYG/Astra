// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Ir;

namespace SharpEmu.ShaderCompiler.Resources;

// A finite overestimate of a selector. Unsupported paths retain the full-domain scan.
public sealed class IndirectSelectorValues
{
    private const int MaximumValues = 4096;
    private const int MaximumCombinations = 65536;
    internal static bool WritesMask(Gen5ShaderInstruction instruction, Gen5Operand mask) => Builder.WritesSavedMask(instruction, mask);
    private sealed record Expression(uint[]? Values = null, ScalarValue? RuntimeValue = null,
        ScalarOperation Operation = ScalarOperation.None, Expression[]? Inputs = null);
    private readonly Expression _root;
    private readonly WaveMaskSelectorBounds? _waveBounds;

    // Enumerate the actual dispatch domain, never a guessed workgroup ID. The
    // original scalar loads and their run-time image selector remain in the shader.
    internal sealed record WorkgroupDescriptor(ScalarValue Handle, ScalarValue Input, ScalarValue? Key)
    {
        internal static bool TryCreate(ShaderResourcePlan plan, ScalarValue handle, ScalarValue? key,
            out WorkgroupDescriptor result)
        {
            result = null!;
            if (plan.Stage != ShaderStage.Compute ||
                handle.Kind is not (ScalarValueKind.ImageHandle or ScalarValueKind.SamplerHandle) ||
                handle.Operands.Length != (handle.Kind == ScalarValueKind.ImageHandle ? 8 : 4)) return false;
            ScalarValue? input = null;
            var pending = new Stack<ScalarValue>(handle.Operands);
            if (key is not null) pending.Push(key);
            var visited = new HashSet<ScalarValue>();
            while (pending.TryPop(out var value))
            {
                if (!visited.Add(value)) continue;
                if (value.Kind == ScalarValueKind.WorkgroupId)
                {
                    if (value.Payload > 2 || input is not null && !ReferenceEquals(input, value)) return false;
                    input = value;
                }
                // No lane-dependent proof or branch-dependent descriptor is implied.
                else if (value.Kind == ScalarValueKind.FirstLane) return false;
                foreach (var operand in value.Operands) pending.Push(operand);
            }
            if (input is null) return false;
            var replacements = new Dictionary<ScalarValue, ScalarValue> { [input] = plan.Graph.Constant(0u) };
            var memo = new Dictionary<ScalarValue, ScalarValue>();
            foreach (var word in handle.Operands)
                if (!plan.ValidateRuntimeValue(plan.Graph.Substitute(word, replacements, memo))) return false;
            if (key is not null && (key.Type != ScalarValueType.U32 ||
                !plan.ValidateRuntimeValue(plan.Graph.Substitute(key, replacements, memo)))) return false;
            result = new(handle, input, key);
            return true;
        }

        internal bool TryEvaluate(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
            out uint[] keys, out uint[][] descriptors)
        {
            keys = []; descriptors = [];
            if (inputs.ReadCleanMemory is null || inputs.ComputeState is not { } dispatch) return false;
            var count = Input.Payload switch
            {
                0 => dispatch.DispatchGroupsX, 1 => dispatch.DispatchGroupsY, 2 => dispatch.DispatchGroupsZ, _ => 0u,
            };
            if (count == 0 || count > MaximumCombinations || dispatch.DispatchGroupsX == 0 ||
                dispatch.DispatchGroupsY == 0 || dispatch.DispatchGroupsZ == 0) return false;
            var captured = new Dictionary<ulong, uint>();
            bool Read(ulong address, out uint word)
            {
                if (captured.TryGetValue(address, out word)) return true;
                if (address > (1ul << 48) - 4 || !inputs.ReadCleanMemory(address, out word)) return false;
                captured.Add(address, word);
                return true;
            }
            var clean = new ResourceRuntimeInputs { UserData = inputs.UserData, ShaderBase = inputs.ShaderBase,
                ReadMemory = Read, ReadCleanMemory = Read, ReadsClean = true, ComputeState = inputs.ComputeState };
            var candidates = new Dictionary<uint, uint[]>();
            var writes = new HashSet<(ulong Base, ulong Size)>();
            // Every possible shader write must be bounded. Unknown address spaces
            // decline this proof instead of assuming they cannot touch the table.
            var writeHandles = new Dictionary<ScalarValue, ulong>();
            for (var index = 0; index < plan.Memory.Count; index++)
            {
                var memory = plan.Memory[index];
                if (memory.Access is not (MemoryAccess.Write or MemoryAccess.Atomic)) continue;
                // GDS has its own backing allocation, like LDS and scratch. Its
                // offsets are not guest virtual addresses into descriptor memory.
                if (memory.Kind is MemoryResourceKind.LocalDataShare or MemoryResourceKind.Scratch or MemoryResourceKind.GlobalDataShare) continue;
                if (memory.Kind is not (MemoryResourceKind.Buffer or MemoryResourceKind.ScalarBuffer) ||
                    plan.Accesses[index]?.Handle is not { Kind: ScalarValueKind.BufferHandle, Operands.Length: 4 } output)
                    return false;
                var extra = (ulong)memory.Offset + Math.Max(16ul, (ulong)memory.DataBits * memory.DataDwords / 8);
                writeHandles[output] = Math.Max(writeHandles.GetValueOrDefault(output), extra);
            }
            for (uint group = 0; group < count; group++)
            {
                var evaluator = new RuntimeValueEvaluator(plan, clean, Input, group);
                var key = group;
                if (Key is not null && !evaluator.Evaluate(Key, out key)) return false;
                var words = new uint[Handle.Operands.Length];
                for (var component = 0; component < words.Length; component++)
                    if (!evaluator.Evaluate(Handle.Operands[component], out words[component])) return false;
                if (candidates.TryGetValue(key, out var existing))
                {
                    if (!existing.AsSpan().SequenceEqual(words)) return false;
                }
                else if (candidates.Count >= MaximumValues) return false;
                else candidates.Add(key, words);
                foreach (var (output, extra) in writeHandles)
                {
                    if (!PackedPointerDescriptor.EvaluateHandle(output, evaluator, out var outputWords) ||
                        !PackedPointerDescriptor.Range(outputWords, out var address, out var length) ||
                        address + length + extra > 1ul << 48) return false;
                    writes.Add((address, length + extra));
                }
            }
            // Compare after evaluating *all* groups: a later group's output may
            // alias an earlier group's descriptor or one of its pointer dependencies.
            var reads = captured.Keys.Order().ToArray();
            var orderedWrites = writes.OrderBy(range => range.Base).ToArray();
            var readIndex = 0; var writeIndex = 0;
            while (readIndex < reads.Length && writeIndex < orderedWrites.Length)
            {
                var write = orderedWrites[writeIndex];
                if (write.Base + write.Size <= reads[readIndex]) writeIndex++;
                else if (reads[readIndex] + 4 <= write.Base) readIndex++;
                else return false;
            }
            keys = candidates.Keys.ToArray();
            descriptors = candidates.Values.ToArray();
            return descriptors.Length != 0;
        }
    }

    // An indexed raw buffer word supplies a finite data domain even when its
    // lane index is unknown. Runtime enumeration must validate the V# layout,
    // include the out-of-bounds value, and establish write disjointness.
    internal sealed record PackedBufferWordDomain(ScalarValue Handle, uint Component)
    {
        // This is a data-domain proof, not permission to bind a descriptor.
        // The caller must separately exclude writes to every dependency range.
        internal bool TryEvaluate(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
            out uint[] values, out ulong baseAddress, out ulong byteLength)
        {
            values = [];
            baseAddress = byteLength = 0;
            if (inputs.ReadCleanMemory is null || Handle.Operands.Length != 4 || Component >= 2) return false;
            var evaluator = new RuntimeValueEvaluator(plan, inputs.WithReader(inputs.ReadCleanMemory));
            var descriptor = new uint[4];
            for (var index = 0; index < descriptor.Length; index++)
                if (!evaluator.Evaluate(Handle.Operands[index], out descriptor[index])) return false;
            var stride = (descriptor[1] >> 16) & 0x3FFF;
            // Only the unswizzled, record-bounded raw layout is implemented.
            // Other OOB modes must not be interpreted as a record count.
            if (stride != 8 || (descriptor[1] & 0x80000000) != 0 ||
                (descriptor[3] & 0xF0800000) != 0 ||
                ((descriptor[3] >> 12) & 0x7F) == 0 || descriptor[2] > MaximumCombinations) return false;
            baseAddress = ((ulong)(descriptor[1] & 0xFFFF) << 32) | descriptor[0];
            byteLength = (ulong)descriptor[2] * stride;
            if (baseAddress + byteLength > 1ul << 48) return false;
            var found = new HashSet<uint> { 0 }; // Initial and out-of-bounds reads.
            for (uint record = 0; record < descriptor[2]; record++)
            {
                var address = baseAddress + (ulong)record * stride + Component * sizeof(uint);
                if (!inputs.ReadCleanMemory(address, out var word)) return false;
                if (found.Add(word >> 16) && found.Count > MaximumValues) return false;
            }
            values = found.Order().ToArray();
            return true;
        }
    }

    internal sealed record PackedPointerDescriptor(PackedBufferWordDomain Domain,
        ScalarValue MaterialHandle, uint Stride, uint Offset, uint PointerImmediate,
        uint DescriptorImmediate, uint Width, uint SelectorPc, bool BufferFieldsOnly = false)
    {
        internal static bool TryCreateBufferFields(ShaderResourcePlan plan, ScalarValue handle,
            out PackedPointerDescriptor result)
        {
            result = null!;
            if (handle.Kind != ScalarValueKind.BufferHandle || handle.Operands.Length != 4 ||
                !TryGetSingleBufferRead(handle.Operands[0], out var first) ||
                !TryGetOffset(first.Operands[1], out var selector, out var stride, out var offset) ||
                !TryGetPackedBufferWordDomain(plan, selector, out var domain) ||
                !plan.ValidateRuntimeValue(first.Operands[0])) return false;
            var firstMemory = plan.Memory[first.MemoryIndex];
            if ((firstMemory.Offset & 3) != 0) return false;
            for (var component = 0; component < 4; component++)
            {
                if (!TryGetSingleBufferRead(handle.Operands[component], out var read) ||
                    !plan.Graph.Equivalent(first.Operands[0], read.Operands[0]) ||
                    !plan.Graph.Equivalent(first.Operands[1], read.Operands[1])) return false;
                var memory = plan.Memory[read.MemoryIndex];
                if (memory.Kind != MemoryResourceKind.ScalarBuffer || memory.Access != MemoryAccess.Read ||
                    memory.DataBits != 32 || memory.DataDwords != 1 || memory.Pc != firstMemory.Pc ||
                    (ulong)memory.Offset != (ulong)firstMemory.Offset + (uint)component * 4) return false;
            }
            result = new(domain, first.Operands[0], stride, offset, firstMemory.Offset,
                0, 4, (uint)selector.Payload, BufferFieldsOnly: true);
            return true;
        }

        internal static bool TryCreate(ShaderResourcePlan plan, ScalarValue handle,
            out PackedPointerDescriptor result)
        {
            result = null!;
            if (handle.Kind is not (ScalarValueKind.ImageHandle or ScalarValueKind.SamplerHandle) ||
                handle.Operands.Length is not (4 or 8)) return false;
            ScalarValue? address = null;
            uint immediate = 0;
            for (var component = 0; component < handle.Operands.Length; component++)
            {
                var read = handle.Operands[component];
                if (read.Kind != ScalarValueKind.ScalarAddressWord || read.Operands.Length != 2 ||
                    !read.Operands[1].IsConstant || read.Operands[1].ConstantU32 != 0 ||
                    read.MemoryIndex < 0 || read.MemoryIndex >= plan.Memory.Count) return false;
                var memory = plan.Memory[read.MemoryIndex];
                if (memory.Access != MemoryAccess.Read || memory.Kind != MemoryResourceKind.ScalarAddress ||
                    memory.DataBits != 32 || memory.DataDwords != 1 || memory.Offset < component * 4u) return false;
                var current = memory.Offset - (uint)component * 4;
                if (component == 0) { address = read.Operands[0]; immediate = current; }
                else if (current != immediate || !plan.Graph.Equivalent(address!, read.Operands[0])) return false;
            }
            if (address is not { Kind: ScalarValueKind.AddressHandle, Operands.Length: 2 } ||
                !TryGetSingleBufferRead(address.Operands[0], out var low) ||
                !TryGetSingleBufferRead(address.Operands[1], out var high)) return false;
            var lowMemory = plan.Memory[low.MemoryIndex];
            var highMemory = plan.Memory[high.MemoryIndex];
            if ((immediate & 3) != 0 || lowMemory.Offset > uint.MaxValue - 4 ||
                lowMemory.Kind != MemoryResourceKind.ScalarBuffer || highMemory.Kind != MemoryResourceKind.ScalarBuffer ||
                lowMemory.Access != MemoryAccess.Read || highMemory.Access != MemoryAccess.Read ||
                lowMemory.DataBits != 32 || highMemory.DataBits != 32 ||
                lowMemory.DataDwords != 1 || highMemory.DataDwords != 1 ||
                lowMemory.Pc != highMemory.Pc || highMemory.Offset != lowMemory.Offset + 4 ||
                highMemory.ComponentIndex != lowMemory.ComponentIndex + 1 ||
                !plan.Graph.Equivalent(low.Operands[0], high.Operands[0]) ||
                !plan.Graph.Equivalent(low.Operands[1], high.Operands[1]) ||
                !TryGetOffset(low.Operands[1], out var selector, out var stride, out var offset) ||
                !TryGetPackedBufferWordDomain(plan, selector, out var domain) ||
                !plan.ValidateRuntimeValue(low.Operands[0])) return false;
            result = new(domain, low.Operands[0], stride, offset, lowMemory.Offset,
                immediate, (uint)handle.Operands.Length, (uint)selector.Payload);
            return true;
        }

        private static bool TryGetSingleBufferRead(ScalarValue value, out ScalarValue read)
        {
            read = null!;
            var pending = new Stack<ScalarValue>(); pending.Push(value);
            var visited = new HashSet<ScalarValue>();
            while (pending.TryPop(out var current))
            {
                if (!visited.Add(current)) continue;
                if (current.Kind == ScalarValueKind.ScalarBufferWord && current.Operands.Length == 2)
                {
                    if (read is not null && !ReferenceEquals(read, current)) return false;
                    read = current;
                }
                else if (current.Kind == ScalarValueKind.Phi && current.Operands.Length != 0)
                    foreach (var operand in current.Operands) pending.Push(operand);
                else return false;
            }
            return read is not null;
        }

        private static bool TryGetOffset(ScalarValue value, out ScalarValue selector, out uint stride, out uint offset)
        {
            selector = null!; stride = offset = 0;
            if (value.Kind == ScalarValueKind.Phi)
            {
                var alternatives = new HashSet<ScalarValue>();
                var pending = new Stack<ScalarValue>(); pending.Push(value);
                var visited = new HashSet<ScalarValue>();
                while (pending.TryPop(out var current))
                {
                    if (!visited.Add(current)) continue;
                    if (current.Kind == ScalarValueKind.Phi)
                        foreach (var operand in current.Operands) pending.Push(operand);
                    else alternatives.Add(current);
                }
                if (alternatives.Count != 1) return false;
                value = alternatives.Single();
            }
            if (value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.IAdd32)
            {
                if (value.Operands[0].IsConstant) { offset = value.Operands[0].ConstantU32; value = value.Operands[1]; }
                else if (value.Operands[1].IsConstant) { offset = value.Operands[1].ConstantU32; value = value.Operands[0]; }
                else return false;
            }
            if (value.Kind != ScalarValueKind.Operation || value.Operation != ScalarOperation.IMul32) return false;
            if (value.Operands[0].IsConstant) { stride = value.Operands[0].ConstantU32; selector = value.Operands[1]; }
            else if (value.Operands[1].IsConstant) { stride = value.Operands[1].ConstantU32; selector = value.Operands[0]; }
            else return false;
            return stride != 0 && selector.Kind == ScalarValueKind.FirstLane;
        }

        internal bool TryEvaluate(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, out uint[] words)
        {
            words = [];
            if (inputs.ReadCleanMemory is null) return false;
            var dependencies = new List<(ulong Base, ulong Size)>();
            var capturedWords = new Dictionary<ulong, uint>();
            bool Read(ulong address, out uint word)
            {
                if (capturedWords.TryGetValue(address, out word)) return true;
                if (!inputs.ReadCleanMemory(address, out word)) return false;
                dependencies.Add((address, 4));
                capturedWords.Add(address, word);
                return true;
            }
            var clean = new ResourceRuntimeInputs { UserData = inputs.UserData, ShaderBase = inputs.ShaderBase,
                ReadMemory = Read, ReadCleanMemory = Read, ReadsClean = true, ComputeState = inputs.ComputeState };
            if (!Domain.TryEvaluate(plan, clean, out var selectors, out var searchBase, out var searchLength)) return false;
            dependencies.Add((searchBase, searchLength));
            var evaluator = new RuntimeValueEvaluator(plan, clean);
            if (!EvaluateHandle(MaterialHandle, evaluator, out var material) ||
                ((material[1] >> 16) & 0x3FFF) != Stride ||
                !Range(material, out var materialBase, out var materialLength)) return false;
            dependencies.Add((materialBase, materialLength));
            uint[]? candidate = null;
            var writes = new List<(ulong Base, ulong Size)>();
            foreach (var selector in selectors)
            {
                var dynamicOffset = unchecked(selector * Stride + Offset);
                var current = new uint[Width];
                if (BufferFieldsOnly)
                {
                    for (uint component = 0; component < Width; component++)
                        if (!ReadMaterial(dynamicOffset, (ulong)PointerImmediate + component * 4, out current[component])) return false;
                    if (!Range(current, out _, out _)) return false;
                    // Addresses and record counts stay in the executing SGPRs.
                    // Only require agreement on addressing/conversion fields.
                    current = [0, current[1] & 0xFFFF0000, 0, current[3]];
                }
                else
                {
                    if (!ReadMaterial(dynamicOffset, PointerImmediate, out var low) ||
                        !ReadMaterial(dynamicOffset, (ulong)PointerImmediate + 4, out var high)) return false;
                    var pointer = ((ulong)high << 32) | low;
                    if (pointer == 0 || pointer > 0xFFFFFFFFFFFFul ||
                        pointer + DescriptorImmediate + Width * 4ul > 1ul << 48) return false;
                    for (uint component = 0; component < Width; component++)
                        if (!Read((pointer & ~3ul) + DescriptorImmediate + component * 4, out current[component])) return false;
                }
                if (candidate is not null && !candidate.AsSpan().SequenceEqual(current)) return false;
                candidate = current;

                for (var index = 0; index < plan.Memory.Count; index++)
                {
                    var memory = plan.Memory[index];
                    if (memory.Access is not (MemoryAccess.Write or MemoryAccess.Atomic)) continue;
                    if (memory.Kind is MemoryResourceKind.LocalDataShare or MemoryResourceKind.Scratch) continue;
                    if (memory.Kind is not (MemoryResourceKind.Buffer or MemoryResourceKind.ScalarBuffer) ||
                        plan.Accesses[index]?.Handle is not { Kind: ScalarValueKind.BufferHandle, Operands.Length: 4 } handle) return false;
                    uint[] output;
                    if (plan.ValidateRuntimeValue(handle))
                    {
                        if (!EvaluateHandle(handle, evaluator, out output)) return false;
                    }
                    else
                    {
                        output = new uint[4];
                        for (var component = 0; component < 4; component++)
                        {
                            if (!TryGetSingleBufferRead(handle.Operands[component], out var read) ||
                                !plan.Graph.Equivalent(read.Operands[0], MaterialHandle) ||
                                !TryGetOffset(read.Operands[1], out var key, out var stride, out var offset) ||
                                key.Payload != SelectorPc || stride != Stride || offset != Offset ||
                                !ReadMaterial(dynamicOffset, plan.Memory[read.MemoryIndex].Offset, out output[component])) return false;
                        }
                    }
                    if (!Range(output, out var writeBase, out var writeLength) ||
                        writeBase + writeLength + 16 > 1ul << 48) return false;
                    // Include an entire possible formatted element past the last
                    // record, rather than assume stride equals the format width.
                    writes.Add((writeBase, writeLength + 16));
                }
            }
            var protectedRanges = dependencies.Where(range => range.Size != 0).Distinct().OrderBy(range => range.Base).ToArray();
            var writtenRanges = writes.Distinct().OrderBy(range => range.Base).ToArray();
            var protectedIndex = 0;
            var writtenIndex = 0;
            while (protectedIndex < protectedRanges.Length && writtenIndex < writtenRanges.Length)
            {
                var dependency = protectedRanges[protectedIndex];
                var write = writtenRanges[writtenIndex];
                if (write.Base + write.Size <= dependency.Base) writtenIndex++;
                else if (dependency.Base + dependency.Size <= write.Base) protectedIndex++;
                else return false;
            }
            words = candidate ?? [];
            return words.Length == Width;

            bool ReadMaterial(uint dynamicOffset, ulong immediate, out uint word)
            {
                word = 0;
                var relative = (ulong)(dynamicOffset & ~3u) + (immediate & ~3ul);
                if (relative + 4 > materialLength) return false;
                return Read((materialBase & ~3ul) + relative, out word);
            }
        }

        internal static bool EvaluateHandle(ScalarValue handle, RuntimeValueEvaluator evaluator, out uint[] words)
        {
            words = new uint[4];
            if (handle.Operands.Length != 4) return false;
            for (var index = 0; index < 4; index++)
                if (!evaluator.Evaluate(handle.Operands[index], out words[index])) return false;
            return true;
        }

        internal static bool Range(uint[] words, out ulong address, out ulong length)
        {
            address = ((ulong)(words[1] & 0xFFFF) << 32) | words[0];
            var stride = (words[1] >> 16) & 0x3FFF;
            length = (ulong)words[2] * Math.Max(stride, 1u);
            return (words[1] & 0x80000000) == 0 && (words[3] & 0xF0800000) == 0 &&
                address + length <= 1ul << 48;
        }
    }

    internal static bool TryGetPackedBufferWordDomain(ShaderResourcePlan plan, ScalarValue selector,
        out PackedBufferWordDomain domain)
    {
        domain = null!;
        if (selector.Kind != ScalarValueKind.FirstLane) return false;
        var instructions = plan.Graph.Program.Instructions;
        var laneIndex = instructions.ToList().FindIndex(instruction => instruction.Pc == selector.Payload);
        if (laneIndex < 0) return false;
        var laneRead = instructions[laneIndex];
        if (laneRead.Opcode is not ("VReadlaneB32" or "VReadfirstlaneB32")) return false;
        if (laneRead.Sources.Count == 0 || laneRead.Sources[0].Kind != Gen5OperandKind.VectorRegister) return false;
        var packedIndex = -1;
        for (var index = laneIndex - 1; index >= 0; index--)
            if (Builder.WritesRegister(instructions[index], laneRead.Sources[0])) { packedIndex = index; break; }
        if (packedIndex < 0) return false;
        var packed = instructions[packedIndex];
        if (packed is not { Opcode: "VMovB32", Sources.Count: 1,
            Control: Gen5SdwaControl { DestinationSelect: 6, Source0Select: 5,
                Source0SignExtend: false, AbsoluteMask: 0, NegateMask: 0, OutputModifier: 0, Clamp: false } } ||
            packed.Sources[0].Kind != Gen5OperandKind.VectorRegister) return false;
        if (!Dominates(packed.Pc, laneRead.Pc)) return false;
        if (laneRead.Opcode == "VReadlaneB32" && !ReadsInitializedLane()) return false;
        // READFIRSTLANE with empty EXEC has no initialized lane. Such a value
        // cannot become an active resource key if later execution never expands.
        if (laneRead.Opcode == "VReadfirstlaneB32" && instructions.Skip(laneIndex + 1)
                .Any(Builder.MayExpandExecution)) return false;
        for (var index = packedIndex + 1; index < laneIndex; index++)
            if (Builder.MayExpandExecution(instructions[index]) &&
                !instructions[index].Opcode.StartsWith("VCmpx", StringComparison.Ordinal)) return false;
        var vector = packed.Sources[0];
        var initialIndex = -1;
        for (var index = 0; index < packedIndex; index++)
            if (instructions[index] is { Opcode: "VMovB32", Sources.Count: 1, Control: null } initial &&
                initial.Destinations.Contains(vector) && Constant(initial.Sources[0], out var zero) && zero == 0)
            { initialIndex = index; break; }
        if (initialIndex < 0) return false;
        var captureIndex = -1;
        for (var index = initialIndex - 1; index >= 0; index--)
            if (instructions[index] is { Opcode: "SMovB64", Sources.Count: 1, Destinations.Count: 1 } capture &&
                capture.Sources[0] == Gen5Operand.Scalar(126) && capture.Destinations[0] is
                    { Kind: Gen5OperandKind.ScalarRegister, Value: < 126 } saved && (saved.Value & 1) == 0)
            { captureIndex = index; break; }
        if (captureIndex < 0 || !Dominates(instructions[captureIndex].Pc, instructions[initialIndex].Pc)) return false;
        var mask = instructions[captureIndex].Destinations[0];
        ScalarValue? handle = null;
        uint component = 0;
        for (var index = captureIndex + 1; index < packedIndex; index++)
        {
            var instruction = instructions[index];
            if (instruction.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) ||
                instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal) || Builder.WritesSavedMask(instruction, mask)) return false;
            if (Builder.MayExpandExecution(instruction))
            {
                var restrictsCurrent = instruction.Opcode is "SAndn2B64" && instruction.Sources.Count == 2 &&
                    instruction.Sources[0] == Gen5Operand.Scalar(126) || instruction.Opcode == "SAndB64" &&
                    instruction.Sources.Contains(Gen5Operand.Scalar(126));
                if (index < initialIndex || !restrictsCurrent && !instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal) &&
                    (instruction is not { Opcode: "SMovB64", Sources.Count: 1 } ||
                    instruction.Sources[0] != mask || !instruction.Destinations.Contains(Gen5Operand.Scalar(126)))) return false;
            }
            if (index <= initialIndex || !Builder.WritesRegister(instruction, vector)) continue;
            if (instruction is not { Opcode: "BufferLoadDwordx2", Control: Gen5BufferMemoryControl
                { DwordCount: 2, IndexEnabled: true, OffsetEnabled: false, OffsetBytes: 0, Typed: false } buffer } ||
                instruction.Sources.Count < 3 || !Constant(instruction.Sources[2], out var scalarOffset) || scalarOffset != 0 ||
                vector.Value < buffer.VectorData || vector.Value - buffer.VectorData >= 2 ||
                !Dominates(instructions[initialIndex].Pc, instruction.Pc, captureIndex) ||
                !plan.Memory.TryGetIndex(instruction.Pc, 0, out var memoryIndex) ||
                plan.Accesses[memoryIndex]?.Handle is not { Kind: ScalarValueKind.BufferHandle } current ||
                !plan.ValidateRuntimeValue(current)) return false;
            var currentComponent = vector.Value - buffer.VectorData;
            if (handle is not null && (!plan.Graph.Equivalent(handle, current) || component != currentComponent)) return false;
            handle = current; component = currentComponent;
        }
        if (handle is null) return false;
        if (!Dominates(instructions[initialIndex].Pc, packed.Pc, captureIndex)) return false;
        domain = new(handle, component);
        return true;

        static bool Constant(Gen5Operand operand, out uint value)
        {
            value = operand.Value;
            return operand.Kind == Gen5OperandKind.LiteralConstant ||
                operand.Kind == Gen5OperandKind.EncodedConstant && Gen5InlineConstants.TryDecode(operand.Value, out value);
        }

        bool ReadsInitializedLane()
        {
            if (laneRead.Sources.Count < 2 || laneRead.Sources[1].Kind != Gen5OperandKind.ScalarRegister) return false;
            var scanIndex = laneIndex - 1;
            while (scanIndex > packedIndex && !Builder.WritesRegister(instructions[scanIndex], laneRead.Sources[1])) scanIndex--;
            if (scanIndex <= packedIndex || instructions[scanIndex] is not
                { Opcode: "SFF1I32B64", Sources.Count: 1 } scan || scan.Sources[0] is not
                { Kind: Gen5OperandKind.ScalarRegister, Value: < 126 } remaining || (remaining.Value & 1) != 0) return false;
            if (!Dominates(scan.Pc, laneRead.Pc)) return false;
            var capture = scanIndex - 1;
            while (capture > packedIndex && !Builder.WritesSavedMask(instructions[capture], remaining)) capture--;
            if (capture <= packedIndex || instructions[capture] is not { Opcode: "SMovB64", Sources.Count: 1 } copy ||
                copy.Sources[0] != Gen5Operand.Scalar(126) || !copy.Destinations.Contains(remaining) ||
                !Dominates(copy.Pc, laneRead.Pc)) return false;
            for (var index = packedIndex + 1; index < capture; index++)
                if (Builder.MayExpandExecution(instructions[index])) return false;
            // The first scan must be non-empty. A fallthrough EXECZ guard in the
            // preceding block establishes this without inferring it from data.
            var guarded = false;
            for (var index = packedIndex - 1; index >= 0; index--)
            {
                var instruction = instructions[index];
                if (instruction.Opcode == "SCbranchExecz" && Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var target) &&
                    instructions.Any(exit => exit.Pc == target && exit.Opcode == "SEndpgm") &&
                    Dominates(instruction.Pc, packed.Pc)) { guarded = true; break; }
                if (Builder.MayExpandExecution(instruction) || Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out _)) break;
            }
            if (!guarded) return false;
            for (var index = capture + 1; index < instructions.Count; index++)
            {
                var instruction = instructions[index];
                if (instruction.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal) ||
                    Builder.WritesRegister(instruction, laneRead.Sources[0])) return false;
                if (Builder.WritesSavedMask(instruction, remaining) &&
                    (instruction is not { Opcode: "SAndn2B64", Sources.Count: 2 } || instruction.Sources[0] != remaining ||
                        !instruction.Destinations.Contains(remaining))) return false;
                if (!Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var target) || target > instruction.Pc) continue;
                if (target <= copy.Pc) return false;
                if (target > instructions[scanIndex].Pc) continue;
                // Each repeat is conditional on the remaining mask's SCC result.
                if (instruction.Opcode != "SCbranchScc1" || target != scan.Pc) return false;
                var setter = index - 1;
                while (setter > capture && instructions[setter].Opcode == "SMovB64") setter--;
                if (setter <= capture || instructions[setter] is not { Opcode: "SAndn2B64", Sources.Count: 2 } reduction ||
                    reduction.Sources[0] != remaining || !reduction.Destinations.Contains(remaining)) return false;
            }
            return true;
        }

        bool Dominates(uint definition, uint use, int zeroCapture = -1)
        {
            var flow = plan.Graph.ControlFlow;
            var definitionBlock = Enumerable.Range(0, flow.Blocks.Count).First(index => definition >= flow.Blocks[index].StartPc && definition < flow.Blocks[index].EndPc);
            var useBlock = Enumerable.Range(0, flow.Blocks.Count).First(index => use >= flow.Blocks[index].StartPc && use < flow.Blocks[index].EndPc);
            if (definitionBlock == useBlock) return definition <= use;
            var pending = new Queue<int>(); pending.Enqueue(0);
            var visited = new HashSet<int>();
            while (pending.TryDequeue(out var block))
            {
                if (block == definitionBlock || !visited.Add(block)) continue;
                if (block == useBlock) return false;
                uint? zeroTarget = null;
                if (zeroCapture >= 0)
                {
                    var last = instructions.LastOrDefault(instruction => instruction.Pc >= flow.Blocks[block].StartPc && instruction.Pc < flow.Blocks[block].EndPc);
                    if (last is { Opcode: "SCbranchExecz" } && last.Pc > instructions[zeroCapture].Pc && last.Pc < definition &&
                        instructions.Skip(zeroCapture + 1).TakeWhile(instruction => instruction.Pc < last.Pc)
                            .All(instruction => !Builder.MayExpandExecution(instruction)) &&
                        Gen5IrBranchResolver.Instance.TryGetBranchTarget(last, out var target) &&
                        target != last.Pc + (uint)last.Words.Count * 4) zeroTarget = target;
                }
                foreach (var next in flow.Successors[block])
                    if (zeroTarget is null || flow.Blocks[next].StartPc != zeroTarget) pending.Enqueue(next);
            }
            return true;
        }
    }

    private IndirectSelectorValues(Expression root, WaveMaskSelectorBounds? waveBounds)
    {
        _root = root;
        _waveBounds = waveBounds;
    }

    internal static IndirectSelectorValues? Create(ShaderResourcePlan plan, ScalarValue selector)
    {
        if (selector.Kind != ScalarValueKind.FirstLane) return null;
        var instruction = plan.Graph.Program.Instructions.FirstOrDefault(candidate => candidate.Pc == selector.Payload);
        if (instruction is not { Opcode: "VReadfirstlaneB32", Sources.Count: 1 }) return null;
        var builder = new Builder(plan);
        var root = builder.Read(instruction.Sources[0], instruction.Pc);
        return root is null || !builder.HasBitScan ? null : new IndirectSelectorValues(root,
            WaveMaskSelectorBounds.TryCreate(plan, instruction));
    }

    // Only constant reaching definitions qualify here. Runtime memory/user data,
    // unknown writes and execution-mask expansion must not manufacture a bound.
    internal static bool TryGetConstantValues(ShaderResourcePlan plan, ScalarValue selector, out uint[] values)
    {
        values = [];
        if (selector.Kind != ScalarValueKind.FirstLane) return false;
        var instruction = plan.Graph.Program.Instructions.FirstOrDefault(candidate => candidate.Pc == selector.Payload);
        if (instruction is null) return false;
        var before = instruction.Pc;
        if (instruction.Opcode == "VReadlaneB32")
        {
            var index = plan.Graph.Program.Instructions.ToList().IndexOf(instruction);
            if (!ResourceTracker.TryGetStableLaneReadStart(plan.Graph.Program.Instructions, index, out before)) return false;
        }
        else if (instruction is not { Opcode: "VReadfirstlaneB32", Sources.Count: 1 }) return false;
        var root = new Builder(plan).Read(instruction.Sources[0], before);
        if (root is null) return false;
        var pending = new Stack<Expression>();
        pending.Push(root);
        while (pending.TryPop(out var expression))
        {
            if (expression.RuntimeValue is not null) return false;
            if (expression.Inputs is { } operands)
                foreach (var operand in operands) pending.Push(operand);
        }
        return new IndirectSelectorValues(root, null).TryEvaluate(plan, new ResourceRuntimeInputs(), out values);
    }

    internal bool TryEvaluate(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, out uint[] values,
        IndirectSelectorDiagnostic? diagnostic = null)
    {
        bool ReadCapturedWord(ulong address, out uint word)
        {
            var succeeded = inputs.ReadCleanMemory!(address, out word);
            if (!succeeded) diagnostic!.FailedReadAddress ??= address;
            if (diagnostic!.MemoryReads.Count < 256)
                diagnostic.MemoryReads.Add(new(address, succeeded, succeeded ? word : null));
            else diagnostic.OmittedMemoryReads++;
            return succeeded;
        }
        var reader = inputs.ReadCleanMemory;
        if (diagnostic is not null && reader is not null) reader = ReadCapturedWord;
        var evaluator = new RuntimeValueEvaluator(plan, inputs.WithReader(reader));
        if (_waveBounds is not null && _waveBounds.TryEvaluate(evaluator, inputs.ComputeState, out values))
            return true;
        var cache = new Dictionary<Expression, uint[]>();
        uint[]? Decline(string reason)
        {
            if (diagnostic is not null) diagnostic.EvaluationFailure ??= reason;
            return null;
        }
        uint[]? Evaluate(Expression expression)
        {
            if (cache.TryGetValue(expression, out var previous)) return previous;
            if (expression.Values is { } constants) return constants;
            if (expression.RuntimeValue is { } runtime)
                return evaluator.Evaluate(runtime, out var value) ? [value] : Decline("runtime_value_unavailable");
            var operands = expression.Inputs!.Select(Evaluate).ToArray();
            if (operands.Any(operand => operand is null)) return null;
            var results = new HashSet<uint>();
            if (expression.Operation == ScalarOperation.None)
            {
                foreach (var operand in operands)
                    foreach (var value in operand!)
                        if (results.Add(value) && results.Count > MaximumValues) return Decline("value_limit");
            }
            else
            {
                if ((long)operands[0]!.Length * operands[1]!.Length > MaximumCombinations) return Decline("combination_limit");
                Span<ulong> pair = stackalloc ulong[2];
                foreach (var left in operands[0]!)
                    foreach (var right in operands[1]!)
                    {
                        pair[0] = left;
                        pair[1] = right;
                        if (!ScalarOperationSemantics.TryEvaluate(expression.Operation, pair, out var result)) return Decline("unsupported_operation");
                        if (results.Add((uint)result) && results.Count > MaximumValues) return Decline("value_limit");
                    }
            }
            return cache[expression] = results.ToArray();
        }
        values = Evaluate(_root) ?? [];
        return values.Length != 0;
    }

    private sealed class Builder(ShaderResourcePlan plan)
    {
        private readonly HashSet<(Gen5Operand Operand, uint Address)> _active = [];
        private int _requests;
        public bool HasBitScan { get; private set; }

        public Expression? Read(Gen5Operand operand, uint before)
        {
            if (++_requests > 512 || !_active.Add((operand, before))) return null;
            try
            {
                if (operand.Kind == Gen5OperandKind.LiteralConstant) return new(Values: [operand.Value]);
                if (operand.Kind == Gen5OperandKind.EncodedConstant)
                    return Gen5InlineConstants.TryDecode(operand.Value, out var constant) ? new(Values: [constant]) : null;
                var flow = plan.Graph.ControlFlow;
                var initial = Enumerable.Range(0, flow.Blocks.Count).FirstOrDefault(
                    index => before >= flow.Blocks[index].StartPc && before < flow.Blocks[index].EndPc, -1);
                if (initial < 0) return null;
                var pending = new Queue<(int Block, uint Before)>();
                var visited = new HashSet<(int Block, uint Before)>();
                var definitions = new List<Expression>();
                pending.Enqueue((initial, before));
                while (pending.TryDequeue(out var position))
                {
                    if (!visited.Add(position)) continue;
                    var block = flow.Blocks[position.Block];
                    var found = false;
                    foreach (var instruction in plan.Graph.Program.Instructions.Reverse())
                    {
                        if (instruction.Pc < block.StartPc || instruction.Pc >= position.Before) continue;
                        if (instruction.Opcode.Contains("Movreld", StringComparison.Ordinal) ||
                            instruction.Opcode.Contains("Movrelsd", StringComparison.Ordinal) ||
                            instruction.Opcode.Contains("Swaprel", StringComparison.Ordinal) ||
                            instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal)) return null;
                        if (WritesRegister(instruction, operand))
                        {
                            var definition = Define(instruction, operand);
                            if (definition is null) return null;
                            definitions.Add(definition);
                            found = true;
                            break;
                        }
                        // A later active lane must also have been active at the defining write.
                        if (operand.Kind == Gen5OperandKind.VectorRegister && MayExpandExecution(instruction))
                        {
                            var restored = ReadBeforeSavedExecRestore(operand, instruction);
                            if (restored is null) return null;
                            definitions.Add(restored);
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                    if (position.Block == 0)
                    {
                        if (operand.Kind != Gen5OperandKind.ScalarRegister || operand.Value < plan.UserDataBase ||
                            operand.Value - plan.UserDataBase >= plan.UserDataCount) return null;
                        definitions.Add(new(RuntimeValue: plan.Graph.UserData(operand.Value)));
                    }
                    if (flow.Predecessors[position.Block].Count == 0 && position.Block != 0) return null;
                    foreach (var predecessor in flow.Predecessors[position.Block])
                        pending.Enqueue((predecessor, flow.Blocks[predecessor].EndPc));
                }
                return definitions.Count == 0 ? null : new(Inputs: definitions.ToArray());
            }
            finally { _active.Remove((operand, before)); }
        }

        private Expression? Define(Gen5ShaderInstruction instruction, Gen5Operand destination)
        {
            if (!instruction.Destinations.Contains(destination)) return null;
            // Packed descriptor selectors may extract the upper word with SDWA.
            // A full-dword destination discards the previous value; partial
            // destinations and source modifiers require a separate proof.
            if (instruction is { Opcode: "VMovB32", Sources.Count: 1,
                Control: Gen5SdwaControl { DestinationSelect: 6, Source0Select: 5,
                    Source0SignExtend: false, AbsoluteMask: 0, NegateMask: 0,
                    OutputModifier: 0, Clamp: false } })
            {
                var packed = Read(instruction.Sources[0], instruction.Pc);
                return packed is null ? null : new(Operation: ScalarOperation.And32, Inputs:
                    [new(Operation: ScalarOperation.ShiftRightLogical32, Inputs:
                        [packed, new(Values: [16])]), new(Values: [0xFFFF])]);
            }
            if (instruction.Control is Gen5Vop3Control { AbsoluteMask: not 0 } or Gen5Vop3Control { NegateMask: not 0 } or
                Gen5Vop3Control { Clamp: true } or Gen5Vop3Control { OutputModifier: not 0 } or Gen5Vop3Control { OperandSelect: not 0 } or
                Gen5DppControl or Gen5Dpp8Control or Gen5Vop3pControl) return null;
            if (instruction.Control is Gen5SdwaControl sdwa &&
                (instruction.Opcode != "VCndmaskB32" || sdwa.DestinationSelect != 6 ||
                 sdwa.Source0Select != 6 || sdwa.Source1Select != 6 ||
                 sdwa.Source0SignExtend || sdwa.Source1SignExtend ||
                 sdwa.AbsoluteMask != 0 || sdwa.NegateMask != 0 || sdwa.OutputModifier != 0 || sdwa.Clamp)) return null;
            if (instruction.Opcode is "SFF1I32B32" or "VFfblB32")
            {
                HasBitScan = true;
                return new(Values: Enumerable.Range(0, 32).Select(value => (uint)value).Append(uint.MaxValue).ToArray());
            }
            if (instruction.Encoding == Gen5ShaderEncoding.Smem)
            {
                var component = instruction.Destinations.ToList().IndexOf(destination);
                if (component < 0 || !plan.Memory.TryGetIndex(instruction.Pc, (uint)component, out var memoryIndex)) return null;
                var read = plan.Graph.Accesses[memoryIndex]?.Read;
                return read is not null && plan.ValidateRuntimeValue(read) ? new(RuntimeValue: read) : null;
            }
            if (instruction.Opcode is "SMovB32" or "VMovB32") return Read(instruction.Sources[0], instruction.Pc);
            if (instruction.Opcode == "VCndmaskB32" && instruction.Sources.Count >= 2)
            {
                var falseArm = Read(instruction.Sources[0], instruction.Pc);
                var trueArm = Read(instruction.Sources[1], instruction.Pc);
                return falseArm is null || trueArm is null ? null : new(Inputs: [falseArm, trueArm]);
            }
            if (instruction.Opcode is "VMadU32U24" or "VMulU32U24" &&
                instruction.Sources.Count == (instruction.Opcode == "VMadU32U24" ? 3 : 2))
            {
                var operands = instruction.Sources.Select(source => Read(source, instruction.Pc)).ToArray();
                if (operands.Any(operand => operand is null)) return null;
                var mask = new Expression(Values: [0x00FF_FFFF]);
                var multiplicand = new Expression(Operation: ScalarOperation.And32, Inputs: [operands[0]!, mask]);
                var multiplier = new Expression(Operation: ScalarOperation.And32, Inputs: [operands[1]!, mask]);
                var product = new Expression(Operation: ScalarOperation.IMul32, Inputs: [multiplicand, multiplier]);
                if (instruction.Opcode == "VMulU32U24") return product;
                return new(Operation: ScalarOperation.IAdd32, Inputs: [product, operands[2]!]);
            }
            var operation = instruction.Opcode switch
            {
                "SAddU32" or "SAddI32" or "VAddU32" or "VAddI32" or "VAdd3U32" => ScalarOperation.IAdd32,
                "SLshlB32" => ScalarOperation.ShiftLeft32,
                _ => ScalarOperation.None,
            };
            if (operation == ScalarOperation.None || instruction.Sources.Count < 2) return null;
            var left = Read(instruction.Sources[0], instruction.Pc);
            var right = Read(instruction.Sources[1], instruction.Pc);
            if (left is null || right is null) return null;
            var result = new Expression(Operation: operation, Inputs: [left, right]);
            if (instruction.Opcode != "VAdd3U32") return result;
            var third = Read(instruction.Sources[2], instruction.Pc);
            return third is null ? null : new(Operation: ScalarOperation.IAdd32, Inputs: [result, third]);
        }

        // Restoring a saved mask exposes both unchanged lanes and lanes written
        // while restricted. Retain every reaching value until a full-mask overwrite.
        private Expression? ReadBeforeSavedExecRestore(Gen5Operand vector, Gen5ShaderInstruction restore)
        {
            if (restore is not { Opcode: "SMovB64", Sources.Count: 1 } ||
                !restore.Destinations.Contains(Gen5Operand.Scalar(126)) ||
                restore.Sources[0] is not { Kind: Gen5OperandKind.ScalarRegister } saved ||
                saved.Value >= 126 || (saved.Value & 1) != 0) return null;
            var instructions = plan.Graph.Program.Instructions;
            var restoreIndex = instructions.ToList().IndexOf(restore);
            // A later full-mask definition can kill every value from an earlier
            // waterfall. Start there only when its saved-mask restoration cannot
            // be bypassed and the defining write dominates this restoration.
            if (TryFindFullSavedMaskOverwrite(vector, saved, restoreIndex, out var overwrite))
                return ReadAfterFullSavedMaskOverwrite(vector, saved, overwrite, restoreIndex);
            var saveIndex = -1;
            for (var index = restoreIndex - 1; index >= 0; index--)
            {
                var instruction = instructions[index];
                if (!WritesSavedMask(instruction, saved)) continue;
                if (instruction is not { Sources.Count: 1 } || instruction.Opcode is not ("SMovB64" or "SAndSaveexecB64" or "SAndn1SaveexecB64") ||
                    !instruction.Destinations.Contains(saved)) return null;
                if (instruction.Opcode == "SMovB64" && instruction.Sources[0] != Gen5Operand.Scalar(126))
                {
                    var alias = instruction.Sources[0];
                    if (alias is not { Kind: Gen5OperandKind.ScalarRegister } || alias.Value >= 126 || (alias.Value & 1) != 0)
                        return null;
                    var restoredAlias = false;
                    for (var previous = index - 1; previous >= 0; previous--)
                    {
                        var prior = instructions[previous];
                        if (WritesSavedMask(prior, alias) || Gen5IrBranchResolver.Instance.TryGetBranchTarget(prior, out _)) return null;
                        if (!MayExpandExecution(prior) && prior.Opcode is not ("SAndSaveexecB64" or "SAndSaveexecB32")) continue;
                        restoredAlias = prior is { Opcode: "SMovB64", Sources.Count: 1 } &&
                            prior.Destinations.Contains(Gen5Operand.Scalar(126)) && prior.Sources[0] == alias;
                        if (instructions.Any(edge => Gen5IrBranchResolver.Instance.TryGetBranchTarget(edge, out var target) &&
                            target > prior.Pc && target <= instruction.Pc)) return null;
                        break;
                    }
                    if (!restoredAlias) return null;
                }
                saveIndex = index;
                break;
            }
            if (saveIndex < 0) return null;
            // A waterfall can recapture the same mask on every iteration. If this
            // register is invariant, read it before the loop rather than following
            // a cyclic reaching definition through the loop's EXEC restoration.
            if (instructions[saveIndex].Opcode == "SAndSaveexecB64")
            {
                for (var readIndex = saveIndex - 1; readIndex >= 0; readIndex--)
                {
                    var laneRead = instructions[readIndex];
                    if (laneRead.Opcode != "VReadlaneB32") continue;
                    if (!ResourceTracker.TryGetStableLaneReadStart(instructions, readIndex, vector, out var start) ||
                        start >= instructions[saveIndex].Pc) break;
                    var region = instructions.Where(candidate => candidate.Pc >= start && candidate.Pc < restore.Pc).ToArray();
                    var unchanged = region.All(candidate => !WritesRegister(candidate, vector) &&
                        !candidate.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) &&
                        !candidate.Opcode.Contains("GprIdx", StringComparison.Ordinal) &&
                        (candidate == instructions[saveIndex] || !WritesSavedMask(candidate, saved)) &&
                        (candidate.Pc >= instructions[saveIndex].Pc || !MayExpandExecution(candidate) &&
                            candidate.Opcode is not ("SAndSaveexecB64" or "SAndSaveexecB32")));
                    var bypass = instructions.Any(edge => (edge.Pc < start || edge.Pc >= restore.Pc) &&
                        Gen5IrBranchResolver.Instance.TryGetBranchTarget(edge, out var target) &&
                        target > start && target <= restore.Pc);
                    if (unchanged && !bypass) return Read(vector, start);
                    break;
                }
            }
            // An incoming edge must not bypass the saved mask or the defining write.
            foreach (var instruction in instructions)
                if ((instruction.Pc < instructions[saveIndex].Pc || instruction.Pc >= restore.Pc) &&
                    Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var target) &&
                    target > instructions[saveIndex].Pc && target <= restore.Pc) return null;
            var fullMask = instructions[saveIndex].Opcode == "SMovB64";
            Expression? values = Read(vector, instructions[saveIndex].Pc);
            for (var index = saveIndex + 1; index < restoreIndex; index++)
            {
                var instruction = instructions[index];
                if (instruction.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal) || WritesSavedMask(instruction, saved)) return null;
                if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var target))
                {
                    if (target <= instruction.Pc) return null;
                    // An edge leaving this region cannot reach this restoration.
                    if (target <= restore.Pc)
                    {
                        if (fullMask) return null;
                        var nextRestore = instructions.Skip(index + 1).FirstOrDefault(candidate =>
                            candidate.Pc <= restore.Pc && candidate is { Opcode: "SMovB64", Sources.Count: 1 } &&
                            candidate.Destinations.Contains(Gen5Operand.Scalar(126)) && candidate.Sources[0] == saved);
                        if (nextRestore is null || target > nextRestore.Pc) return null;
                    }
                }
                var writesVector = WritesRegister(instruction, vector);
                if (MayExpandExecution(instruction) || instruction.Opcode is "SAndSaveexecB64" or "SAndSaveexecB32")
                {
                    if (instruction is { Opcode: "SMovB64", Sources.Count: 1 } &&
                        instruction.Destinations.Contains(Gen5Operand.Scalar(126)) && instruction.Sources[0] == saved)
                        fullMask = true;
                    else fullMask = false;
                }
                if (writesVector)
                {
                    var written = Define(instruction, vector);
                    if (fullMask) values = written;
                    else if (values is null || written is null) values = null;
                    else values = new(Inputs: [values, written]);
                }
            }
            return values;
        }

        private bool TryFindFullSavedMaskOverwrite(Gen5Operand vector, Gen5Operand saved,
            int restoreIndex, out int overwrite)
        {
            overwrite = -1;
            var instructions = plan.Graph.Program.Instructions;
            var end = instructions[restoreIndex].Pc;
            for (var start = restoreIndex - 1; start >= 0; start--)
            {
                var restored = instructions[start];
                if (restored is not { Opcode: "SMovB64", Sources.Count: 1 } ||
                    !restored.Destinations.Contains(Gen5Operand.Scalar(126)) || restored.Sources[0] != saved) continue;
                if (instructions.Skip(start + 1).Take(restoreIndex - start - 1).Any(instruction =>
                    WritesSavedMask(instruction, saved) || instruction.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal))) continue;
                var candidate = -1;
                for (var index = start + 1; index < restoreIndex; index++)
                {
                    var instruction = instructions[index];
                    if (MayExpandExecution(instruction) || instruction.Opcode.Contains("Saveexec", StringComparison.Ordinal)) break;
                    // A retry before initialization must cross this restoration
                    // again. Incoming edges that bypass it are rejected below.
                    if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var retry) &&
                        retry > restored.Pc) break;
                    if (WritesRegister(instruction, vector)) candidate = index;
                }
                if (candidate < 0 || Define(instructions[candidate], vector) is null) continue;
                var definition = instructions[candidate].Pc;
                var bypass = instructions.Any(edge => Gen5IrBranchResolver.Instance.TryGetBranchTarget(edge, out var target) &&
                    ((target > restored.Pc && target <= definition && (edge.Pc < restored.Pc || edge.Pc >= end)) ||
                     (target > definition && target <= end && (edge.Pc < definition || edge.Pc >= end)) ||
                     (edge.Pc > definition && edge.Pc < end && target <= edge.Pc && target > restored.Pc)));
                if (bypass) continue;
                overwrite = candidate;
                return true;
            }
            return false;
        }

        private Expression? ReadAfterFullSavedMaskOverwrite(Gen5Operand vector, Gen5Operand saved,
            int overwrite, int restoreIndex)
        {
            var instructions = plan.Graph.Program.Instructions;
            var values = Define(instructions[overwrite], vector);
            for (var index = overwrite + 1; index < restoreIndex; index++)
            {
                var instruction = instructions[index];
                if (MayExpandExecution(instruction))
                {
                    var restoresSavedMask = instruction is { Opcode: "SMovB64", Sources.Count: 1 } &&
                        instruction.Destinations.Contains(Gen5Operand.Scalar(126)) && instruction.Sources[0] == saved;
                    if (!restoresSavedMask && !instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal)) return null;
                }
                if (!WritesRegister(instruction, vector)) continue;
                var written = Define(instruction, vector);
                // Later branches may bypass a restore. Keep all possible earlier values
                // rather than assuming a subsequent write replaces every saved lane.
                if (values is null || written is null) values = null;
                else values = new(Inputs: [values, written]);
            }
            return values;
        }

        internal static bool WritesSavedMask(Gen5ShaderInstruction instruction, Gen5Operand saved)
        {
            // RDNA2 CMPX updates EXEC only, leaving the explicit condition SGPRs intact.
            if (instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal) && saved.Value < 126) return false;
            var pairEnd = saved.Value + 1;
            var width = instruction.Opcode.Contains("64", StringComparison.Ordinal) ? 2u : 1u;
            if (instruction.Destinations.Any(destination => destination.Kind == Gen5OperandKind.ScalarRegister &&
                destination.Value <= pairEnd && destination.Value + width > saved.Value)) return true;
            if (instruction.Control is Gen5Vop3Control { ScalarDestination: { } vop } && vop <= pairEnd && vop + 1 >= saved.Value ||
                instruction.Control is Gen5SdwaControl { ScalarDestination: { } sdwa } && sdwa <= pairEnd && sdwa + 1 >= saved.Value)
                return !instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal);
            return saved.Value is 106 or 107 && instruction.Opcode.StartsWith('V') &&
                (instruction.Opcode.StartsWith("VCmp", StringComparison.Ordinal) && !instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal) ||
                 instruction.Opcode.Contains("Co", StringComparison.Ordinal) || instruction.Opcode.Contains("Vcc", StringComparison.OrdinalIgnoreCase));
        }

        internal static bool MayExpandExecution(Gen5ShaderInstruction instruction)
        {
            if (instruction.Opcode is "SAndSaveexecB64" or "SAndSaveexecB32") return false;
            return instruction.Opcode.Contains("Saveexec", StringComparison.Ordinal) ||
                instruction.Opcode.Contains("Wrexec", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal) ||
                WritesRegister(instruction, Gen5Operand.Scalar(126)) || WritesRegister(instruction, Gen5Operand.Scalar(127));
        }

        internal static bool WritesRegister(Gen5ShaderInstruction instruction, Gen5Operand register)
        {
            if (register.Kind == Gen5OperandKind.ScalarRegister && register.Value is 106 or 107 &&
                instruction.Opcode.StartsWith('V')) return true;
            if (instruction.Control is Gen5Vop3Control { ScalarDestination: { } scalarDestination } &&
                register.Kind == Gen5OperandKind.ScalarRegister && register.Value >= scalarDestination && register.Value - scalarDestination < 2) return true;
            return instruction.Destinations.Any(destination => destination == register ||
                (destination.Kind == register.Kind && instruction.Opcode.Contains("64", StringComparison.Ordinal) &&
                    register.Value > destination.Value && register.Value - destination.Value == 1));
        }
    }
}
