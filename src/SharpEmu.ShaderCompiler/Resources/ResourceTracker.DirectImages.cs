// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

public sealed partial class ResourceTracker
{
    private bool TryMakeDirectImage(ScalarValue handle, out IndirectImagePlan plan)
    {
        plan = null!;
        if (handle.Kind != ScalarValueKind.ImageHandle || handle.Operands.Length != 8)
            return false;

        var reads = handle.Operands;
        var memoryIndices = new int[8];
        var canSuppressMemoryReads = true;
        for (var component = 0; component < reads.Length; component++)
        {
            var read = reads[component];
            if (read.Kind != ScalarValueKind.ScalarAddressWord || read.MemoryIndex < 0 ||
                read.MemoryIndex >= _plan.Memory.Count || !MemoryIndexBelongsTo(read.MemoryIndex, read) ||
                !UsesOnly(read, [handle])) return false;
            var memory = _plan.Memory[read.MemoryIndex];
            if (memory.Kind != MemoryResourceKind.ScalarAddress || memory.DataBits != 32 || memory.DataDwords != 1)
                return false;
            canSuppressMemoryReads &= HasOnlyImageConsumers(memory, handle);
            memoryIndices[component] = read.MemoryIndex;
        }

        var keyRead = reads[0];
        var keyMemory = _plan.Memory[keyRead.MemoryIndex];
        var keyInstruction = _graph.Program.Instructions.First(instruction => instruction.Pc == keyMemory.Pc);
        if (keyMemory.ComponentIndex != 0 || keyInstruction.Control is not Gen5ScalarMemoryControl { DynamicOffsetRegister: not null })
            return false;
        var block = _graph.ControlFlow.Blocks.First(candidate => keyMemory.Pc >= candidate.StartPc && keyMemory.Pc < candidate.EndPc);
        if (memoryIndices.Any(index => _plan.Memory[index].Pc < block.StartPc || _plan.Memory[index].Pc >= block.EndPc))
            return false;

        // Include the all-zero input result unless the scan's incoming edge proves it cannot occur.
        var pending = new Stack<ScalarValue>(reads.Select(read => read.Operands[1]));
        var visited = new HashSet<ScalarValue>();
        ScalarValue? selector = null;
        while (pending.TryPop(out var value))
        {
            if (!visited.Add(value)) continue;
            if (value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.FindLowestBit32)
            {
                if (selector is not null && !ReferenceEquals(selector, value)) return false;
                selector = value;
                continue;
            }
            if (!value.IsConstant && value.Kind != ScalarValueKind.Operation) return false;
            foreach (var operand in value.Operands) pending.Push(operand);
        }
        if (selector is null) return false;

        var candidates = new List<DirectImageCandidate>();
        var sources = new List<DescriptorSource>();
        var keys = new HashSet<uint>();
        var results = Enumerable.Range(0, 32).Select(index => (uint)index);
        if (!_graph.HasNonZeroBitScanInput(selector)) results = results.Append(uint.MaxValue);
        foreach (var result in results)
        {
            var replacements = new Dictionary<ScalarValue, ScalarValue> { [selector] = _graph.Constant(result) };
            var memo = new Dictionary<ScalarValue, ScalarValue>();
            var key = _graph.Substitute(keyRead.Operands[1], replacements, memo);
            if (!key.IsConstant || key.Type != ScalarValueType.U32 || !keys.Add(key.ConstantU32)) return false;
            var source = new DescriptorSource
            {
                Dwords = reads.Select(read => _graph.Substitute(read, replacements, memo)).ToArray(),
            };
            if (!ValidateSource(source, out _)) return false;
            sources.Add(source);
            candidates.Add(new DirectImageCandidate(key.ConstantU32, 0));
        }

        for (var index = 0; index < candidates.Count; index++)
            candidates[index] = candidates[index] with { Source = InternSource(sources[index]) };
        var imageSource = new DescriptorSource
        {
            Dwords = sources[0].Dwords,
            IndirectImage = new IndirectImageSelector(0, 0, 0, 0, 0) { DirectCandidates = candidates },
        };
        plan = new IndirectImagePlan
        {
            Handle = handle,
            Source = InternSource(imageSource),
            Key = keyRead,
            KeyIsAddressOffset = true,
            SuppressMemoryReads = canSuppressMemoryReads,
            Memory = memoryIndices,
            Reads = reads,
        };
        return true;
    }

    // Resource-graph uses omit ordinary shader arithmetic; check those reads before removing a load.
    private bool HasOnlyImageConsumers(MemoryAccessInfo memory, ScalarValue handle)
    {
        var instructions = _graph.Program.Instructions;
        var load = instructions.First(instruction => instruction.Pc == memory.Pc);
        if (memory.ComponentIndex >= load.Destinations.Count) return false;
        var destination = load.Destinations[(int)memory.ComponentIndex];
        if (destination.Kind != Gen5OperandKind.ScalarRegister || destination.Value >= 106) return false;
        var flow = _graph.ControlFlow;
        var initialBlock = Enumerable.Range(0, flow.Blocks.Count)
            .First(index => load.Pc >= flow.Blocks[index].StartPc && load.Pc < flow.Blocks[index].EndPc);
        var pending = new Queue<(int Block, uint Start)>();
        var visited = new HashSet<(int Block, uint Start)>();
        pending.Enqueue((initialBlock, load.Pc + (uint)load.Words.Count * sizeof(uint)));
        while (pending.TryDequeue(out var position))
        {
            if (!visited.Add(position)) continue;
            var block = flow.Blocks[position.Block];
            var overwritten = false;
            foreach (var instruction in instructions.Where(instruction => instruction.Pc >= position.Start && instruction.Pc < block.EndPc))
            {
                // RDNA2 V_MOVREL* indexes only VGPRs through M0. It cannot
                // alias the SGPR descriptor being checked here. Its explicit
                // scalar operands are still checked by ReadsScalar below.
                var vectorRelativeMove = instruction.Encoding == Gen5ShaderEncoding.Vop1 &&
                    instruction.Opcode is "VMovrelsB32" or "VMovreldB32" or "VMovrelsdB32" or "VMovrelsd2B32";
                if (instruction.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) && !vectorRelativeMove ||
                    instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal)) return false;
                var allowedImage = instruction.Control is Gen5ImageControl &&
                    _plan.Memory.TryGetIndex(instruction.Pc, 0, out var imageIndex) &&
                    ReferenceEquals(_plan.Accesses[imageIndex]?.Handle, handle);
                if (!allowedImage && (ReadsScalar(instruction, destination.Value) ||
                    (instruction.Encoding == Gen5ShaderEncoding.Sopk && instruction.Opcode is "SAddkI32" or "SMulkI32" &&
                        instruction.Destinations.Contains(destination)))) return false;
                overwritten = instruction.Destinations.Any(target => target.Kind == Gen5OperandKind.ScalarRegister &&
                    (target == destination || (instruction.Opcode.Contains("64", StringComparison.Ordinal) && target.Value + 1 == destination.Value))) ||
                    instruction.Control is Gen5Vop3Control { ScalarDestination: { } scalarDestination } &&
                        destination.Value >= scalarDestination && destination.Value - scalarDestination < 2 ||
                    instruction.Control is Gen5SdwaControl { ScalarDestination: { } compareDestination } &&
                        destination.Value >= compareDestination && destination.Value - compareDestination < 2;
                if (overwritten) break;
            }
            if (!overwritten)
                foreach (var successor in flow.Successors[position.Block])
                    pending.Enqueue((successor, flow.Blocks[successor].StartPc));
        }
        return true;
    }

    private static bool ReadsScalar(Gen5ShaderInstruction instruction, uint register)
    {
        bool InRange(uint first, uint width) => register >= first && register - first < width;
        var width = instruction.Opcode.Contains("64", StringComparison.Ordinal) &&
            instruction.Opcode != "SBitreplicateB64B32" ? 2u : 1u;
        if (instruction.Sources.Any(source => source.Kind == Gen5OperandKind.ScalarRegister && InRange(source.Value, width)))
            return true;
        return instruction.Control switch
        {
            Gen5ImageControl image => InRange(image.ScalarResource, 8) || InRange(image.ScalarSampler, 4),
            Gen5BufferMemoryControl buffer => InRange(buffer.ScalarResource, 4),
            Gen5GlobalMemoryControl global => InRange(global.ScalarAddress, 2),
            Gen5ScalarMemoryControl scalar =>
                instruction.Sources.Count > 0 && InRange(instruction.Sources[0].Value,
                    instruction.Opcode.StartsWith("SBuffer", StringComparison.Ordinal) ? 4u : 2u) ||
                scalar.DynamicOffsetRegister == register,
            _ => false,
        };
    }
}

// Proves a finite descriptor-index domain from a uniform comparison that clears
// EXEC before an image instruction. Unknown control flow or an EXEC expansion
// declines the proof; the normal strict resource-planning failure then applies.
internal static class CountedImageBound
{
    public static bool TryGet(ShaderResourcePlan plan, ScalarValue key, uint imagePc, uint firstReadPc, out ScalarValue bound)
    {
        bound = null!;
        var graph = plan.Graph;
        foreach (var (comparePc, condition) in graph.ScalarCompareConditions)
        {
            if (condition.Kind != ScalarValueKind.Operation ||
                condition.Operation is not (ScalarOperation.ULessThan32 or ScalarOperation.SLessThan32) ||
                condition.Operands.Length != 2 ||
                !(graph.Equivalent(condition.Operands[0], key) ||
                  comparePc > imagePc && IsIncrementOfKey(graph, condition.Operands[0], key)))
                continue;

            var candidate = graph.ResolveInvariantPhi(condition.Operands[1]);
            if (candidate is null || !plan.ValidateRuntimeValue(candidate) ||
                (condition.Operation == ScalarOperation.SLessThan32 && !IsZeroBasedIncrementing(key)) ||
                !(comparePc < imagePc &&
                    (ProvesGuard(graph, comparePc, imagePc, firstReadPc) ||
                     ProvesScalarBranchGuard(graph, comparePc, imagePc, firstReadPc)) ||
                  comparePc > imagePc && ProvesPostTestLoopGuard(graph, key, candidate,
                      comparePc, imagePc, firstReadPc)))
                continue;

            bound = candidate;
            return true;
        }
        return false;
    }

    // A do-while loop reads index zero once, increments by one, and branches
    // back only while the incremented index is below a positive fixed bound.
    // Removing the latch block must leave no path back to the descriptor load.
    private static bool ProvesPostTestLoopGuard(
        ScalarValueGraph graph, ScalarValue key, ScalarValue bound, uint comparePc, uint imagePc, uint firstReadPc)
    {
        if (!bound.IsConstant || bound.ConstantU32 is 0 or > 4096 ||
            !IsZeroBasedIncrementing(key)) return false;

        var instructions = graph.Program.Instructions;
        var compareIndex = instructions.ToList().FindIndex(instruction => instruction.Pc == comparePc);
        if (compareIndex < 0 || compareIndex + 1 >= instructions.Count) return false;
        var branch = instructions[compareIndex + 1];
        if (branch.Opcode != "SCbranchScc1") return false;
        var targetPc = unchecked((uint)((long)branch.Pc + 4 +
            (long)unchecked((short)(branch.Words[0] & 0xFFFF)) * 4));
        if (targetPc > firstReadPc) return false;

        var flow = graph.ControlFlow;
        var readBlock = FindBlock(flow, firstReadPc);
        var imageBlock = FindBlock(flow, imagePc);
        var compareBlock = FindBlock(flow, comparePc);
        var targetBlock = FindBlock(flow, targetPc);
        var exitBlock = FindBlock(flow, branch.Pc + 4);
        if (readBlock < 0 || imageBlock < 0 || compareBlock < 0 || targetBlock < 0 ||
            exitBlock < 0 ||
            !Dominates(flow, readBlock, imageBlock) ||
            !Dominates(flow, targetBlock, readBlock) ||
            branch.Pc <= imagePc ||
            ReachableWithoutGuard(flow, exitBlock, readBlock, compareBlock)) return false;

        foreach (var successor in flow.Successors[imageBlock])
        {
            if (ReachableWithoutGuard(flow, successor, readBlock, compareBlock)) return false;
        }
        return true;
    }

    // A scalar branch on SCC=0 skips the descriptor loads and the image access.
    // For signed comparisons, the zero-based unit increment proves the key
    // cannot become negative before a bounded iteration count is reached.
    private static bool ProvesScalarBranchGuard(ScalarValueGraph graph, uint comparePc, uint imagePc, uint firstReadPc)
    {
        var flow = graph.ControlFlow;
        var compareBlock = FindBlock(flow, comparePc);
        var readBlock = FindBlock(flow, firstReadPc);
        var imageBlock = FindBlock(flow, imagePc);
        if (compareBlock < 0 || readBlock < 0 || imageBlock < 0 ||
            !Dominates(flow, compareBlock, readBlock) || !Dominates(flow, compareBlock, imageBlock))
            return false;

        var instructions = graph.Program.Instructions;
        var compareIndex = instructions.ToList().FindIndex(instruction => instruction.Pc == comparePc);
        if (compareIndex < 0 || compareIndex + 1 >= instructions.Count)
            return false;
        var branch = instructions[compareIndex + 1];
        if (branch.Opcode != "SCbranchScc0" || FindBlock(flow, branch.Pc) != compareBlock ||
            branch.Pc >= firstReadPc)
            return false;

        var displacement = unchecked((short)(branch.Words[0] & 0xFFFF));
        var targetPc = unchecked((uint)((long)branch.Pc + 4 + (long)displacement * 4));
        var targetBlock = FindBlock(flow, targetPc);
        return targetPc > imagePc && targetBlock >= 0 &&
            !ReachableWithoutGuard(flow, targetBlock, readBlock, compareBlock) &&
            !ReachableWithoutGuard(flow, targetBlock, imageBlock, compareBlock);
    }

    private static bool IsZeroBasedIncrementing(ScalarValue key)
    {
        if (key.Kind != ScalarValueKind.Phi || key.Operands.Length != 2)
            return false;
        var start = key.Operands[0];
        var next = key.Operands[1];
        if (!start.IsConstant || start.ConstantU32 != 0 ||
            next.Kind != ScalarValueKind.Operation || next.Operation != ScalarOperation.IAdd32 ||
            next.Operands.Length != 2)
            return false;
        return (ReferenceEquals(CollapseSelfPhi(next.Operands[0]), key) && next.Operands[1].IsConstant && next.Operands[1].ConstantU32 == 1) ||
            (ReferenceEquals(CollapseSelfPhi(next.Operands[1]), key) && next.Operands[0].IsConstant && next.Operands[0].ConstantU32 == 1);
    }

    private static bool IsIncrementOfKey(ScalarValueGraph graph, ScalarValue value, ScalarValue key) =>
        value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.IAdd32 &&
        value.Operands.Length == 2 &&
        ((graph.Equivalent(CollapseSelfPhi(value.Operands[0]), key) && value.Operands[1].IsConstant && value.Operands[1].ConstantU32 == 1) ||
         (graph.Equivalent(CollapseSelfPhi(value.Operands[1]), key) && value.Operands[0].IsConstant && value.Operands[0].ConstantU32 == 1));

    private static ScalarValue CollapseSelfPhi(ScalarValue value)
    {
        while (value.Kind == ScalarValueKind.Phi && value.Operands.Length == 2)
        {
            if (ReferenceEquals(value.Operands[0], value)) value = value.Operands[1];
            else if (ReferenceEquals(value.Operands[1], value)) value = value.Operands[0];
            else break;
        }
        return value;
    }

    private static bool ProvesGuard(ScalarValueGraph graph, uint comparePc, uint imagePc, uint firstReadPc)
    {
        var flow = graph.ControlFlow;
        var compareBlock = FindBlock(flow, comparePc);
        var imageBlock = FindBlock(flow, imagePc);
        if (compareBlock < 0 || imageBlock < 0 ||
            !Dominates(flow, compareBlock, imageBlock)) return false;

        var block = flow.Blocks[compareBlock];
        var instructions = graph.Program.Instructions;
        var compareIndex = instructions.ToList().FindIndex(instruction => instruction.Pc == comparePc);
        if (compareIndex < 0) return false;
        var selector = -1;
        for (var index = compareIndex + 1; index < instructions.Count && instructions[index].Pc < block.EndPc; index++)
        {
            var instruction = instructions[index];
            if (instruction.Opcode == "SCselectB64" && instruction.Sources.Count == 2 &&
                instruction.Destinations.Count == 1 && instruction.Destinations[0].Kind == Gen5OperandKind.ScalarRegister &&
                instruction.Sources[1].Kind == Gen5OperandKind.EncodedConstant &&
                Gen5InlineConstants.TryDecode(instruction.Sources[1].Value, out var zero) && zero == 0)
            {
                selector = index;
                break;
            }
            if (WritesScc(instruction)) return false;
        }
        if (selector < 0) return false;

        var selected = instructions[selector].Destinations[0].Value;
        for (var index = selector + 1; index < instructions.Count && instructions[index].Pc < block.EndPc; index++)
        {
            var instruction = instructions[index];
            if (instruction.Opcode == "SMovB64" && instruction.Destinations.Count == 1 &&
                instruction.Destinations[0] == Gen5Operand.Scalar(126) &&
                instruction.Sources.Count == 1 && instruction.Sources[0] == Gen5Operand.Scalar(selected))
            {
                var afterGuard = instruction.Pc + (uint)instruction.Words.Count * sizeof(uint);
                return ZeroMaskPersists(graph, compareBlock, afterGuard, imagePc) &&
                    ZeroMaskSkipsScalarReads(graph, compareBlock, afterGuard, firstReadPc);
            }
            if (WritesScalarPair(instruction, selected) || MayExpandExec(instruction)) return false;
        }
        return false;
    }

    // Scalar memory instructions ignore EXEC. A zero mask is sufficient for the
    // vector image operation only when an EXECZ branch bypasses its descriptor
    // loads as well. Re-entering the comparison may establish a new mask.
    private static bool ZeroMaskSkipsScalarReads(
        ScalarValueGraph graph, int guardBlock, uint afterGuard, uint firstReadPc)
    {
        if (firstReadPc < afterGuard) return false;
        var flow = graph.ControlFlow;
        var readBlock = FindBlock(flow, firstReadPc);
        if (readBlock < 0) return false;
        foreach (var instruction in graph.Program.Instructions)
        {
            if (instruction.Pc < afterGuard || instruction.Pc >= firstReadPc ||
                instruction.Opcode != "SCbranchExecz") continue;
            var branchBlock = FindBlock(flow, instruction.Pc);
            if (branchBlock < 0 || !Dominates(flow, branchBlock, readBlock) ||
                (branchBlock == guardBlock && instruction.Pc < afterGuard)) continue;
            var displacement = unchecked((short)(instruction.Words[0] & 0xFFFF));
            var targetPc = unchecked((uint)((long)instruction.Pc + 4 + (long)displacement * 4));
            var targetBlock = FindBlock(flow, targetPc);
            if (targetBlock >= 0 && targetBlock != guardBlock &&
                !ReachableWithoutGuard(flow, targetBlock, readBlock, guardBlock))
                return true;
        }
        return false;
    }

    private static bool ReachableWithoutGuard(
        SharpEmu.ShaderCompiler.Ir.IrControlFlowGraph flow, int start, int target, int guard)
    {
        var pending = new Stack<int>();
        var visited = new HashSet<int>();
        pending.Push(start);
        while (pending.TryPop(out var block))
        {
            if (block == guard || !visited.Add(block)) continue;
            if (block == target) return true;
            foreach (var successor in flow.Successors[block]) pending.Push(successor);
        }
        return false;
    }

    private static bool ZeroMaskPersists(ScalarValueGraph graph, int guardBlock, uint afterGuard, uint imagePc)
    {
        var flow = graph.ControlFlow;
        var pending = new Stack<(int Block, uint Start)>();
        var visited = new HashSet<(int Block, uint Start)>();
        var imageBlock = FindBlock(flow, imagePc);
        var canReachImage = new HashSet<int>();
        var reverse = new Stack<int>();
        reverse.Push(imageBlock);
        while (reverse.TryPop(out var block))
        {
            if (!canReachImage.Add(block) || block == guardBlock) continue;
            foreach (var predecessor in flow.Predecessors[block]) reverse.Push(predecessor);
        }
        pending.Push((guardBlock, afterGuard));
        var foundImage = false;
        while (pending.TryPop(out var position))
        {
            if (!visited.Add(position)) continue;
            if (position.Block == guardBlock && position.Start == flow.Blocks[guardBlock].StartPc) continue;
            if (!canReachImage.Contains(position.Block)) continue;
            var range = flow.Blocks[position.Block];
            var reachedImage = false;
            foreach (var instruction in graph.Program.Instructions)
            {
                if (instruction.Pc < position.Start || instruction.Pc >= range.EndPc) continue;
                if (instruction.Pc == imagePc)
                {
                    foundImage = true;
                    reachedImage = true;
                    break;
                }
                if (MayExpandExec(instruction)) return false;
            }
            if (reachedImage) continue;
            foreach (var successor in flow.Successors[position.Block])
                pending.Push((successor, flow.Blocks[successor].StartPc));
        }
        return foundImage;
    }

    private static int FindBlock(SharpEmu.ShaderCompiler.Ir.IrControlFlowGraph flow, uint pc) =>
        Enumerable.Range(0, flow.Blocks.Count).FirstOrDefault(index =>
            pc >= flow.Blocks[index].StartPc && pc < flow.Blocks[index].EndPc, -1);

    private static bool Dominates(SharpEmu.ShaderCompiler.Ir.IrControlFlowGraph flow, int guard, int target)
    {
        if (guard == 0) return true;
        var pending = new Stack<int>();
        var visited = new HashSet<int>();
        pending.Push(0);
        while (pending.TryPop(out var block))
        {
            if (block == guard || !visited.Add(block)) continue;
            if (block == target) return false;
            foreach (var successor in flow.Successors[block]) pending.Push(successor);
        }
        return true;
    }

    private static bool WritesScc(Gen5ShaderInstruction instruction) =>
        instruction.Encoding is Gen5ShaderEncoding.Sopc or Gen5ShaderEncoding.Sop2 or Gen5ShaderEncoding.Sopk ||
        instruction.Opcode.Contains("Saveexec", StringComparison.Ordinal);

    private static bool WritesScalarPair(Gen5ShaderInstruction instruction, uint first) =>
        instruction.Destinations.Any(destination => destination.Kind == Gen5OperandKind.ScalarRegister &&
            (destination.Value == first || destination.Value + 1 == first || destination.Value == first + 1));

    private static bool MayExpandExec(Gen5ShaderInstruction instruction)
    {
        if (instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal)) return false;
        if (instruction.Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64" ||
            instruction.Opcode.Contains("Wrexec", StringComparison.Ordinal) ||
            instruction.Opcode.Contains("Saveexec", StringComparison.Ordinal)) return true;
        if (instruction.Control is Gen5Vop3Control { ScalarDestination: { } scalarDestination } &&
            scalarDestination <= 127 && scalarDestination + 1 >= 126) return true;
        return WritesScalarPair(instruction, 126);
    }
}

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
