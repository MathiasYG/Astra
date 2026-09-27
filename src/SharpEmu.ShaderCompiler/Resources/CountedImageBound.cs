// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

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
            if (comparePc >= imagePc || condition.Kind != ScalarValueKind.Operation ||
                condition.Operation != ScalarOperation.ULessThan32 || condition.Operands.Length != 2 ||
                !graph.Equivalent(condition.Operands[0], key))
                continue;

            var candidate = graph.ResolveInvariantPhi(condition.Operands[1]);
            if (candidate is null || !plan.ValidateRuntimeValue(candidate) ||
                !ProvesGuard(graph, comparePc, imagePc, firstReadPc))
                continue;

            bound = candidate;
            return true;
        }
        return false;
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
            if (targetBlock >= 0 && !ReachableWithoutGuard(flow, targetBlock, readBlock, guardBlock))
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
