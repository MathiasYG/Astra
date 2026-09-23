// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Ir;

namespace SharpEmu.ShaderCompiler.Resources;

public sealed record ResourceBranchBlock(ScalarValue? Condition, int[] Successors, uint[] Sources)
{
    internal static IReadOnlyList<ResourceBranchBlock> Build(
        ShaderResourcePlan plan,
        Func<ScalarValue, ScalarValue> rewrite,
        out bool canPruneResourceSources)
    {
        // A shader write can invalidate host-side resource pruning, but it does not prevent
        // resolving a predecessor when the branch predicate is independent of guest memory.
        canPruneResourceSources = !plan.Memory.Entries.Any(access => access.Access is MemoryAccess.Write or MemoryAccess.Atomic);
        var instructions = plan.Graph.Program.Instructions;
        if (instructions.Any(instruction => instruction.Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64" or
            "SCbranchJoin" or "SCbranchIFork" or "SCbranchGFork"))
        {
            canPruneResourceSources = false;
            return [];
        }

        var flow = plan.Graph.ControlFlow;
        var blocks = new ResourceBranchBlock[flow.Blocks.Count];
        var hasCondition = false;
        for (var blockIndex = 0; blockIndex < blocks.Length; blockIndex++)
        {
            var range = flow.Blocks[blockIndex];
            var last = instructions.LastOrDefault(instruction => instruction.Pc >= range.StartPc && instruction.Pc < range.EndPc);
            if (last is null) return [];
            var successors = flow.Successors[blockIndex].ToArray();
            ScalarValue? condition = null;
            var resolver = Gen5IrBranchResolver.Instance;
            if (resolver.IsConditional(last) || Gen5IrBranchResolver.IsUnconditionalBranch(last))
            {
                if (!resolver.TryGetBranchTarget(last, out var target) || !instructions.Any(instruction => instruction.Pc == target)) return [];
                if (resolver.IsConditional(last))
                {
                    if (blockIndex + 1 >= blocks.Length) return [];
                    // Keep taken and fallthrough edges separate even when both name the same block.
                    successors = [flow.BlockByStartPc[target], blockIndex + 1];
                    if (plan.Graph.BranchConditions.TryGetValue(last.Pc, out var original))
                    {
                        var candidate = rewrite(original);
                        if (plan.ValidateRuntimeValue(candidate) &&
                            (canPruneResourceSources || !DependsOnGuestMemory(candidate)))
                        {
                            condition = candidate;
                        }
                    }
                }
            }

            var sources = new HashSet<uint>();
            foreach (var access in plan.Memory.Entries)
            {
                if (access.Pc < range.StartPc || access.Pc >= range.EndPc || access.PlanningOnly) continue;
                if (access.Kind is MemoryResourceKind.Buffer or MemoryResourceKind.ScalarBuffer && access.Resource < plan.Info.Buffers.Count)
                    sources.Add(plan.Info.Buffers[(int)access.Resource].Source);
                if (access.Kind == MemoryResourceKind.Image && access.Resource < plan.Info.Images.Count)
                {
                    sources.Add(plan.Info.Images[(int)access.Resource].Source);
                    if (access.NeedsSampler && access.Sampler < plan.Info.Samplers.Count)
                        sources.Add(plan.Info.Samplers[(int)access.Sampler].Source);
                }
            }

            blocks[blockIndex] = new ResourceBranchBlock(condition, successors, sources.Order().ToArray());
            hasCondition |= condition is not null;
        }

        if (!hasCondition)
        {
            canPruneResourceSources = false;
            return [];
        }

        return blocks;
    }

    private static bool DependsOnGuestMemory(ScalarValue value)
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

            if (current.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord or ScalarValueKind.ResourceTableWord)
            {
                return true;
            }

            foreach (var operand in current.Operands)
            {
                pending.Push(operand);
            }
        }

        return false;
    }
}
