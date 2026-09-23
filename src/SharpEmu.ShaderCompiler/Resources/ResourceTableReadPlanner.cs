// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;

namespace SharpEmu.ShaderCompiler.Resources;

// Collects scalar reads reachable from descriptor handles. Host-evaluable reads with
// immediate offsets receive compact flattened-table slots; dynamic or unresolved
// addresses stay explicit for shader-side execution.
public sealed class ResourceTableReadPlanner
{
    private readonly ScalarValueGraph _graph;
    private readonly ShaderStage _stage;
    private readonly ulong _hash;
    private readonly List<ResourceTableRead> _reads = [];
    private readonly List<ScalarValue> _dynamicReads = [];
    private readonly List<(ScalarValue Read, uint Slot)> _patches = [];
    private readonly Dictionary<int, uint> _flattenedSlotByMemoryIndex = [];
    private readonly HashSet<int> _explicitAddressReadMemoryIndices = [];
    private readonly List<ScalarValue> _visiting = [];
    private readonly HashSet<ScalarValue> _visited = [];
    private readonly Dictionary<ScalarValue, ScalarValue> _replacements = [];

    private ResourceTableReadPlanner(ScalarValueGraph graph, ShaderStage stage, ulong hash)
    {
        _graph = graph;
        _stage = stage;
        _hash = hash;
    }

    public sealed record Result(
        IReadOnlyList<ResourceTableRead> Reads,
        IReadOnlyList<ScalarValue> DynamicReads,
        IReadOnlyDictionary<ScalarValue, ScalarValue> Replacements,
        IReadOnlyDictionary<int, uint> FlattenedSlotByMemoryIndex);

    public static Result Plan(ScalarValueGraph graph, ShaderStage stage, ulong hash) =>
        new ResourceTableReadPlanner(graph, stage, hash).Run();

    private Result Run()
    {
        foreach (var value in _graph.Values)
        {
            if (value.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord &&
                value.MemoryIndex < _graph.Memory.Count)
            {
                var kind = _graph.Memory[value.MemoryIndex].Kind;
                var crosswired = (value.Kind == ScalarValueKind.ScalarAddressWord && kind == MemoryResourceKind.ScalarBuffer) ||
                    (value.Kind == ScalarValueKind.ScalarBufferWord && kind == MemoryResourceKind.ScalarAddress);
                if (crosswired)
                {
                    Fail(_graph.Memory[value.MemoryIndex].Pc, $"{value.Kind} has incompatible scalar memory metadata");
                }
            }

            // A scalar-address read is only safe to flatten when its address is
            // available to the host at resource-materialization time. In
            // particular, a loop-carried address may be a valid run-time shader
            // value while having no single host-evaluable value. Make the choice
            // per memory instruction, since one instruction can have multiple
            // graph nodes from different data-flow iterations. A non-invariant
            // phi is not a host-evaluable address even when its defined arms are
            // individually uniform: materialization has no guest predecessor for
            // a loop-carried or otherwise ambiguous merge.
            if (value.Kind == ScalarValueKind.ScalarAddressWord && IsRawRead(value) &&
                value.Operands.Length >= 2 && value.Operands[1].IsConstant &&
                (!new RuntimeValueValidator(_graph, _graph.UserDataBase, _graph.UserDataCount, 0).Validate(value) ||
                 ContainsNonInvariantPhi(value.Operands[0]) || ContainsNonInvariantPhi(value.Operands[1])))
            {
                _explicitAddressReadMemoryIndices.Add(value.MemoryIndex);
            }
        }

        foreach (var access in _graph.Accesses)
        {
            if (access?.Handle is { } handle)
            {
                foreach (var dword in handle.Operands)
                {
                    Collect(dword, 0);
                }
            }

            if (access?.SamplerHandle is { } sampler)
            {
                foreach (var dword in sampler.Operands)
                {
                    Collect(dword, 0);
                }
            }
        }

        var validatorReadCount = _reads.Count;
        foreach (var access in _graph.Accesses)
        {
            if (access?.Read is { Kind: ScalarValueKind.ScalarAddressWord } read &&
                IsRawRead(read) && read.Operands[1].IsConstant &&
                new RuntimeValueValidator(_graph, _graph.UserDataBase, _graph.UserDataCount, validatorReadCount).Validate(read))
            {
                Collect(read, _graph.Memory[read.MemoryIndex].Pc);
            }
        }

        PatchReads();
        return new Result(_reads, _dynamicReads, _replacements, _flattenedSlotByMemoryIndex);
    }

    private void Fail(uint pc, string message) =>
        throw new ResourcePlanException(
            $"shader resource-table planning failed: hash=0x{_hash:X16} stage={_stage} pc=0x{pc:X8} {message}");

    private bool IsRawRead(ScalarValue value) =>
        new RuntimeValueValidator(_graph, _graph.UserDataBase, _graph.UserDataCount, 0).IsRawRead(value);

    private bool ContainsNonInvariantPhi(ScalarValue root)
    {
        var pending = new Stack<ScalarValue>();
        var visited = new HashSet<ScalarValue>();
        pending.Push(root);
        while (pending.Count != 0)
        {
            var current = pending.Pop();
            if (!visited.Add(current))
            {
                continue;
            }

            if (current.Kind == ScalarValueKind.Phi && _graph.ResolveInvariantPhi(current) is null)
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

    private void Collect(ScalarValue value, uint usePc)
    {
        if (value.IsConstant)
        {
            return;
        }

        var cycle = _visiting.IndexOf(value);
        if (cycle >= 0)
        {
            if (_visiting.Skip(cycle).Any(pending => pending.Kind == ScalarValueKind.Phi))
            {
                return;
            }

            Fail(usePc, $"cyclic planning value {value.Kind} without a phi");
        }

        if (!_visited.Add(value))
        {
            return;
        }

        _visiting.Add(value);
        foreach (var operand in value.Operands)
        {
            Collect(operand, usePc);
        }

        _visiting.RemoveAt(_visiting.Count - 1);
        if (!IsRawRead(value))
        {
            return;
        }

        var offset = value.Operands[1];
        if (!offset.IsConstant)
        {
            AddDynamicRead(value);

            return;
        }

        if (value.Kind == ScalarValueKind.ScalarAddressWord &&
            _explicitAddressReadMemoryIndices.Contains(value.MemoryIndex))
        {
            AddDynamicRead(value);

            return;
        }

        for (var slot = 0; slot < _reads.Count; slot++)
        {
            if (_graph.Equivalent(value, _reads[slot].Value))
            {
                _patches.Add((value, (uint)slot));
                return;
            }
        }

        var newSlot = (uint)_reads.Count;
        _reads.Add(new ResourceTableRead(value, newSlot));
        _patches.Add((value, newSlot));
    }

    private void AddDynamicRead(ScalarValue value)
    {
        if (!_dynamicReads.Contains(value))
        {
            _dynamicReads.Add(value);
        }
    }

    // Equivalent reads share one host read, but each instruction must load its destination
    // from that slot instead of reading guest memory again on the device.
    private void PatchReads()
    {
        foreach (var (read, slot) in _patches)
        {
            _replacements[read] = _graph.ResourceTableWord(slot);
            if (read.MemoryIndex < _graph.Memory.Count)
            {
                _graph.Memory[read.MemoryIndex].PlanningOnly = true;
                _flattenedSlotByMemoryIndex.Add(read.MemoryIndex, slot);
            }
        }
    }
}
