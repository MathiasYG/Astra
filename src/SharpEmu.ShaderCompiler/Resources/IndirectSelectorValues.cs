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

        private static bool MayExpandExecution(Gen5ShaderInstruction instruction)
        {
            if (instruction.Opcode is "SAndSaveexecB64" or "SAndSaveexecB32") return false;
            return instruction.Opcode.Contains("Saveexec", StringComparison.Ordinal) ||
                instruction.Opcode.Contains("Wrexec", StringComparison.Ordinal) ||
                instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal) ||
                WritesRegister(instruction, Gen5Operand.Scalar(126)) || WritesRegister(instruction, Gen5Operand.Scalar(127));
        }

        private static bool WritesRegister(Gen5ShaderInstruction instruction, Gen5Operand register)
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
