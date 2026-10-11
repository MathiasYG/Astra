// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Gpu.Scheduling;

// A GPU feedback resource belongs to one submission until its completion callback
// has consumed it. Later submissions must never reuse its mapped storage before then.
//
// With a reset, a consumed resource is returned to its new state and kept for a later
// submission instead of being destroyed: one per submission was created and destroyed
// a thousand times a second. A failed read still destroys it.
internal sealed class TickOwnedResources<T>(Action<T>? reset = null) : IDisposable where T : class, IDisposable
{
    private const int MaxSpare = 8;
    private readonly Dictionary<ulong, T> _resources = [];
    private readonly Stack<T> _spare = new();
    private readonly object _gate = new();

    public T Acquire(ulong tick, Func<T> create)
    {
        lock (_gate)
        {
            if (!_resources.TryGetValue(tick, out var resource))
            {
                resource = _spare.TryPop(out var spare) ? spare : create();
                _resources.Add(tick, resource);
            }
            return resource;
        }
    }

    public void Complete(ulong tick, Action<T> consume)
    {
        lock (_gate)
        {
            if (!_resources.Remove(tick, out var resource)) return;
            var consumed = false;
            try
            {
                consume(resource);
                consumed = true;
            }
            finally
            {
                if (consumed && reset is not null && _spare.Count < MaxSpare)
                {
                    reset(resource);
                    _spare.Push(resource);
                }
                else
                {
                    resource.Dispose();
                }
            }
        }
    }

    // The caller must wait for GPU completion before disposing pending resources.
    public void Dispose()
    {
        lock (_gate)
        {
            foreach (var resource in _resources.Values) resource.Dispose();
            _resources.Clear();
            while (_spare.TryPop(out var spare)) spare.Dispose();
        }
    }
}
