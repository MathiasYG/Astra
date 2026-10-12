// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;

internal static unsafe partial class VulkanVideoPresenter
{
    private sealed partial class Presenter : IConcurrentResolveHost
    {
        // This partial serves the guest readers of program resolution on a thread resolving draws
        // ahead of the render thread. That thread never synchronizes GPU-owned memory: a word the
        // GPU owns abandons the resolution, which the render thread then does itself. Every
        // ownership answer it relied on is recorded and checked again when the result is taken.

        private enum ConcurrentQuery : byte
        {
            DirtyBytes,
            CleanWord,
            Resident,
            ResidentClean,
            // Bytes read without an ownership question: checked only against later GPU stores.
            Read,
        }

        private sealed class ConcurrentAssumptions
        {
            public long BufferVersion;
            public long WritebackEpoch;
            public List<(ConcurrentQuery Query, ulong Address, ulong Size, bool Answer)> Answers = new();
            public bool ReadsImageOwnership;

            public void Begin(long bufferVersion, long writebackEpoch)
            {
                BufferVersion = bufferVersion;
                WritebackEpoch = writebackEpoch;
                ReadsImageOwnership = false;
                Answers.Clear();
            }

            public void Record(ConcurrentQuery query, ulong address, ulong size, bool answer)
            {
                Answers.Add((query, address, size, answer));
                ReadsImageOwnership |= query is ConcurrentQuery.CleanWord or ConcurrentQuery.ResidentClean;
            }
        }

        // Set only on the resolving thread, around one resolution.
        [ThreadStatic]
        private static ConcurrentAssumptions? _concurrentAssumptions;

        // A resolution records one answer per guest read, so a new list per resolution regrew to
        // thousands of entries every draw. Released sets keep their capacity for the next one.
        // Slots rather than a locked stack: the resolving thread takes and the render thread
        // returns a set per draw, and the lock between them was contended (~1.5 % of the render
        // thread at ~6,000 draws per frame).
        private const int MaxSpareAssumptions = 16;
        private const int MaxRetainedAnswers = 1 << 16;
        private readonly ConcurrentAssumptions?[] _spareAssumptions = new ConcurrentAssumptions?[MaxSpareAssumptions];

        void IConcurrentResolveHost.BeginConcurrentResolve()
        {
            ConcurrentAssumptions? assumptions = null;
            for (var slot = 0; slot < _spareAssumptions.Length && assumptions is null; slot++)
            {
                if (Volatile.Read(ref _spareAssumptions[slot]) is not null)
                {
                    assumptions = Interlocked.Exchange(ref _spareAssumptions[slot], null);
                }
            }

            assumptions ??= new ConcurrentAssumptions();
            assumptions.Begin(_bufferCache.GpuModifiedVersion, GpuWritebackEpoch.Value);
            _concurrentAssumptions = assumptions;
        }

        void IConcurrentResolveHost.ReleaseAssumptions(object assumptionsObject)
        {
            var assumptions = (ConcurrentAssumptions)assumptionsObject;
            if (assumptions.Answers.Capacity > MaxRetainedAnswers)
            {
                assumptions.Answers = new();
            }

            assumptions.Answers.Clear();
            for (var slot = 0; slot < _spareAssumptions.Length; slot++)
            {
                if (Volatile.Read(ref _spareAssumptions[slot]) is null &&
                    Interlocked.CompareExchange(ref _spareAssumptions[slot], assumptions, null) is null)
                {
                    return;
                }
            }
        }

        object IConcurrentResolveHost.EndConcurrentResolve()
        {
            var assumptions = _concurrentAssumptions ?? throw new InvalidOperationException("No concurrent resolution is open.");
            _concurrentAssumptions = null;
            return assumptions;
        }

        // GPU data stored since into what the resolution read may have changed it; otherwise the
        // answers stand while the ownership they came from has not changed. Any store used to
        // void every resolution in flight: with a few dozen readbacks a frame, ~6 % of the
        // prefetched draws were resolved again, none of them because of what they had read.
        bool IConcurrentResolveHost.IsStillCurrent(object assumptionsObject)
        {
            var assumptions = (ConcurrentAssumptions)assumptionsObject;
            if (assumptions.WritebackEpoch != GpuWritebackEpoch.Value && !MissesWritebacksSince(assumptions))
            {
                return false;
            }

            if (assumptions.BufferVersion == _bufferCache.GpuModifiedVersion && !assumptions.ReadsImageOwnership)
            {
                return true;
            }

            foreach (var (query, address, size, answer) in assumptions.Answers)
            {
                if (query == ConcurrentQuery.Read)
                {
                    continue;
                }

                var now = query switch
                {
                    ConcurrentQuery.DirtyBytes => _bufferCache.HasGpuDirtyBytes(address, size),
                    ConcurrentQuery.CleanWord => IsGpuOwnedForCleanRead(address, size),
                    ConcurrentQuery.Resident => IsGpuOwnedForResidentRead(address, size, clean: false),
                    _ => IsGpuOwnedForResidentRead(address, size, clean: true),
                };
                if (now != answer)
                {
                    return false;
                }
            }

            return true;
        }

        private readonly List<(ulong Address, ulong Size)> _writebacksSince = new();

        private bool MissesWritebacksSince(ConcurrentAssumptions assumptions)
        {
            _writebacksSince.Clear();
            if (!GpuWritebackEpoch.TryCollectSince(assumptions.WritebackEpoch, _writebacksSince))
            {
                return false;
            }

            foreach (var (store, storeSize) in _writebacksSince)
            {
                foreach (var (_, address, size, _) in assumptions.Answers)
                {
                    if (address < store + storeSize && store < address + size)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        // Shader code the resolution hashed; a later store into it voids the resolution.
        void SharpEmu.Libs.Gpu.Pipelines.IShaderPipelineHost.NoteConcurrentRead(ulong address, ulong size) =>
            _concurrentAssumptions?.Record(ConcurrentQuery.Read, address, size, false);

        // An image whose contents went to its buffer view is caught by the buffer checks.
        private bool IsGpuOwnedForCleanRead(ulong address, ulong size) =>
            (_bufferCache.MayHaveGpuDirtyPages(address, size) && _bufferCache.HasGpuDirtyPages(address, size)) ||
            _bufferCache.HasGpuDirtyBytes(address, size) ||
            _imageCache.HasUnsynchronizedGpuImage(address, size);

        private bool IsGpuOwnedForResidentRead(ulong address, ulong size, bool clean) =>
            _bufferCache.HasGpuDirtyBytes(address, size) ||
            (clean && (_bufferCache.HasGpuDirtyPages(address, size) || _imageCache.HasGpuModifiedImageBytes(address, size)));

        private bool TryReadGuestWordConcurrent(ulong address, out uint word, ConcurrentAssumptions assumptions)
        {
            word = 0;
            if (!_guestMemory.CanRead(address, sizeof(uint)))
            {
                return false;
            }

            var dirty = _bufferCache.HasGpuDirtyBytes(address, sizeof(uint));
            assumptions.Record(ConcurrentQuery.DirtyBytes, address, sizeof(uint), dirty);
            if (dirty)
            {
                throw new SubmissionScheduler.AbandonedWorkException($"The resolution reads GPU-owned memory: address=0x{address:X16}.");
            }

            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            if (!_guestMemory.TryRead(address, bytes))
            {
                return false;
            }

            word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
            return true;
        }

        private bool TryReadResidentGuestBytesConcurrent(ulong address, Span<byte> destination, bool clean, ConcurrentAssumptions assumptions)
        {
            var size = (ulong)destination.Length;
            if (!_guestMemory.CanRead(address, size))
            {
                return false;
            }

            var owned = IsGpuOwnedForResidentRead(address, size, clean);
            assumptions.Record(clean ? ConcurrentQuery.ResidentClean : ConcurrentQuery.Resident, address, size, owned);
            return !owned && _guestMemory.TryRead(address, destination);
        }
    }
}
