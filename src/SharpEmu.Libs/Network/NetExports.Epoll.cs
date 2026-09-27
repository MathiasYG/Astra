// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Net.Sockets;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Network;

public static partial class NetExports
{
    private const uint EpollIn = 0x0001;
    private const uint EpollOut = 0x0002;
    private const uint EpollError = 0x0008;
    private const uint EpollHangup = 0x0010;
    private const int EpollEventSize = 32;
    private static readonly ConcurrentDictionary<int, EpollInstance> EpollInstances = new();
    private static int _nextEpollId = 0x6000;

    private sealed class EpollInstance
    {
        public readonly object Gate = new();
        public readonly Dictionary<int, byte[]> Watches = new();
        public bool Aborted;
        public bool Destroyed;
    }

    [SysAbiExport(Nid = "SF47kB2MNTo", ExportName = "sceNetEpollCreate",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNet")]
    public static int NetEpollCreate(CpuContext ctx)
    {
        if (!_initialized)
            return SetNetError(ctx, NetErrorNotInitialized, NetErrnoNotInitialized);

        var nameAddress = ctx[CpuRegister.Rdi];
        var flags = unchecked((int)ctx[CpuRegister.Rsi]);
        if (flags != 0 || !TryReadUtf8Z(ctx, nameAddress, MaxNameLength, out _))
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);

        var id = Interlocked.Increment(ref _nextEpollId);
        EpollInstances[id] = new EpollInstance();
        TraceNet("epoll.create", id, unchecked((ulong)flags), 0, 0);
        return ctx.SetReturn(id);
    }

    [SysAbiExport(ExportName = "sceNetEpollControl",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNet")]
    public static int NetEpollControl(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var operation = unchecked((int)ctx[CpuRegister.Rsi]);
        var socketId = unchecked((int)ctx[CpuRegister.Rdx]);
        var eventAddress = ctx[CpuRegister.Rcx];
        if (!EpollInstances.TryGetValue(id, out var instance))
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        if (operation is < 1 or > 3)
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
        if (!_sockets.ContainsKey(socketId))
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);

        var eventBytes = new byte[EpollEventSize];
        if (operation != 3 && (eventAddress == 0 || !ctx.Memory.TryRead(eventAddress, eventBytes)))
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);

        lock (instance.Gate)
        {
            if (instance.Destroyed)
                return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);

            var exists = instance.Watches.ContainsKey(socketId);
            if ((operation == 1 && exists) || (operation != 1 && !exists))
                return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);

            if (operation == 3)
                instance.Watches.Remove(socketId);
            else
                instance.Watches[socketId] = eventBytes;
        }

        TraceNet("epoll.control", id, unchecked((ulong)operation), unchecked((ulong)socketId), eventAddress);
        return ctx.SetReturn(0);
    }

    [SysAbiExport(ExportName = "sceNetEpollWait",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNet")]
    public static int NetEpollWait(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        var eventsAddress = ctx[CpuRegister.Rsi];
        var maxEvents = unchecked((int)ctx[CpuRegister.Rdx]);
        var timeoutMicroseconds = unchecked((int)ctx[CpuRegister.Rcx]);
        if (!EpollInstances.TryGetValue(id, out var instance))
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        if (eventsAddress == 0 || maxEvents <= 0 || maxEvents > 4096 || timeoutMicroseconds < -1)
            return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);

        var deadline = timeoutMicroseconds < 0
            ? DateTime.MaxValue
            : DateTime.UtcNow.AddTicks((long)timeoutMicroseconds * 10);
        while (true)
        {
            KeyValuePair<int, byte[]>[] watches;
            lock (instance.Gate)
            {
                if (instance.Destroyed)
                    return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
                if (instance.Aborted)
                {
                    instance.Aborted = false;
                    return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
                }
                watches = instance.Watches.ToArray();
            }

            var count = 0;
            foreach (var watch in watches)
            {
                if (count >= maxEvents)
                    break;
                if (!_sockets.TryGetValue(watch.Key, out var socket))
                    continue;

                uint ready = 0;
                try
                {
                    var requested = BinaryPrimitives.ReadUInt32LittleEndian(watch.Value);
                    if ((requested & EpollIn) != 0 && socket.Poll(0, SelectMode.SelectRead))
                        ready |= EpollIn;
                    if ((requested & EpollOut) != 0 && socket.Poll(0, SelectMode.SelectWrite))
                        ready |= EpollOut;
                    if (socket.Poll(0, SelectMode.SelectError))
                        ready |= EpollError;
                }
                catch (SocketException)
                {
                    ready = EpollError | EpollHangup;
                }
                catch (ObjectDisposedException)
                {
                    ready = EpollHangup;
                }

                if (ready == 0)
                    continue;

                var result = (byte[])watch.Value.Clone();
                BinaryPrimitives.WriteUInt32LittleEndian(result, ready);
                if (!ctx.Memory.TryWrite(eventsAddress + (ulong)(count * EpollEventSize), result))
                    return SetNetError(ctx, NetErrorInvalidArgument, NetErrnoInvalidArgument);
                count++;
            }

            if (count != 0 || DateTime.UtcNow >= deadline)
                return ctx.SetReturn(count);

            Thread.Sleep(1);
        }
    }

    [SysAbiExport(ExportName = "sceNetEpollDestroy",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNet")]
    public static int NetEpollDestroy(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        if (!EpollInstances.TryRemove(id, out var instance))
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        lock (instance.Gate)
            instance.Destroyed = true;
        return ctx.SetReturn(0);
    }

    [SysAbiExport(ExportName = "sceNetEpollAbort",
        Target = Generation.Gen4 | Generation.Gen5, LibraryName = "libSceNet")]
    public static int NetEpollAbort(CpuContext ctx)
    {
        var id = unchecked((int)ctx[CpuRegister.Rdi]);
        if (!EpollInstances.TryGetValue(id, out var instance))
            return SetNetError(ctx, NetErrorBadFileDescriptor, NetErrnoBadFileDescriptor);
        lock (instance.Gate)
            instance.Aborted = true;
        return ctx.SetReturn(0);
    }

    private static void ClearEpollInstances()
    {
        foreach (var instance in EpollInstances.Values)
        {
            lock (instance.Gate)
                instance.Destroyed = true;
        }
        EpollInstances.Clear();
    }
}
