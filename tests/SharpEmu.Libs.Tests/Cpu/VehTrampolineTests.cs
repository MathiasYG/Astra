// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Core.Cpu.Native;
using SharpEmu.HLE;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

public sealed class VehTrampolineTests
{
    private static bool Supported => OperatingSystem.IsWindows() &&
        RuntimeInformation.ProcessArchitecture == Architecture.X64;

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate int Handler(nint exceptionInfo);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool VirtualFree(nint address, nuint size, uint freeType);

    [Fact]
    public async Task ConcurrentCallbacksDoNotSerializeAcrossManagedWaits()
    {
        if (!Supported) return;

        using var backend = new DirectExecutionBackend(new ModuleManager());
        using var firstEntered = new ManualResetEventSlim();
        using var secondEntered = new ManualResetEventSlim();
        using var releaseFirst = new ManualResetEventSlim();
        var calls = 0;
        Handler handler = _ =>
        {
            if (Interlocked.Increment(ref calls) == 1)
            {
                firstEntered.Set();
                releaseFirst.Wait(TimeSpan.FromSeconds(10));
            }
            else
            {
                secondEntered.Set();
            }
            return -1;
        };
        var thunk = backend.CreateExceptionHandlerTrampoline(Marshal.GetFunctionPointerForDelegate(handler));
        Assert.NotEqual(0, thunk);
        Task<int>? first = null;
        Task<int>? second = null;
        try
        {
            first = StartCallback(thunk);
            Assert.True(firstEntered.Wait(TimeSpan.FromSeconds(5)), "First callback did not enter.");
            second = StartCallback(thunk);
            Assert.True(secondEntered.Wait(TimeSpan.FromSeconds(5)),
                "A fault on another thread was blocked by the first managed callback.");
        }
        finally
        {
            releaseFirst.Set();
            if (first is not null) await first;
            if (second is not null) await second;
            VirtualFree(thunk, 0, 0x8000);
            GC.KeepAlive(handler);
        }
        Assert.Equal(-1, await first);
        Assert.Equal(-1, await second!);
    }

    [Theory]
    [InlineData(0xE0434352u)] // CLR exception must never re-enter managed code.
    [InlineData(0xE06D7363u)] // Native C++ exception.
    [InlineData(0xC00000FDu)] // Stack overflow.
    public void RuntimeExceptionsBypassManagedCallback(uint exceptionCode)
    {
        if (!Supported) return;
        using var backend = new DirectExecutionBackend(new ModuleManager());
        var called = false;
        Handler handler = _ => { called = true; return -1; };
        var thunk = backend.CreateExceptionHandlerTrampoline(Marshal.GetFunctionPointerForDelegate(handler));
        Assert.NotEqual(0, thunk);
        try
        {
            Assert.Equal(0, Invoke(thunk, exceptionCode));
            Assert.False(called);
        }
        finally
        {
            VirtualFree(thunk, 0, 0x8000);
            GC.KeepAlive(handler);
        }
    }

    [Fact]
    public unsafe void ManagedCallbackPreservesWin64NonvolatileRegisters()
    {
        if (!Supported) return;

        using var backend = new DirectExecutionBackend(new ModuleManager());
        Handler handler = _ =>
        {
            // Force ordinary managed work and volatile-register traffic while
            // the generated thunk is active. The wrapper below checks the ABI
            // registers after this reverse P/Invoke returns.
            var text = string.Concat("VEH-", Environment.CurrentManagedThreadId, "-", DateTime.UtcNow.Ticks);
            GC.KeepAlive(text);
            return -1;
        };
        var thunk = backend.CreateExceptionHandlerTrampoline(Marshal.GetFunctionPointerForDelegate(handler));
        Assert.NotEqual(0, thunk);
        var wrapper = CreateRegisterProbe(thunk);
        var output = stackalloc ulong[8];
        var exceptionInfo = CreateExceptionInfo(0xC000001D);
        try
        {
            var result = ((delegate* unmanaged<nint, nint, int>)wrapper)(exceptionInfo, (nint)output);
            Assert.Equal(-1, result);
            Assert.Equal(0x1111111111111111UL, output[0]); // rbx
            Assert.Equal(0x2222222222222222UL, output[1]); // rbp
            Assert.Equal(0x3333333333333333UL, output[2]); // rsi
            Assert.Equal(0x4444444444444444UL, output[3]); // rdi
            Assert.Equal(0x5555555555555555UL, output[4]); // r12
            Assert.Equal(0x6666666666666666UL, output[5]); // r13
            Assert.Equal(0x7777777777777777UL, output[6]); // r14
            Assert.Equal(0x8888888888888888UL, output[7]); // r15
        }
        finally
        {
            NativeMemory.Free((void*)exceptionInfo);
            HostMemory.Free((void*)wrapper, 0, HostMemory.MEM_RELEASE);
            GC.KeepAlive(handler);
        }
    }

    private static Task<int> StartCallback(nint thunk) => Task.Factory.StartNew(
        () => Invoke(thunk, 0xC000001D), CancellationToken.None,
        TaskCreationOptions.LongRunning, TaskScheduler.Default);

    private static unsafe int Invoke(nint thunk, uint code)
    {
        // Exercise the actual emitted Win64 entry on the host stack without
        // installing a process-wide VEH or raising a real hardware exception.
        byte* record = stackalloc byte[0xA0];
        byte* context = stackalloc byte[0x4D0];
        new Span<byte>(record, 0xA0).Clear();
        new Span<byte>(context, 0x4D0).Clear();
        *(uint*)record = code;
        nint* pointers = stackalloc nint[2];
        pointers[0] = (nint)record;
        pointers[1] = (nint)context;
        return ((delegate* unmanaged<nint, int>)thunk)((nint)pointers);
    }

    private static unsafe nint CreateExceptionInfo(uint code)
    {
        // The wrapper only needs a valid ExceptionCode field and ContextRecord
        // pointer; the thunk's native pre-filter reads exactly those members.
        var pointers = (nint*)NativeMemory.Alloc((nuint)(2 * sizeof(nint) + 0xA0 + 0x4D0));
        var record = (byte*)(pointers + 2);
        var context = record + 0xA0;
        new Span<byte>(record, 0xA0).Clear();
        new Span<byte>(context, 0x4D0).Clear();
        *(uint*)record = code;
        pointers[0] = (nint)record;
        pointers[1] = (nint)context;
        return (nint)pointers;
    }

    private static unsafe nint CreateRegisterProbe(nint thunk)
    {
        var code = new List<byte>();
        void U64(ulong value) { code.AddRange(BitConverter.GetBytes(value)); }
        void MovReg(byte rex, byte opcode, ulong value) { code.Add(rex); code.Add(opcode); U64(value); }
        // Preserve caller nonvolatiles, then install sentinels and call thunk.
        code.AddRange(new byte[] { 0x53, 0x56, 0x57, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57 });
        code.AddRange(new byte[] { 0x48, 0x83, 0xEC, 0x38 });
        code.AddRange(new byte[] { 0x48, 0x89, 0x54, 0x24, 0x30 }); // [rsp+30] = output
        MovReg(0x49, 0xBC, 0x5555555555555555UL);
        MovReg(0x49, 0xBD, 0x6666666666666666UL);
        MovReg(0x49, 0xBE, 0x7777777777777777UL);
        MovReg(0x49, 0xBF, 0x8888888888888888UL);
        MovReg(0x48, 0xBB, 0x1111111111111111UL);
        MovReg(0x48, 0xBD, 0x2222222222222222UL);
        MovReg(0x48, 0xBE, 0x3333333333333333UL);
        MovReg(0x48, 0xBF, 0x4444444444444444UL);
        code.AddRange(new byte[] { 0x48, 0xB8 }); U64((ulong)thunk); code.AddRange(new byte[] { 0xFF, 0xD0 });
        code.AddRange(new byte[] { 0x89, 0x44, 0x24, 0x28 });       // [rsp+28] = callback result
        code.AddRange(new byte[] { 0x48, 0x8B, 0x44, 0x24, 0x30 }); // rax = output
        code.AddRange(new byte[] { 0x48, 0x89, 0x98, 0, 0, 0, 0 });
        code.AddRange(new byte[] { 0x48, 0x89, 0xA8, 8, 0, 0, 0 });
        code.AddRange(new byte[] { 0x48, 0x89, 0xB0, 16, 0, 0, 0 });
        code.AddRange(new byte[] { 0x48, 0x89, 0xB8, 24, 0, 0, 0 });
        code.AddRange(new byte[] { 0x4C, 0x89, 0xA0, 32, 0, 0, 0 });
        code.AddRange(new byte[] { 0x4C, 0x89, 0xA8, 40, 0, 0, 0 });
        code.AddRange(new byte[] { 0x4C, 0x89, 0xB0, 48, 0, 0, 0 });
        code.AddRange(new byte[] { 0x4C, 0x89, 0xB8, 56, 0, 0, 0 });
        code.AddRange(new byte[] { 0x8B, 0x44, 0x24, 0x28 });       // eax = callback result
        code.AddRange(new byte[] { 0x48, 0x83, 0xC4, 0x38, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5D, 0x5F, 0x5E, 0x5B, 0xC3 });
        var ptr = (byte*)HostMemory.Alloc(null, (nuint)code.Count, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
        code.ToArray().AsSpan().CopyTo(new Span<byte>(ptr, code.Count));
        return (nint)ptr;
    }
}
