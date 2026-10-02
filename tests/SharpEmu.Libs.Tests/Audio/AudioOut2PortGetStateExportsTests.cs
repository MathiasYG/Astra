// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Audio;
using Xunit;

using System.Reflection;
using SharpEmu.HLE.Host;

namespace SharpEmu.Libs.Tests.Audio;

public sealed class AudioOut2PortGetStateExportsTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const ulong StateAddress = MemoryBase + 0x100;

    private static CpuContext CreateContext(out FakeCpuMemory memory)
    {
        memory = new FakeCpuMemory(MemoryBase, 0x1000);
        return new CpuContext(memory, Generation.Gen5);
    }

    [Fact]
    public void PortGetState_WritesFixedSizeIgnoringPollutedR9()
    {
        var ctx = CreateContext(out var memory);
        // Paint the buffer so we can see the write footprint.
        Span<byte> paint = stackalloc byte[0x100];
        paint.Fill(0xAB);
        Assert.True(memory.TryWrite(StateAddress, paint));

        ctx[CpuRegister.Rdi] = 0xDE1FF6800001UL;
        ctx[CpuRegister.Rsi] = StateAddress;
        ctx[CpuRegister.Rdx] = StateAddress + 0x200;
        // Polluted GetSize leftover — must NOT enlarge the write.
        ctx[CpuRegister.R9] = 0x180;

        var result = AudioOut2Exports.AudioOut2PortGetState(ctx);

        Assert.Equal(0, result);
        Span<byte> state = stackalloc byte[0x100];
        Assert.True(memory.TryRead(StateAddress, state));
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(state));
        Assert.Equal(2, state[2]);
        // Bytes past the fixed 0x20 header must remain untouched.
        Assert.Equal(0xAB, state[0x20]);
        Assert.Equal(0xAB, state[0x7F]);
    }

    [Fact]
    public void PortGetState_SkipsGuestStackOutBuffer()
    {
        var ctx = CreateContext(out _);
        const ulong stackOut = 0x00007FFFDE1FF688UL;
        ctx[CpuRegister.Rdi] = 0xDE1FF688004DUL;
        ctx[CpuRegister.Rsi] = stackOut;
        ctx[CpuRegister.Rdx] = 0;

        var result = AudioOut2Exports.AudioOut2PortGetState(ctx);

        Assert.Equal(0, result);
    }

    [Fact]
    public void GetSpeakerInfo_WritesFixedSizeToRdiNotRsiTypeFlag()
    {
        var ctx = CreateContext(out var memory);
        Span<byte> paint = stackalloc byte[0x80];
        paint.Fill(0xCD);
        Assert.True(memory.TryWrite(StateAddress, paint));

        ctx[CpuRegister.Rdi] = StateAddress;
        ctx[CpuRegister.Rsi] = 1;
        ctx[CpuRegister.Rdx] = StateAddress + 0x200;
        ctx[CpuRegister.R8] = 0x840;
        ctx[CpuRegister.R9] = 0x10C;

        var result = AudioOut2Exports.AudioOut2GetSpeakerInfo(ctx);

        Assert.Equal(0, result);
        Span<byte> info = stackalloc byte[0x80];
        Assert.True(memory.TryRead(StateAddress, info));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(info));
        Assert.Equal(48000u, BinaryPrimitives.ReadUInt32LittleEndian(info[4..]));
        Assert.Equal(0xCD, info[0x20]);
    }
}

[Collection(AudioOutStateCollection.Name)]
public sealed class AudioOut2PcmLifetimeTests : IDisposable
{
    private const ulong Base = 0x100000000;
    private const ulong Source = Base + 0x1000;
    private readonly FakeCpuMemory _memory = new(Base, 0x2000);
    private readonly CpuContext _ctx;
    private readonly RecordingStream _stream = new();
    private readonly ulong _context;
    private readonly ulong _port;
    private readonly object? _savedBackend;
    private readonly object? _savedContext;
    private static FieldInfo Field(string name) => typeof(AudioOut2Exports).GetField(name, BindingFlags.NonPublic | BindingFlags.Static)!;

    public AudioOut2PcmLifetimeTests()
    {
        _ctx = new CpuContext(_memory, Generation.Gen5);
        byte[] parameters = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(parameters.AsSpan(0x10), 64);
        _memory.TryWrite(Base, parameters);
        _ctx[CpuRegister.Rdi] = Base;
        _ctx[CpuRegister.Rsi] = Base + 0x400;
        _ctx[CpuRegister.Rdx] = 0x100;
        _ctx[CpuRegister.Rcx] = Base + 0x100;
        Assert.Equal(0, AudioOut2Exports.AudioOut2ContextCreate(_ctx));
        _context = ReadHandle(Base + 0x100);
        parameters.AsSpan().Clear();
        BinaryPrimitives.WriteUInt16LittleEndian(parameters, 0x100);
        BinaryPrimitives.WriteUInt32LittleEndian(parameters.AsSpan(4), 0x100);
        _memory.TryWrite(Base, parameters);
        _ctx[CpuRegister.Rdi] = _context;
        _ctx[CpuRegister.Rsi] = Base;
        _ctx[CpuRegister.Rdx] = Base + 0x100;
        Assert.Equal(0, AudioOut2Exports.AudioOut2PortCreate(_ctx));
        _port = ReadHandle(Base + 0x100);
        _savedBackend = Field("PrimaryBackend").GetValue(null);
        _savedContext = Field("PrimaryContextHandle").GetValue(null);
        Field("PrimaryBackend").SetValue(null, _stream);
        Field("PrimaryContextHandle").SetValue(null, _context);
    }

    [Fact]
    public void TemporaryPcmRemainsValidAfterCallerReusesStorage()
    {
        SetSource(0.5f);
        Assert.Equal(0, SetAttribute(Source));
        // Simulate returned stack memory being reused for non-PCM data.
        byte[] reused = new byte[64 * sizeof(float)];
        reused.AsSpan().Fill(0xFF);
        _memory.TryWrite(Source, reused);
        Push();
        var submitted = Assert.Single(_stream.Submissions);
        var sample = BinaryPrimitives.ReadInt16LittleEndian(submitted);
        Assert.InRange(sample, (short)16383, (short)16384);
        for (var offset = 0; offset < submitted.Length; offset += 2)
            Assert.Equal(sample, BinaryPrimitives.ReadInt16LittleEndian(submitted.AsSpan(offset)));
        Push();
        Assert.Single(_stream.Submissions);
    }

    [Fact]
    public void ReplacementUsesNewestCapturedGrainAndNullClearsPending()
    {
        SetSource(0.5f);
        Assert.Equal(0, SetAttribute(Source));
        SetSource(-0.25f);
        Assert.Equal(0, SetAttribute(Source));
        SetSource(0);
        Push();
        var submitted = Assert.Single(_stream.Submissions);
        Assert.InRange(BinaryPrimitives.ReadInt16LittleEndian(submitted), (short)-8192, (short)-8191);
        Assert.Equal(0, SetAttribute(Source));
        Assert.Equal(0, SetAttribute(0));
        Push();
        Assert.Single(_stream.Submissions);
    }

    [Fact]
    public void UnreadablePcmIsReportedBeforeItCanReachTheHost()
    {
        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT, SetAttribute(Base + 0x2000));
        Push();
        Assert.Empty(_stream.Submissions);
    }

    private int SetAttribute(ulong address)
    {
        Span<byte> attribute = stackalloc byte[24];
        attribute.Clear();
        BinaryPrimitives.WriteUInt64LittleEndian(attribute[8..], Base + 0x300);
        BinaryPrimitives.WriteUInt64LittleEndian(attribute[16..], 8);
        _memory.TryWrite(Base + 0x200, attribute);
        Span<byte> pointer = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(pointer, address);
        _memory.TryWrite(Base + 0x300, pointer);
        _ctx[CpuRegister.Rdi] = _port;
        _ctx[CpuRegister.Rsi] = Base + 0x200;
        _ctx[CpuRegister.Rdx] = 1;
        return AudioOut2Exports.AudioOut2PortSetAttributes(_ctx);
    }

    private void SetSource(float value)
    {
        var samples = new byte[64 * sizeof(float)];
        for (var offset = 0; offset < samples.Length; offset += 4)
            BinaryPrimitives.WriteInt32LittleEndian(samples.AsSpan(offset), BitConverter.SingleToInt32Bits(value));
        _memory.TryWrite(Source, samples);
    }

    private ulong ReadHandle(ulong address)
    {
        Span<byte> bytes = stackalloc byte[8];
        Assert.True(_memory.TryRead(address, bytes));
        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }

    private void Push()
    {
        _ctx[CpuRegister.Rdi] = _context;
        _ctx[CpuRegister.Rsi] = 0;
        Assert.Equal(0, AudioOut2Exports.AudioOut2ContextPush(_ctx));
    }

    public void Dispose()
    {
        _ctx[CpuRegister.Rdi] = _port;
        AudioOut2Exports.AudioOut2PortDestroy(_ctx);
        _ctx[CpuRegister.Rdi] = _context;
        AudioOut2Exports.AudioOut2ContextDestroy(_ctx);
        Field("PrimaryBackend").SetValue(null, _savedBackend);
        Field("PrimaryContextHandle").SetValue(null, _savedContext);
    }

    private sealed class RecordingStream : IHostAudioStream
    {
        public List<byte[]> Submissions { get; } = [];
        public bool Submit(ReadOnlySpan<byte> bytes) { Submissions.Add(bytes.ToArray()); return true; }
        public void Dispose() { }
    }
}
