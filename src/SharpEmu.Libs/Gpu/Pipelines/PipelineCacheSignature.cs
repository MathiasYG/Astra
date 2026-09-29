// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.IO.Hashing;
using System.Text;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The prefix of a saved driver pipeline cache: wrapper format and device/driver identity, then the payload hash.
public static class PipelineCacheSignature
{
    public const int UuidSize = 16;

    public static string Build(uint vendorId, uint deviceId, uint driverVersion, ReadOnlySpan<byte> pipelineCacheUuid)
    {
        var uuid = new StringBuilder(UuidSize * 2);
        for (var index = 0; index < UuidSize && index < pipelineCacheUuid.Length; index++)
        {
            uuid.Append(pipelineCacheUuid[index].ToString("x2"));
        }

        return $"SharpEmuPC2:{vendorId:x8}:{deviceId:x8}:{driverVersion:x8}:{uuid}\n";
    }

    // The signature line, the payload hash and the payload.
    public static byte[] Wrap(string signature, ReadOnlySpan<byte> payload)
    {
        var prefix = Encoding.ASCII.GetBytes(signature);
        var file = new byte[prefix.Length + sizeof(ulong) + payload.Length];
        prefix.CopyTo(file, 0);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(prefix.Length), XxHash3.HashToUInt64(payload));
        payload.CopyTo(file.AsSpan(prefix.Length + sizeof(ulong)));
        return file;
    }

    // The payload when the signature and the hash match; false for any other file.
    // Read PC1 caches when their device/driver identity matches. Their build tag did not
    // affect Vulkan's opaque cache contents and unnecessarily discarded caches after commits.
    public static bool TryUnwrap(string signature, ReadOnlySpan<byte> file, out byte[] payload)
    {
        payload = [];
        if (file.Length < sizeof(ulong) + 1 || file.Length > int.MaxValue)
        {
            return false;
        }

        var newline = file.IndexOf((byte)'\n');
        if (newline < 0 || !HasCompatibleSignature(signature, file[..newline]))
        {
            return false;
        }

        var payloadOffset = newline + 1 + sizeof(ulong);
        if (file.Length < payloadOffset)
        {
            return false;
        }

        var expectedHash = BinaryPrimitives.ReadUInt64LittleEndian(file[(newline + 1)..]);
        var data = file[payloadOffset..];
        if (XxHash3.HashToUInt64(data) != expectedHash)
        {
            return false;
        }

        payload = data.ToArray();
        return true;
    }

    private static bool HasCompatibleSignature(string signature, ReadOnlySpan<byte> fileHeader)
    {
        var expectedHeader = Encoding.ASCII.GetBytes(signature.TrimEnd('\n'));
        if (fileHeader.SequenceEqual(expectedHeader))
        {
            return true;
        }

        const string currentPrefix = "SharpEmuPC2:";
        const string legacyPrefix = "SharpEmuPC1:";
        var currentPrefixBytes = Encoding.ASCII.GetBytes(currentPrefix);
        var legacyPrefixBytes = Encoding.ASCII.GetBytes(legacyPrefix);
        if (!expectedHeader.AsSpan().StartsWith(currentPrefixBytes) || !fileHeader.StartsWith(legacyPrefixBytes))
        {
            return false;
        }

        var legacyVersionEnd = fileHeader[legacyPrefixBytes.Length..].IndexOf((byte)':');
        if (legacyVersionEnd < 0)
        {
            return false;
        }

        var expectedDeviceIdentity = expectedHeader.AsSpan(currentPrefixBytes.Length);
        var legacyDeviceIdentity = fileHeader[(legacyPrefixBytes.Length + legacyVersionEnd + 1)..];
        return legacyDeviceIdentity.SequenceEqual(expectedDeviceIdentity);
    }
}
