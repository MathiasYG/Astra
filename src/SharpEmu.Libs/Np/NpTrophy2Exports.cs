// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using System.Buffers.Binary;

namespace SharpEmu.Libs.Np;

public static class NpTrophy2Exports
{
    private static int _nextContext = 1;
    private static int _nextHandle = 1;

    [SysAbiExport(
        Nid = "Bagshr7OQ6Q",
        ExportName = "sceNpTrophy2CreateContext",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2CreateContext(CpuContext ctx)
    {
        return WriteIdAndReturn(ctx, ctx[CpuRegister.Rdi], ref _nextContext);
    }

    [SysAbiExport(
        Nid = "Gz1rmUZpROM",
        ExportName = "sceNpTrophy2CreateHandle",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2CreateHandle(CpuContext ctx)
    {
        return WriteIdAndReturn(ctx, ctx[CpuRegister.Rdi], ref _nextHandle);
    }

    [SysAbiExport(
        Nid = "sysY2FHYff4",
        ExportName = "sceNpTrophy2DestroyContext",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2DestroyContext(CpuContext ctx) => ReturnOk(ctx);

    [SysAbiExport(
        Nid = "d8P11CI40KE",
        ExportName = "sceNpTrophy2DestroyHandle",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2DestroyHandle(CpuContext ctx) => ReturnOk(ctx);

    [SysAbiExport(
        Nid = "fYapWA9xVmA",
        ExportName = "sceNpTrophy2AbortHandle",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2AbortHandle(CpuContext ctx) => ReturnOk(ctx);

    [SysAbiExport(
        Nid = "bIDov3wBu5Q",
        ExportName = "sceNpTrophy2RegisterContext",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2RegisterContext(CpuContext ctx) => ReturnOk(ctx);

    [SysAbiExport(
        Nid = "sUXGfNMalIo",
        ExportName = "sceNpTrophy2RegisterUnlockCallback",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2RegisterUnlockCallback(CpuContext ctx) => ReturnOk(ctx);

    [SysAbiExport(
        Nid = "wVqxM58sIKs",
        ExportName = "sceNpTrophy2UnregisterUnlockCallback",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2UnregisterUnlockCallback(CpuContext ctx) => ReturnOk(ctx);

    [SysAbiExport(
        Nid = "EHQEDVXZ0TI",
        ExportName = "sceNpTrophy2ShowTrophyList",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2ShowTrophyList(CpuContext ctx) => ReturnOk(ctx);

    /// <summary>
    /// Gen5 ABI: context, handle, trophy id, then SceNpTrophy2Details and
    /// SceNpTrophy2Data output pointers.
    /// </summary>
    /// <remarks>
    /// Reports "no such trophy" rather than succeeding. Succeeding would require
    /// filling both output structures, and their exact layouts are not confirmed
    /// here — a title that trusted zeroed details would read an empty name and a
    /// grade of zero as real data. NOT_FOUND is a documented outcome that callers
    /// must already handle, so it degrades along a path the game tests.
    /// </remarks>
    [SysAbiExport(
        Nid = "EwNylPdWUTM",
        ExportName = "sceNpTrophy2GetTrophyInfo",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2GetTrophyInfo(CpuContext ctx) =>
        SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_FOUND);

    [SysAbiExport(
        Nid = "y3zHpdZO6ME",
        ExportName = "sceNpTrophy2GetTrophyInfoArray",
        Target = Generation.Gen5,
        LibraryName = "libSceNpTrophy2")]
    public static int NpTrophy2GetTrophyInfoArray(CpuContext ctx)
    {
        var offset = (uint)ctx[CpuRegister.Rdx];
        var limit = (uint)ctx[CpuRegister.Rcx];
        var detailsAddress = ctx[CpuRegister.R8];
        var dataAddress = ctx[CpuRegister.R9];
        if ((detailsAddress == 0 && dataAddress == 0) ||
            !ctx.TryGetImportStackArgument(0, out var countAddress) || countAddress == 0)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        // The count is how many trophies the package really holds from offset on, capped by
        // limit. Reporting limit itself makes a title walk past its own arrays.
        var trophies = LoadTrophies();
        var start = (int)Math.Min(offset, (uint)trophies.Length);
        var count = (int)Math.Min(limit, (uint)(trophies.Length - start));
        Span<byte> details = stackalloc byte[TrophyDetailsSize];
        Span<byte> data = stackalloc byte[TrophyDataSize];
        for (var index = 0; index < count; index++)
        {
            var trophy = trophies[start + index];
            if (detailsAddress != 0)
            {
                trophy.WriteDetails(details);
                if (!ctx.Memory.TryWrite(detailsAddress + (ulong)index * TrophyDetailsSize, details))
                {
                    return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }
            }

            if (dataAddress != 0)
            {
                // Nothing is unlocked for the emulated user.
                data.Clear();
                BinaryPrimitives.WriteInt32LittleEndian(data, trophy.Id);
                if (!ctx.Memory.TryWrite(dataAddress + (ulong)index * TrophyDataSize, data))
                {
                    return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }
            }
        }

        Span<byte> countBytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(countBytes, (uint)count);
        return ctx.Memory.TryWrite(countAddress, countBytes)
            ? SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK)
            : SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
    }

    // SceNpTrophy2Details is 1312 bytes: id/grade/group at +0/+4/+8, hidden at +12, has-reward at
    // +13, progress target (type, value) at +16, then name[128], description[1024], reward[128].
    // SceNpTrophy2Data is 32 bytes: id at +0, unlocked flag at +4, progress at +8, timestamp at +24.
    private const int TrophyDetailsSize = 1312;
    private const int TrophyDataSize = 32;

    private static readonly object TrophyGate = new();
    private static TrophyInfo[]? _trophies;

    private sealed record TrophyInfo(
        int Id, int Grade, int GroupId, bool Hidden, bool HasReward, string Name, string Description)
    {
        public void WriteDetails(Span<byte> destination)
        {
            destination.Clear();
            BinaryPrimitives.WriteInt32LittleEndian(destination, Id);
            BinaryPrimitives.WriteInt32LittleEndian(destination[4..], Grade);
            BinaryPrimitives.WriteInt32LittleEndian(destination[8..], GroupId);
            destination[12] = Hidden ? (byte)1 : (byte)0;
            destination[13] = HasReward ? (byte)1 : (byte)0;
            WriteText(destination.Slice(32, 128), Name);
            WriteText(destination.Slice(160, 1024), Description);
        }

        private static void WriteText(Span<byte> field, string text)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(text);
            var length = Math.Min(bytes.Length, field.Length - 1);
            bytes.AsSpan(0, length).CopyTo(field);
        }
    }

    private static TrophyInfo[] LoadTrophies()
    {
        lock (TrophyGate)
        {
            return _trophies ??= ReadTrophyPackage();
        }
    }

    // sce_sys/trophy2/trophy00.ucp is an unencrypted container: a big-endian header, a table of
    // 0x40-byte entries (name[0x20], offset, length), and JSON files for the trophy list and the
    // localized texts.
    private static TrophyInfo[] ReadTrophyPackage()
    {
        try
        {
            var path = Kernel.KernelMemoryCompatExports.ResolveGuestPath("/app0/sce_sys/trophy2/trophy00.ucp");
            if (!File.Exists(path))
            {
                return [];
            }

            var package = File.ReadAllBytes(path);
            if (package.Length < 0x40 ||
                BinaryPrimitives.ReadUInt32BigEndian(package) != 0xB228C60A)
            {
                return [];
            }

            var fileCount = BinaryPrimitives.ReadUInt32BigEndian(package.AsSpan(0x10));
            var tableOffset = BinaryPrimitives.ReadUInt32BigEndian(package.AsSpan(0x14));
            var files = new Dictionary<string, (long Offset, long Length)>();
            for (var i = 0; i < Math.Min(fileCount, 4096u); i++)
            {
                var entry = (long)tableOffset + 0x20 + i * 0x40L;
                if (entry + 0x40 > package.Length)
                {
                    break;
                }

                var nameBytes = package.AsSpan((int)entry, 0x20);
                var nameLength = nameBytes.IndexOf((byte)0);
                var name = System.Text.Encoding.UTF8.GetString(nameBytes[..(nameLength < 0 ? nameBytes.Length : nameLength)]);
                var offset = (long)BinaryPrimitives.ReadUInt64BigEndian(package.AsSpan((int)entry + 0x20));
                var length = (long)BinaryPrimitives.ReadUInt64BigEndian(package.AsSpan((int)entry + 0x28));
                if (offset >= 0 && length >= 0 && offset + length <= package.Length)
                {
                    files[name] = (offset, length);
                }
            }

            System.Text.Json.JsonDocument? Json(string name) =>
                files.TryGetValue(name, out var file)
                    ? System.Text.Json.JsonDocument.Parse(package.AsMemory((int)file.Offset, (int)file.Length))
                    : null;

            using var conf = Json("tropconf.json");
            if (conf is null || !conf.RootElement.TryGetProperty("trophies", out var list))
            {
                return [];
            }

            var language = conf.RootElement.TryGetProperty("defaultLanguage", out var lang)
                ? lang.GetString() ?? "en-US"
                : "en-US";
            using var meta = Json("tropmeta_en-US.json") ?? Json($"tropmeta_{language}.json");
            var texts = new Dictionary<int, (string Name, string Detail)>();
            if (meta is not null &&
                meta.RootElement.TryGetProperty("metadata", out var metadata) &&
                metadata.TryGetProperty("trophyMetadata", out var trophyTexts))
            {
                foreach (var text in trophyTexts.EnumerateArray())
                {
                    if (TryReadId(text, "id", out var id))
                    {
                        texts[id] = (
                            text.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "",
                            text.TryGetProperty("detail", out var d) ? d.GetString() ?? "" : "");
                    }
                }
            }

            var result = new SortedDictionary<int, TrophyInfo>();
            foreach (var definition in list.EnumerateArray())
            {
                if (!TryReadId(definition, "id", out var id) ||
                    !definition.TryGetProperty("grade", out var gradeElement) ||
                    gradeElement.GetString() is not { Length: 1 } grade)
                {
                    continue;
                }

                var gradeIndex = "PGSB".IndexOf(grade[0]);
                if (gradeIndex < 0)
                {
                    continue;
                }

                var groupId = TryReadId(definition, "groupId", out var group) ? group : 0;
                texts.TryGetValue(id, out var trophyText);
                result[id] = new TrophyInfo(
                    id,
                    gradeIndex + 1,
                    groupId,
                    definition.TryGetProperty("hidden", out var hidden) && hidden.ValueKind == System.Text.Json.JsonValueKind.True,
                    definition.TryGetProperty("hasReward", out var reward) && reward.ValueKind == System.Text.Json.JsonValueKind.True,
                    trophyText.Name ?? "",
                    trophyText.Detail ?? "");
            }

            return [.. result.Values];
        }
        catch (Exception ex) when (ex is IOException or System.Text.Json.JsonException or InvalidOperationException or ArgumentException)
        {
            return [];
        }
    }

    private static bool TryReadId(System.Text.Json.JsonElement element, string property, out int id)
    {
        id = 0;
        if (!element.TryGetProperty(property, out var value))
        {
            return false;
        }

        return value.ValueKind == System.Text.Json.JsonValueKind.Number
            ? value.TryGetInt32(out id)
            : int.TryParse(value.GetString(), out id);
    }


    private static int WriteIdAndReturn(CpuContext ctx, ulong outAddress, ref int nextId)
    {
        if (outAddress == 0)
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT);
        }

        Span<byte> idBytes = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(idBytes, nextId);
        if (!ctx.Memory.TryWrite(outAddress, idBytes))
        {
            return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        nextId++;
        return SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);
    }

    private static int ReturnOk(CpuContext ctx) => SetReturn(ctx, OrbisGen2Result.ORBIS_GEN2_OK);

    private static int SetReturn(CpuContext ctx, OrbisGen2Result result)
    {
        ctx[CpuRegister.Rax] = unchecked((ulong)(int)result);
        return (int)result;
    }
}
