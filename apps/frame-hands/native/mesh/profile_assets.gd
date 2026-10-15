extends RefCounted

static func load_verified(index_path: String) -> Array:
    var index = JSON.parse_string(FileAccess.get_file_as_string(index_path))
    if not index is Dictionary or index.get("version",0) != 2 or not index.get("profiles") is Array or index.profiles.size() != 2: return []
    var result := []
    for side in 2:
        var entry = index.profiles[side]
        if not entry is Dictionary or int(entry.get("side",-1)) != side: return []
        var filename := str(entry.get("file",""))
        if filename != "side%d.json" % side: return []
        var path := index_path.get_base_dir().path_join(filename)
        var bytes := FileAccess.get_file_as_bytes(path)
        var hash := HashingContext.new()
        if hash.start(HashingContext.HASH_SHA256) != OK or hash.update(bytes) != OK: return []
        if hash.finish().hex_encode() != str(entry.get("sha256","")): return []
        var asset = JSON.parse_string(bytes.get_string_from_utf8())
        if not asset is Dictionary or int(asset.get("profile_side",-1)) != side or asset.get("profile_sha256","") != entry.get("profile_sha256",""): return []
        result.append(asset)
    return result
