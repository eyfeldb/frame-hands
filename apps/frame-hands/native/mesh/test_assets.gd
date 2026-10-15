extends SceneTree
const Assets = preload("res://profile_assets.gd")
func _initialize():
    var original = JSON.parse_string(FileAccess.get_file_as_string("res://profile-rigs.json"))
    if Assets.load_verified("res://profile-rigs.json").size() != 2: quit(2);return
    var cases := []
    var changed = original.duplicate(true)
    changed.profiles[0].sha256 = "0".repeat(64)
    cases.append(changed)
    changed = original.duplicate(true)
    changed.profiles[0].side = 1
    cases.append(changed)
    changed = original.duplicate(true)
    changed.profiles[0].profile_sha256 = "0".repeat(64)
    cases.append(changed)
    changed = original.duplicate(true)
    changed.profiles[0].file = "../side0.json"
    cases.append(changed)
    cases.append({"version":2,"profiles":42})
    cases.append({"version":2,"profiles":[null,null]})
    for bad in cases:
        var file = FileAccess.open("res://invalid-index.json",FileAccess.WRITE)
        file.store_string(JSON.stringify(bad))
        file.close()
        if not Assets.load_verified("res://invalid-index.json").is_empty(): quit(3);return
    DirAccess.remove_absolute(ProjectSettings.globalize_path("res://invalid-index.json"))
    print("PASS: both profile assets verified; six invalid index controls rejected")
    quit()
