extends SceneTree

func _init() -> void:
    assert(Engine.has_singleton("FrameHandsNative"),"Native optical bridge did not load")
    var native = Engine.get_singleton("FrameHandsNative")
    var observation: Dictionary = native.sample()
    assert(not observation.valid,"XR-disabled viewer must not return valid world poses")
    assert(observation.reason == "OpenXR capture-time transform unavailable")
    print("PASS: native extension loaded; XR-disabled mode rejects world poses")
    quit()
