extends SceneTree

func _init() -> void:
    call_deferred("check_scene")

func check_scene() -> void:
    var scene = load("res://main.tscn").instantiate()
    root.add_child(scene)
    scene.set_process(false)
    var points := PackedVector3Array()
    for i in range(21): points.append(Vector3(i * 0.01, 0.0, -0.5))
    scene.display_observation({"valid": true, "hands": [{"right": false, "points": points, "stereo": true}]})
    for i in range(21):
        assert(scene.joints[0][i].visible)
        assert(scene.joints[0][i].position.is_equal_approx(points[i]))
        assert(not scene.joints[1][i].visible)
    assert(scene.lines[0][0].visible)
    assert(scene.lines[0][0].position.is_equal_approx((points[0]+points[1])*0.5))
    scene.display_observation({"valid": false})
    for hand in scene.joints:
        for joint in hand: assert(not joint.visible, "Invalid observation must hide old joints")
    for hand in scene.lines:
        for line in hand: assert(not line.visible, "Invalid observation must hide old bones")
    scene.trial_state = {"phase": "prepare", "remaining": 12}
    scene.update_status({"valid": false})
    assert("Starting in 12 seconds" in scene.status.text)
    assert(not scene.xr_initialized, "This regression must run with XR disabled")
    print("PASS: joint placement, bone placement, stale-geometry removal, countdown, XR disabled")
    scene.queue_free()
    quit()
