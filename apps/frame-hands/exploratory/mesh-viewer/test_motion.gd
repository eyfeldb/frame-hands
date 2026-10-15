extends SceneTree
const Motion = preload("res://hand_motion.gd")
const Rig = preload("res://ume_rig.gd")
const Viewer = preload("res://viewer.gd")
var failures := 0

func check(value: bool, description: String):
    if not value:
        failures += 1
        push_error(description)

func observe(state, id: int, t: float, p: Vector3, mode: String = "predict", delay: float = 0.04):
    return state.observe(id,t,t,t+delay,p,Quaternion.IDENTITY,Array(PackedFloat64Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])),mode)

func _initialize():
    call_deferred("run_tests")

func run_tests():
    var state = Motion.new()
    observe(state,1,1.0,Vector3.ZERO)
    observe(state,1,1.033333,Vector3(0.02,0,0))
    var original_capture: float = state.capture_local
    check(not observe(state,1,1.033333,Vector3.ONE),"duplicate must be rejected")
    check(state.capture_local == original_capture,"duplicate renewed hold")
    var early: Dictionary = state.sample(1.14,0.011,"predict")
    check(early.visible and early.opacity < 1.0,"brief dropout must fade")
    check(not state.sample(1.21,0.011,"predict").visible,"stale hand must disappear")
    observe(state,8,2.0,Vector3.ZERO)
    observe(state,8,2.03333,Vector3(0.04,0,0))
    var capped: Dictionary = state.sample(2.1,0.5,"predict")
    check(capped.position.distance_to(state.position) <= 0.025001,"prediction exceeded distance cap")
    observe(state,9,2.06666,Vector3(0.5,0,0))
    check(state.velocity == Vector3.ZERO and state.position.x == 0.5,"new identity inherited velocity")
    check(not observe(state,9,2.1,Vector3(1.2,0,0)),"one-frame teleport was accepted")
    check(observe(state,9,2.13333,Vector3(0.51,0,0)),"valid recovery was rejected")
    check(state.position.x < 0.53,"outlier poisoned the filter")
    var spike_angles := Array(PackedFloat64Array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    check(not state.observe(9,2.16666,2.16666,2.20666,Vector3(0.51,0,0),Quaternion(Vector3.UP,PI),spike_angles,"predict"),"one-frame orientation flip accepted")
    state.clear()
    check(not state.sample(2.15,0.011,"predict").visible,"invalid context did not clear")
    var metrics := {}
    for mode in ["raw","smooth","predict"]:
        var filter = Motion.new()
        var errors := 0.0
        var differences := 0.0
        var previous := 0.0
        var last := -1
        var count := 0
        for i in 270:
            var now := float(i)/90.0
            var k := floori((now-0.045)*30.0+0.00001)
            if k < 0: continue
            if k != last:
                last = k
                var capture := float(k)/30.0
                # Deterministic 1 mm observation jitter around a fixed-speed wrist.
                observe(filter,1,capture,Vector3(0.4*capture+0.001*sin(k*2.3),0,0),mode,now-capture)
            var result: Dictionary = filter.sample(now,1.0/90.0,mode)
            if i > 45 and result.visible:
                errors += pow(result.position.x-0.4*now,2)
                differences += pow((result.position.x-previous)-0.4/90.0,2)
                count += 1
            previous = result.position.x
        metrics[mode] = {"constant_speed_rmse_mm":sqrt(errors/count)*1000,"step_residual_rmse_mm":sqrt(differences/count)*1000}
    check(metrics.predict.step_residual_rmse_mm < metrics.raw.step_residual_rmse_mm,"display prediction failed to reduce stepping")
    check(metrics.predict.constant_speed_rmse_mm < metrics.smooth.constant_speed_rmse_mm,"prediction failed to reduce smoothing lag")
    var stop_filter = Motion.new()
    var stop_max := 0.0
    var stop_last := -1
    for i in 180:
        var now := float(i)/90.0
        var k := floori((now-0.04)*30.0)
        if k < 0: continue
        if k != stop_last:
            stop_last = k
            var capture := float(k)/30.0
            var measured := minf(capture,0.8)*0.8
            observe(stop_filter,1,capture,Vector3(measured,0,0),"predict",now-capture)
        var value: Dictionary = stop_filter.sample(now,1.0/90.0,"predict")
        if now >= 0.8 and value.visible: stop_max = maxf(stop_max,value.position.x-0.64)
    metrics.abrupt_stop_max_overshoot_mm = stop_max*1000
    check(stop_max < 0.025001,"stop prediction overshot its bound")
    check(stop_filter.velocity.length() < 0.001,"stop retained velocity")
    # Hold still with 2 mm input jitter; no source availability changes.
    var smooth = Motion.new()
    var square := 0.0
    for i in 120:
        observe(smooth,1,float(i)/30.0,Vector3(0.002*sin(i*2.1),0,0),"smooth",0.03)
        var sample: Dictionary = smooth.sample(float(i)/30.0+0.03,1.0/30.0,"smooth")
        if i>20:square += sample.position.x*sample.position.x
    metrics.stationary_smoothed_rms_mm = sqrt(square/99.0)*1000
    check(metrics.stationary_smoothed_rms_mm < 1.0,"stationary jitter not sufficiently reduced")
    # GPU rig's actual bone matrices against independent Torch skinning.
    var root_path := OS.get_environment("FRAME_UME_TEST")
    var rig = JSON.parse_string(FileAccess.get_file_as_string(root_path+"/rig.json"))
    var packet = JSON.parse_string(FileAccess.get_file_as_string(root_path+"/pose.json"))
    var expected = JSON.parse_string(FileAccess.get_file_as_string(root_path+"/expected.json"))
    var maximum := 0.0
    for side in 2:
        var hand = Rig.new()
        root.add_child(hand)
        hand.configure(rig,side,StandardMaterial3D.new())
        hand.transform = Viewer.transform_from_rows(packet.hands[side].wrist_world_m)
        hand.apply_angles(packet.hands[side].joint_angles)
        var actual: Array = hand.debug_vertices()
        for i in actual.size():
            maximum = maxf(maximum,Rig.vec(actual[i]).distance_to(Rig.vec(expected[side][i])))
        hand.queue_free()
    metrics.rig_max_difference_mm = maximum*1000
    check(maximum < 0.000005,"GPU skeleton does not match reference mesh")
    metrics.failures = failures
    print(JSON.stringify(metrics))
    var f := FileAccess.open(root_path+"/tests.json",FileAccess.WRITE)
    f.store_string(JSON.stringify(metrics,"  "))
    quit(1 if failures else 0)
