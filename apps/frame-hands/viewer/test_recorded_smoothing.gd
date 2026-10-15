extends SceneTree
const Smoother = preload("res://smooth_hands.gd")

func percentile(values: Array, fraction: float) -> float:
    if values.is_empty(): return 0.0
    values.sort()
    return float(values[int(round((values.size()-1)*fraction))])

func pose_step(hands: Array, previous: Dictionary) -> float:
    var maximum := 0.0
    for hand in hands:
        var points: PackedVector3Array = hand.points
        if previous.has(hand.id):
            var before: PackedVector3Array = previous[hand.id]
            for k in range(21): maximum=maxf(maximum,points[k].distance_to(before[k]))
        previous[hand.id]=points.duplicate()
    return maximum

func _init() -> void:
    var path := OS.get_environment("FRAME_HANDS_REPLAY_POSES")
    assert(not path.is_empty() and FileAccess.file_exists(path))
    var frames: Array = []
    for line in FileAccess.get_file_as_string(path).split("\n",false):
        var fields := line.split(" ",false)
        assert(fields.size()==194)
        var t := float(fields[0])
        if frames.is_empty() or float(frames[-1].time)!=t: frames.append({"time":t,"hands":[]})
        var points := PackedVector3Array()
        for k in range(21):
            var start := 68+k*3 # original --poses filtered head-frame positions
            points.append(Vector3(float(fields[start]),float(fields[start+1]),float(fields[start+2])))
        var hands: Array = frames[-1].hands
        var side := fields[2]=="R"
        var duplicate := false
        for hand in hands:
            if hand.right==side: duplicate=true
        # This clip has only IDs 1/2 in age order, matching publication priority.
        if not duplicate: hands.append({"id":int(fields[1]),"right":side,"stereo":int(fields[3])>=2,"points":points})
    var smooth = Smoother.new()
    var render_frames := 0
    var baseline_changes := 0
    var smooth_changes := 0
    var baseline_steps: Array = []
    var smooth_steps: Array = []
    var costs: Array = []
    var baseline_previous: Dictionary = {}
    var smooth_previous: Dictionary = {}
    var index := 0
    var now: float = frames[0].time+0.012
    var end: float = frames[-1].time+0.012
    while now<=end:
        while index+1<frames.size() and float(frames[index+1].time)+0.012<=now: index+=1
        var obs := {"valid":true,"hands":frames[index].hands}
        var start := Time.get_ticks_usec()
        var result = smooth.sample(obs,1.0/72)
        costs.append(Time.get_ticks_usec()-start)
        assert(result.hands.size()==obs.hands.size(),"Smoother changed measured hand availability")
        for k in range(result.hands.size()):
            assert(result.hands[k].id==obs.hands[k].id and result.hands[k].right==obs.hands[k].right)
        var baseline_step := pose_step(obs.hands,baseline_previous)
        var smooth_step := pose_step(result.hands,smooth_previous)
        baseline_steps.append(baseline_step*1000)
        smooth_steps.append(smooth_step*1000)
        if baseline_step>0.000001: baseline_changes+=1
        if smooth_step>0.000001: smooth_changes+=1
        render_frames+=1
        now+=1.0/72
    var report := {"kind":"Recorded head-frame pose replay, simulated presentation timing; no XR/head-pose replay",
        "recorded_observations":frames.size(),"render_frames":render_frames,
        "baseline_changed_poses":baseline_changes,"smooth_changed_poses":smooth_changes,
        "availability_and_labels_preserved":true,
        "baseline_render_step_p95_mm":percentile(baseline_steps,0.95),
        "smooth_render_step_p95_mm":percentile(smooth_steps,0.95),
        "smoother_cpu_us_p50":percentile(costs,0.5),"smoother_cpu_us_p95":percentile(costs,0.95),
        "simulated_sensor_to_arrival_ms":12,"simulated_render_hz":72,"physical_acceptance_verified":false}
    assert(smooth_changes>render_frames*0.9)
    var output := OS.get_environment("FRAME_HANDS_REPLAY_REPORT")
    if not output.is_empty():
        var file := FileAccess.open(output,FileAccess.WRITE)
        file.store_string(JSON.stringify(report,"  ")+"\n")
        file.close()
    print(JSON.stringify(report))
    quit()
