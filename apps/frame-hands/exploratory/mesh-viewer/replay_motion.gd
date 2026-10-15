extends SceneTree
const Motion = preload("res://hand_motion.gd")
const Viewer = preload("res://viewer.gd")

func percentile(values: Array, fraction: float) -> float:
    if values.is_empty(): return 0
    values.sort()
    return float(values[mini(values.size()-1,int(fraction*(values.size()-1)))])

func _initialize():
    call_deferred("run_replay")

func run_replay():
    var path := OS.get_environment("FRAME_UME_REPLAY")
    var data = JSON.parse_string(FileAccess.get_file_as_string(path))
    var rig = JSON.parse_string(FileAccess.get_file_as_string(OS.get_environment("FRAME_UME_RIG")))
    var report := {"name":data.name,"scope":data.scope,"modes":{}}
    for mode in ["raw","smooth","predict"]:
        var states := [Motion.new(),Motion.new()]
        var index := 0
        var accepted := 0
        var rejected := 0
        var visible := 0
        var held := 0
        var steps := []
        var offset := []
        var all_steps := []
        var identity_changes := 0
        var visibility_losses := 0
        var was_visible := [false,false]
        var gap_start := [-1.0,-1.0]
        var gaps := []
        var gap_lengths := []
        var opacity_sum := 0.0
        var faint_frames := 0
        var last := [{},{}]
        var frames := int(ceil(float(data.events[-1].now)*90.0))+1
        if data.has("ticks"): frames = data.ticks.size()
        for i in frames:
            var now := float(data.ticks[i]) if data.has("ticks") else float(i)/90.0
            var delta := float(data.ticks[i])-float(data.ticks[i-1]) if data.has("ticks") and i > 0 else 1.0/90.0
            while index < data.events.size() and float(data.events[index].now) <= now:
                var event: Dictionary = data.events[index]
                if not event.valid:
                    for state in states: state.clear()
                else:
                    for hand in event.hands:
                        var side := int(hand.side)
                        var id := int(hand.source_id)
                        if states[1-side].source_id == id: states[1-side].clear()
                        var pose := Viewer.transform_from_rows(hand.wrist_world_m)
                        if states[side].observe(id,float(event.stamp),float(event.capture),now,pose.origin,pose.basis.get_rotation_quaternion(),hand.joint_angles,mode):
                            accepted += 1
                        else: rejected += 1
                index += 1
            for side in 2:
                var sample: Dictionary = states[side].sample(now,delta,mode)
                if sample.visible:
                    if gap_start[side] >= 0:
                        var duration_ms := (now-float(gap_start[side]))*1000.0
                        gaps.append({"side":side,"start_s":gap_start[side],"end_s":now,"duration_ms":duration_ms,"censored":false})
                        gap_lengths.append(duration_ms)
                        gap_start[side] = -1.0
                    opacity_sum += float(sample.opacity)
                    faint_frames += int(float(sample.opacity) < 0.25)
                    visible += 1
                    held += int(sample.held)
                    offset.append(sample.position.distance_to(states[side].raw_position)*1000)
                    if not last[side].is_empty():
                        all_steps.append(sample.position.distance_to(last[side].position)*1000)
                        identity_changes += int(last[side].id != sample.source_id)
                    if not last[side].is_empty() and last[side].id == sample.source_id:
                        steps.append(sample.position.distance_to(last[side].position)*1000)
                    last[side] = {"id":sample.source_id,"position":sample.position}
                else:
                    if was_visible[side]:
                        visibility_losses += 1
                        gap_start[side] = now
                    last[side] = {}
                was_visible[side] = sample.visible
        var short_gaps := 0
        for duration in gap_lengths: short_gaps += int(float(duration) <= 25.0)
        var final_time := float(data.ticks[-1]) if data.has("ticks") else float(frames-1)/90.0
        for side in 2:
            if gap_start[side] >= 0:
                gaps.append({"side":side,"start_s":gap_start[side],"end_s":final_time,"duration_ms":(final_time-float(gap_start[side]))*1000.0,"censored":true})
        report.modes[mode] = {"display_frames":frames,"visible_hand_frames":visible,"held_hand_frames":held,"accepted_observations":accepted,"rejected_observations":rejected,
            "wrist_display_step_mm_p95":percentile(steps,.95),"wrist_display_step_mm_max":percentile(steps,1.0),"distance_from_latest_raw_mm_p95":percentile(offset,.95),
            "all_visible_wrist_step_mm_p95":percentile(all_steps,.95),"all_visible_wrist_step_mm_max":percentile(all_steps,1.0),
            "visible_identity_changes":identity_changes,"visibility_losses":visibility_losses,
            "completed_hidden_gaps":gap_lengths.size(),"completed_hidden_gaps_at_most_25ms":short_gaps,
            "completed_hidden_gap_ms_p50":percentile(gap_lengths,.5),"completed_hidden_gap_ms_p95":percentile(gap_lengths,.95),"completed_hidden_gap_ms_max":percentile(gap_lengths,1.0),
            "opacity_weighted_hand_frames":opacity_sum,"mean_opacity_when_visible":opacity_sum/maxi(1,visible),"faint_visible_hand_frames":faint_frames,"hidden_gaps":gaps}
    print(JSON.stringify(report))
    var f := FileAccess.open(path+".results.json",FileAccess.WRITE)
    f.store_string(JSON.stringify(report,"  "))
    quit()
