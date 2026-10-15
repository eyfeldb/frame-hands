extends SceneTree
const Baseline = preload("res://hand_motion.gd")
const PacketPolicy = preload("res://packet_policy.gd")
const Candidate = preload("res://hand_motion.gd")
func percentile(values: Array, p: float):
    if values.is_empty():return 0.0
    values.sort()
    return values[mini(values.size()-1,int(p*(values.size()-1)))]
func pose_from_rows(rows: Array) -> Transform3D:
    return Transform3D(Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2])),Vector3(rows[0][3],rows[1][3],rows[2][3]))
func _initialize():call_deferred("run_replay")
func run_replay():
    var data=JSON.parse_string(FileAccess.get_file_as_string(OS.get_environment("FRAME_MOTION_INPUT")))
    var states := [[Baseline.new(),Baseline.new()],[Candidate.new(),Candidate.new()]]
    var observations := FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT")+".observations.jsonl",FileAccess.WRITE)
    var ambiguous_packets := 0
    var quarantined_packets := 0
    var quarantined_hands := 0
    var policy_enabled := OS.get_environment("FRAME_PACKET_POLICY_DISABLED") != "1"
    var index := 0
    var last := [[{},{}],[{},{}]]
    var steps := [[],[]]
    var visible := [0,0]
    var accepted := [0,0]
    var rejected := [0,0]
    var differs := 0
    var mismatch := 0
    var maximum_offset := 0.0
    var per_frame := FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT")+".frames.jsonl",FileAccess.WRITE)
    var frames := int(ceil(float(data.events[-1].now)*90.0))+1
    if data.has("ticks"):frames=data.ticks.size()
    var began := Time.get_ticks_usec()
    for i in frames:
        var now := float(data.ticks[i]) if data.has("ticks") else float(i)/90.0
        var delta := float(data.ticks[i])-float(data.ticks[i-1]) if data.has("ticks") and i>0 else 1.0/90.0
        while index<data.events.size() and float(data.events[index].now)<=now:
            var event: Dictionary=data.events[index]
            if event.get("ambiguous",false): ambiguous_packets += 1
            for variant in 2:
                if not event.valid:
                    for state in states[variant]:state.clear()
                else:
                    var packet_hands: Array = event.hands
                    if variant==1 and policy_enabled:
                        var routed: Dictionary = PacketPolicy.route(packet_hands)
                        if not routed.valid:
                            for state in states[variant]:state.clear()
                            observations.store_line(JSON.stringify({"event":index,"variant":variant,"now":now,"invalid_clear":true}))
                            continue
                        packet_hands = routed.hands
                        if routed.discarded:
                            quarantined_packets += 1
                            quarantined_hands += int(routed.discarded)
                            observations.store_line(JSON.stringify({"event":index,"variant":variant,"now":now,"quarantined_sides":routed.ambiguous_sides,"discarded":routed.discarded}))
                    var packet_ids := []
                    var packet_sides := []
                    for hand in packet_hands:
                        var side:=int(hand.side)
                        var id:=int(hand.source_id)
                        if side < 0 or side > 1 or id in packet_ids or side in packet_sides:
                            for state in states[variant]:state.clear()
                            observations.store_line(JSON.stringify({"event":index,"variant":variant,"now":now,"ambiguous_clear":true}))
                            break
                        packet_ids.append(id)
                        packet_sides.append(side)
                        if states[variant][1-side].source_id==id:states[variant][1-side].clear()
                        var pose:=pose_from_rows(hand.wrist_world_m)
                        var ok: bool = states[variant][side].observe(id,float(event.stamp),float(event.capture),now,pose.origin,pose.basis.get_rotation_quaternion(),hand.joint_angles,"predict")
                        if ok:accepted[variant]+=1
                        else:rejected[variant]+=1
                        observations.store_line(JSON.stringify({"event":index,"variant":variant,"now":now,"side":side,"id":id,"accepted":ok,"stamp":states[variant][side].stamp,"capture":states[variant][side].capture_local,"pending":states[variant][side].pending}))
            index+=1
        for side in 2:
            var samples:=[]
            var logged:=[]
            for variant in 2:
                var sample: Dictionary=states[variant][side].sample(now,delta,"predict")
                samples.append(sample)
                if sample.visible:
                    visible[variant]+=1
                    if not last[variant][side].is_empty():steps[variant].append(sample.position.distance_to(last[variant][side].position)*1000)
                    last[variant][side]={"position":sample.position,"id":sample.source_id}
                    logged.append({"position":[sample.position.x,sample.position.y,sample.position.z],"orientation":[sample.orientation.w,sample.orientation.x,sample.orientation.y,sample.orientation.z],"angles":Array(sample.angles),"id":sample.source_id,"opacity":sample.opacity,"correcting":sample.get("correcting",false),"age":sample.age,"capture":states[variant][side].capture_local,"held":sample.held,"prediction":sample.predicted_seconds})
                else:
                    last[variant][side]={}
                    logged.append({})
            if samples[0].visible!=samples[1].visible:mismatch+=1
            elif samples[0].visible:
                if samples[0].opacity!=samples[1].opacity or samples[0].source_id!=samples[1].source_id:mismatch+=1
                var distance: float=samples[0].position.distance_to(samples[1].position)*1000
                maximum_offset=maxf(maximum_offset,distance)
                if distance>0.000001:differs+=1
            per_frame.store_line(JSON.stringify({"now":now,"side":side,"values":logged}))
    var report:={"name":data.name,"scope":data.scope,"frames":frames,"visible":visible,"accepted":accepted,"rejected":rejected,"visibility_id_opacity_mismatches":mismatch,"different_position_frames":differs,"max_candidate_offset_mm":maximum_offset,"baseline_step_p95_mm":percentile(steps[0],.95),"candidate_step_p95_mm":percentile(steps[1],.95),"baseline_max_step_mm":percentile(steps[0],1.0),"candidate_max_step_mm":percentile(steps[1],1.0),"ambiguous_packets":ambiguous_packets,"quarantined_packets":quarantined_packets,"quarantined_hands":quarantined_hands,"policy_enabled":policy_enabled,"corrections":states[1][0].corrections+states[1][1].corrections,"paired_replay_wall_ms":(Time.get_ticks_usec()-began)/1000.0,"timing_scope":"Includes both variants, parsing/serialization and file output; not rendering latency."}
    var output:=FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT"),FileAccess.WRITE)
    output.store_string(JSON.stringify(report,"  "));print(JSON.stringify(report))
    quit(1 if not policy_enabled and (mismatch or accepted[0]!=accepted[1] or rejected[0]!=rejected[1]) else 0)
