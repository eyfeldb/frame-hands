extends "res://native_bridge.gd"
var failures:=0
func check(ok: bool,label: String):
    if not ok:failures+=1;push_error(label)
func observation(t: float,id: int=7) -> Dictionary:
    var assets=Assets.load_verified("res://profile-rigs.json")
    var a=[]
    for v in assets[0].limits:a.append((float(v[0])+float(v[1]))*.5)
    return {"sequence":int(t*1000),"capture_ns":int(t*1e9),"context_valid":true,"age_ms":50.0,"published_wall_s":Time.get_unix_time_from_system(),"hands":[{"side":0,"source_id":id,"wrist_world_m":[[1,0,0,-.2],[0,1,0,1],[0,0,1,-.5]],"joint_angles":a}]}
func warm(t: float,id: int=7):
    for i in 5:
        var at:=t+float(i)/30.0
        consume(observation(at,id),at+.05)
        render_output(at+.05,1.0/90)
func _initialize():
    super._initialize()
    # Fresh observations match the accepted native filter; only gaps differ.
    var old=load("res://hand_motion.gd").new()
    var candidate=Motion.new()
    var assets=Assets.load_verified("res://profile-rigs.json")
    candidate.configure_limits(assets[0].limits)
    var angles=observation(1.0).hands[0].joint_angles
    for i in 12:
        var t:=.2+float(i)/30.0
        for state in [old,candidate]:state.observe(20,t,t,t+.05,Vector3(float(i)*.002,0,0),Quaternion.IDENTITY,angles,"predict")
        for tick in 3:
            var now:=t+.05+float(tick)/90.0
            var a: Dictionary=old.sample(now,1.0/90,"predict")
            var c: Dictionary=candidate.sample(now,1.0/90,"predict")
            check(a.visible and c.visible and a.position.distance_to(c.position)<1e-9,"fresh wrist filtering differs")
            check(a.orientation==c.orientation and a.angles==c.angles,"fresh rotation/shape filtering differs")
    warm(1.0)
    var capture: float=states[0].capture_local
    var original: int=captures[0]
    var held:=render_output(capture+.300,1.0/90)
    check(held.size()==1,"steady hand missing at 300 ms")
    if held.size():
        check(held[0].capture_ms==original and held[0].until_ms==original+350,"wire expiry renewed or truncated")
        check(held[0].points.size()==21,"joint reconstruction missing")
    check(states[0].extra_position.length()<=.010001,"gap displacement exceeds one cm")
    check(render_output(capture+.351,1.0/90).is_empty(),"hand survives 350 ms")
    consume(observation(2.0,8),2.05)
    check(render_output(2.171,1.0/90).is_empty(),"new identity receives long hold")
    warm(3.0,9)
    states[0].velocity=Vector3(1.8,0,0);states[0].acceleration=80
    check(is_equal_approx(states[0].visibility_deadline(),.250),"fast hand does not use 250 ms")
    states[0].pending=true
    check(is_equal_approx(states[0].visibility_deadline(),.170),"ambiguous hand receives extended hold")
    consume({"context_valid":false},3.3)
    check(render_output(3.31,.011).is_empty(),"invalid head/context does not immediately clear")
    warm(4.0,10)
    var stamp_before: float=states[0].stamp
    var captures_before: int=captures[0]
    consume(observation(4.0,10),4.3)
    check(states[0].stamp==stamp_before and captures[0]==captures_before,"duplicate capture renews hold")
    var last: float=states[0].capture_local
    render_output(last+.29,.011)
    consume(observation(last+.25,10),last+.30)
    var reacquired:=render_output(last+.30,.011)
    check(reacquired.size()==1 and states[0].corrections>0,"same-ID reacquisition does not blend from held pose")
    print("NATIVE GAP failures=",failures)
    quit(1 if failures else 0)
