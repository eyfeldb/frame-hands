extends SceneTree
const Policy = preload("res://packet_policy.gd")
const Motion = preload("res://hand_motion.gd")
var failures := 0
var checks := 0
func check(ok: bool, label: String):
    checks += 1
    if not ok:
        failures += 1
        push_error(label)
func hand(side: int,id: int) -> Dictionary:
    return {"side":side,"source_id":id,"marker":str(id)}
func angles() -> Array:
    var a := []
    for i in 20:a.append(0.0)
    return a
func _initialize():call_deferred("run_test")
func run_test():
    var a:=hand(0,1)
    var b:=hand(1,2)
    var c:=hand(0,3)
    var clean=[a,b]
    check(Policy.route(clean).hands==clean,"clean packet changed")
    check(Policy.route([]).valid,"empty packet invalid")
    var mixed=[a,b,c]
    var saved=mixed.duplicate(true)
    for packet in [[a,b,c],[c,b,a],[b,a,c],[c,a,b],[a,c,b],[b,c,a]]:
        var r: Dictionary=Policy.route(packet)
        check(r.valid and r.hands==[b] and r.ambiguous_sides==[0] and r.discarded==2,"order selected an ambiguous winner")
    check(mixed==saved,"routing mutated input")
    for bad in [[a,a],[hand(0,1),hand(1,1)],[hand(-1,2)],[hand(2,2)],[hand(0,-1)],[{}],[1],[{"side":0.5,"source_id":1}]]:
        check(not Policy.route(bad).valid,"malformed identity admitted")
    # Ambiguous data must never renew an existing capture clock.
    var states=[Motion.new(),Motion.new()]
    for side in 2:states[side].observe(side+1,0.0,0.0,0.05,Vector3(side,0,0),Quaternion.IDENTITY,angles(),"predict")
    var packet=[hand(0,1),hand(0,2)]
    var visible_before := false
    var hidden_after := true
    for tick in range(5,91):
        var now:=float(tick)/90.0
        var routed: Dictionary=Policy.route(packet)
        check(routed.hands.is_empty(),"ambiguous frame fabricated observation")
        for side in 2:
            var sample: Dictionary=states[side].sample(now,1.0/90.0,"predict")
            check(states[side].capture_local==0.0,"ambiguity renewed age")
            if now<0.1:visible_before=visible_before or sample.visible
            if now>=0.17:hidden_after=hidden_after and not sample.visible
    check(visible_before and hidden_after,"bounded hold did not expire")
    # Whole-packet structural invalidity still requires clear; unique side can update.
    states[0].clear()
    states[1].clear()
    var r: Dictionary=Policy.route(mixed)
    for h in r.hands:states[int(h.side)].observe(int(h.source_id),1.0,1.0,1.05,Vector3.ZERO,Quaternion.IDENTITY,angles(),"predict")
    check(not states[0].active and states[1].source_id==2,"unambiguous side was not isolated")
    var report={"checks":checks,"failures":failures}
    var output:=FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT"),FileAccess.WRITE)
    output.store_string(JSON.stringify(report));print(JSON.stringify(report));quit(1 if failures else 0)
