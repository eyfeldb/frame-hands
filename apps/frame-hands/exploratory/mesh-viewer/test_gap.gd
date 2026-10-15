extends SceneTree
const Motion = preload("res://gap_motion.gd")
var failures := 0
func check(ok: bool, what: String):
    if not ok: failures+=1;push_error(what)
func _initialize():
    var m = Motion.new()
    var angles := []
    for i in 20: angles.append(0.0)
    for i in 5:
        var t := 1.0+float(i)/30.0
        check(m.observe(7,t,t,t+0.06,Vector3(float(i)*0.003,0,0),Quaternion.IDENTITY,angles,"predict"),"accepted warmup")
        m.sample(t+0.06,1.0/90.0,"predict")
    var capture: float=m.capture_local
    var start: Dictionary=m.sample(capture+0.16,1.0/90.0,"predict")
    var grace: Dictionary=m.sample(capture+0.18,1.0/90.0,"predict")
    check(grace.visible and grace.grace,"conditional grace absent")
    check(not grace.input_observed,"predicted grace became observed input")
    check(grace.position.x>start.position.x,"grace flat-held instead of continuing")
    check(m.extra_position.length()<=0.010001,"extra motion unbounded")
    check(m.capture_local==capture,"capture age renewed")
    check(not m.sample(capture+0.221,1.0/90.0,"predict").visible,"expiry unbounded")
    m.observe(8,2.0,2.0,2.05,Vector3.ZERO,Quaternion.IDENTITY,angles,"predict")
    check(m.extra_position==Vector3.ZERO,"source inherited extra velocity")
    check(not m.sample(2.171,.011,"predict").visible,"cold identity received grace")
    m=Motion.new()
    for i in 5: m.observe(1,3.0+i*.033,3.0+i*.033,3.05+i*.033,Vector3.ZERO,Quaternion.IDENTITY,angles,"raw")
    check(not m.sample(m.capture_local+.171,.011,"raw").visible,"raw got grace")
    print("GAP MOTION: ",failures," failures")
    quit(1 if failures else 0)
