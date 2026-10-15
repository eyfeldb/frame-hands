extends SceneTree
const Baseline=preload("res://gap_motion.gd")
const Candidate=preload("res://game_motion.gd")
var failures:=0
func check(ok: bool,what: String):
    if not ok:failures+=1;push_error(what)
func _initialize():
    var old=Baseline.new();var candidate=Candidate.new()
    var angles=[]
    for i in 22:angles.append(0.0)
    for i in 12:
        var t:=1.0+float(i)/30.0
        for state in [old,candidate]:state.observe(7,t,t,t+0.088,Vector3(float(i)*0.002,0,0),Quaternion.IDENTITY,angles,"predict")
        for tick in 3:
            var now:=t+0.088+float(tick)/90
            var a: Dictionary=old.sample(now,1.0/90,"predict")
            var b: Dictionary=candidate.sample(now,1.0/90,"predict")
            check(a.visible and b.visible and a.position.distance_to(b.position)<1e-9,"fresh wrist differs")
            check(a.orientation==b.orientation and a.angles==b.angles,"fresh shape differs")
    var capture: float=candidate.capture_local
    var v: Dictionary=candidate.sample(capture+.300,1.0/90,"predict")
    check(v.visible and not v.input_observed,"300ms grace missing or called observed")
    check(candidate.extra_position.length()<=.010001 and candidate.extra_rotation.length()<=.060001,"unbounded extra motion")
    check(candidate.capture_local==capture,"timestamp renewed")
    check(not candidate.sample(capture+.351,1.0/90,"predict").visible,"350ms expiry missing")
    candidate.observe(8,2.0,2.0,2.088,Vector3.ZERO,Quaternion.IDENTITY,angles,"predict")
    check(not candidate.sample(2.171,.011,"predict").visible,"new identity got hold")
    for i in 5:candidate.observe(9,3.0+i*.033,3.0+i*.033,3.088+i*.033,Vector3.ZERO,Quaternion.IDENTITY,angles,"predict")
    candidate.velocity=Vector3(1.8,0,0);candidate.acceleration=80
    check(is_equal_approx(candidate.visibility_deadline(),.250),"fast confirmed track deadline")
    candidate.pending=true
    check(is_equal_approx(candidate.visibility_deadline(),.170),"ambiguous track got extended hold")
    print("GAME MOTION failures=",failures)
    quit(1 if failures else 0)
