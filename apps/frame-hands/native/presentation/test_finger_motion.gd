extends SceneTree
const MotionControl = preload("res://correction_motion.gd")
const Candidate = preload("res://hand_motion.gd")
var failures := 0
var report := {}
func check(condition: bool,label: String):
    if not condition:
        failures += 1
        push_error(label)
func shape(value: float = 0.0) -> Array:
    var a := []
    for i in 20: a.append(value if i>=4 and i<8 else 0.0)
    return a
func send(state,t: float,a: Array,id: int=1,p: Vector3=Vector3.ZERO,mode: String="predict"):
    return state.observe(id,t,t,t+0.020,p,Quaternion.IDENTITY,a,mode)
func sample(state,t: float,mode: String="predict") -> Dictionary:
    return state.sample(t+0.020,1.0/90.0,mode)
func _initialize():call_deferred("run_tests")
func run_tests():
    var base=MotionControl.new();var candidate=Candidate.new()
    # Gradual fast fist, including reversal, remains bit-identical.
    for i in 30:
        var t:=1.0+float(i)/30.0
        var value:=minf(float(i)*0.4,1.6) if i<15 else maxf(1.6-float(i-15)*0.4,0.0)
        for state in [base,candidate]:send(state,t,shape(value))
        var x:=sample(base,t);var y:=sample(candidate,t)
        check(x.angles==y.angles,"ordinary fist/reversal changed")
        check(x.position==y.position and x.opacity==y.opacity,"wrist/life changed")
    check(candidate.finger_holds==0,"ordinary gesture was held")
    # One-frame finger corruption is removed, with other fingers and wrist intact.
    base=MotionControl.new();candidate=Candidate.new()
    for state in [base,candidate]:send(state,3.0,shape());sample(state,3.0)
    var spike:=shape(1.6);spike[12]=0.3
    for state in [base,candidate]:send(state,3.033333,spike,1,Vector3(.01,0,0))
    var x:=sample(base,3.033333);var y:=sample(candidate,3.033333)
    check(x.angles[4]>.2 and y.angles[4]==0,"single-frame corruption not suppressed")
    check(x.angles[12]==y.angles[12] and x.position==y.position,"unrelated channel held")
    report.spike_control_rad=x.angles[4];report.spike_candidate_rad=y.angles[4]
    check(y.inferred_fingers==[1],"missing inferred marker")
    check(not send(candidate,3.033333,spike),"duplicate renewed finger")
    send(candidate,3.066667,shape());y=sample(candidate,3.066667)
    check(y.angles[4]==0 and y.inferred_fingers.is_empty(),"spike return polluted history")
    # A genuine abrupt gesture confirms on the second update, then eases to baseline.
    base=MotionControl.new();candidate=Candidate.new()
    for state in [base,candidate]:send(state,4.0,shape());sample(state,4.0)
    send(candidate,4.033333,shape(1.6));sample(candidate,4.033333)
    sample(candidate,4.055556) # Actual90Hz display tick before the next30Hz observation.
    send(candidate,4.066667,shape(1.6));y=sample(candidate,4.066667)
    check(candidate.finger_confirmations==1 and y.angles[4]==0,"confirmation not eased from display")
    var previous:=0.0;var maximum_step:=0.0
    for i in range(1,10):
        var t:=4.066667+float(i)/90.0
        if i%3==0: send(candidate,t,shape(1.6))
        y=sample(candidate,t)
        check(y.angles[4]>=previous-0.000001 and y.angles[4]<=1.600001,"confirmation overshoot/reversal")
        maximum_step=maxf(maximum_step,y.angles[4]-previous);previous=y.angles[4]
    check(y.angles[4]>1.5 and y.finger_correcting.is_empty(),"confirmed gesture froze")
    report.confirmed_max_step_rad=maximum_step
    # Pending fingers cannot cross identity, raw mode, expiry or a rejected wrist jump.
    candidate=Candidate.new();send(candidate,5.0,shape());sample(candidate,5.0)
    check(not send(candidate,5.033333,shape(1.6),1,Vector3(.5,0,0)),"wrist corruption accepted")
    check(candidate.finger_holds==0,"rejected wrist mutated finger state")
    send(candidate,5.066667,shape(1.6));sample(candidate,5.066667)
    send(candidate,5.1,shape(1.4),2);y=sample(candidate,5.1)
    check(y.inferred_fingers.is_empty() and y.angles[4]==1.4,"identity inherited finger hold")
    check(not candidate.sample(5.28,1.0/90.0,"predict").visible,"hold extended hand expiry")
    send(candidate,5.3,shape());sample(candidate,5.3)
    send(candidate,5.333333,shape(1.6),2,Vector3.ZERO,"raw");y=sample(candidate,5.333333,"raw")
    check(y.angles[4]==1.6 and y.inferred_fingers.is_empty(),"raw path altered")
    # Conflicting proposals cannot keep a visible finger frozen indefinitely.
    candidate=Candidate.new();send(candidate,6.0,shape());sample(candidate,6.0)
    send(candidate,6.033333,shape(1.3));sample(candidate,6.033333)
    send(candidate,6.066667,shape(2.1));sample(candidate,6.066667)
    sample(candidate,6.088889)
    send(candidate,6.1,shape(1.3));y=sample(candidate,6.1)
    check(candidate.finger_forced_corrections==1 and y.finger_hold_ages.is_empty(),"ambiguous hold deadline extended")
    check(1 in y.inferred_fingers and 1 in y.finger_correcting,"unconfirmed correction lost inference marker")
    for i in range(1,10):
        var t:=6.1+float(i)/90.0
        if i%3==0:send(candidate,t,shape(1.3))
        y=sample(candidate,t)
    check(y.angles[4]>1.2 and y.finger_correcting.is_empty(),"ambiguous release stayed frozen")
    report.failures=failures
    var output=FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT"),FileAccess.WRITE)
    output.store_string(JSON.stringify(report,"  "));print(JSON.stringify(report));quit(1 if failures else 0)
