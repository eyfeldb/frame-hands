extends SceneTree
const Baseline = preload("res://hand_motion.gd")
const Candidate = preload("res://continuity_motion.gd")
var failures := 0
var reports := {}

func check(condition: bool, message: String):
    if not condition:
        failures += 1
        push_error(message)

func angles(value: float = 0.0) -> Array:
    var result := []
    for i in 20: result.append(value)
    return result

func observation(state, id: int, t: float, now: float, p: Vector3, q: Quaternion = Quaternion.IDENTITY, a: Array = [], mode: String = "predict"):
    return state.observe(id,t,t,now,p,q,angles() if a.is_empty() else a,mode)

func compare(a: Dictionary,b: Dictionary, label: String):
    check(a.visible == b.visible,label+" visibility")
    if a.visible and b.visible:
        check(a.position.distance_to(b.position)<0.0000001,label+" position")
        check(a.orientation==b.orientation,label+" rotation")
        for i in 20: check(absf(a.angles[i]-b.angles[i])<0.0000001,label+" angle")
        check(a.opacity==b.opacity,label+" opacity")

func _initialize():
    call_deferred("run_tests")

func run_tests():
    # The candidate must not add lag to uninterrupted motion or abrupt stops.
    var a = Baseline.new()
    var b = Candidate.new()
    var last := -1
    var stop_overshoot := 0.0
    for i in 240:
        var now := float(i)/90.0
        var k := floori((now-0.04)*30.0)
        if k<0: continue
        if k!=last:
            last=k
            var capture := float(k)/30.0
            var position := Vector3(0.8*minf(capture,0.8),0.001*sin(k*2.1),0)
            var rotation := Quaternion(Vector3.UP,0.2*sin(capture))
            for state in [a,b]: observation(state,1,capture,now,position,rotation,angles(0.3*sin(capture)))
        var x: Dictionary = a.sample(now,1.0/90.0,"predict")
        var y: Dictionary = b.sample(now,1.0/90.0,"predict")
        compare(x,y,"ordinary motion")
        if now>=0.8 and y.visible: stop_overshoot=maxf(stop_overshoot,y.position.x-0.64)
    check(b.corrections==0,"ordinary motion started catch-up")
    reports.unchanged_motion_stop_overshoot_mm=stop_overshoot*1000
    # Two consistent observations confirm a 400mm tracking correction.
    a=Baseline.new();b=Candidate.new()
    for state in [a,b]:
        observation(state,1,1.0,1.02,Vector3.ZERO)
        state.sample(1.02,1.0/90.0,"predict")
        check(not observation(state,1,1.033333,1.053333,Vector3(0.4,0,0),Quaternion(Vector3.UP,0.6),angles(0.6)),"first jump not quarantined")
        state.sample(1.075,1.0/90.0,"predict")
        check(observation(state,1,1.066667,1.086667,Vector3(0.4,0,0),Quaternion(Vector3.UP,0.6),angles(0.6)),"confirmed correction not accepted")
    var previous := [Vector3.ZERO,Vector3.ZERO]
    var max_step := [0.0,0.0]
    for i in 8:
        var now := 1.086667+float(i)/90.0
        var samples := [a.sample(now,1.0/90.0,"predict"),b.sample(now,1.0/90.0,"predict")]
        for j in 2:
            var value: Dictionary = samples[j]
            max_step[j]=maxf(max_step[j],value.position.distance_to(previous[j])*1000)
            previous[j]=value.position
            check(value.position.x>=-0.000001 and value.position.x<=0.400001,"correction overshoot")
            for angle in value.angles: check(angle>=-0.000001 and angle<=0.600001,"finger correction overshoot")
        check(samples[0].opacity==samples[1].opacity,"catch-up extended visual life")
        if i==0:check(samples[1].position.length()<0.000001,"correction did not start from displayed pose")
        if i>=6:compare(samples[0],samples[1],"correction deadline")
    check(max_step[1]<max_step[0]*0.4,"correction did not reduce jump")
    check(b.corrections==1,"correction count")
    reports.confirmed_correction_max_step_mm={"baseline":max_step[0],"candidate":max_step[1]}
    # Reset bursts cannot postpone the original deadline; ID changes never blend.
    observation(b,1,1.2,1.21,Vector3(0.4,0,0))
    b.sample(1.21,1.0/90.0,"predict")
    b._observation_now=1.22
    b.reset_pose(1,1.21,1.21,Vector3(0.6,0,0),Quaternion.IDENTITY,PackedFloat64Array(angles()))
    b.sample(1.23,1.0/90.0,"predict")
    b._observation_now=1.24
    b.reset_pose(1,1.23,1.23,Vector3(0.5,0,0),Quaternion.IDENTITY,PackedFloat64Array(angles()))
    var settled: Dictionary=b.sample(1.281,1.0/90.0,"predict")
    check(not settled.correcting,"reset burst extended deadline")
    observation(b,2,1.29,1.30,Vector3.ONE)
    var changed: Dictionary=b.sample(1.30,1.0/90.0,"predict")
    check(not changed.correcting and changed.position==Vector3.ONE,"ID change inherited catch-up")
    check(not observation(b,2,1.29,1.31,Vector3.ZERO),"duplicate accepted")
    check(not b.sample(1.47,1.0/90.0,"predict").visible,"expiry extended")
    observation(b,2,1.48,1.49,Vector3.ZERO)
    check(not b.sample(1.49,1.0/90.0,"predict").correcting,"expired hand inherited catch-up")
    b.clear()
    check(not b.sample(1.5,1.0/90.0,"predict").visible,"clear retained hand")
    # Joint-limit projection occurs before filtering. Raw mode stays diagnostic.
    b=Candidate.new()
    var limits := []
    for i in 20: limits.append([-0.2,0.8])
    check(b.configure_limits(limits),"valid limits rejected")
    check(not b.configure_limits([[1.0,0.0]]),"invalid limits accepted")
    observation(b,1,4.0,4.02,Vector3.ZERO,Quaternion.IDENTITY,angles(2.0))
    var bounded: Dictionary=b.sample(4.02,1.0/90.0,"predict")
    for angle in bounded.angles:check(angle<=0.8,"cold start violates limits")
    observation(b,1,4.033333,4.053333,Vector3.ZERO,Quaternion.IDENTITY,angles(-2.0))
    bounded=b.sample(4.053333,1.0/90.0,"predict")
    for angle in bounded.angles:check(angle>=-0.2 and angle<=0.8,"filtered output violates limits")
    check(b.joint_clamps==40,"projection counter")
    b.clear()
    observation(b,2,5.0,5.02,Vector3.ZERO,Quaternion.IDENTITY,angles(2.0),"raw")
    for angle in b.sample(5.02,1.0/90.0,"raw").angles:check(angle==2.0,"raw mode changed")
    reports.failures=failures
    var output=FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT"),FileAccess.WRITE)
    output.store_string(JSON.stringify(reports,"  "))
    print(JSON.stringify(reports))
    quit(1 if failures else 0)
