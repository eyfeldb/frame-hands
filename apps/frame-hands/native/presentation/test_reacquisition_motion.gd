extends SceneTree
const Baseline = preload("res://hand_motion.gd")
const Candidate = preload("res://reacquisition_motion.gd")
var checks := 0
var failures := 0
func check(ok: bool, label: String):
    checks += 1
    if not ok:
        push_error(label)
        failures += 1
func angles() -> Array:
    var a := []
    for i in 20:a.append(0.0)
    return a
func same(a: Dictionary,b: Dictionary):
    check(a.visible==b.visible,"visibility changed")
    if a.visible and b.visible:
        check(a.position==b.position and a.orientation==b.orientation and a.angles==b.angles,"unaffected pose changed")
        check(a.opacity==b.opacity and a.age==b.age,"freshness changed")
func _initialize():call_deferred("run_test")
func run_test():
    var summary := {}
    for scenario in ["recover","duplicate","new_id","hidden","disabled","raw","fast_clean","rejected","cleared","new_id_then_packet"]:
        var a=Baseline.new()
        var b=Candidate.new()
        if scenario=="disabled":b.reacquisition_enabled=false
        var mode := "raw" if scenario=="raw" else "predict"
        for state in [a,b]:state.observe(1,0.0,0.0,0.05,Vector3.ZERO,Quaternion.IDENTITY,angles(),mode)
        var last_a: Dictionary
        var last_b: Dictionary
        for tick in range(5,14):
            var now:=float(tick)/90.0
            last_a=a.sample(now,1.0/90.0,mode)
            last_b=b.sample(now,1.0/90.0,mode)
            same(last_a,last_b)
        check(last_b.held,"fixture did not reach held display")
        var time := 0.1
        var now := 0.15
        var id := 1
        var target := Vector3(0.25,0,0)
        if scenario=="rejected":target=Vector3(1,0,0)
        if scenario=="cleared":
            a.clear()
            b.clear()
        if scenario=="new_id_then_packet":
            id=2
            for state in [a,b]:state.observe(2,0.08,0.08,0.145,Vector3(0.1,0,0),Quaternion.IDENTITY,angles(),mode)
        if scenario=="duplicate":time=0.0
        if scenario=="new_id":id=2
        if scenario=="hidden":
            now=0.22
            time=0.17
            for state in [a,b]:state.sample(0.18,1.0/90.0,mode)
        if scenario=="fast_clean":
            a.clear()
            b.clear()
            for state in [a,b]:state.observe(1,0.08,0.08,0.13,Vector3.ZERO,Quaternion.IDENTITY,angles(),mode)
            last_a=a.sample(0.14444444,1.0/90.0,mode)
            last_b=b.sample(0.14444444,1.0/90.0,mode)
            time=0.1133333
            now=0.1633333
        var ok_a: bool = a.observe(id,time,time,now,target,Quaternion(Vector3.UP,1.0),angles(),mode)
        var ok_b: bool = b.observe(id,time,time,now,target,Quaternion(Vector3.UP,1.0),angles(),mode)
        check(ok_a==ok_b,"acceptance changed")
        var first_a: Dictionary=a.sample(now,1.0/90.0,mode)
        var first_b: Dictionary=b.sample(now,1.0/90.0,mode)
        if scenario=="recover":
            check(ok_b and b.reacquisition_corrections==1,"ordinary held reacquisition not bridged")
            check(first_b.position==last_b.position,"recovery snapped on first tick")
            check(first_a.position.distance_to(last_a.position)>0.05,"fixture did not exercise jump")
            check(a.capture_local==b.capture_local and first_a.opacity==first_b.opacity,"visibility budget changed")
            var peak := 0.0
            var previous: Vector3=first_b.position
            for k in range(1,7):
                var t:=now+float(k)/90.0
                var sa: Dictionary=a.sample(t,1.0/90.0,mode)
                var sb: Dictionary=b.sample(t,1.0/90.0,mode)
                check(sa.visible==sb.visible and sa.opacity==sb.opacity,"recovery extended visibility")
                peak=maxf(peak,previous.distance_to(sb.position))
                previous=sb.position
                if k==6:check(sa.position==sb.position and not sb.correcting,"catch-up did not finish")
            summary.recovery_first_step_mm=first_a.position.distance_to(last_a.position)*1000.0
            summary.recovery_peak_step_mm=peak*1000.0
        else:
            check(b.reacquisition_corrections==0,"bridge crossed forbidden boundary")
            same(first_a,first_b)
    summary.checks=checks
    summary.failures=failures
    var output:=FileAccess.open(OS.get_environment("FRAME_MOTION_RESULT"),FileAccess.WRITE)
    output.store_string(JSON.stringify(summary,"  "))
    print(JSON.stringify(summary))
    quit(1 if failures else 0)
