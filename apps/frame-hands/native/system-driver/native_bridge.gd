extends SceneTree
const Motion=preload("res://native_game_motion.gd")
const Rig=preload("res://rig_joints.gd")
const Assets=preload("res://profile_assets.gd")
const MAP=[1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25]
var path:=OS.get_environment("FRAME_UME_VIEW")
var states=[]
var rigs=[]
var captures=[0,0]
var sequence: int=-1
var source_ms: int=0
var receipt:=0.0
var context:=false
var stamp:=-1.0
func _initialize():
    Engine.max_fps=90
    var assets=Assets.load_verified("res://profile-rigs.json")
    if assets.size()!=2:quit(2);return
    for side in 2:
        var rig=Rig.new()
        if not rig.configure(assets[side]):quit(3);return
        rigs.append(rig)
        var motion=Motion.new()
        if not motion.configure_limits(assets[side].limits):quit(4);return
        states.append(motion)
static func xf(rows) -> Transform3D:
    return Transform3D(Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2])),Vector3(rows[0][3],rows[1][3],rows[2][3]))
func consume(packet: Dictionary,now: float):
    if not packet.get("context_valid",false):
        context=false
        for state in states:state.clear()
        return
    var timestamp:=float(packet.capture_ns)*1e-9
    if timestamp<=stamp:return
    stamp=timestamp
    context=true
    source_ms=int(packet.capture_ns/1000000)
    var age:=float(packet.age_ms)*.001+maxf(0,Time.get_unix_time_from_system()-float(packet.published_wall_s))
    var ids=[]
    var sides=[]
    for hand in packet.hands:
        var side:=int(hand.side)
        var id:=int(hand.source_id)
        if side<0 or side>1 or id in ids or side in sides:
            context=false
            for state in states:state.clear()
            return
        ids.append(id);sides.append(side)
        if states[1-side].active and states[1-side].source_id==id:states[1-side].clear()
        var pose:=xf(hand.wrist_world_m)
        if states[side].observe(id,timestamp,now-age,now,pose.origin,pose.basis.get_rotation_quaternion(),hand.joint_angles,"predict"):
            captures[side]=int(packet.capture_ns/1000000)
func _process(delta: float) -> bool:
    if states.size()!=2:return false
    var now:=Time.get_ticks_usec()*1e-6
    if FileAccess.file_exists(path):
        var parser=JSON.new()
        if parser.parse(FileAccess.get_file_as_string(path))==OK and parser.data is Dictionary:
            var packet: Dictionary=parser.data
            if int(packet.get("sequence",-1))!=sequence:
                sequence=int(packet.get("sequence",-1));receipt=now
                consume(packet,now)
    if now-receipt>=.250:context=false
    var output=render_output(now,delta)
    print("FHMODEL "+JSON.stringify({"source_ms":source_ms,"wall_s":Time.get_unix_time_from_system(),"hands":output}))
    return false

func render_output(now: float,delta: float) -> Array:
    var output=[]
    for side in 2:
        var display: Dictionary=states[side].sample(now,delta,"predict")
        if context and display.visible:
            var joints: PackedVector3Array=rigs[side].joints(display.angles,Transform3D(Basis(display.orientation),display.position),side)
            var points=[]
            for index in MAP:points.append([joints[index].x,joints[index].y,joints[index].z])
            output.append({"side":side,"id":states[side].source_id,"capture_ms":captures[side],"until_ms":captures[side]+int(states[side].visibility_deadline()*1000),"points":points})
    return output if context else []
