extends SceneTree
const Motion=preload("res://game_motion.gd")
const Rig=preload("res://rig_joints.gd")
const MAP=[1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25]
var path:=OS.get_environment("FRAME_UME_VIEW")
var rig=Rig.new()
var states=[]
var captures=[0,0]
var sequence: int=-1
var source_ms: int=0
var receipt:=0.0
var context:=false
func _initialize():
    Engine.max_fps=90
    var asset=JSON.parse_string(FileAccess.get_file_as_string(OS.get_environment("FRAME_UME_RIG")))
    if not asset is Dictionary or not rig.configure(asset):quit(2);return
    for side in 2:
        var state=Motion.new()
        if not state.configure_limits(asset.get("limits",[])):quit(3);return
        states.append(state)
func xf(rows) -> Transform3D:
    return Transform3D(Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2])),Vector3(rows[0][3],rows[1][3],rows[2][3]))
func _process(delta: float) -> bool:
    if states.size()!=2:return false
    var now:=Time.get_ticks_usec()*1e-6
    if FileAccess.file_exists(path):
        var parser=JSON.new()
        if parser.parse(FileAccess.get_file_as_string(path))==OK and parser.data is Dictionary:
            var packet: Dictionary=parser.data
            if int(packet.get("sequence",-1))!=sequence:
                sequence=int(packet.get("sequence",-1));receipt=now
                source_ms=int(packet.get("pc_sent_ms",0));context=packet.get("context_valid",false)
                var age: float=float(packet.get("age_ms",1000))*.001+maxf(0,Time.get_unix_time_from_system()-float(packet.get("published_wall_s",0)))
                if not context or age<0:
                    for state in states:state.clear()
                elif age<.250:
                    for hand in packet.get("hands",[]):
                        var side:=int(hand.side)
                        if side<0 or side>1:continue
                        var pose:=xf(hand.wrist_world_m)
                        if states[1-side].active and states[1-side].source_id==int(hand.source_id):states[1-side].clear()
                        if states[side].observe(int(hand.source_id),float(packet.capture_ns)*1e-9,now-age,now,pose.origin,pose.basis.get_rotation_quaternion(),hand.joint_angles,"predict"):
                            captures[side]=int(packet.pc_capture_ms)
    var output=[]
    if now-receipt>=.500:context=false
    for side in 2:
        var display: Dictionary=states[side].sample(now,delta,"predict")
        if context and display.visible:
            var joints: PackedVector3Array=rig.joints(display.angles,Transform3D(Basis(display.orientation),display.position),side)
            var points=[]
            for index in MAP:points.append([joints[index].x,joints[index].y,joints[index].z])
            output.append({"side":side,"id":states[side].source_id+1,"capture_ms":captures[side],"until_ms":captures[side]+int(states[side].visibility_deadline()*1000),"points":points})
    print("FHMODEL "+JSON.stringify({"source_ms":source_ms,"wall_s":Time.get_unix_time_from_system(),"hands":output}))
    return false
