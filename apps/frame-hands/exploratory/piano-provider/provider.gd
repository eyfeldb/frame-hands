extends Node
const Motion = preload("res://scripts/frame_provider/gap_motion.gd")
const Rig = preload("res://scripts/frame_provider/rig_joints.gd")
var hands
var camera: Camera3D
var states := []
var rig = Rig.new()
var mapping := Transform3D.IDENTITY
var path := OS.get_environment("FRAME_UME_VIEW")
var sequence := -1
var capture_stamp := 0.0
var observed := [false,false]
var valid := false
var trace: FileAccess

func _ready():
    process_priority=-20
    var asset = JSON.parse_string(FileAccess.get_file_as_string(OS.get_environment("FRAME_UME_RIG")))
    if not asset is Dictionary or not rig.configure(asset):
        push_error("Frame hand rig is unavailable");set_process(false);return
    for side in 2:
        var state=Motion.new()
        if not state.configure_limits(asset.get("limits",[])):set_process(false);return
        states.append(state)
    var target:=OS.get_environment("FRAME_UME_RENDER_TRACE")
    if not target.is_empty():trace=FileAccess.open(target,FileAccess.WRITE)

static func xf(rows) -> Transform3D:
    return Transform3D(Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2])),Vector3(rows[0][3],rows[1][3],rows[2][3]))

static func rigid(rows) -> bool:
    if not rows is Array or rows.size()!=4:return false
    for row in rows:
        if not row is Array or row.size()!=4:return false
        for value in row:
            if not (value is float or value is int) or not is_finite(float(value)):return false
    var basis:=xf(rows).basis
    return absf(basis.determinant()-1.0)<.002 and basis.is_equal_approx(basis.orthonormalized())

func invalidate():
    valid=false
    for side in 2:
        states[side].clear();observed[side]=false
        hands.clear_frame("left" if side==0 else "right")
        hands.clear_visual("left" if side==0 else "right")

func consume(packet: Dictionary, now: float):
    if packet.get("version")!=2 or packet.get("context_valid")!=true:
        invalidate();return
    var stamp: float=float(packet.get("capture_ns",0))*1e-9
    if stamp<=capture_stamp:return
    capture_stamp=stamp
    var age: float=float(packet.get("age_ms",1000))*.001+maxf(0,Time.get_unix_time_from_system()-float(packet.get("published_wall_s",0)))
    if not is_finite(age) or age<0 or age>=.250:
        invalidate();return
    if not rigid(packet.get("head_world_m",[])):invalidate();return
    var entries=packet.get("hands",[])
    if not entries is Array or entries.size()>2:invalidate();return
    var ids:=[];var sides:=[]
    for hand in entries:
        if not hand is Dictionary:invalidate();return
        var side:=int(hand.get("side",-1));var id:=int(hand.get("source_id",-1))
        if side not in [0,1] or id<0 or id in ids or side in sides:invalidate();return
        if hand.get("joint_angles",[]).size()<20 or not rigid(hand.get("wrist_world_m",[])):invalidate();return
        ids.append(id);sides.append(side)
    mapping=(hands.origin.global_transform*XRServer.get_reference_frame()).affine_inverse()*camera.global_transform*xf(packet.head_world_m).affine_inverse()
    valid=true
    observed=[false,false]
    for hand in entries:
        var side:=int(hand.side);var id:=int(hand.source_id)
        if states[side].active and states[side].source_id!=id:hands.clear_frame("left" if side==0 else "right")
        if states[1-side].active and states[1-side].source_id==id:
            states[1-side].clear();hands.clear_frame("right" if side==0 else "left")
        var pose:=xf(hand.wrist_world_m)
        observed[side]=states[side].observe(id,stamp,now-age,now,pose.origin,pose.basis.get_rotation_quaternion(),hand.joint_angles,"predict")
    for side in 2:
        if not observed[side]:hands.clear_frame("left" if side==0 else "right")

func _process(delta: float):
    if states.size()!=2 or not is_instance_valid(hands):return
    var now:=Time.get_ticks_usec()*1e-6
    if FileAccess.file_exists(path):
        var packet=JSON.parse_string(FileAccess.get_file_as_string(path))
        if packet is Dictionary and int(packet.get("sequence",-1))!=sequence:
            sequence=int(packet.get("sequence",-1));consume(packet,now)
    var certainty:=PackedFloat32Array();certainty.resize(26);certainty.fill(1.0)
    for side in 2:
        var name: String="left" if side==0 else "right"
        var motion=states[side]
        var display: Dictionary=motion.sample(now,delta,"predict")
        var input_age: float=now-motion.capture_local
        var input_joints:=PackedVector3Array()
        var visual_joints:=PackedVector3Array()
        if valid and observed[side] and motion.active and input_age>=0 and input_age<.150:
            # No extrapolation for contacts: only the filtered observed pose.
            var pose:=mapping*Transform3D(Basis(motion.orientation),motion.position)
            input_joints=rig.joints(motion.angles,pose,side)
            hands.submit_frame(name,input_joints,certainty,input_age)
        else:hands.clear_frame(name)
        if valid and display.visible:
            var pose:=mapping*Transform3D(Basis(display.orientation),display.position)
            visual_joints=rig.joints(display.angles,pose,side)
            hands.submit_visual(name,visual_joints,certainty,maxf(0,motion.visibility_deadline()-display.age),display.opacity)
        else:hands.clear_visual(name)
        if trace:
            trace.store_line(JSON.stringify({"kind":"render","side":side,"sequence":sequence,"render_tick_s":now,"visible":display.visible,"grace":display.get("grace",false),"input_valid":valid and observed[side] and motion.active and input_age<.150,"age_s":input_age,"source_id":motion.source_id,"input_joints":point_rows(input_joints),"visual_joints":point_rows(visual_joints),"opacity":display.get("opacity",0.0)}))
    if trace:trace.flush()

static func point_rows(points: PackedVector3Array) -> Array:
    var result:=[]
    for point in points:result.append([point.x,point.y,point.z])
    return result
