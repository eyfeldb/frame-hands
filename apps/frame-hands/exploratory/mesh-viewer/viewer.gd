extends Node3D

const Motion = preload("res://hand_motion.gd")
const GapMotion = preload("res://gap_motion.gd")
const ContinuityMotion = preload("res://continuity_motion.gd")
const Rig = preload("res://ume_rig.gd")
var rigs: Array = []
var motion: Array = []
var mapping := Transform3D.IDENTITY
var presentation := OS.get_environment("FRAME_UME_PRESENTATION")
var continuity := OS.get_environment("FRAME_UME_CONTINUITY")
var rig_path := OS.get_environment("FRAME_UME_RIG")
var camera: Camera3D
var status: Label3D
var meshes: Array[MeshInstance3D] = []
var materials: Array[StandardMaterial3D] = []
var path := OS.get_environment("FRAME_UME_VIEW")
var start := Time.get_ticks_msec()
var sequence := -1
var pose_stamp := -1.0
var last_capture := 0.0
var fixture := false
var last_poll := 0
var save_frames := 0
var trace: FileAccess

func _ready():
    fixture = "--fixture" in OS.get_cmdline_user_args()
    if fixture:
        camera = Camera3D.new()
        add_child(camera)
        camera.fov = 72
    else:
        var xr = XRServer.find_interface("OpenXR")
        if xr == null or not xr.initialize():
            push_error("OpenXR initialization failed")
            get_tree().quit(2)
            return
        if "diagnostic_idle" not in xr.get_action_sets():
            push_error("Missing unbound diagnostic action set")
            get_tree().quit(3)
            return
        for action_set in xr.get_action_sets():
            xr.set_action_set_active(action_set, action_set == "diagnostic_idle")
        get_viewport().use_xr = true
        var origin := XROrigin3D.new()
        add_child(origin)
        camera = XRCamera3D.new()
        origin.add_child(camera)
    if presentation.is_empty(): presentation = "predict"
    if continuity.is_empty(): continuity = "native"
    if continuity not in ["accepted","native","bridge"]:
        push_error("Unknown hand continuity mode")
        get_tree().quit(6)
        return
    assert(presentation in ["raw","smooth","predict"])
    camera.near = 0.03
    status = Label3D.new()
    camera.add_child(status)
    status.position = Vector3(0,0.24,-0.85)
    status.font_size = 27
    status.pixel_size = 0.0006
    status.text = "UME TRACK • MESH LAB\nWaiting for hands"
    status.modulate = Color(0.7,0.8,0.9)
    var env := WorldEnvironment.new()
    env.environment = Environment.new()
    env.environment.background_mode = Environment.BG_COLOR
    env.environment.background_color = Color(0.025,0.035,0.055)
    env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    env.environment.ambient_light_color = Color(0.72,0.8,1)
    env.environment.ambient_light_energy = 0.65
    add_child(env)
    for rotation_degrees_value in [Vector3(-35,-25,0),Vector3(20,140,0)]:
        var light := DirectionalLight3D.new()
        light.rotation_degrees = rotation_degrees_value
        light.light_energy = 1.0
        add_child(light)
    for color in [Color(0.12,0.72,0.87),Color(0.96,0.49,0.2)]:
        var node := MeshInstance3D.new()
        var material := StandardMaterial3D.new()
        material.albedo_color = color
        material.roughness = 0.65
        material.cull_mode = BaseMaterial3D.CULL_DISABLED
        node.material_override = material
        node.visible = false
        add_child(node)
        meshes.append(node)
        materials.append(material)
    if not rig_path.is_empty():
        var rig = JSON.parse_string(FileAccess.get_file_as_string(rig_path))
        assert(rig is Dictionary and rig.version == 1)
        for side in 2:
            var hand = Rig.new()
            add_child(hand)
            hand.configure(rig,side,materials[side])
            hand.visible = false
            rigs.append(hand)
            var state = Motion.new() if continuity == "accepted" else (GapMotion.new() if continuity == "bridge" else ContinuityMotion.new())
            if continuity != "accepted" and not state.configure_limits(rig.get("limits",[])):
                push_error("Native-style continuity requires verified model joint limits")
                get_tree().quit(7)
                return
            motion.append(state)
    var trace_path := OS.get_environment("FRAME_UME_RENDER_TRACE")
    if not trace_path.is_empty():
        trace = FileAccess.open(trace_path,FileAccess.WRITE)
    print("UmeTrack mesh viewer ready; one solid layer; cyan left / orange right")

func draw_hand(hand: Dictionary, indices: PackedInt32Array):
    var side := int(hand.side)
    if side < 0 or side > 1: return
    var vertices := PackedVector3Array()
    var normals := PackedVector3Array()
    for v in hand.vertices_head_m:
        vertices.append(Vector3(v[0],v[1],v[2]))
    for v in hand.normals_head:
        normals.append(Vector3(v[0],v[1],v[2]))
    if vertices.size() != normals.size() or vertices.size() < 100: return
    for i in indices:
        if i < 0 or i >= vertices.size(): return
    var arrays := []
    arrays.resize(Mesh.ARRAY_MAX)
    arrays[Mesh.ARRAY_VERTEX] = vertices
    arrays[Mesh.ARRAY_NORMAL] = normals
    arrays[Mesh.ARRAY_INDEX] = indices
    var mesh := ArrayMesh.new()
    mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES,arrays)
    meshes[side].mesh = mesh
    # Place the received head-relative mesh in scene space once. Head movement
    # between inference updates then remains handled by the XR runtime.
    meshes[side].global_transform = camera.global_transform
    meshes[side].visible = true

static func transform_from_rows(rows) -> Transform3D:
    return Transform3D(Basis(Vector3(rows[0][0],rows[1][0],rows[2][0]),Vector3(rows[0][1],rows[1][1],rows[2][1]),Vector3(rows[0][2],rows[1][2],rows[2][2])),Vector3(rows[0][3],rows[1][3],rows[2][3]))

func consume_pose(value: Dictionary, now: float):
    if rigs.size() != 2: return
    if not value.get("context_valid",false):
        for state in motion: state.clear()
        return
    var timestamp := float(value.capture_ns)*0.000000001
    if timestamp <= pose_stamp: return
    pose_stamp = timestamp
    mapping = camera.global_transform*transform_from_rows(value.head_world_m).affine_inverse()
    var transport_age := maxf(0,Time.get_unix_time_from_system()-float(value.published_wall_s))
    var capture := now-float(value.age_ms)*0.001-transport_age
    if fixture: capture = now
    var ids := []
    var sides := []
    for hand in value.hands:
        var side := int(hand.side)
        var id := int(hand.source_id)
        if side < 0 or side > 1 or id in ids or side in sides:
            for state in motion: state.clear()
            return
        ids.append(id)
        sides.append(side)
        if motion[1-side].active and motion[1-side].source_id == id:
            motion[1-side].clear()
        var pose := transform_from_rows(hand.wrist_world_m)
        var accepted: bool = motion[side].observe(id,timestamp,capture,now,pose.origin,pose.basis.get_rotation_quaternion(),hand.joint_angles,presentation)
        if trace: trace.store_line(JSON.stringify({"kind":"observation","sequence":sequence,"capture_ns":value.capture_ns,"side":side,"id":id,"accepted":accepted}))

func render_poses(now: float, delta: float):
    for side in rigs.size():
        if fixture: motion[side].capture_local = now
        var state: Dictionary = motion[side].sample(now,delta,presentation)
        rigs[side].visible = state.visible
        if state.visible:
            rigs[side].global_transform = mapping*Transform3D(Basis(state.orientation),state.position)
            rigs[side].apply_angles(state.angles)
            rigs[side].mesh_node.transparency = 1.0-float(state.opacity)
        if trace:
            var row := {"kind":"render","render_tick_s":now,"sequence":sequence,"side":side,"mode":presentation,"continuity":continuity,"visible":state.visible}
            if state.visible:
                row.merge({"id":state.source_id,"age_s":state.age,"prediction_s":state.predicted_seconds,"held":state.held,"opacity":state.opacity,
                    "wrist_world_m":[state.position.x,state.position.y,state.position.z],"quaternion":[state.orientation.x,state.orientation.y,state.orientation.z,state.orientation.w],"angles":Array(state.angles),"correcting":state.get("correcting",false)})
            trace.store_line(JSON.stringify(row))
    if trace: trace.flush()

func _process(_delta):
    if camera == null: return
    var now := Time.get_ticks_msec()
    # Pose presentation runs on every render tick, independent of camera delivery.
    last_poll = now
    if FileAccess.file_exists(path):
        var file := FileAccess.open(path,FileAccess.READ)
        if file:
            var value = JSON.parse_string(file.get_as_text())
            if value is Dictionary and value.get("version",0) == 2 and int(value.sequence) != sequence:
                sequence = int(value.sequence)
                consume_pose(value,float(now)*0.001)
                status.text = "UME TRACK • MESH LAB\nCyan left  /  Amber right  •  "+presentation
            elif value is Dictionary and value.get("version",0) == 1 and int(value.sequence) != sequence:
                sequence = int(value.sequence)
                last_capture = float(value.capture_wall_s)
                var fresh := fixture or Time.get_unix_time_from_system()-last_capture < 0.25
                for mesh in meshes: mesh.visible = false
                if fresh:
                    var triangles := PackedInt32Array(value.triangles)
                    for hand in value.hands: draw_hand(hand,triangles)
                var count := int(meshes[0].visible)+int(meshes[1].visible)
                status.text = "UME TRACK • MESH LAB\nCyan left  /  Amber right  •  %d hands  •  %.0f ms" % [count,float(value.get("age_ms",0))]
                if not fresh: status.text += "\nWaiting for fresh camera input"
                if trace:
                    trace.store_line(JSON.stringify({"wall_s":Time.get_unix_time_from_system(),"sequence":sequence,"capture_wall_s":last_capture,"hands":count,"camera_transform":str(camera.global_transform)}))
                    trace.flush()
    render_poses(float(now)*0.001,_delta)
    if rigs.is_empty() and not fixture and Time.get_unix_time_from_system()-last_capture > 0.25:
        for mesh in meshes: mesh.visible = false
        status.text = "UME TRACK • MESH LAB\nWaiting for fresh hands"
    if fixture:
        save_frames += 1
        if save_frames == 15:
            assert((meshes[0].visible and meshes[1].visible) or (rigs.size()==2 and rigs[0].visible and rigs[1].visible), "Both fixture meshes must render")
            var output := OS.get_environment("FRAME_UME_SCREENSHOT")
            if not output.is_empty():
                await RenderingServer.frame_post_draw
                get_viewport().get_texture().get_image().save_png(output)
            print("PASS: both solid hand meshes rendered")
            get_tree().quit()
    var limit := int(OS.get_environment("FRAME_UME_SECONDS"))
    if not fixture and limit > 0 and now-start > limit*1000:
        get_tree().quit()
