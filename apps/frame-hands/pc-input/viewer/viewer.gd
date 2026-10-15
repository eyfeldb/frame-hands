extends Node3D

const RAW_PARENTS = [0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
const BONE_PARENTS = [-1,0,1,2,3,4,1,6,7,8,9,1,11,12,13,14,1,16,17,18,19,1,21,22,23,24,0,0,0,0,0]
const COLORS = [Color(0.1,0.85,1.0),Color(1.0,0.48,0.1)]
var camera: XRCamera3D
var status: Label3D
var labels: Array[Label3D] = []
var raw_meshes: Array[MeshInstance3D] = []
var candidate_meshes: Array[MeshInstance3D] = []
var bone_meshes: Array[MeshInstance3D] = []
var axes_meshes: Array[MeshInstance3D] = []
var path = OS.get_environment("FRAME_HANDS_PC_VIEW")
var started = Time.get_ticks_msec()
var limit = 180
var check_only = false
var last_stamp: int = -1
var last_received: int = -1000
var layer = OS.get_environment("FRAME_HANDS_VIEW_LAYER")

func material(color: Color, foreground: bool = false) -> StandardMaterial3D:
    var m = StandardMaterial3D.new()
    m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
    m.albedo_color = color
    m.no_depth_test = foreground
    m.vertex_color_use_as_albedo = true
    return m

func mesh_node(color: Color, foreground: bool = false) -> MeshInstance3D:
    var node = MeshInstance3D.new()
    node.mesh = ImmediateMesh.new()
    node.material_override = material(color, foreground)
    camera.add_child(node)
    return node

func _ready():
    if layer.is_empty(): layer = "all"
    if layer not in ["all", "candidate", "source", "steam"]:
        push_error("Unknown hand display layer")
        get_tree().quit(2)
        return
    check_only = "--check-scene" in OS.get_cmdline_user_args()
    if not check_only:
        var xr = XRServer.find_interface("OpenXR")
        if xr == null or not xr.initialize():
            push_error("OpenXR initialization failed")
            get_tree().quit(1)
            return
        if OS.get_environment("FRAME_HANDS_NO_SCENE_INPUT") == "1":
            if "diagnostic_idle" not in xr.get_action_sets():
                push_error("Missing unbound diagnostic action set")
                get_tree().quit(3)
                return
            for action_set in xr.get_action_sets():
                # Godot requires at least one active set for xrSyncActions.
                # This set has no controller bindings and consumes no inputs.
                var enabled = action_set == "diagnostic_idle"
                xr.set_action_set_active(action_set, enabled)
                if xr.is_action_set_active(action_set) != enabled:
                    push_error("Failed to configure diagnostic scene action set: " + action_set)
                    get_tree().quit(3)
                    return
                print("Diagnostic scene action set: ", action_set, " active=", enabled)
        get_viewport().use_xr = true
    var origin = XROrigin3D.new()
    add_child(origin)
    camera = XRCamera3D.new()
    origin.add_child(camera)
    status = Label3D.new()
    status.position = Vector3(0,0.23,-0.95)
    status.font_size = 38
    status.pixel_size = 0.00065
    status.no_depth_test = true
    camera.add_child(status)
    for side in 2:
        raw_meshes.append(mesh_node(Color.WHITE,true))
        candidate_meshes.append(mesh_node(Color.GREEN,true))
        bone_meshes.append(mesh_node(COLORS[side]))
        axes_meshes.append(mesh_node(Color.WHITE))
        var label = Label3D.new()
        label.font_size = 30
        label.pixel_size = 0.0006
        label.modulate = Color.GREEN if layer == "candidate" else (Color.WHITE if layer == "source" else COLORS[side])
        label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
        label.no_depth_test = true
        camera.add_child(label)
        labels.append(label)
    var configured = OS.get_environment("FRAME_HANDS_PC_SECONDS")
    if configured.is_valid_int(): limit = int(configured)
    print("PC skeleton scene initialized; layer="+layer+"; white=source21 green=candidate cyan=left31 orange=right31")
    if check_only:
        draw_test()
        if not path.is_empty():
            check_only = false
            _process(0)
            var held_test = "--check-held" in OS.get_cmdline_user_args()
            if layer == "candidate":
                for side in 2:
                    assert(candidate_meshes[side].visible and not raw_meshes[side].visible and not bone_meshes[side].visible and not axes_meshes[side].visible)
            elif layer == "source":
                assert(raw_meshes[0].visible and raw_meshes[1].visible == not held_test)
                for side in 2: assert(not candidate_meshes[side].visible and not bone_meshes[side].visible and not axes_meshes[side].visible)
            elif layer == "steam":
                assert(bone_meshes[0].visible and bone_meshes[1].visible == not held_test)
                for side in 2: assert(not candidate_meshes[side].visible and not raw_meshes[side].visible)
            else:
                assert(raw_meshes[0].visible and bone_meshes[0].visible)
                if held_test:
                    assert(not raw_meshes[1].visible and not bone_meshes[1].visible and candidate_meshes[1].visible)
                else:
                    assert(raw_meshes[1].visible and bone_meshes[1].visible)
            assert(labels[0].text == "LEFT  id 7")
            if labels[1].visible: assert(labels[1].text == "RIGHT  id 8")
            print("PASS binary frame decoding and both hand drawings")
        get_tree().quit()

func point(data: PackedByteArray, offset: int) -> Vector3:
    return Vector3(data.decode_float(offset),data.decode_float(offset+4),data.decode_float(offset+8))

func line(mesh: ImmediateMesh, a: Vector3, b: Vector3, color: Color):
    mesh.surface_set_color(color)
    mesh.surface_add_vertex(a)
    mesh.surface_set_color(color)
    mesh.surface_add_vertex(b)

func draw_points(node: MeshInstance3D, points: Array[Vector3], parents: Array, color: Color):
    var mesh: ImmediateMesh = node.mesh
    mesh.clear_surfaces()
    mesh.surface_begin(Mesh.PRIMITIVE_LINES)
    for i in points.size():
        # Aux joints are shown as crosses, without the long root-to-finger links.
        if i > 0 and i < 26: line(mesh,points[parents[i]],points[i],color)
        var radius = 0.0025 if i < 26 else 0.004
        for axis in [Vector3.RIGHT,Vector3.UP,Vector3.BACK]:
            line(mesh,points[i]-axis*radius,points[i]+axis*radius,color)
    mesh.surface_end()

func draw_test():
    for side in 2:
        var points: Array[Vector3] = []
        for i in 31: points.append(Vector3(float(i)*0.005,0,-0.5))
        draw_points(bone_meshes[side],points,BONE_PARENTS,COLORS[side])

func _process(_delta):
    if check_only: return
    var remaining = max(0,limit-int((Time.get_ticks_msec()-started)/1000))
    if remaining <= 0: get_tree().quit(); return
    var data = FileAccess.get_file_as_bytes(path) if FileAccess.file_exists(path) else PackedByteArray()
    var v2 = data.size()==1864 and data.slice(0,7).get_string_from_ascii()=="FHVIEW2"
    var fresh = (v2 or (data.size()==1352 and data.slice(0,7).get_string_from_ascii()=="FHVIEW1")) and data[7]==0
    if fresh:
        var stamp = int(data.decode_u64(8))
        if stamp != last_stamp:
            last_stamp = stamp
            last_received = Time.get_ticks_msec()
        fresh = Time.get_ticks_msec()-last_received < 250
    var lines = ["LEFT cyan   |   RIGHT orange", "White: tracker 21 points   •   Color: SteamVR 31 bones", "Wrist axes: X red, Y green, Z blue   •   %ds remaining" % remaining]
    if OS.get_environment("FRAME_HANDS_STABILITY_PREVIEW") == "1":
        lines.append("Green: stability candidate   •   Yellow: brief held estimate")
    if layer == "candidate": lines = ["Smoothed hands", "Green: tracked   •   Yellow: brief held pose", "%ds remaining" % remaining]
    elif layer == "source": lines = ["Source hands", "White: current measured pose", "%ds remaining" % remaining]
    elif layer == "steam": lines = ["SteamVR hands", "Left: blue   •   Right: orange", "%ds remaining" % remaining]
    for side in 2:
        var base = 16+side*(924 if v2 else 668)
        var flags = int(data.decode_u32(base+4)) if fresh else 0
        if layer == "candidate": flags &= 12
        elif layer == "source": flags &= 1
        elif layer == "steam": flags &= 2
        raw_meshes[side].visible = bool(flags&1)
        bone_meshes[side].visible = bool(flags&2)
        axes_meshes[side].visible = bool(flags&2)
        candidate_meshes[side].visible = bool(flags&4)
        labels[side].visible = bool(flags)
        var raw: Array[Vector3] = []
        var bones: Array[Vector3] = []
        var candidate: Array[Vector3] = []
        if flags&1:
            for i in 21: raw.append(point(data,base+8+12*i))
            draw_points(raw_meshes[side],raw,RAW_PARENTS,Color.WHITE)
        if flags&2:
            for i in 31: bones.append(point(data,base+260+12*i))
            draw_points(bone_meshes[side],bones,BONE_PARENTS,COLORS[side])
            var mesh: ImmediateMesh = axes_meshes[side].mesh
            mesh.clear_surfaces();mesh.surface_begin(Mesh.PRIMITIVE_LINES)
            for a in 3: line(mesh,bones[1],point(data,base+632+12*a),[Color.RED,Color.GREEN,Color.BLUE][a])
            mesh.surface_end()
        if flags&4:
            for i in 21: candidate.append(point(data,base+668+12*i))
            draw_points(candidate_meshes[side],candidate,RAW_PARENTS,Color.YELLOW if flags&8 else Color.GREEN)
        if flags:
            labels[side].position = (bones[1] if flags&2 else (raw[0] if flags&1 else candidate[0]))+Vector3(0,0.055,0)
            var ident = data.decode_u32(base if flags&3 else base+920)
            labels[side].text = ("LEFT" if side==0 else "RIGHT")+"  id %d" % ident
        if layer == "all":
            lines.append(("Left" if side==0 else "Right")+": source "+("live" if flags&1 else "missing")+" / SteamVR "+("live" if flags&2 else "missing"))
        else:
            lines.append(("Left" if side==0 else "Right")+": "+("held" if flags&8 else ("visible" if flags else "missing")))
    if not fresh: lines.append("Waiting for the optical source")
    status.text = "\n".join(lines)
