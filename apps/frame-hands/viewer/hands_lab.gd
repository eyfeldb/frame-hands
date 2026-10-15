extends Node3D

# Diagnostic geometry only. No gesture actions, controller emulation, or prediction.
const LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[0,17],[17,18],[18,19],[19,20]]
var origin: XROrigin3D
var geometry: Node3D
var status: Label3D
var joints: Array = []
var lines: Array = []
var native: Object
var status_clock := 0.0
var elapsed := 0.0
var trial_state: Dictionary = {}
var report_clock := 0.0
var frame_count := 0
var valid_frames := 0
var timeout_seconds := 0.0
var xr_initialized := false
var temporal = preload("res://temporal_hands.gd").new()
var temporal_enabled := false
var smoother = preload("res://smooth_hands.gd").new()
var smoothing_enabled := false
var rendered_hand_frames := 0
var tracking_render_frames := 0
var tracking_elapsed := 0.0
var changed_render_frames := 0
var previous_rendered: Dictionary = {}

func _ready() -> void:
    XRServer.world_scale = 1.0
    origin = XROrigin3D.new()
    add_child(origin)
    var camera := XRCamera3D.new()
    origin.add_child(camera)
    geometry = Node3D.new()
    origin.add_child(geometry)
    status = Label3D.new()
    status.text = "FRAME HANDS LAB\nWaiting for camera observations"
    status.position = Vector3(0,0.18,-0.85)
    status.font_size = 32
    status.pixel_size = 0.0005
    status.no_depth_test = true
    camera.add_child(status)
    for side in range(2):
        var hand_joints: Array = []
        var hand_lines: Array = []
        var material := StandardMaterial3D.new()
        material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
        material.albedo_color = Color(0.15,0.85,1.0) if side == 0 else Color(1.0,0.65,0.18)
        for point in range(21):
            var node := MeshInstance3D.new()
            var sphere := SphereMesh.new()
            sphere.radius = 0.004 if point != 0 else 0.006
            sphere.height = sphere.radius * 2.0
            sphere.radial_segments = 8
            sphere.rings = 4
            node.mesh = sphere
            node.material_override = material
            node.visible = false
            geometry.add_child(node)
            hand_joints.append(node)
        for _link in LINKS:
            var node := MeshInstance3D.new()
            var cylinder := CylinderMesh.new()
            cylinder.top_radius = 0.002
            cylinder.bottom_radius = 0.002
            cylinder.height = 1.0
            cylinder.radial_segments = 6
            node.mesh = cylinder
            node.material_override = material
            node.visible = false
            geometry.add_child(node)
            hand_lines.append(node)
        joints.append(hand_joints)
        lines.append(hand_lines)
    if Engine.has_singleton("FrameHandsNative"):
        native = Engine.get_singleton("FrameHandsNative")
    var xr := XRServer.find_interface("OpenXR")
    if xr and xr.is_initialized():
        get_viewport().use_xr = true
        xr_initialized = true
    timeout_seconds = float(OS.get_environment("FRAME_HANDS_VIEWER_TIMEOUT"))
    temporal_enabled = OS.get_environment("FRAME_HANDS_TEMPORAL_VIEWER")=="1"
    smoothing_enabled = OS.get_environment("FRAME_HANDS_SMOOTH_VIEWER")=="1"

func update_status(observation: Dictionary) -> void:
    var path := OS.get_environment("FRAME_HANDS_TRIAL_STATUS")
    if not path.is_empty() and FileAccess.file_exists(path):
        var parsed = JSON.parse_string(FileAccess.get_file_as_string(path))
        if parsed is Dictionary: trial_state = parsed
    var phase: String = str(trial_state.get("phase", "tracking"))
    var remaining: int = int(trial_state.get("remaining", 0))
    var caption := ""
    match phase:
        "checking": caption = "Keep hands lowered and head still\nChecking cameras"
        "prepare": caption = "Raise both hands, palms toward headset\nStarting in %d seconds" % remaining
        "tracking": caption = "Open and curl your fingers · %d seconds left" % remaining
        "done": caption = "Test finished · lower your hands"
        "error": caption = "Test stopped\n" + str(trial_state.get("detail", ""))
    var detail: String = str(observation.get("reason", "Waiting for observations"))
    if observation.get("valid", false):
        var hands: Array = observation.get("hands", [])
        var stereo := 0
        for hand in hands:
            if hand.stereo: stereo += 1
        detail = "%d hands · %d stereo\nCapture→publish %.1f ms · age %.1f ms" % [hands.size(), stereo, observation.pipeline_ms, observation.age_ms]
    status.text = "FRAME HANDS LAB\n" + caption + "\n" + detail

func write_report(observation: Dictionary) -> void:
    var path := OS.get_environment("FRAME_HANDS_VIEWER_REPORT")
    if path.is_empty(): return
    var report := {"pid": OS.get_process_id(), "xr_initialized": xr_initialized,
        "frames": frame_count, "valid_pose_frames": valid_frames, "elapsed_seconds": elapsed,
        "phase": trial_state.get("phase", "unknown"), "valid": observation.get("valid", false),
        "reason": observation.get("reason", ""), "hands": observation.get("hands", []).size(),
        "pipeline_ms": observation.get("pipeline_ms", 0), "age_ms": observation.get("age_ms", 0),
        "display_latency_measured": false, "temporal_enabled":temporal_enabled,
        "rendered_hand_frames":rendered_hand_frames,"temporal_observations":temporal.accepted_updates,
        "smoothing_enabled":smoothing_enabled,
        "display_lead_ms":1000.0*(float(observation.get("display_seconds",0))-float(observation.get("now_seconds",0))),
        "prediction_horizon_ms":1000.0*(float(observation.get("display_seconds",0))-float(observation.get("capture_seconds",0))) if observation.get("valid",false) else 0.0,
        "label_disagreements":temporal.label_disagreements,"held_render_samples":temporal.held_render_samples,
        "tracking_render_frames":tracking_render_frames,"tracking_elapsed_seconds":tracking_elapsed,
        "tracking_render_hz":tracking_render_frames/maxf(0.001,tracking_elapsed),
        "changed_pose_render_frames":changed_render_frames}
    var file := FileAccess.open(path + ".tmp", FileAccess.WRITE)
    if file:
        file.store_string(JSON.stringify(report) + "\n")
        file.close()
        DirAccess.rename_absolute(path + ".tmp", path)

func hide_hands() -> void:
    for hand in joints:
        for node in hand: node.visible = false
    for hand in lines:
        for node in hand: node.visible = false

func _process(delta: float) -> void:
    elapsed += delta
    frame_count += 1
    var observation: Dictionary = native.sample() if native else {"valid": false, "reason": "Native tracking bridge unavailable"}
    if observation.get("valid", false): valid_frames += 1
    status_clock += delta
    report_clock += delta
    if status_clock >= 0.1:
        status_clock = 0.0
        update_status(observation)
    if report_clock >= 0.25:
        report_clock = 0.0
        write_report(observation)
    if timeout_seconds > 0.0 and elapsed >= timeout_seconds:
        get_tree().quit()
    var rendered: Dictionary = observation
    if temporal_enabled:
        rendered = temporal.sample(observation,float(observation.get("now_seconds",0)),float(observation.get("display_seconds",0)))
    elif smoothing_enabled:
        rendered = smoother.sample(observation,delta)
    if rendered.get("hands",[]).size()>0: rendered_hand_frames += 1
    if trial_state.get("phase", "") == "tracking":
        tracking_render_frames += 1
        tracking_elapsed += delta
        var changed := false
        for hand in rendered.get("hands",[]):
            var key: int = 1 if hand.right else 0
            var points: PackedVector3Array = hand.points
            if not previous_rendered.has(key): changed = true
            else:
                var before: PackedVector3Array = previous_rendered[key]
                for k in range(21):
                    if points[k].distance_squared_to(before[k])>0.000000000001: changed = true
            previous_rendered[key] = points.duplicate()
        if changed: changed_render_frames += 1
    display_observation(rendered)

func display_observation(observation: Dictionary) -> void:
    hide_hands()
    geometry.transform = XRServer.get_reference_frame()
    if not observation.get("valid",false):
        return
    var stereo_count := 0
    var hands: Array = observation.get("hands",[])
    for hand in hands:
        var side: int = 1 if hand.right else 0
        var points: PackedVector3Array = hand.points
        if points.size() != 21: continue
        var opacity: float = hand.get("opacity",1.0)
        if hand.stereo: stereo_count += 1
        for i in range(21):
            joints[side][i].position = points[i]
            joints[side][i].visible = true
            joints[side][i].transparency = 1.0-opacity
        for i in range(LINKS.size()):
            var a: Vector3 = points[LINKS[i][0]]
            var b: Vector3 = points[LINKS[i][1]]
            var direction := b-a
            var length := direction.length()
            if length < 0.0001 or length > 0.25: continue
            var basis := Basis(Quaternion(Vector3.UP,direction/length))
            lines[side][i].transform = Transform3D(basis.scaled_local(Vector3(1,length,1)),(a+b)*0.5)
            lines[side][i].visible = true
            lines[side][i].transparency = 1.0-opacity
