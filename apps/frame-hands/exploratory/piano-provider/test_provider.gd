extends SceneTree
const Provider=preload("res://scripts/frame_provider/provider.gd")
const Rig=preload("res://scripts/frame_provider/rig_joints.gd")
var failures:=0
var checks:=0
func check(ok: bool, text: String):
    checks+=1
    if not ok:failures+=1;push_error(text)
func _initialize():call_deferred("run")
func run():
    var fixture=JSON.parse_string(FileAccess.get_file_as_string(OS.get_environment("FRAME_PIANO_FIXTURE")))
    var rig=Rig.new();check(rig.configure(fixture.rig),"configure")
    var maximum:=0.0
    for row in fixture.cases:
        var points: PackedVector3Array=rig.joints(row.angles,Transform3D(Basis.IDENTITY,Vector3(.1,.25,-.5)),int(row.side))
        var corresponding := {0:20,1:5,3:6,4:7,5:0,7:8,8:9,9:10,10:1,12:11,13:12,14:13,15:2,17:14,18:15,19:16,20:3,22:17,23:18,24:19,25:4}
        for joint in corresponding:
            maximum=maxf(maximum,points[joint].distance_to(Rig.v(row.expected[corresponding[joint]])))
    check(maximum<.000005,"joint mapping differs from independent skinning")
    var app=load("res://scenes/main.tscn").instantiate();root.add_child(app)
    await process_frame
    app.set_process(false);app.hands.set_process(false);app.audio.enabled=false
    app.hands.enabled=true
    var confidence:=PackedFloat32Array();confidence.resize(26);confidence.fill(1)
    var points: PackedVector3Array=rig.joints(fixture.cases[0].angles,Transform3D(Basis.IDENTITY,Vector3(.1,.25,-.5)),1)
    app.hands.clear_frame("right")
    app.hands.submit_visual("right",points,confidence,.05,1)
    app.hands._process(.011)
    check(app.hands.models.right.visible,"visual grace not rendered")
    check(not app.hands.available and app.detector.owners.is_empty(),"visual grace created input")
    for i in points.size():points[i].y-=.1
    app.hands.submit_visual("right",points,confidence,.05,1);app.hands._process(.011)
    check(app.detector.owners.is_empty() and not app.hands.pinching,"predicted movement created press/pinch")
    app.hands.visual_frames.right.until=0;app.hands._process(.011)
    check(not app.hands.models.right.visible,"stale visual remained")
    app.hands.submit_frame("right",points,confidence,.151)
    check(not app.hands.external_frames.has("right"),"stale input accepted")
    app.build_xr_dashboard();app.xr_panel.visible=false
    app.xr_active=true;app.performance.autoplay=false;app.performance.hand="right"
    app.performance.playing=true
    app.hands.submit_visual("right",points,confidence,.05,1)
    var score_time: float=app.performance.time
    app._process(.011)
    check(app.performance.playing and app.performance.time==score_time,"short grace opened menu or advanced score without input")
    app.hands.visual_frames.right.until=0
    app._process(.011)
    check(not app.performance.playing,"long loss failed to pause practice")
    app.xr_active=false
    check(not Provider.rigid([[0]]),"malformed transform accepted")
    app.queue_free()
    print("PIANO PROVIDER: ",checks," checks, ",failures," failures; FK error mm=",maximum*1000)
    quit(1 if failures else 0)
