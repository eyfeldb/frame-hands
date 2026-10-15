extends SceneTree
const Rig=preload("res://rig_joints.gd")
const Visual=preload("res://profile_rig.gd")
const Assets=preload("res://profile_assets.gd")
const MAP=[1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25]
const LANDMARK_MAP={0:5,2:6,3:7,4:0,5:8,6:9,7:10,8:1,9:11,10:12,11:13,12:2,13:14,14:15,15:16,16:3,17:17,18:18,19:19,20:4}
func _initialize():
    var assets=Assets.load_verified("res://profile-rigs.json")
    assert(assets.size()==2)
    var packets=[]
    var max_error:=0.0
    for sample in 41:
        var hands=[]
        for side in 2:
            var a=assets[side]
            var rig=Rig.new();assert(rig.configure(a))
            var visual=Visual.new();assert(visual.configure(a,side,StandardMaterial3D.new()))
            var angles=[]
            for i in 20:
                var fraction:=float((sample*7+i*3)%41)/40.0
                angles.append(lerpf(a.limits[i][0],a.limits[i][1],fraction))
            assert(visual.apply_angles(angles))
            var wrist:=Transform3D(Basis(Quaternion(Vector3(1,2,3).normalized(),float(sample)*.02)),Vector3(-.2 if side==0 else .2,1.0,-.5))
            visual.transform=wrist
            var reference=visual.debug_landmarks()
            var joints: PackedVector3Array=rig.joints(angles,wrist,side)
            var points=[]
            for i in 21:
                var point: Vector3=joints[MAP[i]]
                var expected: Vector3
                if i==1:
                    expected=Vector3(a.rest_m[0][0],a.rest_m[0][1],a.rest_m[0][2])
                    if side==1:expected.x*=-1
                    expected=wrist*expected
                else:
                    var row=reference[LANDMARK_MAP[i]]
                    expected=Vector3(row[0],row[1],row[2])
                max_error=maxf(max_error,point.distance_to(expected))
                assert(point.distance_to(expected)<.000005)
                points.append([point.x,point.y,point.z])
            hands.append({"side":side,"id":side+1,"capture_ms":1000+sample*10,"until_ms":1170+sample*10,"points":points})
            visual.free()
        packets.append({"source_ms":1000+sample*10,"wall_s":1.0,"hands":hands})
    var output:=OS.get_cmdline_user_args()[0]
    var file=FileAccess.open(output,FileAccess.WRITE)
    file.store_string(JSON.stringify(packets));file.close()
    print("PASS native fixed-rig joints vs accepted visual: 82 poses, max_error_mm=",max_error*1000)
    quit()
