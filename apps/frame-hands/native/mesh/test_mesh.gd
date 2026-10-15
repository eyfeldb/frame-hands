extends SceneTree
const Rig = preload("res://profile_rig.gd")
func _initialize():
    var fixture = JSON.parse_string(FileAccess.get_file_as_string("res://fixtures.json"))
    var container := Node3D.new()
    root.add_child(container)
    var output := []
    for row in fixture.cases:
        var asset = JSON.parse_string(FileAccess.get_file_as_string("res://"+row.asset))
        var hand = Rig.new()
        container.add_child(hand)
        if not hand.configure(asset,int(row.side),StandardMaterial3D.new()): quit(2);return
        var w = row.wrist
        hand.transform = Transform3D(Basis(Vector3(w[0][0],w[1][0],w[2][0]),Vector3(w[0][1],w[1][1],w[2][1]),Vector3(w[0][2],w[1][2],w[2][2])),Vector3(w[0][3],w[1][3],w[2][3]))
        if not hand.apply_angles(row.angles): quit(3);return
        output.append({"vertices":hand.debug_vertices(),"landmarks":hand.debug_landmarks()})
        var invalid: Array = row.angles.duplicate()
        invalid[0] = 99.0
        if hand.apply_angles(invalid): quit(4);return
        if asset.profile_side != -1 and Rig.accepts(asset,1-int(row.side)): quit(5);return
        hand.free()
    var file := FileAccess.open(OS.get_environment("FRAME_MESH_RESULT"),FileAccess.WRITE)
    if file == null: quit(6);return
    file.store_string(JSON.stringify({"cases":output,"invalid_angles_rejected":true,"wrong_profile_side_rejected":true}))
    file.close()
    print("PASS: ",output.size()," fixed-profile skinning cases; side/angle controls")
    quit()
