extends SceneTree
const Rig = preload("res://profile_rig.gd")
var frames := 0
func _initialize():
    var scene := Node3D.new()
    root.add_child(scene)
    var camera := Camera3D.new()
    scene.add_child(camera)
    camera.position = Vector3(0,0.04,1.35)
    camera.fov = 36
    camera.current = true
    var environment := WorldEnvironment.new()
    environment.environment = Environment.new()
    environment.environment.background_mode = Environment.BG_COLOR
    environment.environment.background_color = Color(0.035,0.045,0.06)
    environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    environment.environment.ambient_light_color = Color(0.8,0.86,1)
    environment.environment.ambient_light_energy = 0.55
    scene.add_child(environment)
    for direction in [Vector3(-35,-25,0),Vector3(25,150,0)]:
        var light := DirectionalLight3D.new()
        light.rotation_degrees = direction
        light.light_energy = 1.1
        scene.add_child(light)
    for col in 4:
        var side := 0 if col < 2 else 1
        var generic := col % 2 == 0
        var asset = JSON.parse_string(FileAccess.get_file_as_string("res://generic.json" if generic else "res://side%d.json" % side))
        for row in 2:
            var hand = Rig.new()
            scene.add_child(hand)
            var material := StandardMaterial3D.new()
            material.albedo_color = Color(0.55,0.6,0.67) if generic else (Color(0.12,0.72,0.87) if side == 0 else Color(0.96,0.49,0.2))
            material.roughness = 0.65
            material.cull_mode = BaseMaterial3D.CULL_DISABLED
            if not hand.configure(asset,side,material): quit(2);return
            var basis := Basis(Vector3(0,1,0),Vector3(0,0,1),Vector3(1,0,0))
            if side == 1: basis = basis*Basis(Vector3(0,1,0),PI)
            hand.transform = Transform3D(basis,Vector3(-0.405+col*0.27,0.08-row*0.32,0))
            var angles := []
            for i in 20:
                var value := 0.0
                if row == 1 and i >= 4 and i%4 != 0: value = 0.7 if i%4 == 3 else 1.2
                angles.append(clampf(value,float(asset.limits[i][0]),float(asset.limits[i][1])))
            if not hand.apply_angles(angles): quit(3);return
    var canvas := CanvasLayer.new()
    root.add_child(canvas)
    for col in 4:
        var label := Label.new()
        canvas.add_child(label)
        label.position = Vector2(90+col*290,30)
        label.text = ["LEFT / generic","LEFT / fitted palm","RIGHT / generic","RIGHT / fitted palm"][col]
        label.add_theme_font_size_override("font_size",20)
    var note := Label.new()
    canvas.add_child(note)
    note.position = Vector2(36,842)
    note.text = "Static geometry check: open / curled fingers. Approximate surface; not a tracking or collision test."
    note.add_theme_font_size_override("font_size",18)
func _process(_delta):
    frames += 1
    if frames == 30:
        await RenderingServer.frame_post_draw
        var result := root.get_texture().get_image().save_png(OS.get_environment("FRAME_MESH_SCREENSHOT"))
        if result != OK: quit(4);return false
        print("PASS: profile comparison rendered")
        quit()
    return false
