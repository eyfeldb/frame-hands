class_name UmeRig
extends Node3D
var skeleton: Skeleton3D
var mesh_node: MeshInstance3D
var reflected: Node3D
var axes: Array[Vector3] = []
var rest: Array[Vector3] = []
var vertices: PackedVector3Array
var weights: Array
var side := 0

static func vec(value) -> Vector3:
    return Vector3(value[0],value[1],value[2])

func configure(rig: Dictionary, hand_side: int, material: StandardMaterial3D):
    side = hand_side
    reflected = Node3D.new()
    reflected.name = "Handedness"
    add_child(reflected)
    if side == 1: reflected.scale.x = -1
    skeleton = Skeleton3D.new()
    skeleton.name = "Bones"
    reflected.add_child(skeleton)
    var skin := Skin.new()
    for i in 17:
        skeleton.add_bone("bone_%d" % i)
        skeleton.set_bone_rest(i,Transform3D.IDENTITY)
        skin.add_bind(i,Transform3D.IDENTITY)
    vertices = PackedVector3Array()
    var normals := PackedVector3Array()
    var bone_ids := PackedInt32Array()
    var bone_weights := PackedFloat32Array()
    weights = rig.weights
    for v in rig.vertices_m: vertices.append(vec(v))
    for n in rig.normals: normals.append(vec(n))
    for row in weights:
        var count := 0
        for i in row.size():
            if float(row[i]) > 0:
                bone_ids.append(i)
                bone_weights.append(float(row[i]))
                count += 1
        assert(count <= 4,"Skin has too many influences")
        while count < 4:
            bone_ids.append(0)
            bone_weights.append(0)
            count += 1
    for v in rig.axes: axes.append(vec(v))
    for v in rig.rest_m: rest.append(vec(v))
    var arrays := []
    arrays.resize(Mesh.ARRAY_MAX)
    arrays[Mesh.ARRAY_VERTEX] = vertices
    arrays[Mesh.ARRAY_NORMAL] = normals
    arrays[Mesh.ARRAY_INDEX] = PackedInt32Array(rig.triangles)
    arrays[Mesh.ARRAY_BONES] = bone_ids
    arrays[Mesh.ARRAY_WEIGHTS] = bone_weights
    var mesh := ArrayMesh.new()
    mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES,arrays)
    mesh_node = MeshInstance3D.new()
    reflected.add_child(mesh_node)
    mesh_node.mesh = mesh
    mesh_node.skin = skin
    mesh_node.skeleton = mesh_node.get_path_to(skeleton)
    mesh_node.material_override = material

func apply_angles(angles):
    skeleton.set_bone_pose(0,Transform3D.IDENTITY)
    skeleton.set_bone_pose(1,Transform3D.IDENTITY)
    for finger in 5:
        var transform := Transform3D.IDENTITY
        for joint in 4:
            var i := finger*4+joint
            var rotation := Basis(Quaternion(axes[i].normalized(),float(angles[i])))
            transform = transform*Transform3D(rotation,rest[i]-rotation*rest[i])
            if joint > 0: skeleton.set_bone_pose(2+finger*3+joint-1,transform)

func debug_vertices() -> Array:
    var result := []
    for i in vertices.size():
        var v := Vector3.ZERO
        for b in 17:
            if float(weights[i][b]) != 0.0:
                v += (skeleton.get_bone_pose(b)*vertices[i])*float(weights[i][b])
        if side == 1: v.x *= -1
        v = transform*v
        result.append([v.x,v.y,v.z])
    return result
