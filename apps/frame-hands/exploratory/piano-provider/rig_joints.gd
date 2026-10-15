extends RefCounted
# UmeTrack fixed-rig FK; joint names from pinned lib/common/hand.py.
var rig: Dictionary
func configure(value: Dictionary) -> bool:
    if value.get("landmarks_m",[]).size()!=21 or value.get("landmark_weights",[]).size()!=21: return false
    rig=value
    return true
static func v(a) -> Vector3: return Vector3(a[0],a[1],a[2])
func joints(angles, wrist: Transform3D, side: int) -> PackedVector3Array:
    var bones := [Transform3D.IDENTITY,Transform3D.IDENTITY]
    for finger in 5:
        var xf := Transform3D.IDENTITY
        for joint in 4:
            var i := finger*4+joint
            var rest := v(rig.rest_m[i])
            var rotation := Basis(Quaternion(v(rig.axes[i]).normalized(),float(angles[i])))
            xf=xf*Transform3D(rotation,rest-rotation*rest)
            if joint>0:bones.append(xf)
    var landmarks := PackedVector3Array()
    for i in 21:
        var point := Vector3.ZERO
        for b in 17:point+=(bones[b]*v(rig.landmarks_m[i]))*float(rig.landmark_weights[i][b])
        if side==1:point.x*=-1.0
        landmarks.append(wrist*point)
    var result := PackedVector3Array()
    result.resize(26)
    result[0]=landmarks[20];result[1]=landmarks[5]
    # Thumb CMC uses its fixed articulation pivot; MCP/IP/tip have landmarks.
    var thumb := v(rig.rest_m[0])
    if side==1:thumb.x*=-1.0
    result[2]=wrist*thumb
    result[3]=landmarks[6];result[4]=landmarks[7];result[5]=landmarks[0]
    for finger in 4:
        var index := 6+finger*5
        var lm := 8+finger*3
        # UmeTrack has no separate non-thumb metacarpal landmark. Interpolate
        # this visual-only joint; actual fingertips use exact model landmarks.
        result[index]=landmarks[5].lerp(landmarks[lm],0.35)
        result[index+1]=landmarks[lm];result[index+2]=landmarks[lm+1]
        result[index+3]=landmarks[lm+2];result[index+4]=landmarks[finger+1]
    return result
