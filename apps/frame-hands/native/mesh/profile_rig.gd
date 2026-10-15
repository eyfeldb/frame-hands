extends "res://base_rig.gd"
var landmark_rest: Array = []
var landmark_weights: Array = []
var angle_limits: Array = []
var profile_sha := ""

static func accepts(asset: Dictionary, hand_side: int) -> bool:
    return asset.get("version",0) == 1 and hand_side in [0,1] and int(asset.get("profile_side",-2)) in [-1,hand_side] and str(asset.get("profile_sha256","")).length() == 64 and asset.get("axes",[]).size() == 20 and asset.get("rest_m",[]).size() == 20 and asset.get("limits",[]).size() == 20 and asset.get("landmarks_m",[]).size() == 21 and asset.get("landmark_weights",[]).size() == 21

func configure(asset: Dictionary, hand_side: int, material: StandardMaterial3D):
    if not accepts(asset,hand_side): return false
    profile_sha = asset.profile_sha256
    landmark_rest = asset.landmarks_m
    landmark_weights = asset.landmark_weights
    angle_limits = asset.limits
    super.configure(asset,hand_side,material)
    return true

func valid_angles(angles) -> bool:
    if angles.size() != 20: return false
    for i in 20:
        var value := float(angles[i])
        if not is_finite(value) or value < float(angle_limits[i][0])-1e-8 or value > float(angle_limits[i][1])+1e-8: return false
    return true

func apply_angles(angles):
    if not valid_angles(angles): return false
    super.apply_angles(angles)
    return true

func debug_landmarks() -> Array:
    var result := []
    for i in 21:
        var point := Vector3.ZERO
        for b in 17:
            var weight := float(landmark_weights[i][b])
            if weight != 0.0: point += (skeleton.get_bone_pose(b)*vec(landmark_rest[i]))*weight
        if side == 1: point.x *= -1.0
        point = transform*point
        result.append([point.x,point.y,point.z])
    return result
