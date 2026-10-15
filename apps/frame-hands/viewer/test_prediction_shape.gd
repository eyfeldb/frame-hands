extends SceneTree
const Temporal = preload("res://temporal_hands.gd")
const PARENTS = [-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]

func finger_hand(angle: float) -> Dictionary:
    var p := PackedVector3Array()
    p.resize(21)
    p[0]=Vector3(0,0,-0.5)
    for finger in range(5):
        var first := 1+finger*4
        p[first]=p[0]+Vector3((finger-2)*0.018,0.05,0)
        p[first+1]=p[first]+Vector3(0,0.03,0)
        var direction := Vector3(0,cos(angle),sin(angle))
        p[first+2]=p[first+1]+direction*0.025
        p[first+3]=p[first+2]+direction*0.018
    return {"id":1,"right":false,"stereo":true,"points":p}

func _init() -> void:
    var worst_bone_change := 0.0
    var worst_displacement := 0.0
    for angle in [0.25,0.5,0.8,1.2,-0.8]:
        var filter = Temporal.new()
        filter.sample({"valid":true,"sequence":1,"capture_seconds":1.0,"hands":[finger_hand(0)]},1.01,1.02)
        var current := finger_hand(angle)
        var obs := {"valid":true,"sequence":2,"capture_seconds":1.033333,"hands":[current]}
        var rendered = filter.sample(obs,1.05,1.083333).hands[0].points
        for k in range(1,21):
            var parent: int = PARENTS[k]
            var measured: float = current.points[k].distance_to(current.points[parent])
            var predicted: float = rendered[k].distance_to(rendered[parent])
            worst_bone_change=maxf(worst_bone_change,absf(predicted-measured))
            worst_displacement=maxf(worst_displacement,rendered[k].distance_to(current.points[k]))
    print("Prediction shape: max bone-length change=",worst_bone_change," m; max displacement=",worst_displacement," m")
    if worst_bone_change>0.002 or worst_displacement>0.050001:
        push_error("Prediction distorts measured bones or exceeds its displacement bound")
        quit(1)
        return
    print("PASS: articulated-finger prediction preserves bone lengths and displacement bounds")
    quit()
