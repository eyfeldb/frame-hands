extends RefCounted

# Presentation-only, causal interpolation. No velocity prediction, relabeling,
# hidden-hand persistence, or extra rejection of valid optical observations.
const RESPONSE_SECONDS := 0.020
const PARENTS = [-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
var displayed: Dictionary = {}

func sample(observation: Dictionary, delta: float) -> Dictionary:
    if not observation.get("valid",false):
        displayed.clear()
        return {"valid":false,"hands":[]}
    var hands: Array = []
    var alive: Array = []
    var blend := 1.0-exp(-maxf(delta,0.0)/RESPONSE_SECONDS)
    for hand in observation.get("hands",[]):
        var id: int = int(hand.id)
        alive.append(id)
        var target: PackedVector3Array = hand.points
        var points := target.duplicate()
        if displayed.has(id):
            var before: PackedVector3Array = displayed[id]
            points[0]=before[0].lerp(target[0],blend)
            for k in range(1,21):
                var parent: int = PARENTS[k]
                var old_bone := before[k]-before[parent]
                var target_bone := target[k]-target[parent]
                if old_bone.length_squared()<0.000000001 or target_bone.length_squared()<0.000000001:
                    points[k]=points[parent]+old_bone.lerp(target_bone,blend)
                else:
                    var direction := old_bone.normalized().slerp(target_bone.normalized(),blend)
                    var length := lerpf(old_bone.length(),target_bone.length(),blend)
                    points[k]=points[parent]+direction.normalized()*length
        displayed[id]=points.duplicate()
        hands.append({"id":id,"right":hand.right,"points":points,"stereo":hand.stereo,
            "input_valid":false,"predicted":false,"held":false})
    for id in displayed.keys():
        if not id in alive: displayed.erase(id)
    return {"valid":true,"hands":hands}
