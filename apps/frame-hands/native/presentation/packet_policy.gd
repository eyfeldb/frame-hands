extends RefCounted
# Opt-in presentation routing. Duplicate-side hypotheses supply no fresh pose.
# No relabeling, confidence creation, winner-by-order or visibility-clock renewal.
static func route(hands: Array) -> Dictionary:
    var ids := {}
    var counts := [0,0]
    for hand in hands:
        if not hand is Dictionary or not hand.has("side") or not hand.has("source_id"):
            return {"valid":false,"hands":[],"ambiguous_sides":[],"discarded":0}
        var side := int(hand.side)
        var id := int(hand.source_id)
        if side < 0 or side > 1 or float(hand.side) != side or id <= 0 or float(hand.source_id) != id or ids.has(id):
            return {"valid":false,"hands":[],"ambiguous_sides":[],"discarded":0}
        ids[id] = true
        counts[side] += 1
    var result := []
    var ambiguous := []
    for side in 2:
        if counts[side] > 1:ambiguous.append(side)
    for hand in hands:
        if counts[int(hand.side)] == 1:result.append(hand)
    return {"valid":true,"hands":result,"ambiguous_sides":ambiguous,"discarded":hands.size()-result.size()}
