extends RefCounted

# Visual continuity only. These predicted/held poses must never trigger input.
# Times are CLOCK_MONOTONIC seconds supplied by the native bridge.
const PREDICT_SECONDS := 0.050
const FADE_START := 0.100
const EXPIRE_SECONDS := 0.250
const IDENTITY_SECONDS := 0.500
const INITIAL_LABEL_SECONDS := 0.500
const PARENTS = [-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
var slots: Array = [{}, {}]
var last_sequence := -1
var accepted_updates := 0
var label_disagreements := 0
var held_render_samples := 0

func reset() -> void:
    slots = [{}, {}]
    last_sequence = -1

func palm(points: PackedVector3Array) -> Vector3:
    return (points[0]+points[5]+points[9]+points[13]+points[17])*0.2

func predict_points(source: PackedVector3Array, velocity: PackedVector3Array, horizon: float) -> PackedVector3Array:
    var amount := horizon
    for _attempt in range(5):
        var points := source.duplicate()
        for k in range(21): points[k] += (velocity[k]*amount).limit_length(0.05)
        # Independent Cartesian predictions stretch bones during finger bends.
        # Keep the most recent measured segment lengths, without fitting a new
        # anatomy model or inventing measurements.
        for k in range(1,21):
            var parent: int = PARENTS[k]
            var length := source[k].distance_to(source[parent])
            var direction := points[k]-points[parent]
            if direction.length_squared()<0.0000000001: direction=source[k]-source[parent]
            points[k]=points[parent]+direction.normalized()*length
        var maximum := 0.0
        for k in range(21): maximum=maxf(maximum,points[k].distance_to(source[k]))
        if maximum<=0.050001: return points
        amount *= 0.049/maximum
    return source.duplicate()

func match_cost(state: Dictionary, hand: Dictionary, capture: float) -> float:
    var dt: float = capture-state.time
    if dt <= 0.0 or dt > IDENTITY_SECONDS: return INF
    var points: PackedVector3Array = hand.points
    var old: PackedVector3Array = state.points
    var velocity: PackedVector3Array = state.velocity
    var expected := palm(old)+palm(velocity)*minf(dt,PREDICT_SECONDS)
    var distance := expected.distance_to(palm(points))
    if distance > minf(0.35,0.10+3.0*dt): return INF
    var cost := distance
    for k in [0,5,9,13,17]:
        cost += 0.05*((points[k]-palm(points))-(old[k]-palm(old))).length()
    if int(hand.id)==int(state.id): cost -= 0.06
    return cost

func ingest(observation: Dictionary) -> void:
    var capture: float = observation.capture_seconds
    var hands: Array = observation.get("hands", [])
    var mapping := [-1,-1]
    var best_cost := INF
    # Exhaustive two-slot assignment avoids greedy swaps when hands cross.
    for a in range(-1,hands.size()):
        for b in range(-1,hands.size()):
            if a >= 0 and a==b: continue
            var indices := [a,b]
            var cost := 0.0
            for side in range(2):
                var state: Dictionary = slots[side]
                var index: int = indices[side]
                if state.is_empty():
                    if index >= 0: cost = INF
                elif index < 0:
                    cost += 0.20
                else:
                    cost += match_cost(state,hands[index],capture)
            if cost < best_cost:
                best_cost = cost
                mapping = indices
    var used := []
    for index in mapping:
        if index >= 0: used.append(index)
    # Only an empty identity slot can take an unmatched hand based on its label.
    for index in range(hands.size()):
        if index in used: continue
        var side: int = 1 if hands[index].right else 0
        if slots[side].is_empty() and mapping[side]<0:
            mapping[side]=index
            used.append(index)
    for side in range(2):
        if mapping[side]<0: continue
        var hand: Dictionary = hands[mapping[side]]
        var points: PackedVector3Array = hand.points
        if points.size()!=21: continue
        var velocity := PackedVector3Array()
        velocity.resize(21)
        var state: Dictionary = slots[side]
        if not state.is_empty():
            var dt: float = capture-state.time
            if dt>0.001 and dt<0.1:
                for k in range(21):
                    var measured: Vector3 = ((points[k]-state.points[k])/dt).limit_length(3.0)
                    var previous: Vector3 = state.velocity[k]
                    # Do not let the velocity smoother carry momentum through a
                    # newly observed stop or reversal.
                    if measured.dot(previous)<=0.0 or measured.length()<previous.length()*0.25:
                        velocity[k]=measured
                    else:
                        velocity[k] = previous.lerp(measured,0.65)
        if bool(hand.right)!=(side==1): label_disagreements += 1
        var voted_side: bool = bool(hand.right)
        var streak: int = int(state.get("side_streak",0))+1 if state.get("voted_side",voted_side)==voted_side else 1
        slots[side]={"id":hand.id,"points":points.duplicate(),"velocity":velocity,
            "time":capture,"stereo":hand.stereo,"born":state.get("born",capture),
            "voted_side":voted_side,"side_streak":streak}
    # Correct an initially wrong classifier label only during acquisition, after
    # several consistent observations. Established identities ignore later flips.
    var corrections := [false,false]
    for side in range(2):
        var state: Dictionary = slots[side]
        if not state.is_empty():
            corrections[side] = capture-float(state.born)<=INITIAL_LABEL_SECONDS and int(state.side_streak)>=4 and bool(state.voted_side)!=(side==1)
    if (corrections[0] and (slots[1].is_empty() or corrections[1])) or (corrections[1] and slots[0].is_empty()):
        var swap: Dictionary = slots[0]
        slots[0]=slots[1]
        slots[1]=swap
    accepted_updates += 1

func sample(observation: Dictionary, now: float, display: float) -> Dictionary:
    for side in range(2):
        if not slots[side].is_empty() and now-float(slots[side].time)>IDENTITY_SECONDS:
            slots[side]={}
    if not observation.get("valid",false):
        # Invalid head poses, packets, clock conversions and stopped producers
        # are not ordinary per-hand losses; never cover those with old geometry.
        reset()
        return {"valid":false,"hands":[]}
    var sequence: int = int(observation.sequence)
    if sequence!=last_sequence:
        ingest(observation)
        last_sequence=sequence
    var result: Array = []
    for side in range(2):
        var state: Dictionary = slots[side]
        if state.is_empty(): continue
        var age: float = maxf(0.0,now-state.time)
        if age>=EXPIRE_SECONDS: continue
        var horizon: float = clampf(display-state.time,0.0,PREDICT_SECONDS)
        var opacity := 1.0-clampf((age-FADE_START)/(EXPIRE_SECONDS-FADE_START),0.0,1.0)
        var points := predict_points(state.points,state.velocity,horizon)
        var held := age>0.060
        if held: held_render_samples += 1
        result.append({"id":state.id,"right":side==1,"points":points,"stereo":state.stereo,
            "opacity":opacity,"held":held,"predicted":horizon>0.0,"input_valid":false})
    return {"valid":true,"hands":result}
