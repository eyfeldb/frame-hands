extends "res://correction_motion.gd"
# Experimental rendering only. Hand presence/identity/input confidence are untouched.
# These are visual policy thresholds, not physiological speed limits.
const FINGER_JUMP_RADIANS := 0.75
const FINGER_SPEED_ALLOWANCE := 7.5
const FINGER_CONFIRM_RADIANS := 0.35
const FINGER_EASE_SECONDS := 0.060
var finger_guard_enabled := true
var finger_holds := 0
var finger_confirmations := 0
var finger_forced_corrections := 0
var _finger_pending := {}
var _finger_transitions := {}
var _finger_reset := false

func clear():
    super.clear()
    _finger_pending.clear()
    _finger_transitions.clear()

func reset_pose(id: int, time: float, capture: float, p: Vector3, q: Quaternion, a: PackedFloat64Array):
    _finger_reset = true
    _finger_pending.clear()
    _finger_transitions.clear()
    super.reset_pose(id,time,capture,p,q,a)

func observe(id: int, time: float, capture: float, now: float, p: Vector3, q: Quaternion, values: Array, mode: String) -> bool:
    var old_raw: PackedFloat64Array = raw_angles.duplicate()
    var old_angles: PackedFloat64Array = angles.duplicate()
    var old_derivatives: PackedFloat64Array = derivatives.duplicate()
    var dt: float = time-stamp
    _finger_reset = false
    var accepted: bool = super.observe(id,time,capture,now,p,q,values,mode)
    if not accepted: return false
    if not finger_guard_enabled or mode == "raw" or _finger_reset:
        _finger_pending.clear()
        _finger_transitions.clear()
        return true
    for finger in 5:
        var start: int = finger*4
        var change := 0.0
        for j in range(start,start+4): change = maxf(change,absf(float(values[j])-old_raw[j]))
        if change <= FINGER_JUMP_RADIANS+FINGER_SPEED_ALLOWANCE*dt:
            _finger_pending.erase(finger)
            continue
        var consistent := false
        if _finger_pending.has(finger):
            var distance := 0.0
            for j in 4: distance = maxf(distance,absf(float(values[start+j])-float(_finger_pending[finger].proposal[j])))
            consistent = distance <= FINGER_CONFIRM_RADIANS
        var timed_out: bool = _finger_pending.has(finger) and now-float(_finger_pending[finger].first) >= FINGER_EASE_SECONDS
        if consistent or timed_out:
            _finger_pending.erase(finger)
            if consistent: finger_confirmations += 1
            else: finger_forced_corrections += 1
            if not _last_display.is_empty() and now-_last_display_time <= 0.030:
                if not _finger_transitions.has(finger) or now >= float(_finger_transitions[finger].start)+FINGER_EASE_SECONDS:
                    _finger_transitions[finger] = {"start":now,"from":_last_display.angles.slice(start,start+4),"confirmed":consistent}
        else:
            var first: float = float(_finger_pending[finger].first) if _finger_pending.has(finger) else now
            _finger_pending[finger] = {"first":first,"proposal":values.slice(start,start+4)}
            finger_holds += 1
            # Revert only the unsupported articulation update. The parent has accepted
            # actual hand presence/wrist/capture time; duplicates/stale data cannot enter.
            for j in range(start,start+4):
                raw_angles[j] = old_raw[j]
                angles[j] = old_angles[j]
                derivatives[j] = old_derivatives[j]
    return true

func sample(now: float, delta: float, mode: String) -> Dictionary:
    var result: Dictionary = super.sample(now,delta,mode)
    if not result.visible: return result
    var correcting := []
    var inferred: Array = _finger_pending.keys()
    for finger in _finger_transitions.keys():
        var transition: Dictionary = _finger_transitions[finger]
        var progress := clampf((now-float(transition.start))/FINGER_EASE_SECONDS,0.0,1.0)
        if progress >= 1.0 or mode == "raw":
            _finger_transitions.erase(finger)
            continue
        var blend := progress*progress*(3.0-2.0*progress)
        for j in 4:
            var index: int = int(finger)*4+j
            result.angles[index] = lerpf(float(transition.from[j]),result.angles[index],blend)
        correcting.append(finger)
        if not bool(transition.confirmed) and finger not in inferred: inferred.append(finger)
    var hold_ages := {}
    for finger in _finger_pending: hold_ages[finger] = maxf(0,now-float(_finger_pending[finger].first))
    result.inferred_fingers = inferred
    result.finger_hold_ages = hold_ages
    result.finger_correcting = correcting
    # Parent correction must start from the actual displayed angles.
    _last_display = result.duplicate(true)
    return result
