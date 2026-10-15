extends "res://baseline_motion.gd"
# Optional presentation-only catch-up. Inputs must already be in tracking-world coordinates.
const CORRECTION_SECONDS := 0.060
var correction_enabled := true
var corrections := 0
var _observation_now := 0.0
var _mode := "predict"
var _transition_start := -1.0
var _transition_from := {}
var _last_display := {}
var _last_display_time := -1.0

func clear():
    super.clear()
    _transition_start = -1.0
    _transition_from = {}
    _last_display = {}
    _last_display_time = -1.0

func observe(id: int, time: float, capture: float, now: float, p: Vector3, q: Quaternion, values: Array, mode: String) -> bool:
    _observation_now = now
    _mode = mode
    return super.observe(id,time,capture,now,p,q,values,mode)

func reset_pose(id: int, time: float, capture: float, p: Vector3, q: Quaternion, a: PackedFloat64Array):
    # A source ID is a continuity hypothesis, not proof of physical identity.
    # Do not carry a visible pose across a side/ID change, expiry or hidden display.
    var can_continue: bool = correction_enabled and _mode != "raw" and active and id == source_id and not _last_display.is_empty() and _observation_now >= _last_display_time and _observation_now-_last_display_time <= 0.030 and _observation_now-capture_local < HIDE_AGE
    if can_continue:
        # Preserve the original deadline during a burst of reset confirmations.
        if _transition_start < 0.0 or _observation_now >= _transition_start+CORRECTION_SECONDS:
            _transition_start = _observation_now
            _transition_from = _last_display.duplicate(true)
        corrections += 1
    else:
        _transition_start = -1.0
        _transition_from = {}
    super.reset_pose(id,time,capture,p,q,a)

func sample(now: float, delta: float, mode: String) -> Dictionary:
    var result: Dictionary = super.sample(now,delta,mode)
    if not result.visible:
        _last_display = {}
        _transition_start = -1.0
        _transition_from = {}
        return result
    var correcting := false
    if mode != "raw" and _transition_start >= 0.0:
        var progress := clampf((now-_transition_start)/CORRECTION_SECONDS,0.0,1.0)
        if progress < 1.0:
            var blend := progress*progress*(3.0-2.0*progress)
            result.position = _transition_from.position.lerp(result.position,blend)
            result.orientation = _transition_from.orientation.slerp(result.orientation,blend).normalized()
            var mixed: PackedFloat64Array = result.angles.duplicate()
            for i in 20: mixed[i] = lerpf(_transition_from.angles[i],mixed[i],blend)
            result.angles = mixed
            correcting = true
        else:
            _transition_start = -1.0
            _transition_from = {}
    result.correcting = correcting
    # Visibility, opacity, source timestamps and gesture confidence are not extended.
    _last_display = result.duplicate(true)
    _last_display_time = now
    return result
