extends "res://continuity_motion.gd"
# Display-only grace. Never renew capture time or confer input confidence.
const GRACE_END := 0.220
const GRACE_FADE := 0.170
const EXTRA_METERS := 0.010
const EXTRA_RADIANS := 0.060
var good_observations := 0
var cadence := 1.0/30.0
var receipt := -1.0
var receipt_age := 1.0
var extra_position := Vector3.ZERO
var extra_rotation := Vector3.ZERO

func clear():
    super.clear()
    good_observations = 0
    receipt = -1.0
    extra_position = Vector3.ZERO
    extra_rotation = Vector3.ZERO

func observe(id: int, time: float, capture: float, now: float, p: Vector3, q: Quaternion, values: Array, mode: String) -> bool:
    var continuing: bool = active and id == source_id and time > stamp and time-stamp <= 0.1 and now-capture_local < GRACE_END
    var interval: float = time-stamp
    var accepted: bool = super.observe(id,time,capture,now,p,q,values,mode)
    if accepted:
        if not continuing:
            extra_position = Vector3.ZERO
            extra_rotation = Vector3.ZERO
        good_observations = good_observations+1 if continuing else 1
        cadence = lerpf(cadence,interval,0.2) if continuing else 1.0/30.0
        receipt = now
        receipt_age = now-capture
    return accepted

func eligible() -> bool:
    return _mode != "raw" and good_observations >= 3 and not pending and receipt_age < 0.150 and cadence <= 0.075 and acceleration < 40.0 and velocity.length() < 1.5 and angular_velocity.length() < 4.0

func visibility_deadline() -> float:
    return GRACE_END if eligible() else HIDE_AGE

func fade_start() -> float:
    return GRACE_FADE if eligible() else HOLD_AGE

func sample(now: float, delta: float, mode: String) -> Dictionary:
    # Gentle continuation after an expected update is late; velocity decays to
    # zero. The extra displacement is bounded even if the source disappears.
    var target_p := Vector3.ZERO
    var target_r := Vector3.ZERO
    if eligible() and mode == "predict" and receipt >= 0:
        var late := maxf(0.0,now-receipt-maxf(0.025,cadence*1.25))
        var tau := 0.025
        # Integral of a smoothly starting, exponentially damped velocity.
        var horizon := tau*(1.0-(1.0+late/tau)*exp(-late/tau))
        target_p = (velocity*horizon).limit_length(EXTRA_METERS)
        target_r = (angular_velocity*horizon).limit_length(EXTRA_RADIANS)
    var blend := 1.0-exp(-maxf(delta,0.00001)/0.020)
    extra_position = extra_position.lerp(target_p,blend)
    extra_rotation = extra_rotation.lerp(target_r,blend)
    var result: Dictionary = super.sample(now,delta,mode)
    if not result.visible: return result
    if mode != "raw":
        result.position += extra_position
        if extra_rotation.length() > 0.000001:
            result.orientation = (Quaternion(extra_rotation.normalized(),extra_rotation.length())*result.orientation).normalized()
    result.grace = result.age >= HIDE_AGE
    result.input_observed = result.age < 0.150
    # Store final displayed pose for a later confirmed-correction transition.
    _last_display = result.duplicate(true)
    return result
