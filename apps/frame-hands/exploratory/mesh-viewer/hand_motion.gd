class_name HandMotion
extends RefCounted
# Presentation-only state, never an input/gesture confidence source.
const HOLD_AGE := 0.100
const HIDE_AGE := 0.170
const PREDICT_SECONDS := 0.025
const MAX_PREDICT_METERS := 0.025
const MAX_PREDICT_RADIANS := 0.12
var active := false
var source_id := -1
var stamp := -1.0
var capture_local := -1.0
var position := Vector3.ZERO
var orientation := Quaternion.IDENTITY
var angles := PackedFloat64Array()
var raw_position := Vector3.ZERO
var raw_orientation := Quaternion.IDENTITY
var raw_angles := PackedFloat64Array()
var velocity := Vector3.ZERO
var angular_velocity := Vector3.ZERO
var derivatives := PackedFloat64Array()
var acceleration := 0.0
var rendered_position := Vector3.ZERO
var rendered_orientation := Quaternion.IDENTITY
var rendered_angles := PackedFloat64Array()
var pending := false
var pending_position := Vector3.ZERO
var pending_orientation := Quaternion.IDENTITY

func clear():
    active = false
    source_id = -1
    velocity = Vector3.ZERO
    angular_velocity = Vector3.ZERO
    pending = false

static func alpha(cutoff: float, dt: float) -> float:
    return 1.0 / (1.0 + 1.0 / (TAU * cutoff * dt))

static func rotation_vector(q: Quaternion) -> Vector3:
    var canonical := q.normalized()
    if canonical.w < 0: canonical = -canonical
    var angle := canonical.get_angle()
    if angle < 0.000001: return Vector3.ZERO
    return canonical.get_axis() * angle

func pose_angles(values: Array) -> PackedFloat64Array:
    var result := PackedFloat64Array()
    for i in 20:
        var value := float(values[i])
        result.append(value)
    return result

func reset_pose(id: int, time: float, capture: float, p: Vector3, q: Quaternion, a: PackedFloat64Array):
    active = true
    source_id = id
    stamp = time
    capture_local = capture
    position = p
    orientation = q
    angles = a.duplicate()
    raw_position = p
    raw_orientation = q
    raw_angles = a.duplicate()
    rendered_position = p
    rendered_orientation = q
    rendered_angles = a.duplicate()
    velocity = Vector3.ZERO
    angular_velocity = Vector3.ZERO
    derivatives.resize(20)
    derivatives.fill(0)
    acceleration = 0.0
    pending = false

func observe(id: int, time: float, capture: float, now: float, p: Vector3, q: Quaternion, values: Array, mode: String) -> bool:
    if values.size() < 20 or not p.is_finite() or not q.is_finite(): return false
    for v in values:
        if not is_finite(float(v)): return false
    if now-capture > HIDE_AGE: return false
    # A duplicate/old observation cannot renew the hold budget or reassign a side.
    if active and time <= stamp: return false
    var a := pose_angles(values)
    var dt := time-stamp
    if not active or id != source_id or dt > 0.150 or now-capture_local > visibility_deadline():
        reset_pose(id,time,capture,p,q,a)
        return true
    if dt < 0.001: return false
    var distance := p.distance_to(raw_position)
    if mode != "raw" and (distance > 0.10+3.5*dt or rotation_vector(q*raw_orientation.inverse()).length() > 1.4+8.0*dt):
        if pending and p.distance_to(pending_position) < 0.060 and rotation_vector(q*pending_orientation.inverse()).length() < 0.35:
            reset_pose(id,time,capture,p,q,a)
            return true
        pending = true
        pending_position = p
        pending_orientation = q
        return false
    pending = false
    var v := (p-raw_position)/dt
    var omega := rotation_vector(q*raw_orientation.inverse())/dt
    var derivative_alpha := alpha(6.0,dt)
    acceleration = (v-velocity).length()/dt
    # Stop/reversal evidence immediately removes momentum in the old direction.
    if v.dot(velocity) < 0.0 or v.length() < 0.03:
        velocity = v
    else:
        velocity = velocity.lerp(v,derivative_alpha)
    if omega.dot(angular_velocity) < 0.0:
        angular_velocity = omega
    else:
        angular_velocity = angular_velocity.lerp(omega,derivative_alpha)
    velocity = velocity.limit_length(2.0)
    angular_velocity = angular_velocity.limit_length(6.0)
    var p_alpha := alpha(4.0+15.0*velocity.length(),dt)
    var q_alpha := alpha(4.0+1.5*angular_velocity.length(),dt)
    position = p if mode == "raw" else position.lerp(p,p_alpha)
    orientation = q if mode == "raw" else orientation.slerp(q,q_alpha).normalized()
    for i in 20:
        var derivative := (a[i]-raw_angles[i])/dt
        derivatives[i] = lerpf(derivatives[i],derivative,derivative_alpha)
        angles[i] = a[i] if mode == "raw" else lerpf(angles[i],a[i],alpha(5.0+0.8*absf(derivatives[i]),dt))
    raw_position = p
    raw_orientation = q
    raw_angles = a
    stamp = time
    capture_local = capture
    return true

func visibility_deadline() -> float:
    return HIDE_AGE

func fade_start() -> float:
    return HOLD_AGE

func sample(now: float, delta: float, mode: String) -> Dictionary:
    if not active: return {"visible":false}
    var age := maxf(0,now-capture_local)
    if age >= visibility_deadline():
        clear()
        return {"visible":false}
    var target_position := position
    var target_orientation := orientation
    var horizon := 0.0
    if mode == "predict":
        # Never extrapolate across a long dropout. Constant capped offset freezes
        # after 25 ms; it cannot run away while measurements are absent.
        horizon = minf(age,PREDICT_SECONDS)/(1.0+acceleration/40.0)
        target_position += (velocity*horizon).limit_length(MAX_PREDICT_METERS)
        var rotation := (angular_velocity*horizon).limit_length(MAX_PREDICT_RADIANS)
        if rotation.length() > 0.000001:
            target_orientation = (Quaternion(rotation.normalized(),rotation.length())*orientation).normalized()
    var blend := 1.0 if mode == "raw" else 1.0-exp(-maxf(delta,0.00001)/0.012)
    rendered_position = rendered_position.lerp(target_position,blend)
    rendered_orientation = rendered_orientation.slerp(target_orientation,blend).normalized()
    for i in 20: rendered_angles[i] = lerpf(rendered_angles[i],angles[i],blend)
    return {"visible":true,"position":rendered_position,"orientation":rendered_orientation,
        "angles":rendered_angles,"opacity":1.0-clampf((age-fade_start())/(visibility_deadline()-fade_start()),0,1),
        "age":age,"predicted_seconds":horizon,"held":age>HOLD_AGE,"source_id":source_id}
