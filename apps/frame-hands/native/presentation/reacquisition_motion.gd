extends "res://hand_motion.gd"
# Opt-in presentation experiment: ordinary accepted updates can reacquire a held
# hand without calling reset_pose(). Reuse the same60ms complete-pose catch-up.
var reacquisition_enabled := true
var reacquisition_corrections := 0

func observe(id: int, time: float, capture: float, now: float, p: Vector3, q: Quaternion, values: Array, mode: String) -> bool:
    var recover: bool = reacquisition_enabled and correction_enabled and mode != "raw" and active and id == source_id and time > stamp and capture > capture_local and not _last_display.is_empty() and int(_last_display.get("source_id",-1)) == id and _last_display.get("held",false) and now >= _last_display_time and now-_last_display_time <= 0.030 and now-capture_local < HIDE_AGE
    var accepted: bool = super.observe(id,time,capture,now,p,q,values,mode)
    # Preserve any already-running correction deadline. Rejected/duplicate data,
    # new identities, missing context and hidden hands never start this bridge.
    if accepted and recover and (_transition_start < 0.0 or now >= _transition_start+CORRECTION_SECONDS):
        _transition_start = now
        _transition_from = _last_display.duplicate(true)
        corrections += 1
        reacquisition_corrections += 1
    return accepted
