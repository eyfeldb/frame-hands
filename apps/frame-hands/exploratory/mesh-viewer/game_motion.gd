extends "res://gap_motion.gd"
# Game-only gap policy. Fresh-pose filtering/prediction is unchanged.
# No fresh timestamps or controller presses are created during a hold.
func visibility_deadline() -> float:
    if eligible():return 0.350
    if _mode != "raw" and good_observations>=3 and not pending and receipt_age<0.150 and cadence<=0.075 and acceleration<120.0 and velocity.length()<=2.001 and angular_velocity.length()<=6.001:
        return 0.250
    return HIDE_AGE
