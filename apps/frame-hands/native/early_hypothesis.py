"""Single-hypothesis early-exit probe using only its own fit and detector prior.

Experimental fixed screen, off by default. Thresholds fixed after inspecting
training marginal distributions, before development causal evaluation. Full
alternative remains required when the first candidate does not pass.
"""
import math
MAX_REPROJECTION_PX=10.0
MAX_RAW_SIGMA=.012
MIN_DETECTOR_CONFIDENCE=.8
MIN_PRESENCE=.8


def first_side(raw_right):
    if not math.isfinite(raw_right) or not 0<=raw_right<=1:raise ValueError('Invalid detector side score')
    return int(raw_right>=.5)


def early_score(raw_right,presence,first_error,first_sigma):
    if not all(math.isfinite(v) for v in (raw_right,presence,first_error,first_sigma)):return None
    if not 0<=raw_right<=1 or not MIN_PRESENCE<=presence<=1 or first_error<0 or first_sigma<0:return None
    if max(raw_right,1-raw_right)<MIN_DETECTOR_CONFIDENCE or first_error>MAX_REPROJECTION_PX or first_sigma>MAX_RAW_SIGMA:return None
    # Deliberately conservative relative to the maximum .98/.02 full policy.
    # This is not a calibrated probability or a guarantee about omitted evidence.
    return .90 if first_side(raw_right) else .10
