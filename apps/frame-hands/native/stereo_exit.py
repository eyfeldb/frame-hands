"""Finish omitted evidence when provisional ownership lacks a stereo assignment.

The probe uses a copy of pre-frame association state and current available
scores only. It must not advance real state or consult skipped hypotheses.
"""
import copy


def needs_completion(ownership,observation,world):
    probe=copy.copy(ownership)
    probe.tracks=copy.deepcopy(ownership.tracks)
    probe.last_debug=copy.deepcopy(ownership.last_debug)
    probe.assign(observation,world)
    assignments=probe.last_debug.get('assignments',[])
    return bool(probe.last_debug.get('reason')) or not assignments or any(len(row['views'])!=2 for row in assignments)
