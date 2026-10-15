"""Summarize unique diagnostic observations, never count repeated poll rejections."""
from collections import Counter
import json
from pathlib import Path
import sys
from review_recording import read_audit


def summarize(path):
    observations=read_audit(path)
    counts=Counter()
    previous={}
    flips=0
    duplicate=0
    stereo=0
    valid_both=0
    for item in observations.values():
        counts[item['count']]+=1
        sides=[]
        for identity,hand in item['hands'].items():
            side=hand['right']>.5
            sides.append(side)
            if identity in previous and previous[identity]!=side:flips+=1
            previous[identity]=side
            stereo+=hand['stereo']>=2
        duplicate+=len(sides)!=len(set(sides))
        valid_both+=len(sides)==2 and len(set(sides))==2
    return dict(observations=len(observations),hand_count_histogram=dict(counts),
                duplicate_handedness_observations=duplicate,unique_left_and_right=valid_both,
                side_flips_within_same_track=flips,stereo_hand_observations=stereo,
                distinct_track_ids=len(previous),acceptance_verified=False)


if __name__=='__main__':
    print(json.dumps({str(Path(p).name):summarize(Path(p)) for p in sys.argv[1:]},indent=2))
