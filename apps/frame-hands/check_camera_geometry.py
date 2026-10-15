"""Bounded in-memory side-camera identity check. Saves numerical evidence only.

Scores factory-calibrated epipolar geometry on multiple fresh synchronized pairs.
This checks image identity, not hand accuracy. Exit 0=as named, 3=swapped,
2=inconclusive. Thresholds are provisional engineering gates.
"""
import argparse
import json
from pathlib import Path
import sys
import time

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'deps/python'),str(BASE/'upstream')]
from tools.ring import Ring
from tools.check_sides import matches,score,load_cams,PIPES


def check(seconds=8):
    ring=Ring();calibration=load_cams();cameras={};samples=[]
    result=dict(verified=False,kind='camera identity, not tracking accuracy',samples=samples)
    for camera in ring.cams:
        pipe=Path(f'/sys/class/video4linux/video{camera.node}/name').read_text().strip()
        name=PIPES.get(pipe)
        if not name or camera.name.endswith('-dark'):continue
        if (camera.width,camera.height)!=(1056,1024):
            return dict(result,reason='Side-camera dimensions do not match this Frame calibration'),2
        if name in cameras:return dict(result,reason='Ambiguous side-camera entries'),2
        cameras[name]=camera
    if set(cameras)!={'slam_left','slam_right'}:
        return dict(result,reason='Forward NV12 camera pair is missing; corrected reader required'),2
    seen=set();deadline=time.monotonic()+seconds
    votes=[]
    while time.monotonic()<deadline and len(samples)<8:
        if not ring.alive(.5):return dict(result,reason='Camera reader heartbeat expired'),2
        left=ring.read(cameras['slam_left']);right=ring.read(cameras['slam_right'])
        if left is None or right is None:
            time.sleep(.02);continue
        key=(left.frame,right.frame);now=time.monotonic_ns()
        ages=[now-left.publish_ns,now-right.publish_ns]
        if key in seen or min(ages)<0 or max(ages)>100000000 or abs(left.capture_ns-right.capture_ns)>3000000:
            time.sleep(.02);continue
        seen.add(key)
        a,b=matches(left.image,right.image)
        named=score(calibration['slam_left'],calibration['slam_right'],a,b)
        swapped=score(calibration['slam_right'],calibration['slam_left'],a,b)
        decisive=len(a)>=30 and max(named,swapped)>=.5 and abs(named-swapped)>=.2
        samples.append(dict(frames=key,matches=len(a),named=named,swapped=swapped,decisive=decisive))
        if decisive:votes.append('named' if named>swapped else 'swapped')
        time.sleep(.15)
    if len(votes)<3 or len(set(votes))!=1:
        return dict(result,reason='Insufficient consistent geometry; use a well-lit textured scene'),2
    ordering=votes[0]
    return dict(result,verified=True,ordering=ordering,reason='Multiple fresh pairs agree with factory geometry'),0 if ordering=='named' else 3


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:result,code=check()
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        result,code=dict(verified=False,reason=str(exc)),2
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(code)
