"""Bounded local camera diagnostic: two rows, raw then equalized; no upload."""
import argparse
import json
from pathlib import Path
import sys
import time

BASE=Path('/home/steamos/devkit-game/FrameHands')
sys.path[:0]=[str(BASE/'deps/python'),str(BASE/'upstream')]
import cv2
import numpy as np
from tools.ring import Ring
from tools.check_sides import matches,score,load_cams,PIPES

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--allow-stale',action='store_true')
args=parser.parse_args()
ring=Ring()
if not args.allow_stale and not ring.alive():
    raise SystemExit('No live camera reader; use --allow-stale only to inspect a labeled historical sample.')
args.output.mkdir(parents=True,exist_ok=False)
frames={}
raw=[];equal=[]
report=dict(live=ring.alive(),cameras=[],purpose='camera geometry/visibility diagnostic, not accuracy ground truth')
for camera in ring.cams:
    frame=ring.read(camera)
    if frame is None:continue
    pipe=Path(f'/sys/class/video4linux/video{camera.node}/name').read_text().strip()
    name=PIPES.get(pipe,camera.name)
    frames[name]=frame.image
    cv2.imwrite(str(args.output/f'video{camera.node}.png'),frame.image)
    a=cv2.cvtColor(frame.image,cv2.COLOR_GRAY2BGR)
    b=cv2.cvtColor(cv2.createCLAHE(2.,(8,8)).apply(frame.image),cv2.COLOR_GRAY2BGR)
    cv2.putText(a,f'{name} video{camera.node}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
    cv2.putText(b,'CLAHE diagnostic only',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
    raw.append(a);equal.append(b)
    report['cameras'].append(dict(node=camera.node,name=name,frame=frame.frame,
        age_since_publication_ms=(time.monotonic_ns()-frame.publish_ns)/1e6,
        width=camera.width,height=camera.height,mean=float(frame.image.mean()),
        p05=float(np.percentile(frame.image,5)),p95=float(np.percentile(frame.image,95))))
if raw:cv2.imwrite(str(args.output/'montage.png'),np.vstack([np.hstack(raw),np.hstack(equal)]))
if 'slam_left' in frames and 'slam_right' in frames:
    cameras=load_cams();a,b=matches(frames['slam_left'],frames['slam_right'])
    report['pair']=dict(matches=len(a),named=score(cameras['slam_left'],cameras['slam_right'],a,b),
                       swapped=score(cameras['slam_right'],cameras['slam_left'],a,b),
                       caveat='Single image pair; require several fresh textured samples for camera identity.')
(args.output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
