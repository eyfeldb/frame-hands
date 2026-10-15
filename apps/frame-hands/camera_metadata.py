"""Analyze existing camera buffers in memory on Frame; emit numbers only."""
from pathlib import Path
import json
import sys
import time
BASE=Path('/home/steamos/devkit-game/FrameHands')
sys.path[:0]=[str(BASE/'deps/python'),str(BASE/'upstream')]
import numpy as np
from tools.ring import Ring
from tools.check_sides import matches,score,load_cams,PIPES
ring=Ring();frames={};result=dict(live=ring.alive(),cameras=[])
for camera in ring.cams:
    frame=ring.read(camera)
    if frame is None:continue
    pipe=Path(f'/sys/class/video4linux/video{camera.node}/name').read_text().strip()
    name=PIPES.get(pipe,camera.name);frames[name]=frame.image
    result['cameras'].append(dict(node=camera.node,name=name,frame=frame.frame,
        age_ms=(time.monotonic_ns()-frame.publish_ns)/1e6,
        mean=float(frame.image.mean()),p05=float(np.percentile(frame.image,5)),
        p95=float(np.percentile(frame.image,95))))
if 'slam_left' in frames and 'slam_right' in frames:
    cameras=load_cams();a,b=matches(frames['slam_left'],frames['slam_right'])
    result['pair']=dict(matches=len(a),named=score(cameras['slam_left'],cameras['slam_right'],a,b),
                       swapped=score(cameras['slam_right'],cameras['slam_left'],a,b))
print(json.dumps(result,indent=2))
