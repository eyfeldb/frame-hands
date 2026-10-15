"""Render a manual/exhaustive offline crop diagnostic; never label it LIVE."""
import argparse
import json
from pathlib import Path
import review_recording as review

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('probe',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
probe=json.loads(args.probe.read_text())
for index,images in enumerate(review.read_sets(args.directory/'sets.bin')):
    if index==probe['record_index']:break
else:raise ValueError('Requested frame is missing')
observation=dict(hands={},views=[])
for index,view in enumerate(probe['views']):
    if not view['method'].startswith('manual_seed_'):continue
    observation['hands'][index]=dict(right=view['right'],stereo=1)
    observation['views'].append(dict(camera=view['camera'],hand=index,presence=view['presence'],points=review.np.array(view['points'])))
image=review.draw(images,observation,0,label='MANUAL CROP, NOT TRACKING')
if not review.cv2.imwrite(str(args.output),image):raise OSError('Image write failed')
