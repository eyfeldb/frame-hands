"""Compare factory transforms numerically; do not export device identifiers."""
import json
import sys
from pathlib import Path
sys.path.insert(0,'/home/steamos/devkit-game/FrameHands/deps/python')
import numpy as np


def pose(d, scale=1.):
    x=np.asarray(d['plus_x']);z=np.asarray(d['plus_z']);t=np.eye(4)
    t[:3,:3]=np.column_stack((x,np.cross(z,x),z))
    t[:3,3]=np.asarray(d['position'])*scale
    return t


rig=json.loads(Path('/persist/xrservice.json').read_text())
dev=json.loads(Path('/persist/device_config.json').read_text())
cad_from_zero=pose(dev['cv']['cad_from_cal'])
head_from_cad=np.linalg.inv(pose(dev['head']))
result=[]
for i,camera in enumerate(rig['cameras']):
    direct=cad_from_zero@pose(camera['extrinsics'],.001)
    reference=pose(rig['cad']['cameras'][i],.001)
    entry=dict(name=camera['sourceCamera'],parent_annotation=camera.get('parent'),
        direct_position_error_mm=float(np.linalg.norm(direct[:3,3]-reference[:3,3])*1000),
        direct_rotation_max_error=float(np.max(np.abs(direct[:3,:3]-reference[:3,:3]))),
        origin_head_m=(head_from_cad@direct)[:3,3].tolist(),
        optical_axis_head=(head_from_cad@direct)[:3,2].tolist())
    if camera.get('parent'):
        parent=int(camera['parent'][3:])
        chained=cad_from_zero@pose(rig['cameras'][parent]['extrinsics'],.001)@pose(camera['extrinsics'],.001)
        entry['chained_position_error_mm']=float(np.linalg.norm(chained[:3,3]-reference[:3,3])*1000)
        entry['chained_rotation_max_error']=float(np.max(np.abs(chained[:3,:3]-reference[:3,:3])))
    result.append(entry)
print(json.dumps(dict(cameras=result,note='Factory self-consistency only; not image-to-camera identity or measured optical accuracy.'),indent=2))
