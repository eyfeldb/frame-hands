"""Prepare existing private poses for the actual Godot presentation replay; no camera/XR."""
import json
from pathlib import Path
import numpy as np
from mesh_packet import pose_packet, rig_data
from live_backend import Backend
from frame_data import capture_poses
BASE=Path(__file__).resolve().parents[3]/'.local/hand-alternatives'
out=BASE/'motion-test'
out.mkdir(exist_ok=True)
live=BASE/'results/live-mesh-20261005T030445Z'
render={r['sequence']:r for line in (live/'render.jsonl').open() if (r:=json.loads(line))}
predictions=[json.loads(line) for line in (live/'predictions.jsonl').open()]
first=min(r['wall_s'] for r in render.values())
events=[]
for p in predictions:
    r=render.get(p['sequence'])
    if r is None:continue
    packet=pose_packet(p,np.eye(4),p['sequence'],p['capture_mono_ns'])
    events.append(dict(now=r['wall_s']-first,capture=r['capture_wall_s']-first,stamp=p['capture_mono_ns']*1e-9,hands=packet['hands'],valid=p['capture_pose']['valid'] and p['current_pose']['valid']))
case=dict(name='observed_7hz_mesh_trial',scope='Observed model poses and viewer arrival times; not physical ground truth',events=events)
(out/'observed.json').write_text(json.dumps(case))
# A separately labeled 30 Hz offline series checks behavior once delivery is fixed.
rows=[json.loads(line) for line in (BASE/'results/v1-maestro/predictions.jsonl').open()]
first=rows[0]['capture_raw_ns']
events=[]
for p in rows:
    t=(p['capture_raw_ns']-first)*1e-9
    hands=[dict(h,side=int(h['side']=='R')) for h in p['hands']]
    packet=pose_packet(dict(hands=hands),np.eye(4),len(events)+1,p['capture_raw_ns'])
    events.append(dict(now=t+.05,capture=t,stamp=t,hands=packet['hands'],valid=p['has_capture_pose']))
(out/'assumed-30hz.json').write_text(json.dumps(dict(name='recorded_30hz_model_assumed_50ms_delivery',scope='Counterfactual delivery timing on old recorded poses; not achieved live frame rate or accuracy',events=events)))
print('Prepared exact observed delivery and separately labeled 30 Hz timing scenario')

# Reproducible GPU rig fixture against the independent Torch skinning path.
backend=Backend(BASE/'UmeTrack',{})
reference=next(p for p in rows if p['seconds']>51 and len(p['hands'])==2)
manifest=json.loads((BASE/'results/v1-maestro/manifest.json').read_text())
head=capture_poses(Path(manifest['args']['session']))[reference['capture_mono_ns']]
hands=[dict(h,side=int(h['side']=='R')) for h in reference['hands']]
expected=[(backend.mesh(h['joint_angles'],h['wrist_world_mm'],h['side'])*.001).tolist() for h in hands]
(out/'rig.json').write_text(json.dumps(rig_data(backend.hand)))
(out/'pose.json').write_text(json.dumps(pose_packet(dict(hands=hands),head,1,reference['capture_mono_ns'])))
(out/'expected.json').write_text(json.dumps(expected))
