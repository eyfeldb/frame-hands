"""Observation-only stereo startup comparison on frozen recorded model outputs."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from frame_data import load_calibration
from mercury_native_solver import NativeSolver, camera_transforms
from run_mesh_trial import pose_matrix
from mercury_stereo_init import initialize_from_stereo
from probe_mercury_solver import project, stats
from replay_mercury_solver import to_world
from frame_data import read_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','upstream','dll','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream.resolve()));a.output.mkdir(parents=True,exist_ok=False)
    declared=json.loads((a.reference/'protocol.json').read_text());source=Path(declared['source']);parent=json.loads((source/'protocol.json').read_text());origin=json.loads((Path(parent['source'])/'protocol.json').read_text())
    assert hashlib.sha256((source/'frames.jsonl').read_bytes()).hexdigest()==declared['source_frames_sha256']
    baseline=Path(origin['baseline']);assert hashlib.sha256(baseline.read_bytes()).hexdigest()==declared['baseline_sha256']
    poses={}
    with baseline.open() as f:
        for line in f:
            row=json.loads(line);poses[row['capture_raw_ns']]=row['capture_pose']
    cameras=load_calibration(Path(origin['session'])/'calibration');pose,head_from_left=camera_transforms(cameras)
    audit=read_audit(Path(origin['session'])/'frame/live-views.txt')
    source_hands={}
    for line in baseline.read_text().splitlines():
        row=json.loads(line);source_hands[row['capture_raw_ns']]={h['side']:h for h in row['hands']}
    protocol=dict(reference=str(a.reference.resolve()),source=str(source.resolve()),session=origin['session'],purpose='Use current stereo palm observations only for candidate startup; same frozen observations, fixed90mm size, smoothing2 and original reset schedule.',branches=declared['branches'],comparison='All original fits must remain exact. Candidate is initialized only at original reset events; continuing fits preserve independent transported history. Rejected stereo geometry retains original cold initialization.',limitations='No detector, side reassignment, current ML change or native acquisition. Independent ncnn agreement and step size are proxies, not physical accuracy. No truth initialization.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_stereo_init.py','mercury_native_solver.py','mercury_stereo_init.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    counts=Counter();old_camera={};last_angles={};candidates={}
    with ExitStack() as stack:
        solvers={(b,s):stack.enter_context(NativeSolver(a.dll,pose,s)) for b in declared['branches'] for s in (0,1)}
        reference=stack.enter_context((a.reference/'frames.jsonl').open());images=stack.enter_context((source/'frames.jsonl').open());output=stack.enter_context((a.output/'frames.jsonl').open('x'))
        for line in reference:
            row=json.loads(line);image=json.loads(next(images));assert row['stamp']==image['stamp'];fits=[]
            if row['fits']:
                capture=poses[row['stamp']];assert capture['valid'];world=pose_matrix(capture['capture']);current=world.copy();current[:3,3]*=.001;current=current@head_from_left
            for ref in row['fits']:
                key=(ref['branch'],ref['side']);solver=solvers[key];reset=ref['reset_reason'] is not None
                views={name:next(v['model'] for v in image['observations'] if v['side']==ref['side'] and v['camera']==name and v['branch']==source_branch) for name,source_branch in ref['crop_modes'].items()}
                diagnostic=None
                if not reset:
                    solver.transport(np.linalg.inv(current)@old_camera[key]);diagnostic=solver.finger_stability(views)
                    np.testing.assert_array_equal(diagnostic[:,:3],last_angles[key]);counts['diagnosed_nonreset_fits']+=1
                fit=solver.run(views,reset=reset,smoothing=2);assert fit==ref['fit'], 'Original control changed'
                if reset:
                    if key in candidates:candidates[key].close()
                    candidates[key]=stack.enter_context(NativeSolver(a.dll,pose,key[1]))
                    initialization=initialize_from_stereo(candidates[key],views)
                    counts['initialization_'+initialization['reason']]+=1
                else:
                    candidates[key].transport(np.linalg.inv(current)@old_camera[key]);initialization=None
                candidate=candidates[key].run(views,smoothing=2)
                projected=project(candidate,head_from_left,cameras);disagreement={};hand=source_hands[row['stamp']][key[1]]
                for camera,uv in projected.items():
                    source_id=next((v['source_id'] for v in hand['source_views'] if v['camera']==camera),None)
                    reference_view=next((v for v in audit.get(row['stamp'],{}).get('views',[]) if v['camera']==camera and v['id']==source_id),None)
                    if reference_view is not None:disagreement[camera]=stats(np.linalg.norm(uv-reference_view['points'],axis=1))
                after=solver.fingers();last_angles[key]=after.copy();old_camera[key]=current;counts['exact_'+key[0]]+=1;counts['exact_total']+=1;counts['candidate_accepted']+=bool(candidate['native_active'] and candidate['passes_native_reprojection_gate'])
                fits.append(dict(**ref,initialization=initialization,candidate_fit=candidate,candidate_world_joints_mm=to_world(candidate,head_from_left,world).tolist(),candidate_projected={k:v.tolist() for k,v in projected.items()},candidate_ncnn_disagreement_px=disagreement))
            output.write(json.dumps(dict(seconds=row['seconds'],stamp=row['stamp'],fits=fits),allow_nan=False)+'\n');counts['frames']+=1
            if counts['frames']%150==0:output.flush();print('frames',counts['frames'],'exact fits',counts['exact_total'],flush=True)
    summary=dict(protocol=protocol,counts=counts,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(counts))


if __name__=='__main__':main()
