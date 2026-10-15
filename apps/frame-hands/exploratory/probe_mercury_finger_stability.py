"""Read-only audit of native finger state against frozen image-derived curl outputs."""
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
    protocol=dict(reference=str(a.reference.resolve()),source=str(source.resolve()),purpose='Expose native prior finger angles, learned curl average, unmodified stability multiplier and per-view variance without altering fits.',branches=declared['branches'],comparison='Every fit must equal the frozen reference exactly. Stability diagnostic applies only to non-reset steps, before fitting the current observations.',limitations='Network variance is an uncalibrated output, not measured uncertainty. No causal claim or changed weighting from correlations alone.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('probe_mercury_finger_stability.py','mercury_native_solver.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    counts=Counter();old_camera={};last_angles={}
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
                fit=solver.run(views,reset=reset,smoothing=2);assert fit==ref['fit'], 'Read-only diagnostics changed a fit'
                after=solver.fingers();last_angles[key]=after.copy();old_camera[key]=current;counts['exact_'+key[0]]+=1;counts['exact_total']+=1
                fits.append(dict(branch=key[0],side=key[1],reset_reason=ref['reset_reason'],crop_switch=ref['crop_switch'],crop_modes=ref['crop_modes'],before_diagnostic=None if diagnostic is None else diagnostic.tolist(),after_angles=after.tolist(),reprojection_error=fit['reprojection_error'],world_joints_mm=ref['world_joints_mm'],ncnn_disagreement_px=ref['ncnn_disagreement_px']))
            output.write(json.dumps(dict(seconds=row['seconds'],stamp=row['stamp'],fits=fits),allow_nan=False)+'\n');counts['frames']+=1
            if counts['frames']%150==0:output.flush();print('frames',counts['frames'],'exact fits',counts['exact_total'],flush=True)
    summary=dict(protocol=protocol,counts=counts,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},columns=['prior_proximal','prior_pip','prior_dip','prior_sum','learned_average','native_multiplier','left_curl','left_variance','right_curl','right_variance'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(counts))


if __name__=='__main__':main()
