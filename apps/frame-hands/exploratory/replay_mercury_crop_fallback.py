"""Frozen-observation test of geometry-only fallback and crop-switch history."""
import argparse
from copy import deepcopy
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from frame_data import Recording, load_calibration, read_audit
from mercury_native_solver import NativeSolver, camera_transforms
from probe_mercury_solver import project, stats
from probe_mercury_keypoints import draw
from replay_mercury_solver import reset_reason, to_world
from run_mesh_trial import pose_matrix

CAMERAS=('slam_left','slam_right')
BRANCHES=('box','rays','fallback_carry','fallback_reset')


def choose_geometry_views(observations):
    """Fallback on missing crop construction, never on a low model confidence."""
    views={};choices={}
    for camera in CAMERAS:
        source='box' if camera in observations.get('box',{}) else 'rays'
        if camera in observations.get(source,{}):
            views[camera]=observations[source][camera];choices[camera]=source
    return views,choices


def constant_finger_stability(solver,views):
    """Diagnostic ablation only: hold the existing four curl weights at1.

    Direct curl residuals/initialization must remain disabled in pinned native
    sources. Only model curl values are overridden; all geometric observations
    and learned variances remain unchanged. Never called on a reset step.
    """
    result=deepcopy(views);sums=solver.fingers().sum(axis=1,dtype=np.float32)
    for view in result.values():view['curls'][1:5]=sums.tolist()
    np.testing.assert_array_equal(solver.finger_stability(result)[:,5],np.ones(4))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','upstream','dll','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--constant-stability',action='store_true',help='Offline causal ablation; add constant finger-weight branches')
    a=p.parse_args();branches=BRANCHES+(('box_constant','fallback_constant') if a.constant_stability else ());sys.path.insert(0,str(a.upstream.resolve()));cv2.setNumThreads(1)
    source_protocol=json.loads((a.source/'protocol.json').read_text());source_summary=json.loads((a.source/'summary.json').read_text())
    protocol=json.loads((Path(source_protocol['source'])/'protocol.json').read_text());baseline_path=Path(protocol['baseline'])
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==protocol['baseline_sha256']
    if not a.constant_stability:assert hashlib.sha256(a.dll.read_bytes()).hexdigest()==source_summary['dll_sha256']
    base={}
    with baseline_path.open() as f:
        for line in f:
            row=json.loads(line);base[row['capture_raw_ns']]=dict(capture_pose=row['capture_pose'],hands={h['side']:dict(source_id=h['source_id'],source_views=h['source_views']) for h in row['hands']})
    a.output.mkdir(parents=True,exist_ok=False)
    declared=dict(source=str(a.source.resolve()),source_frames_sha256=hashlib.sha256((a.source/'frames.jsonl').read_bytes()).hexdigest(),baseline_sha256=protocol['baseline_sha256'],branches=branches,constant_stability_ablation=a.constant_stability,start=0,end=30,size_m=.09,smoothing=2,
        selection='Use each original box crop whenever construction succeeded, regardless of presence. Only missing box geometry selects rays. No crop/fit-quality ranking or threshold changes.',
        history='Carry uses ordinary capture/generation/gap resets. Reset additionally clears optimizer history when the selected per-camera crop method changes; both preserve independent state.',
        hypothesis='Recover geometry-lost views without replacing supported image crops; audit collateral history effects after returning to original crops.',
        controls='Original box/rays replay must reproduce all original native fits exactly. ML observations are frozen; no model calls, new captures or negative-image retuning.')
    (a.output/'protocol.json').write_text(json.dumps(declared,indent=2)+'\n');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_crop_fallback.py','mercury_native_solver.py','replay_mercury_solver.py','frame_data.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    session=Path(protocol['session']);rec=Recording(session/'frame/sets.bin');cameras=load_calibration(session/'calibration');pose,head_from_left=camera_transforms(cameras);audit=read_audit(session/'frame/live-views.txt')
    counts=Counter();last_review=-1
    with ExitStack() as stack:
        solvers={(branch,side):stack.enter_context(NativeSolver(a.dll,pose,side)) for branch in branches for side in (0,1)};previous={};old_camera={};old_modes={}
        source=stack.enter_context((a.source/'frames.jsonl').open());output=stack.enter_context((a.output/'frames.jsonl').open('x'))
        writer=None
        try:
            for seconds,stamp,images in rec.frames(0,30):
                recorded=json.loads(next(source));assert recorded['stamp']==stamp
                baseline=base.get(stamp);valid=baseline is not None and baseline['capture_pose']['valid'];world=pose_matrix(baseline['capture_pose']['capture']) if valid else None
                if valid:current=world.copy();current[:3,3]*=.001;current=current@head_from_left
                fits=[];selection=[]
                for side in (0,1):
                    observations={b:{v['camera']:v['model'] for v in recorded['observations'] if v['branch']==b and v['side']==side} for b in ('box','rays')}
                    selected,choices=choose_geometry_views(observations);selection.append(dict(side=side,choices=choices))
                    for branch in branches:
                        key=(branch,side);observation_branch='box' if branch=='box_constant' else branch;views=observations[observation_branch] if observation_branch in observations else selected;modes={name:observation_branch for name in views} if observation_branch in observations else choices
                        if not valid or len(views)!=2 or not all(v['active'] for v in views.values()):previous[key]=None;continue
                        hand=baseline['hands'][side];generation=(side,hand['source_id']);reason=reset_reason(previous.get(key),stamp,generation,valid)
                        changed=reason is None and modes!=old_modes.get(key)
                        if branch=='fallback_reset' and changed:reason='Crop construction changed'
                        if reason is None:solvers[key].transport(np.linalg.inv(current)@old_camera[key])
                        supplied=constant_finger_stability(solvers[key],views) if branch.endswith('_constant') and reason is None else views
                        fit=solvers[key].run(supplied,reset=reason is not None,smoothing=2);accepted=fit['native_active'] and fit['passes_native_reprojection_gate']
                        previous[key]=(stamp,generation) if accepted else None;old_camera[key]=current;old_modes[key]=modes.copy()
                        projected=project(fit,head_from_left,cameras);disagreement={}
                        for name,uv in projected.items():
                            source_id=next((v['source_id'] for v in hand['source_views'] if v['camera']==name),None)
                            reference=next((v for v in audit.get(stamp,{}).get('views',[]) if v['camera']==name and v['id']==source_id),None)
                            if reference is not None:disagreement[name]=stats(np.linalg.norm(uv-reference['points'],axis=1))
                        if branch in ('box','rays'):
                            original=next(f for f in recorded['fits'] if f['side']==side and f['branch']==branch);assert fit==original['fit'];counts[branch+'_exact_fits']+=1
                        counts[branch+'_fits']+=1;counts[branch+'_accepted']+=int(accepted);counts[branch+'_crop_switches']+=int(changed)
                        fits.append(dict(branch=branch,side=side,generation=generation,accepted=accepted,fit=fit,reset_reason=reason,crop_modes=modes,crop_switch=changed,
                                         finger_angles=solvers[key].fingers().tolist() if a.constant_stability else None,world_joints_mm=to_world(fit,head_from_left,world).tolist(),projected={k:v.tolist() for k,v in projected.items()},ncnn_disagreement_px=disagreement))
                output.write(json.dumps(dict(seconds=seconds,stamp=stamp,selection=selection,fits=fits),allow_nan=False)+'\n');counts['frames']+=1
                if a.constant_stability or int(seconds)!=last_review or 16<=seconds<20:
                    panels=[]
                    for name in CAMERAS:
                        image=images[name][0];columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)]
                        for branch in (('box','box_constant','fallback_carry','fallback_constant') if a.constant_stability else ('box','fallback_carry','fallback_reset')):
                            canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                            for fit in fits:
                                if fit['branch']==branch and fit['accepted']:canvas=draw(canvas,fit['projected'][name],(255,255,0) if fit['side']==0 else (0,140,255))
                            columns.append(canvas)
                        panel=cv2.hconcat([cv2.resize(c,(384,round(384*image.shape[0]/image.shape[1]))) for c in columns]);cv2.putText(panel,name+' '+str(round(seconds,3))+(' raw / box / box-constant / carry / carry-constant' if a.constant_stability else ' raw / box / carry / reset'),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
                    sheet=cv2.vconcat(panels)
                    if int(seconds)!=last_review:cv2.imwrite(str(a.output/('review-'+str(int(seconds)).zfill(2)+'.jpg')),sheet);last_review=int(seconds)
                    if a.constant_stability or 16<=seconds<20:
                        if writer is None:
                            writer=cv2.VideoWriter(str(a.output/('constant-trajectory-0-30.mp4' if a.constant_stability else 'crossing-16-20.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(sheet.shape[1],sheet.shape[0]))
                            if not writer.isOpened():raise RuntimeError('Review video writer failed')
                        writer.write(sheet)
                if counts['frames']%150==0:output.flush();print('frames',counts['frames'],'carry',counts['fallback_carry_accepted'],'reset',counts['fallback_reset_accepted'],flush=True)
        finally:
            if writer is not None:writer.release()
    summary=dict(protocol=declared,counts=counts,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),control_dll_sha256=source_summary['dll_sha256'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},limitations='Frozen current-image estimates, external seed and identity. No learned feedback or native acquisition. Model disagreement is not physical truth. New independently joined ncnn references include previously unsupported cameras; compare exact paired subsets. No live changes.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(counts))


if __name__=='__main__':main()
