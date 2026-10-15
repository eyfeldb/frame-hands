"""Offline paired initial-pose experiment: baseline, stateless mono and stereo.

Only unused unique current-camera correspondences are proposed. Every replacement
is hypothesis-only, never routed to a live tracker or accepted publication gate.
The mono arm controls for removing the baseline model's temporal memory.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np,torch
from frame_data import Recording,read_audit,load_calibration,NAMES
from live_backend import Backend
from feature_reuse import PreparedRegressor
from replay_raw_chirality import prepare
from run_mesh_trial import pose_matrix
from stereo_acquisition import unused_matches
from probe_roi_presence import PresenceModel,verify_hand
from score_roi_presence import supported_views


def infer(backend,regressor,hand,views,images,world,stamp):
    prepared=prepare(backend.cameras,images,dict(views=views));crops=[];intrinsics=[];extrinsics=[];sources=[]
    for v in sorted(views,key=lambda v:NAMES.index(v['camera'])):
        item=prepared.get((v['camera'],v['id']))
        if item is None:return None
        pixels,meta=item;side=hand['side'];crops.append(pixels[:,::-1].copy() if side else pixels)
        intrinsics.append(meta['intrinsics']);xf=world@np.asarray(meta['camera_to_head_mm'])
        if side:xf[:,0]*=-1
        xf=np.linalg.inv(xf);xf[:3,3]*=.001;extrinsics.append(xf);sources.append(dict(camera=v['camera'],source_id=v['id']))
    frame=backend.Frame(torch.tensor(np.stack(crops)/255.,dtype=torch.float32),torch.tensor(np.stack(intrinsics),dtype=torch.float32),torch.tensor(np.stack(extrinsics),dtype=torch.float32))
    side=torch.tensor([hand['side']]);desc=backend.Desc(torch.tensor([[0,len(crops)]]),side,torch.tensor([False]),side)
    with torch.inference_mode():
        pred=regressor.regress(frame,desc,regressor.extract(frame.left_images));wrist=pred.wrist_xfs[0].clone();wrist[:3,3]*=1000;angles=pred.joint_angles[0]
    return dict(hand,memory=False,hypothesis_only=True,identity_basis='Unchanged baseline side hypothesis; geometry alone does not establish anatomy',source_views=sources,joint_angles=angles.tolist(),wrist_world_mm=wrist.tolist(),vertices_world_mm=backend.mesh(angles,wrist,hand['side']).tolist(),uncertainty_raw=pred.landmark_uncertainty_sigmas[0].tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','baseline','upstream','models','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    if a.device=='cuda':
        if not torch.cuda.is_available():p.error('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    cal=a.session/'calibration';b=Backend(a.upstream,load_calibration(cal if cal.exists() else a.session/'frame/snapshot/device/persist'))
    regressor=PreparedRegressor(b.model,b.skeleton,a.device);regressor.warmup();verifier=PresenceModel(a.models)
    rec=Recording(a.session/'frame/sets.bin');frames=iter(rec.frames(0,float('inf')));current=next(frames,None);audits=read_audit(a.session/'frame/live-views.txt')
    established=set();counts=dict(frames=0,baseline_hands=0,initial_mono=0,matched=0,proposed=0);comparisons=[];presence={arm:[] for arm in ('baseline','mono','stereo')}
    with a.baseline.open() as src,(a.output/'predictions.jsonl').open('x') as output,(a.output/'comparisons.jsonl').open('x') as trace:
        for line in src:
            row=json.loads(line);stamp=row['capture_raw_ns']
            while current is not None and current[1]<stamp:current=next(frames,None)
            if current is None or current[1]!=stamp:raise ValueError('No exact saved pixels')
            seconds,_,images=current;world=pose_matrix(row['capture_pose']['capture']) if row['capture_pose']['valid'] else None
            eligible=set()
            for h in row['hands']:
                if len(h['source_views'])>=2:established.add(h['source_id'])
                elif h['source_id'] not in established:eligible.add(h['source_id']);counts['initial_mono']+=1
            matches=unused_matches(row['hands'],audits[stamp],b.cameras,world,eligible) if world is not None else {}
            counts['frames']+=1;counts['baseline_hands']+=len(row['hands']);counts['matched']+=len(matches);hands=[]
            for h in row['hands']:
                m=matches.get(h['side'])
                if m is None:hands.append(h);continue
                started=time.perf_counter();mono=infer(b,regressor,h,[m['source']],images,world,stamp);stereo=infer(b,regressor,h,[m['source'],m['other']],images,world,stamp)
                if mono is None or stereo is None:hands.append(h);continue
                counts['proposed']+=1;hands.append(stereo);arms={}
                for arm,pose in [('baseline',h),('mono',mono),('stereo',stereo)]:
                    evidence,_=verify_hand(verifier,b.hand,pose,b.cameras,images,world)
                    rows=[dict(stamp=stamp,seconds=seconds,**v) for v in evidence];presence[arm].extend(rows)
                    arms[arm]=dict(supported_cameras=[v['camera'] for v in supported_views(evidence)],evidence=rows,wrist_world_mm=pose['wrist_world_mm'],joint_angles=pose['joint_angles'])
                result=dict(seconds=seconds,stamp=stamp,side=h['side'],source_id=h['source_id'],pair_error=m['error'],arms=arms,probe_ms=(time.perf_counter()-started)*1000)
                comparisons.append(result);trace.write(json.dumps(result,allow_nan=False)+'\n')
            output.write(json.dumps(dict(row,seconds=seconds,hands=hands,stereo_acquisition_hypothesis=True),allow_nan=False)+'\n')
    for arm,rows in presence.items():(a.output/f'{arm}-presence.json').write_text(json.dumps(rows)+'\n')
    report=dict(scope=__doc__+' Output processing_ms is unchanged baseline metadata, not candidate latency.',counts=counts,both_views_supported={arm:sum(len(x['arms'][arm]['supported_cameras'])==2 for x in comparisons) for arm in presence},runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,ncnn=verifier.version),args={k:str(v) for k,v in vars(a).items()},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('counts','both_views_supported')}))

if __name__=='__main__':main()
