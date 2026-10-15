"""Offline falsification of short monocular rigid continuation; not a runtime policy.

Artificially withhold a camera from stereo records; never fit the held-out view.
History is exact recorded native rig output. No latest accepted holdout access.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses,load_calibration
from compare_replay import COMMON
from audit_native_lengths import sha,stats
NAMES=('slam_left','slam_right')
PALM_RIG=np.array([5,8,11,14,17]);PALM_MP=np.array([0,5,9,13,17])
RI,MI=np.array(COMMON).T


def transform(points,matrix):return points@matrix[:3,:3].T+matrix[:3,3]


def continue_rigid(camera,seed_head,pixels,age,mode):
    """Only current one-view palm pixels and a compensated old pose enter this fit."""
    eye=camera.world_to_eye(seed_head)
    if not 0<age<=.100000001 or np.any(eye[:,2]<=1):return seed_head.copy(),dict(accepted=False,reason='age_or_depth')
    if pixels.shape!=(5,2) or not np.isfinite(pixels).all():raise ValueError('Five finite palm observations required')
    cap=min(50.,2000*age);center=eye[PALM_RIG].mean(0)
    def candidate(x):
        if mode=='plane':return eye+np.array([x[0],x[1],0.])
        return (eye-center)@Rotation.from_rotvec(x[3:]).as_matrix().T+center+x[:3]
    dimension=2 if mode=='plane' else 6
    bounds=np.full(dimension,cap)
    if mode=='rigid':bounds[3:]=.25
    def residual(x):
        points=candidate(x)[PALM_RIG]
        if np.any(points[:,2]<=1):return np.full(10+dimension,1e6)
        reprojection=(camera.eye_to_window(points)-pixels).reshape(-1)/3.
        prior=x/30. if mode=='plane' else x/np.array([30,30,30,.15,.15,.15])
        return np.r_[reprojection,prior]
    begin=time.perf_counter_ns()
    fitted=least_squares(residual,np.zeros(dimension),bounds=(-bounds,bounds),loss='huber',f_scale=3.,max_nfev=20)
    corrected=candidate(fitted.x)
    displacement=float(np.max(np.linalg.norm(corrected-eye,axis=1)))
    output=transform(corrected,camera.camera_to_world_xf)
    # Do not consume hidden view, next pose, or reference error in admission.
    before=float(np.sqrt(np.mean((camera.eye_to_window(eye[PALM_RIG])-pixels)**2)))
    after=float(np.sqrt(np.mean((camera.eye_to_window(corrected[PALM_RIG])-pixels)**2)))
    accepted=bool(np.isfinite(output).all() and displacement<=50. and after<=before)
    return (output if accepted else seed_head.copy()),dict(accepted=accepted,observed_before_px=before,observed_after_px=after,max_step_mm=displacement,parameters=fitted.x.tolist(),evaluations=int(fitted.nfev),host_us=(time.perf_counter_ns()-begin)/1000)


def census(path):
    audit=read_audit(path);previous={};counts=dict(frames=len(audit),stereo=0,monocular=0,recent_stereo_mono=0,other=0)
    for stamp,row in sorted(audit.items()):
        for source,hand in row['hands'].items():
            views=[v for v in row['views'] if v['id']==source and v['camera'] in NAMES]
            n=len(set(v['camera'] for v in views));side=int(hand['right']>.5)
            if hand['stereo']>=2 and n==2:counts['stereo']+=1;previous[source]=(side,stamp)
            elif hand['stereo']<2 and n==1:
                counts['monocular']+=1;old=previous.get(source)
                counts['recent_stereo_mono']+=bool(old and old[0]==side and stamp-old[1]<=100_000_000)
            else:counts['other']+=1
    return dict(path=str(path),sha256=sha(path),counts=counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('session','native','upstream','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--census',type=Path,action='append',default=[])
    args=parser.parse_args()
    if any('143912Z' in str(path) for path in [args.session,*args.census]):raise ValueError('Latest accepted holdout excluded')
    args.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(args.upstream))
    audit=read_audit(args.session/'frame/live-views.txt');heads=capture_poses(args.session)
    cameras=load_calibration(args.session/'frame/snapshot/device/persist')
    records=[json.loads(line) for line in (args.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
    fitted=np.loadtxt(args.native/'rig-stage-v2/recorded-output.txt')
    if len(records)!=len(fitted) or fitted.shape[1]!=104:raise ValueError('Unmatched native poses')
    ordered=sorted(audit);frame_index={stamp:i for i,stamp in enumerate(ordered)};poses={}
    for row,fit in zip(records,fitted):
        mono=audit[row['raw_ns']]['mono_ns']
        if mono in heads:poses[(row['raw_ns'],row['source_id'])]=dict(side=row['side'],world=transform(fit[41:].reshape(21,3),heads[mono]))
    metrics={mode:[] for mode in ('hold','velocity','plane','rigid')};rows=[];skipped=0
    # Fixed every-tenth-audit-frame sample spans the full recording, both hands,
    # both camera choices and each1/2/3frame artificial outage. No threshold search.
    for frame in range(3,len(ordered),10):
        stamp=ordered[frame];ob=audit[stamp];mono=ob['mono_ns']
        if mono not in heads:continue
        inverse=np.linalg.inv(heads[mono])
        for source,hand in ob['hands'].items():
            target=poses.get((stamp,source));views={v['camera']:v for v in ob['views'] if v['id']==source and v['camera'] in NAMES}
            if target is None or len(views)!=2 or len([h for h in ob['hands'].values() if int(h['right']>.5)==target['side']])!=1:continue
            for horizon in (1,2,3):
                prior_stamp=ordered[frame-horizon];prior=poses.get((prior_stamp,source));age=(stamp-prior_stamp)/1e9
                contiguous=all((ordered[i],source) in poses and poses[(ordered[i],source)]['side']==target['side'] for i in range(frame-horizon,frame+1))
                if prior is None or not contiguous or age>.100000001:skipped+=1;continue
                seed=transform(prior['world'],inverse);reference=transform(target['world'],inverse)
                older=poses.get((ordered[frame-horizon-1],source)) if frame-horizon>0 else None
                velocity=np.zeros(3)
                if older and older['side']==prior['side']:
                    dt=(prior_stamp-ordered[frame-horizon-1])/1e9
                    velocity=(prior['world'][5]-older['world'][5])/dt
                    velocity*=min(1.,2000/max(1e-12,np.linalg.norm(velocity)))
                step=velocity*min(age,.025);step*=min(1.,25/max(1e-12,np.linalg.norm(step)))
                predicted=transform(prior['world']+step,inverse)
                for view,name in enumerate(NAMES):
                    camera=cameras[name];hidden=cameras[NAMES[1-view]];uv=views[name]['points'][PALM_MP]
                    outputs={'hold':(seed,{}),'velocity':(predicted,{})}
                    for mode in ('plane','rigid'):outputs[mode]=continue_rigid(camera,seed,uv,age,mode)
                    case=dict(frame=frame,raw_ns=stamp,prior_raw_ns=prior_stamp,source=source,side=target['side'],view=name,horizon=horizon,age_ms=age*1000,observed_palm_px=uv.tolist(),reference_head_mm=reference.tolist(),methods={})
                    for mode,(points,detail) in outputs.items():
                        withheld=np.linalg.norm(hidden.eye_to_window(hidden.world_to_eye(points[RI]))-views[NAMES[1-view]]['points'][MI],axis=1)
                        metric=dict(hidden_mean_px=float(withheld.mean()),wrist_disagreement_mm=float(np.linalg.norm(points[5]-reference[5])),tip_disagreement_mm=float(np.linalg.norm(points[:5]-reference[:5],axis=1).mean()),**detail)
                        metrics[mode].append(metric);case['methods'][mode]=dict(head_mm=points.tolist(),**metric)
                    rows.append(case)
    with (args.output/'cases.jsonl').open('x') as output:
        for row in rows:output.write(json.dumps(row,allow_nan=False)+'\n')
    summary={}
    for mode,values in metrics.items():
        summary[mode]={key:stats([x[key] for x in values]) for key in ('hidden_mean_px','wrist_disagreement_mm','tip_disagreement_mm')}
        if mode in ('plane','rigid'):
            delta=np.array([x['hidden_mean_px'] for x in values])-np.array([x['hidden_mean_px'] for x in metrics['hold']])
            summary[mode].update(accepted=sum(x['accepted'] for x in values),hidden_better_over_5px=int((delta<-5).sum()),hidden_worse_over_5px=int((delta>5).sum()),hidden_delta_px=stats(delta),host_us=stats([x['host_us'] for x in values if 'host_us' in x]))
    report=dict(scope=__doc__,cases=len(rows),skipped_histories=skipped,sample='every10th audit frame starting at3, both sources/cameras, horizons1/2/3;100ms maximum',methods=summary,census=[census(args.session/'frame/live-views.txt'),*[census(path) for path in args.census]],source_sha256=sha(Path(__file__)),inputs_sha256={str(path):sha(path) for path in (args.session/'frame/live-views.txt',args.native/'rig-stage-v2/recorded-output.txt',args.native/'palm-profile-viewer-v2/poses.jsonl')},limitations=['Artificial one-view loss in otherwise stereo-visible intervals is easier than actual occlusion.','Withheld camera has detector labels, not ground truth; reference3D pose itself uses both cameras.','No actual temporal presenter/visibility policy, neural rerun, native runtime or wearer test.','Fresh HMD compensation is assumed; tracker adapter currently lacks HMD history for native monocular fitting.','Fixed old finger articulation can be wrong when gestures change; no joint visibility inferred.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
