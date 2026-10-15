"""Paired fixed-scale replay audit and frozen independent landmark comparison."""
import argparse,itertools,json,sys,hashlib
from pathlib import Path
import numpy as np
from frame_data import Recording,load_calibration
from live_backend import Backend
from fixed_scale_backend import ScaledBackend
from run_mesh_trial import pose_matrix
from probe_roi_presence import PresenceModel,verify_hand
from probe_wrist_refinement import landmarks,frozen_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','baseline','candidate','upstream','models','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scale',type=float,required=True);p.add_argument('--interval',nargs=2,type=float,action='append',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.upstream));cal=a.session/'calibration';cameras=load_calibration(cal if cal.exists() else a.session/'frame/snapshot/device/persist')
    b=Backend(a.upstream,cameras);c=ScaledBackend(a.upstream,cameras,hand_scale=a.scale);verifier=PresenceModel(a.models);recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0]
    rows=[];frames=hands=0;shift=[];angle_change=[];steps=[];old={}
    with a.baseline.open() as f,a.candidate.open() as g,(a.output/'comparison.jsonl').open('x') as trace:
        for x,y in itertools.zip_longest(f,g):
            if x is None or y is None:raise ValueError('Frame counts differ')
            original,candidate=json.loads(x),json.loads(y)
            if original['capture_raw_ns']!=candidate['capture_raw_ns'] or original['capture_pose']!=candidate['capture_pose']:raise ValueError('Capture context differs')
            stamp=original['capture_raw_ns'];seconds=(stamp-first)/1e9;frames+=1
            if len(original['hands'])!=len(candidate['hands']):raise ValueError('Hand counts differ')
            inspect=any(lo<=seconds<hi for lo,hi in a.interval) and original['capture_pose']['valid']
            if inspect:
                _,saved_stamp,images=next(recording.frames(seconds-.00001,seconds+.00001))
                if saved_stamp!=stamp:raise ValueError('Wrong raw capture')
                world=pose_matrix(original['capture_pose']['capture']);world_cameras={n:v.copy(camera_to_world_xf=world@v.camera_to_world_xf) for n,v in cameras.items()}
            for ha,hb in zip(original['hands'],candidate['hands']):
                for key in ['side','source_id','source_views','memory','right_score']:
                    if ha[key]!=hb[key]:raise ValueError('Identity/input changed: '+key)
                hands+=1;wa=np.asarray(ha['wrist_world_mm'])[:3,3];wb=np.asarray(hb['wrist_world_mm'])[:3,3]
                shift.append(float(np.linalg.norm(wa-wb)));angle_change.append(float(np.max(np.abs(np.asarray(ha['joint_angles'])-hb['joint_angles']))))
                key=(ha['side'],ha['source_id'])
                if key in old:
                    t,pa,pb=old[key]
                    if 0<stamp-t<=100000000:steps.append([float(np.linalg.norm(wa-pa)),float(np.linalg.norm(wb-pb))])
                old[key]=(stamp,wa,wb)
                if inspect:
                    evidence,_=verify_hand(verifier,b.hand,ha,cameras,images,world)
                    comparisons=frozen_comparison(landmarks(b,ha),landmarks(c,hb),world_cameras,evidence)
                    result=dict(stamp=stamp,seconds=seconds,side=ha['side'],source_id=ha['source_id'],frozen= comparisons,baseline_verifier=evidence)
                    rows.append(result);trace.write(json.dumps(result,allow_nan=False)+'\n')
    qualified=[v for r in rows for v in r['frozen'] if v['presence']>=.5]
    def metric(key):
        before=np.array([v['before_'+key] for v in qualified]);after=np.array([v['after_'+key] for v in qualified]);delta=after-before
        return dict(views=len(delta),improved=int(np.sum(delta<0)),worsened=int(np.sum(delta>0)),delta_px_p05_p50_p95=np.percentile(delta,[5,50,95]).tolist() if len(delta) else None)
    report=dict(scope=__doc__+' Model agreement is not ground-truth accuracy. Recorded baseline verification crops are fixed for both poses.',frames=frames,identity_source_memory_equal_hands=hands,fixed_scale=a.scale,wrist_shift_mm_p50_p95_max=np.percentile(shift,[50,95,100]).tolist(),maximum_angle_change_per_hand_rad_p50_p95_max=np.percentile(angle_change,[50,95,100]).tolist(),raw_wrist_steps_baseline_candidate_mm_p50_p95_max=np.percentile(steps,[50,95,100],axis=0).tolist(),verification_hands=len(rows),all20=metric('median_px'),nonpalm15=metric('heldout_median_px'),args={k:str(v) for k,v in vars(a).items()},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['args','scope','source_sha256']},indent=2))

if __name__=='__main__':main()
