"""Known native finger trajectories: response and pose error with paired inputs."""
import argparse,hashlib,json,shutil
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import numpy as np
from mercury_forward import NativeHandGenerator
from mercury_native_solver import NativeSolver,ML_TO_XR
from replay_mercury_crop_fallback import constant_finger_stability
from test_mercury_native_solver import POSE,observations_from_joints

OPEN=np.array([-.1,-.05,-.05]);CLOSED=np.array([-1.2,-1.2,-.8])


def trajectory(index,duration):
    closing=np.clip((index-59)/duration,0,1);opening=np.clip((index-109)/duration,0,1)
    fraction=float(closing-opening)
    return np.tile(OPEN+(CLOSED-OPEN)*fraction,(4,1)),fraction


def stats(values):
    if not values:return None
    return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))


def response(frames,duration):
    result={}
    for event,start,end,rising in [('close',60,110,True),('open',110,180,False)]:
        truth=next(i for i in range(start,end) if (frames[i]['truth_fraction']>=.5 if rising else frames[i]['truth_fraction']<=.5))
        crossing=next((i for i in range(start,end) if (frames[i]['estimated_fraction']>=.5 if rising else frames[i]['estimated_fraction']<=.5)),None)
        result[event]=dict(truth_frame=truth,estimated_frame=crossing,delay_ms=None if crossing is None else (crossing-truth)/30*1000)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(frames=180,fps=30,closing_start_frame=60,opening_start_frame=110,ramp_durations_frames=[6,30],settling_frames=60,size_m=.09,side=[0,1],conditions=['clean','gaussian'],noise=dict(seed=20261005,xy_crop_sigma_pixels=.5,curl_sigma_rad=.4,reported_curl_variance=.16),clean_curl_variance=.01,wrist_internal_xyzw_xyz=[.70710678,0,0,.70710678,.04,-.03,-.5],observations='Exact analytic native21joint stereo projections at6cm parallel baseline, crop stereographic radius0.4; no relative-depth observations. Curls derive from known commanded angles. Gaussian condition adds independent per-view current2D and curl noise once before sharing packets across arms.',arms=['native','constant'],metric='Absolute26->21joint error and native finger-angle error against generator truth;50percent closure/opening threshold response at30Hz. No calibrated camera/ML or physical-human claim. Initial bias reported; threshold delay alone is unreliable if poses are wrong.',selection='All conditions and trajectory/noise parameters fixed before generation; no tuning to outcomes.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('probe_mercury_gesture_response.py','mercury_forward.py','mercury_native_solver.py','replay_mercury_crop_fallback.py','test_mercury_native_solver.py'):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');gen=NativeHandGenerator(a.dll);reports=[]
    for side in (0,1):
        for duration in (6,30):
            for condition in ('clean','gaussian'):
                rng=np.random.default_rng(20261005+100*side+duration);frames={name:[] for name in ('native','constant')};case=f'side{side}-ramp{duration}-{condition}'
                with ExitStack() as stack:
                    solvers={name:stack.enter_context(NativeSolver(a.dll,POSE,side)) for name in frames};output=stack.enter_context((a.output/(case+'.jsonl')).open('x'))
                    for index in range(180):
                        angles,fraction=trajectory(index,duration);truth=gen.generate(protocol['wrist_internal_xyzw_xyz'],angles,side)[ML_TO_XR,:3];views=observations_from_joints(truth,side)
                        for view in views.values():
                            view['curls'][1:5]=angles.sum(1).tolist();view['curls'][5:]=[0.]*5
                            if condition=='gaussian':
                                view['points_crop']=(np.asarray(view['points_crop'])+rng.normal(0,.5,(21,2))).tolist()
                                view['curls'][1:5]=(angles.sum(1)+rng.normal(0,.4,4)).tolist();view['curls'][6:10]=[.15]*4
                        row=dict(index=index,seconds=index/30,truth_fraction=fraction,truth_angles=angles.tolist(),truth_joints_m=truth.tolist(),views=views,arms={})
                        for name,solver in solvers.items():
                            supplied=constant_finger_stability(solver,views) if name=='constant' and index>0 else views
                            fit=solver.run(supplied,reset=index==0,smoothing=2,depth_weight=0);estimated=np.asarray(fit['joints_left_camera_m'])[ML_TO_XR];fingers=solver.fingers();progress=float((OPEN.sum()-fingers.sum(1).mean())/(OPEN.sum()-CLOSED.sum()))
                            record=dict(index=index,truth_fraction=fraction,estimated_fraction=progress,max_joint_error_mm=float(np.linalg.norm(estimated-truth,axis=1).max()*1000),mean_joint_error_mm=float(np.linalg.norm(estimated-truth,axis=1).mean()*1000),max_angle_error_rad=float(abs(fingers-angles).max()),angles=fingers.tolist(),joints_m=estimated.tolist(),reprojection_error=fit['reprojection_error'],accepted=fit['native_active'] and fit['passes_native_reprojection_gate'])
                            frames[name].append(record);row['arms'][name]=record
                        output.write(json.dumps(row,allow_nan=False)+'\n')
                arms={}
                for name,sequence in frames.items():
                    moving=[r for r in sequence if 60<=r['index']<60+duration or 110<=r['index']<110+duration];settled=sequence[40:60]+sequence[150:180]
                    arms[name]=dict(accepted=sum(r['accepted'] for r in sequence),initial_settled_max_joint_error_mm=stats([r['max_joint_error_mm'] for r in sequence[40:60]]),moving_max_joint_error_mm=stats([r['max_joint_error_mm'] for r in moving]),moving_max_angle_error_rad=stats([r['max_angle_error_rad'] for r in moving]),settled_max_joint_error_mm=stats([r['max_joint_error_mm'] for r in settled]),response=response(sequence,duration))
                report=dict(case=case,side=side,duration_frames=duration,condition=condition,arms=arms);reports.append(report);print(case,json.dumps({k:dict(initial=v['initial_settled_max_joint_error_mm']['p95'],moving=v['moving_max_joint_error_mm']['p95'],response=v['response']) for k,v in arms.items()}),flush=True)
    result=dict(protocol=protocol,reports=reports,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')});(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
