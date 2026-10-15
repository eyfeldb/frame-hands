"""Offline factorial ablation of native optimizer history and saved model conditioning.

Uses fixed observations from the previous paired keypoint experiment. It does not
claim native end-to-end feedback, head-motion compensated state, or ground truth.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
from contextlib import ExitStack
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from frame_data import Recording, load_calibration
from run_mesh_trial import pose_matrix
from mercury_native_solver import NativeSolver, camera_transforms, ML_TO_XR
from probe_mercury_solver import project, stats
from probe_mercury_keypoints import draw

ARMS={f'{source}-{mode}':(source,mode) for source in ('off','on') for mode in ('cold','warm','smooth','compensated')}


def reset_reason(previous,stamp,generation,valid=True):
    if not valid:return 'Invalid capture pose'
    if previous is None:return 'No accepted previous fit'
    old_stamp,old_generation=previous
    if stamp<=old_stamp:return 'Nonincreasing capture'
    if stamp-old_stamp>100000000:return 'Capture gap over100ms'
    if generation!=old_generation:return 'Generation changed'
    return None


def to_world(result,head_from_left,world_from_head):
    points=np.asarray(result['joints_left_camera_m'])[ML_TO_XR]
    head=(points@head_from_left[:3,:3].T+head_from_left[:3,3])*1000
    return head@world_from_head[:3,:3].T+world_from_head[:3,3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observations','dll','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--constant-stability',action='store_true',help='Add diagnostic constant finger-weight branches')
    a=p.parse_args();arms=dict(ARMS)
    if a.constant_stability:
        from replay_mercury_crop_fallback import constant_finger_stability
        arms.update({source+'-constant':(source,'constant') for source in ('off','on')})
    sys.path.insert(0,str(a.upstream.resolve()));a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(a.observations.read_text());protocol=json.loads((a.observations.parent/'protocol.json').read_text())
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_solver.py','mercury_native_solver.py','probe_mercury_solver.py','frame_data.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    if a.constant_stability:shutil.copy2(Path(__file__).with_name('replay_mercury_crop_fallback.py'),snapshot/'replay_mercury_crop_fallback.py')
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json')
    reports=[];groups=sorted({r['case']['name'].rsplit('-',1)[0] for r in source['reports']});cv2.setNumThreads(1)
    for group in groups:
        frames=[r for r in source['reports'] if r['case']['name'].rsplit('-',1)[0]==group];session=Path(frames[0]['case']['session']);cal=session/'calibration'
        cameras=load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist');pose,head_from_left=camera_transforms(cameras);rec=Recording(session/'frame/sets.bin')
        with ExitStack() as stack:
            solvers={name:stack.enter_context(NativeSolver(a.dll,pose,frames[0]['case']['side'])) for name in arms};previous={name:None for name in arms};previous_camera={name:None for name in arms};old_head=None;writers={}
            for frame in frames:
                case=frame['case'];stamp=frame['stamp'];folder=Path(protocol['case_root'])/case['name'];base=json.loads((folder/'baseline.jsonl').read_text());reference=json.loads((folder/'report.json').read_text());generation=tuple(frame['generation'])
                paired={v['camera']:v for v in frame['views'] if 'off' in v and 'on' in v};valid=bool(base['capture_pose']['valid']);world=pose_matrix(base['capture_pose']['capture']) if valid else None
                head_motion=None
                if world is not None and old_head is not None:
                    head_motion=dict(translation_mm=float(np.linalg.norm(world[:3,3]-old_head[:3,3])),rotation_deg=float(np.rad2deg(Rotation.from_matrix(old_head[:3,:3].T@world[:3,:3]).magnitude())))
                old_head=world
                out=dict(case=case,stamp=stamp,seconds=frame['seconds'],generation=generation,head_step=head_motion,arms={})
                if len(paired)!=2 or not valid:
                    out['skipped']='Paired stereo unavailable or capture pose invalid';previous={name:None for name in arms};reports.append(out);continue
                for name,(branch,mode) in arms.items():
                    views={camera:v[branch] for camera,v in paired.items()};reason=reset_reason(previous[name],stamp,generation,valid)
                    if not all(view['active'] for view in views.values()):
                        previous[name]=None;out['arms'][name]=dict(skipped='Inactive model view');continue
                    current_camera=world.copy();current_camera[:3,3]*=.001;current_camera=current_camera@head_from_left
                    if mode in ('compensated','constant') and reason is None:solvers[name].transport(np.linalg.inv(current_camera)@previous_camera[name])
                    supplied=constant_finger_stability(solvers[name],views) if mode=='constant' and reason is None else views
                    start=time.perf_counter();fit=solvers[name].run(supplied,reset=mode=='cold' or reason is not None,smoothing=2 if mode in ('smooth','compensated','constant') else 0);elapsed=(time.perf_counter()-start)*1000
                    previous_camera[name]=current_camera
                    accepted=fit['native_active'] and fit['passes_native_reprojection_gate'];previous[name]=(stamp,generation) if accepted else None
                    projected=project(fit,head_from_left,cameras);compare={}
                    for camera,uv in projected.items():
                        ncnn=np.asarray(next(r['model_points'] for r in reference['before_verifier'] if r['camera']==camera))
                        compare[camera]=dict(ncnn_disagreement_px=stats(np.linalg.norm(uv-ncnn,axis=1)),input_disagreement_px=stats(np.linalg.norm(uv-np.asarray(views[camera]['points_raw']),axis=1)))
                    out['arms'][name]=dict(fit=fit,finger_angles=solvers[name].fingers().tolist() if a.constant_stability else None,world_joints_mm=to_world(fit,head_from_left,world).tolist(),projected={k:v.tolist() for k,v in projected.items()},compare=compare,reset=mode=='cold' or reason is not None,reset_reason='Independent cold fit' if mode=='cold' else reason,solver_ms=elapsed)
                _,actual,images=next(rec.frames(frame['seconds']-.000001,frame['seconds']+.000001))
                if actual!=stamp:raise ValueError('Raw capture timestamp mismatch')
                for branch in ('off','on'):
                    panels=[]
                    for camera in ('slam_left','slam_right'):
                        image=images[camera][0];raw_points=np.asarray(paired[camera][branch]['points_raw']);ref=np.asarray(next(r['model_points'] for r in reference['before_verifier'] if r['camera']==camera))
                        roi=np.vstack((paired[camera]['off']['points_raw'],ref));lo=np.floor(roi.min(0)-35).astype(int);hi=np.ceil(roi.max(0)+35).astype(int);x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[image.shape[1],image.shape[0]])
                        columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),draw(image,raw_points,(0,255,0))]
                        for mode,color in ([('cold',(0,140,255)),('warm',(255,0,255)),('smooth',(255,255,0)),('compensated',(255,140,0))]+([('constant',(80,255,80))] if a.constant_stability else [])):
                            arm=out['arms'][branch+'-'+mode];columns.append(draw(image,arm['projected'][camera],color) if 'projected' in arm else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
                        panel=cv2.hconcat([cv2.resize(c[y0:y1,x0:x1],(256,256)) for c in columns]);cv2.putText(panel,camera+' '+str(round(frame['seconds'],3))+' '+branch,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
                    combined=cv2.vconcat(panels)
                    if branch not in writers:
                        writer=cv2.VideoWriter(str(a.output/(group+'-'+branch+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(combined.shape[1],combined.shape[0]))
                        if not writer.isOpened():raise RuntimeError('Video writer unavailable')
                        stack.callback(writer.release);writers[branch]=writer
                    writers[branch].write(combined)
                    if int(case['name'].rsplit('-',1)[1]) in (0,10,20,28,29):cv2.imwrite(str(a.output/(case['name']+'-'+branch+'.jpg')),combined)
                reports.append(out)
        print(group,'complete',flush=True)
    summary=dict(constant_stability_ablation=a.constant_stability,scope='Frozen89frames;8original arms plus optional2constant-finger-stability ablations; saved external model priorOFF/ON x native cold/warm/no-smoothing/smooth/head-compensated. Fixed90mm; native default smoothing2. Paired stereo only. Head-compensated arms re-express history using exact previous/current capture poses; other arms retain native left-camera-relative state. Output metrics use recorded world poses.',legend='Columns:raw,green model,orange cold native,magenta warm native with smoothing0,cyan warm native with smoothing2, blue head-compensated smoothing2, optional final green constant finger stability. Videos play supported frames at nominal30fps; skipped frames are not represented, so consult timestamps and summary.',observations_sha256=hashlib.sha256(a.observations.read_bytes()).hexdigest(),dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},reports=reports)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
