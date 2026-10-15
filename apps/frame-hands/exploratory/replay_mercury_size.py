"""Isolate native size refinement using frozen stereo model observations."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from contextlib import ExitStack
import cv2,numpy as np
from frame_data import Recording,load_calibration
from run_mesh_trial import pose_matrix
from mercury_native_solver import NativeSolver,camera_transforms
from mercury_size import NativeSizeRefinement
from replay_mercury_solver import reset_reason,to_world
from probe_mercury_solver import project,stats
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observations','protocol','dll','upstream','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream.resolve()));a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    source=json.loads(a.observations.read_text());protocol=json.loads(a.protocol.read_text());snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_size.py','mercury_size.py','mercury_native_solver.py','replay_mercury_solver.py','probe_mercury_solver.py','frame_data.py'):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');reports=[];parity=0
    for group in sorted({r['case']['name'].rsplit('-',1)[0] for r in source['reports']}):
        frames=[r for r in source['reports'] if r['case']['name'].rsplit('-',1)[0]==group];session=Path(frames[0]['case']['session']);cal=session/'calibration';side=frames[0]['case']['side']
        cameras=load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist');pose,head_from_left=camera_transforms(cameras);rec=Recording(session/'frame/sets.bin')
        with ExitStack() as stack:
            solvers={name:stack.enter_context(NativeSolver(a.dll,pose,side)) for name in ('fixed','native_size')};scale=stack.enter_context(NativeSizeRefinement(a.dll));previous={name:None for name in solvers};old_camera={};seen=np.zeros(2,bool);writer=None
            for row in frames:
                case=row['case'];stamp=row['stamp'];generation=tuple(row['generation'])
                if 'skipped' in row:
                    pre=scale.pre(np.zeros((2,2),bool),seen);post=scale.finish();previous={name:None for name in solvers};reports.append(dict(case=case,stamp=stamp,seconds=row['seconds'],generation=generation,skipped=row['skipped'],scale_pre=pre,scale_post=post));continue
                folder=Path(protocol['case_root'])/case['name'];base=json.loads((folder/'baseline.jsonl').read_text());ref=json.loads((folder/'report.json').read_text());world=pose_matrix(base['capture_pose']['capture']);current=world.copy();current[:3,3]*=.001;current=current@head_from_left
                views={v['camera']:v['feedback'] for v in row['views']};found=np.zeros((2,2),bool)
                for index,name in enumerate(('slam_left','slam_right')):found[index,side]=views[name]['active']
                pre=scale.pre(found,seen);arms={}
                for branch,solver in solvers.items():
                    if not all(v['active'] for v in views.values()):previous[branch]=None;arms[branch]=dict(skipped='Inactive stereo model view');continue
                    reason=reset_reason(previous[branch],stamp,generation,base['capture_pose']['valid'])
                    if reason is None:solver.transport(np.linalg.inv(current)@old_camera[branch])
                    options={key:pre[key] for key in ('optimize_size','target_size','size_error')} if branch=='native_size' else {}
                    fit=solver.run(views,reset=reason is not None,smoothing=2,**options);accepted=fit['native_active'] and fit['passes_native_reprojection_gate'];previous[branch]=(stamp,generation) if accepted else None;old_camera[branch]=current
                    uv=project(fit,head_from_left,cameras);disagreement={name:stats(np.linalg.norm(points-np.asarray(next(v['model_points'] for v in ref['before_verifier'] if v['camera']==name)),axis=1)) for name,points in uv.items()}
                    arms[branch]=dict(fit=fit,accepted=accepted,world_joints_mm=to_world(fit,head_from_left,world).tolist(),projected={name:points.tolist() for name,points in uv.items()},ncnn_disagreement_px=disagreement,reset_reason=reason)
                    if branch=='fixed':
                        if fit!=row['arms']['feedback']['fit']:raise AssertionError('Frozen90mm native baseline changed')
                        parity+=1
                    elif accepted:
                        scale.post(fit['hand_size_m'],fit['reprojection_error'],[views[name]['confidence_xy'] for name in ('slam_left','slam_right')]);seen[side]=True
                post=scale.finish();reports.append(dict(case=case,stamp=stamp,seconds=row['seconds'],generation=generation,scale_pre=pre,scale_post=post,arms=arms))
                _,actual,images=next(rec.frames(row['seconds']-.000001,row['seconds']+.000001))
                if stamp!=actual:raise ValueError('Capture mismatch')
                panels=[]
                for name in ('slam_left','slam_right'):
                    image=images[name][0];ncnn=np.asarray(next(v['model_points'] for v in ref['before_verifier'] if v['camera']==name));roi=np.vstack((views[name]['points_raw'],ncnn));lo=np.floor(roi.min(0)-35).astype(int);hi=np.ceil(roi.max(0)+35).astype(int);x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[image.shape[1],image.shape[0]])
                    columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),draw(image,views[name]['points_raw'],(0,255,0))]
                    for branch,color in [('fixed',(0,140,255)),('native_size',(255,255,0))]:columns.append(draw(image,arms[branch]['projected'][name],color) if 'projected' in arms[branch] else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
                    panel=cv2.hconcat([cv2.resize(c[y0:y1,x0:x1],(320,320)) for c in columns]);cv2.putText(panel,name+' '+str(round(row['seconds'],3))+' size='+str(round(post['target_size']*1000,1))+'mm',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
                combined=cv2.vconcat(panels)
                if writer is None:
                    writer=cv2.VideoWriter(str(a.output/(group+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,640));stack.callback(writer.release)
                    if not writer.isOpened():raise RuntimeError('No video writer')
                writer.write(combined)
                if int(case['name'].rsplit('-',1)[1]) in (0,5,10,15,20,25,28,29):cv2.imwrite(str(a.output/(case['name']+'.jpg')),combined)
        print(group,'complete',post,flush=True)
    result=dict(scope='Native size refinement on frozen predicted-crop model outputs; single selected hand per independent one-second window, other hand reported absent. Stereo-only. Calibration persists through missing frames, solver history resets. No learned/crop feedback from calibrated size, no full-recording calibration claim.',legend='raw / green frozen observations / orange90mm / cyan native size-refined; videos omit unavailable frames at nominal30fps with capture timestamps.',input_sha256=hashlib.sha256(a.observations.read_bytes()).hexdigest(),dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),exact_fixed_baseline_fits=parity,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},reports=reports)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
