"""Declared calibration prefix and frozen-size evaluation on an existing recording."""
import argparse,hashlib,json,shutil,sys,time
from contextlib import ExitStack
from pathlib import Path
import cv2,numpy as np,torch
from frame_data import Recording,load_calibration,read_audit
from run_mesh_trial import pose_matrix
from replay_umetrack import COMMON
from probe_roi_presence import crop_matrix
from mercury_keypoints import StereoCrop,estimate
from mercury_probe import MercuryModels
from mercury_native_solver import NativeSolver,camera_transforms
from mercury_size import NativeSizeRefinement
from replay_mercury_solver import reset_reason,to_world
from probe_mercury_solver import project,stats
from probe_mercury_keypoints import draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','dll','upstream','models','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.runtime.resolve()),str(a.upstream.resolve())];torch.set_num_threads(1);cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=False)
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    protocol=json.loads(a.protocol.read_text());baseline_path=Path(protocol['baseline'])
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=protocol['baseline_sha256']:raise ValueError('Declared source changed')
    base={}
    with baseline_path.open() as f:
        for line in f:
            row=json.loads(line);base[row['capture_raw_ns']]={k:row[k] for k in ('capture_pose','hands')}
            for hand in base[row['capture_raw_ns']]['hands']:hand.pop('vertices_world_mm',None)
    session=Path(protocol['session']);rec=Recording(session/'frame/sets.bin');cameras=load_calibration(session/'calibration');pose,head_from_left=camera_transforms(cameras);audit=read_audit(session/'frame/live-views.txt')
    raw=json.loads((a.upstream/'dataset/generic_hand_model.json').read_text());hand_model=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in raw.items()});model=MercuryModels(a.models)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_size_long.py','mercury_keypoints.py','mercury_native_solver.py','mercury_size.py','mercury_probe.py','frame_data.py','probe_roi_presence.py','replay_mercury_solver.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.protocol,a.output/'protocol.json');shutil.copy2(a.dll.parent/'build.json',a.output/'build.json')
    frame_count=0;fit_count=0;frozen=None;last_scale=dict(target_size=.09,schedule_x=0);last_review=-1;counts=dict(missing_baseline=0,invalid_capture=0,model_views=0,active_model_views=0,geometry_rejections=0)
    with ExitStack() as stack:
        output=stack.enter_context((a.output/'frames.jsonl').open('x'));solvers={(branch,side):stack.enter_context(NativeSolver(a.dll,pose,side)) for branch in ('fixed','calibrated') for side in (0,1)};scale=stack.enter_context(NativeSizeRefinement(a.dll));previous={key:None for key in solvers};old_camera={};seen=np.zeros(2,bool);writer=None
        for seconds,stamp,images in rec.frames(protocol['calibration_start_seconds'],protocol['evaluation_end_seconds']):
            start=time.perf_counter();phase='calibration' if seconds<protocol['freeze_seconds'] else 'evaluation'
            if phase=='evaluation' and frozen is None:
                frozen=dict(size_m=last_scale['target_size'],schedule_x=last_scale['schedule_x'],native_complete=last_scale['schedule_x']>100,freeze_before_stamp=stamp,freeze_seconds=seconds)
                print('FROZEN',json.dumps(frozen),flush=True)
            reference=base.get(stamp);valid=reference is not None and reference['capture_pose']['valid'];views_by_side={};side_rows=[];found=np.zeros((2,2),bool);world=pose_matrix(reference['capture_pose']['capture']) if valid else None
            if reference is None:counts['missing_baseline']+=1
            elif not valid:counts['invalid_capture']+=1
            if valid:
                worldcams={name:cam.copy(camera_to_world_xf=world@cam.camera_to_world_xf) for name,cam in cameras.items()};current=world.copy();current[:3,3]*=.001;current=current@head_from_left
                for hand in reference['hands']:
                    side=hand['side'];wrist=torch.tensor(hand['wrist_world_mm'],dtype=torch.float32)
                    if side:wrist[:,0]*=-1
                    with torch.inference_mode():xyz=skin_landmarks(hand_model,torch.tensor(hand['joint_angles']),wrist).numpy()
                    views={};details=[]
                    for index,name in enumerate(('slam_left','slam_right')):
                        cam=worldcams[name];uv=cam.eye_to_window(cam.world_to_eye(xyz));points=np.full((21,2),np.nan)
                        for native,mp in COMMON:points[mp]=uv[native]
                        try:
                            matrix,size=crop_matrix(points);center=np.array([112.,112.,1.])@matrix.T;crop=StereoCrop(cameras[name],center,size,side);crop.sample(images[name][0])
                        except ValueError as error:
                            counts['geometry_rejections']+=1;details.append(dict(camera=name,skipped=str(error)));continue
                        result,pixels=estimate(model,crop,images[name][0]);views[name]=result;found[index,side]=result['active'];counts['model_views']+=1;counts['active_model_views']+=int(result['active'])
                        source_id=next((v['source_id'] for v in hand['source_views'] if v['camera']==name),None);ncnn=next((v['points'].tolist() for v in audit.get(stamp,{}).get('views',[]) if v['camera']==name and v['id']==source_id),None)
                        details.append(dict(camera=name,model=result,raw_sha256=hashlib.sha256(images[name][0].tobytes()).hexdigest(),crop_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),ncnn_points=ncnn))
                    views_by_side[side]=views;side_rows.append(dict(side=side,source_id=hand['source_id'],views=details))
            pre=scale.pre(found,seen) if phase=='calibration' else dict(target_size=frozen['size_m'],size_error=1.,optimize_size=False,schedule_x=frozen['schedule_x']);fits=[]
            present_sides=set(views_by_side)
            for side in (0,1):
                views=views_by_side.get(side,{})
                if len(views)!=2 or not all(v['active'] for v in views.values()):
                    for branch in ('fixed','calibrated'):previous[(branch,side)]=None
                    continue
                side_row=next(r for r in side_rows if r['side']==side);generation=(side,side_row['source_id'])
                for branch in ('fixed','calibrated'):
                    key=(branch,side);reason=reset_reason(previous[key],stamp,generation,valid)
                    if reason is None:solvers[key].transport(np.linalg.inv(current)@old_camera[key])
                    options={k:pre[k] for k in ('target_size','size_error','optimize_size')} if branch=='calibrated' else {}
                    fit=solvers[key].run(views,reset=reason is not None,smoothing=2,**options);accepted=fit['native_active'] and fit['passes_native_reprojection_gate'];previous[key]=(stamp,generation) if accepted else None;old_camera[key]=current
                    projected=project(fit,head_from_left,cameras);disagreement={}
                    for name,uv in projected.items():
                        ncnn=next(v.get('ncnn_points') for v in side_row['views'] if v['camera']==name)
                        if ncnn is not None:disagreement[name]=stats(np.linalg.norm(uv-np.asarray(ncnn),axis=1))
                    fits.append(dict(branch=branch,side=side,generation=generation,accepted=accepted,fit=fit,world_joints_mm=to_world(fit,head_from_left,world).tolist(),projected={name:v.tolist() for name,v in projected.items()},ncnn_disagreement_px=disagreement,reset_reason=reason));fit_count+=1
                    if branch=='calibrated' and accepted:
                        if phase=='calibration':scale.post(fit['hand_size_m'],fit['reprojection_error'],[views[name]['confidence_xy'] for name in ('slam_left','slam_right')])
                        seen[side]=True
            if phase=='calibration':last_scale=scale.finish()
            row=dict(seconds=seconds,stamp=stamp,phase=phase,valid_capture=bool(valid),scale_pre=pre,scale_post=last_scale,observations=side_rows,fits=fits,experiment_ms=(time.perf_counter()-start)*1000);output.write(json.dumps(row,allow_nan=False)+'\n');frame_count+=1
            if int(seconds)!=last_review:
                panels=[]
                for name in ('slam_left','slam_right'):
                    image=images[name][0];columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)]
                    for branch in ('fixed','calibrated'):
                        canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                        for fit in fits:
                            if fit['branch']==branch:canvas=draw(canvas,fit['projected'][name],(255,255,0) if fit['side']==0 else (0,140,255))
                        columns.append(canvas)
                    panel=cv2.hconcat([cv2.resize(c,(512,round(512*image.shape[0]/image.shape[1]))) for c in columns]);cv2.putText(panel,name+' '+str(round(seconds,3))+' '+phase+' target='+str(round(pre['target_size']*1000,1)),(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
                cv2.imwrite(str(a.output/('review-'+str(int(seconds)).zfill(2)+'.jpg')),cv2.vconcat(panels));last_review=int(seconds)
            if frame_count%150==0:output.flush();print('frames',frame_count,'seconds',round(seconds,2),'progress',round(last_scale['schedule_x'],2),flush=True)
    summary=dict(protocol=protocol,frames=frame_count,fits=fit_count,counts=counts,frozen=frozen,last_scale=last_scale,runtime=model.version,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},legend='Review columns:raw/fixed90mm/refined-or-frozen. Leftcyan,rightorange. No true pose annotations.',limitations='Stateless Mercury on current external pose-derived crops. Calibration shared across available stereo hands; monocular active view inhibits refinement. No calibrated hand dimensions or3Dground truth; no live or acquisition claim.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
