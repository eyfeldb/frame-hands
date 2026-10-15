"""Causal native-pose feedback on frozen recorded crops with image controls.

This is not native acquisition: anatomical side and initial/reacquisition crops are fixed by the
previous experiment; optional continuation crops follow earlier native poses. All feedback uses earlier fitted world poses exclusively.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
from contextlib import ExitStack
import cv2,numpy as np
from frame_data import Recording,load_calibration
from run_mesh_trial import pose_matrix
from mercury_probe import MercuryModels
from mercury_keypoints import StereoCrop,estimate
from mercury_prior import EarlierPosePrior
from mercury_native_solver import NativeSolver,camera_transforms
from replay_mercury_solver import reset_reason,to_world
from probe_mercury_solver import project,stats
from probe_mercury_keypoints import draw


def encoded_prior(crop,camera,world_points):
    if world_points is None:return None,'No two-frame prediction'
    try:
        result=crop.normalized_coordinates(camera.world_to_eye(world_points))
        if np.max(np.abs(result))>1:return None,'Predicted point outside fixed crop'
        return result,'Strictly earlier native fits'
    except ValueError as error:return None,str(error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observations','dll','upstream','models','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--crop-mode',choices=('fixed','predicted'),default='fixed')
    a=p.parse_args();sys.path[:0]=[str(a.runtime.resolve()),str(a.upstream.resolve())];a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    original=json.loads(a.observations.read_text());protocol=json.loads((a.observations.parent/'protocol.json').read_text());model=MercuryModels(a.models)
    backgrounds={name:cv2.imread(str(a.observations.parent/(name+'-native-background.png')),cv2.IMREAD_GRAYSCALE) for name in ('slam_left','slam_right')}
    for name,pixels in backgrounds.items():
        if pixels is None or pixels.shape!=(128,128) or hashlib.sha256(pixels.tobytes()).hexdigest()!=original['background_sha256'][name]:raise ValueError('Background control pixels changed')
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for name in ('replay_mercury_feedback.py','mercury_keypoints.py','mercury_prior.py','mercury_native_solver.py','mercury_probe.py','replay_mercury_solver.py','probe_mercury_solver.py','frame_data.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(a.dll.parent/'build.json',a.output/'build.json');reports=[];parity=0
    groups=sorted({r['case']['name'].rsplit('-',1)[0] for r in original['reports']})
    for group in groups:
        frames=[r for r in original['reports'] if r['case']['name'].rsplit('-',1)[0]==group];session=Path(frames[0]['case']['session']);cal=session/'calibration'
        cameras=load_calibration(cal if cal.exists() else session/'frame/snapshot/device/persist');pose,head_from_left=camera_transforms(cameras);rec=Recording(session/'frame/sets.bin');side=frames[0]['case']['side']
        with ExitStack() as stack:
            solvers={name:stack.enter_context(NativeSolver(a.dll,pose,side)) for name in ('off','feedback')};previous={name:None for name in solvers};previous_camera={};history=EarlierPosePrior();writer=None
            for frame in frames:
                case=frame['case'];stamp=frame['stamp'];generation=tuple(frame['generation']);folder=Path(protocol['case_root'])/case['name'];base=json.loads((folder/'baseline.jsonl').read_text());ref=json.loads((folder/'report.json').read_text());valid=bool(base['capture_pose']['valid']);paired={v['camera']:v for v in frame['views'] if 'off' in v}
                if len(paired)!=2 or not valid:
                    history.observe(stamp,generation,None);previous={name:None for name in solvers};reports.append(dict(case=case,stamp=stamp,seconds=frame['seconds'],generation=generation,skipped='Paired stereo unavailable or invalid capture'));continue
                world=pose_matrix(base['capture_pose']['capture']);worldcams={name:cam.copy(camera_to_world_xf=world@cam.camera_to_world_xf) for name,cam in cameras.items()};current_camera=world.copy();current_camera[:3,3]*=.001;current_camera=current_camera@head_from_left
                earlier_stamps=[r[0] for r in history.rows];prior,reason=history.predict(stamp,generation,valid)
                if prior is not None and (len(earlier_stamps)!=2 or max(earlier_stamps)>=stamp):raise AssertionError('Noncausal feedback')
                _,actual,images=next(rec.frames(frame['seconds']-.000001,frame['seconds']+.000001))
                if actual!=stamp:raise ValueError('Capture mismatch')
                rows=[];observations={'off':{},'feedback':{}};start=time.perf_counter()
                for name in ('slam_left','slam_right'):
                    saved=paired[name]['off'];crop=StereoCrop.from_geometry(cameras[name],saved['rotation'],saved['stereographic_radius'],side);image=images[name][0]
                    off,pixels=estimate(model,crop,image)
                    if off!=saved:raise AssertionError('Reconstructed crop/model changed saved OFF result')
                    parity+=1;feedback_crop=crop;crop_info=dict(mode='fixed',reason='Fixed control or fewer than two earlier fits');crop_error=None
                    if a.crop_mode=='predicted' and prior is not None:
                        try:
                            feedback_crop,geometry=StereoCrop.from_predicted_points(cameras[name],worldcams[name].world_to_eye(prior),side)
                            feedback_crop.sample(image) # Validate the entire existing forward-hemisphere remap before inference.
                            crop_info=dict(mode='predicted',**geometry)
                        except ValueError as error:crop_error=str(error);crop_info=dict(mode='rejected',reason=crop_error)
                    if crop_error is None:
                        encoded,encoding_reason=encoded_prior(feedback_crop,worldcams[name],prior);feedback,feedback_pixels=estimate(model,feedback_crop,image,prior=encoded)
                        bg_off,_=estimate(model,feedback_crop,image,pixels_override=backgrounds[name]);bg_feedback,_=estimate(model,feedback_crop,image,prior=encoded,pixels_override=backgrounds[name]);blank,_=estimate(model,feedback_crop,image,prior=encoded,pixels_override=np.zeros((128,128),np.uint8))
                        crop_info.update(rotation=feedback_crop.rotation.tolist(),radius=feedback_crop.radius,pixel_sha256=hashlib.sha256(feedback_pixels.tobytes()).hexdigest())
                    else:
                        encoded=None;encoding_reason='Rejected predicted crop';feedback=dict(active=False,reason=crop_error)
                        bg_off=bg_feedback=blank=dict(active=False,reason='Control not run: rejected predicted crop')
                    observations['off'][name]=off;observations['feedback'][name]=feedback
                    rows.append(dict(camera=name,feedback_crop=crop_info,prior_used=encoded is not None,prior_reason=encoding_reason,prior_normalized=None if encoded is None else encoded.tolist(),off=off,feedback=feedback,background_off=bg_off,background_feedback=bg_feedback,blank=blank,raw_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),crop_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
                arms={}
                for name,views in observations.items():
                    reset=reset_reason(previous[name],stamp,generation,valid)
                    if not all(v['active'] for v in views.values()):
                        previous[name]=None;arms[name]=dict(skipped='Inactive conditioned view');continue
                    if reset is None:solvers[name].transport(np.linalg.inv(current_camera)@previous_camera[name])
                    fit=solvers[name].run(views,reset=reset is not None,smoothing=2);previous_camera[name]=current_camera
                    accepted=fit['native_active'] and fit['passes_native_reprojection_gate'];previous[name]=(stamp,generation) if accepted else None
                    points=to_world(fit,head_from_left,world);uv=project(fit,head_from_left,cameras);compare={}
                    for camera,projected in uv.items():
                        ncnn=np.asarray(next(v['model_points'] for v in ref['before_verifier'] if v['camera']==camera));compare[camera]=stats(np.linalg.norm(projected-ncnn,axis=1))
                    arms[name]=dict(fit=fit,accepted=accepted,world_joints_mm=points.tolist(),projected={camera:points.tolist() for camera,points in uv.items()},ncnn_disagreement_px=compare,reset_reason=reset)
                # Only the completed current feedback fit enters the next frame's history.
                fitted=arms['feedback'];history.observe(stamp,generation,np.asarray(fitted['world_joints_mm']) if fitted.get('accepted') else None)
                reports.append(dict(case=case,stamp=stamp,seconds=frame['seconds'],generation=generation,prior_source_stamps=earlier_stamps,prior_reason=reason,prior_world_mm=None if prior is None else prior.tolist(),views=rows,arms=arms,experiment_ms=(time.perf_counter()-start)*1000))
                panels=[]
                for view in rows:
                    name=view['camera'];image=images[name][0];ncnn=np.asarray(next(v['model_points'] for v in ref['before_verifier'] if v['camera']==name));roi=np.vstack((view['off']['points_raw'],ncnn));lo=np.floor(roi.min(0)-35).astype(int);hi=np.ceil(roi.max(0)+35).astype(int);x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[image.shape[1],image.shape[0]])
                    columns=[cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),(draw(image,view['feedback']['points_raw'],(0,255,0)) if 'points_raw' in view['feedback'] else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))]
                    for branch,color in [('off',(0,140,255)),('feedback',(255,255,0))]:
                        result=arms[branch];columns.append(draw(image,result['projected'][name],color) if 'projected' in result else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
                    panel=cv2.hconcat([cv2.resize(c[y0:y1,x0:x1],(320,320)) for c in columns]);cv2.putText(panel,name+' '+str(round(frame['seconds'],3)),(6,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
                combined=cv2.vconcat(panels)
                if writer is None:
                    writer=cv2.VideoWriter(str(a.output/(group+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,640))
                    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
                    stack.callback(writer.release)
                writer.write(combined)
                if int(case['name'].rsplit('-',1)[1]) in (0,5,10,15,20,25,28,29):cv2.imwrite(str(a.output/(case['name']+'.jpg')),combined)
        print(group,'complete',flush=True)
    summary=dict(crop_mode=a.crop_mode,scope='Fixed sides and seed/reacquisition crops; optional native predicted continuation crop (expansion1.7,palm floor);  native state with measured-camera transport and smoothing2; world-space gain0.4prior from its own two strictly earlier accepted fits. Paired stereo only; presence>=.97per view and native residual<=15 admit fits. Background controls borrow the real trajectory prior for one counterfactual frame and do not update it; blank controls stop before inference.',legend='Rows:left/right camera; columns:raw/green conditioned keypoints/orange native without learned prior/cyan native with own-pose feedback. Videos omit skipped frames at nominal30fps; read capture timestamps.',runtime=model.version,dll_sha256=hashlib.sha256(a.dll.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(a.observations.read_bytes()).hexdigest(),background_sha256=original['background_sha256'],exact_off_camera_results=parity,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},reports=reports)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print('Exact unconditioned camera results:',parity)

if __name__=='__main__':main()
