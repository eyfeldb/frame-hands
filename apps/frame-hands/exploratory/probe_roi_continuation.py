"""Bounded offline closed-loop ROI hypotheses, never accepted tracking output.

Seed only from a prior saved prediction. Later detector poses/labels never repair
this probe. Blank-pixel controls expose hallucination; sigma remains uncalibrated.
"""
import argparse,copy,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
from frame_data import Recording,load_calibration,warp,NAMES
from live_backend import Backend
from feature_reuse import PreparedRegressor
from run_mesh_trial import pose_matrix


def read_reference(rows,start,duration):
    """Keep one causal seed and only head/capture metadata after it."""
    metadata={};seed=None;previous_stamp=None;previous_seconds=None
    for row in rows:
        seconds=float(row['seconds']);stamp=int(row['capture_raw_ns'])
        if not np.isfinite(seconds) or (previous_stamp is not None and (stamp<=previous_stamp or seconds<=previous_seconds)):
            raise ValueError('Reference captures must be finite and strictly ordered')
        previous_stamp=stamp;previous_seconds=seconds
        if seconds>start+duration:break
        if seconds<=start:seed=row
        if start<=seconds<=start+duration:
            metadata[stamp]={key:row[key] for key in ('capture_pose','current_pose','capture_mono_ns')}
    if seed is None or not seed['hands'] or not seed['capture_pose']['valid'] or not 0<=start-seed['seconds']<=.100:
        raise ValueError('No fresh supported seed at start')
    sides=[h['side'] for h in seed['hands']]
    if len(sides)!=len(set(sides)) or any(side not in (0,1) for side in sides):
        raise ValueError('Seed sides must be unique anatomical slots')
    return seed,metadata


def pose_crop(backend,hand,cam,world,prepared_points=None):
    from lib.tracker.perspective_crop import _get_crop_points_from_hand_pose,landmarks_from_hand_pose
    from lib.tracker.tracking_result import SingleHandPose
    from lib.common.crop import gen_crop_parameters_from_points
    from lib.common.affine import make_look_at_matrix
    if prepared_points is None:
        prior=SingleHandPose(np.array(hand['joint_angles'],np.float32),np.array(hand['wrist_world_mm'],np.float32),hand_confidence=0.)
        with torch.inference_mode():
            points=_get_crop_points_from_hand_pose(backend.hand,prior,hand['side'],63)
            landmarks=landmarks_from_hand_pose(backend.hand,prior,hand['side'])
    else:
        points=np.asarray(prepared_points)
        if points.shape!=(63,3) or not np.isfinite(points).all():raise ValueError('Invalid prepared pose geometry')
        # Pinned upstream order is current, neutral, open: current is first 21.
        landmarks=points[:21]
    eye=cam.world_to_eye(landmarks);uv=cam.eye_to_window(eye)
    projected=(eye[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
    if projected.sum()<19:return None
    center=(points.min(0)+points.max(0))*.5
    unrotated=np.linalg.inv(make_look_at_matrix(np.linalg.inv(cam.camera_to_world_xf),center,0))
    down=world[:3,:3]@np.array([0.,-1.,0.])
    angle=np.degrees(np.arctan2(-np.dot(down,unrotated[:3,0]),np.dot(down,unrotated[:3,1])))
    return gen_crop_parameters_from_points(cam,points,(96,96),mirror_img_x=hand['side']==1,camera_angle=angle,focal_multiplier=.95)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,default=4.)
    p.add_argument('--pixels',choices=('real','zero'),default='real');p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args()
    if not np.isfinite(a.start) or not 0<a.duration<=10:p.error('Probe duration must be in (0,10] seconds')
    if a.device=='cuda':
        if not torch.cuda.is_available():p.error('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.upstream))
    with a.reference.open() as stream:
        seed,metadata=read_reference((json.loads(line) for line in stream),a.start,a.duration)
    a.output.mkdir(parents=True,exist_ok=False);snapshot=a.output/'snapshot';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    backend=Backend(a.upstream,load_calibration(a.session/'frame/snapshot/device/persist'))
    regressor=PreparedRegressor(backend.model,backend.skeleton,device=a.device);regressor.warmup()
    states={h['side']:copy.deepcopy(h) for h in seed['hands']};last_stamp=seed['capture_raw_ns'];use_memory=False
    recording=Recording(a.session/'frame/sets.bin');sigmas=[];steps=[];counts=dict(frames=0,hypotheses=0,invalid_pose=0,capture_gap_reset=0,crop_rejected=0);durations=[]
    manifest=dict(scope=__doc__,seed_seconds=seed['seconds'],seed_raw_ns=seed['capture_raw_ns'],seed_sides=sorted(states),seed_source_ids=[h['source_id'] for h in states.values()],args={k:str(v) for k,v in vars(a).items()},source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.glob('*.py')})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (a.output/'predictions.jsonl').open('x') as out:
        for seconds,stamp,images in recording.frames(a.start,a.start+a.duration):
            if stamp<=seed['capture_raw_ns']:continue
            ref=metadata.get(stamp)
            if ref is None:raise ValueError('Reference lacks current pose metadata')
            counts['frames']+=1;started=time.perf_counter();hands=[]
            if not ref['capture_pose']['valid'] or not 0<stamp-last_stamp<=100_000_000:
                states={};use_memory=False
                counts['invalid_pose' if not ref['capture_pose']['valid'] else 'capture_gap_reset']+=1
            world=pose_matrix(ref['capture_pose']['capture']) if ref['capture_pose']['valid'] else None
            crops=[];intrinsics=[];extrinsics=[];ranges=[];included=[];view_names=[]
            if states and world is not None:
                cameras={name:c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for name,c in backend.cameras.items()}
                for side,hand in sorted(states.items()):
                    first=len(crops);names=[]
                    for name in NAMES:
                        try:crop=pose_crop(backend,hand,cameras[name],world)
                        except ValueError:crop=None
                        if crop is None:counts['crop_rejected']+=1;continue
                        pixels=warp(cameras[name],crop,images[name][0])
                        if a.pixels=='zero':pixels=np.zeros_like(pixels)
                        crops.append(pixels);intrinsics.append(crop.uv_to_window_matrix());xf=np.linalg.inv(crop.camera_to_world_xf);xf[:3,3]*=.001;extrinsics.append(xf);names.append(name)
                    if len(crops)>first:ranges.append((first,len(crops)));included.append(side);view_names.append(names)
            if included:
                frame=backend.Frame(torch.tensor(np.stack(crops)/255.,dtype=torch.float32),torch.tensor(np.stack(intrinsics),dtype=torch.float32),torch.tensor(np.stack(extrinsics),dtype=torch.float32))
                sides=torch.tensor(included);desc=backend.Desc(torch.tensor(ranges),sides,torch.full((len(included),),use_memory,dtype=torch.bool),sides)
                with torch.inference_mode():
                    prediction=regressor.regress(frame,desc,regressor.extract(frame.left_images))
                    for i,side in enumerate(included):
                        wrist=prediction.wrist_xfs[i].clone();wrist[:3,3]*=1000;angles=prediction.joint_angles[i];sigma=prediction.landmark_uncertainty_sigmas[i]
                        step=float(np.linalg.norm(wrist.numpy()[:3,3]-np.array(states[side]['wrist_world_mm'])[:3,3]));steps.append(step);sigmas.append(float(sigma.mean()))
                        hands.append(dict(side=side,source_id=states[side]['source_id'],memory=use_memory,hypothesis_only=True,identity_basis='Prior seed, not fresh detector classification',source_views=[dict(camera=name,origin='previous-pose ROI') for name in view_names[i]],vertices_world_mm=backend.mesh(angles,wrist,side).tolist(),joint_angles=angles.tolist(),wrist_world_mm=wrist.tolist(),uncertainty_raw=sigma.tolist()))
            states={h['side']:h for h in hands};use_memory=bool(hands);last_stamp=stamp;elapsed=(time.perf_counter()-started)*1000
            counts['hypotheses']+=len(hands);durations.append(elapsed)
            out.write(json.dumps(dict(seconds=seconds,capture_raw_ns=stamp,input_control=a.pixels,processing_ms=elapsed,hands=hands,**ref),allow_nan=False)+'\n')
    report=dict(scope=__doc__,counts=counts,processing_ms_p50_p95_max=np.percentile(durations,[50,95,100]).tolist(),mean_raw_sigma_p05_p50_p95=np.percentile(sigmas,[5,50,95]).tolist() if sigmas else [],wrist_step_mm_p50_p95_max=np.percentile(steps,[50,95,100]).tolist() if steps else [])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
