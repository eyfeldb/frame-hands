"""Offline identity comparison on complete saved camera/audit/pose recordings.

Reads original recordings in place. No annotation inputs, future frames, camera
access or missing-head-pose fallback. This is not replay of a game's rendered hands.
"""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from frame_data import Recording,read_audit,load_calibration,capture_poses,crop_camera,warp
from live_backend import Backend
from chirality_evidence import ChiralityEvidence
from feature_reuse import PreparedRegressor


def prepare(cameras,images,observation):
    result={}
    for view in observation['views']:
        name=view['camera']
        if name not in cameras or name not in images or view['presence']<.5:continue
        if images[name][0].shape!=(cameras[name].height,cameras[name].width):raise ValueError('Camera dimensions changed')
        try:crop=crop_camera(cameras[name],view['points'],0,np.eye(4))
        except ValueError:continue
        result[name,view['id']]=(warp(cameras[name],crop,images[name][0]),dict(intrinsics=crop.uv_to_window_matrix().tolist(),camera_to_head_mm=crop.camera_to_world_xf.tolist()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('fresh','temporal','chirality'),default='chirality')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--fixed-hand-scale',type=float,default=1.,help='Offline fixed pose-skeleton/mesh scale; identity remains unscaled')
    a=p.parse_args()
    if not np.isfinite(a.fixed_hand_scale) or not .5<=a.fixed_hand_scale<=1.5:p.error('Invalid fixed scale')
    if a.device=='cuda' and (a.mode!='chirality' or not torch.cuda.is_available()):p.error('CUDA requires the isolated chirality path and CUDA environment')
    if a.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    frame=a.session/'frame';calibration=frame/'snapshot/device/persist'
    cameras=load_calibration(calibration);audits=read_audit(frame/'live-views.txt');poses=capture_poses(a.session);recording=Recording(frame/'sets.bin')
    from fixed_scale_backend import ScaledBackend
    backend=ScaledBackend(a.upstream,cameras,association='fresh' if a.mode=='fresh' else 'temporal',hand_scale=a.fixed_hand_scale)
    evidence=ChiralityEvidence(a.upstream,cameras,optimized=True,device=a.device) if a.mode=='chirality' else None
    if evidence:
        backend.prepared_regressor=PreparedRegressor(backend.model,backend.skeleton,device=a.device)
        evidence.regressor.warmup();backend.prepared_regressor.warmup()
    counts=dict(frames=0,missing_audit=0,missing_capture_pose=0,hands=0,valid_pose_frames=0,frames_with_two_outputs=0)
    durations=[];crop_durations=[]
    manifest=dict(scope=__doc__,args={k:str(v) for k,v in vars(a).items()},torch=torch.__version__,cuda=torch.version.cuda,source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.glob('*.py')},calibration_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in calibration.glob('*.json')},image_bytes=recording.path.stat().st_size)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (a.output/'predictions.jsonl').open('x') as output,(a.output/'evidence.jsonl').open('x') as trace:
        for seconds,stamp,images in recording.frames(0,float('inf')):
            counts['frames']+=1;obs=audits.get(stamp)
            if obs is None:
                counts['missing_audit']+=1;backend.previous={}
                if a.mode!='fresh':backend.ownership.reset()
                continue
            world=poses.get(obs['mono_ns']);valid=world is not None
            counts['missing_capture_pose']+=int(not valid);counts['valid_pose_frames']+=int(valid)
            start=time.perf_counter();prepared=prepare(cameras,images,obs) if valid else {};crop_ms=(time.perf_counter()-start)*1000
            scores=[]
            if evidence and valid:obs,scores,_=evidence.score(obs,world,prepared)
            result=backend.predict(obs,{},world,prepared,feature_cache=evidence.features if evidence and valid else None)
            elapsed=(time.perf_counter()-start)*1000;result['processing_ms']=elapsed
            if a.mode=='fresh':
                for hand in result['hands']:
                    hand['source_views']=[dict(camera=v['camera'],source_id=v['id']) for v in obs['views'] if v['id']==hand['source_id'] and (v['camera'],v['id']) in prepared]
            pose=np.eye(4) if not valid else world.copy();pose[:3,3]*=.001
            output.write(json.dumps(dict(seconds=seconds,capture_raw_ns=stamp,capture_mono_ns=obs['mono_ns'],capture_pose=dict(valid=valid,capture=pose[:3].reshape(-1).tolist()),current_pose=dict(valid=valid),**result),allow_nan=False)+'\n')
            trace.write(json.dumps(dict(stamp=stamp,views=scores,association=getattr(backend.ownership,'last_debug',{})),allow_nan=False)+'\n')
            counts['hands']+=len(result['hands']);counts['frames_with_two_outputs']+=int(len(result['hands'])==2)
            if valid:durations.append(elapsed);crop_durations.append(crop_ms)
    report=dict(fixed_hand_scale=a.fixed_hand_scale,mode=a.mode,device=a.device,counts=counts,processing_ms_p50_p95_max=np.percentile(durations,[50,95,100]).tolist(),crop_ms_p50_p95=np.percentile(crop_durations,[50,95]).tolist(),scope=__doc__+' Counts are outputs, not correctness. Processing includes source-image crop construction; excludes capture/transport/display.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
