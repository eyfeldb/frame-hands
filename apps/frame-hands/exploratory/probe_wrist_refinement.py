"""Offline paired rigid-wrist probe on predeclared saved-recording intervals."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch,scipy
from frame_data import Recording,read_audit,load_calibration
from live_backend import Backend
from run_mesh_trial import pose_matrix
from replay_umetrack import COMMON
from wrist_refinement import fit_wrist,apply_correction,transform,PALM_DETECTOR
from probe_roi_presence import PresenceModel,verify_hand
from review_mesh_recording import render


def landmarks(backend,h):
    wrist=torch.tensor(h['wrist_world_mm'],dtype=torch.float32)
    if h['side']:wrist[:,0]*=-1
    with torch.inference_mode():return backend.landmarks(backend.hand,torch.tensor(h['joint_angles']),wrist).numpy()


def frozen_comparison(xyz,corrected,cameras,evidence):
    rows=[];native=[i for i,j in COMMON];mp=[j for i,j in COMMON]
    held=[n for n,j in enumerate(mp) if j not in PALM_DETECTOR]
    for v in evidence:
        cam=cameras[v['camera']];target=np.asarray(v['model_points'])[mp]
        before=np.linalg.norm(cam.eye_to_window(cam.world_to_eye(xyz[native]))-target,axis=1)
        after=np.linalg.norm(cam.eye_to_window(cam.world_to_eye(corrected[native]))-target,axis=1)
        rows.append(dict(camera=v['camera'],presence=v['presence'],crop_size_px=v['crop_size_px'],before_median_px=float(np.median(before)),after_median_px=float(np.median(after)),before_heldout_median_px=float(np.median(before[held])),after_heldout_median_px=float(np.median(after[held]))))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('session','baseline','upstream','models','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--interval',nargs=2,type=float,action='append',required=True);a=p.parse_args()
    if any(not 0<=x<y for x,y in a.interval):p.error('Invalid interval')
    sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/'snapshot';snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    cal=a.session/'calibration';b=Backend(a.upstream,load_calibration(cal if cal.exists() else a.session/'frame/snapshot/device/persist'))
    verifier=PresenceModel(a.models);audit=read_audit(a.session/'frame/live-views.txt');first=min(audit)
    records=iter(Recording(a.session/'frame/sets.bin').frames(min(x for x,y in a.interval)-.01,max(y for x,y in a.interval)+.01));current=next(records,None)
    counts=dict(frames=0,hands=0,stereo_fits=0,corrected=0);reasons={};all_rows=[];previewed=set()
    with a.baseline.open() as source,(a.output/'predictions.jsonl').open('x') as output,(a.output/'comparisons.jsonl').open('x') as trace:
        for line in source:
            row=json.loads(line);stamp=row['capture_raw_ns'];seconds=(stamp-first)/1e9
            if not any(x<=seconds<y for x,y in a.interval):continue
            while current is not None and current[1]<stamp:current=next(records,None)
            if current is None or current[1]!=stamp:raise ValueError('No exact raw capture')
            images=current[2];world=pose_matrix(row['capture_pose']['capture']) if row['capture_pose']['valid'] else None
            cameras={n:c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for n,c in b.cameras.items()} if world is not None else {}
            views={(v['camera'],v['id']):v for v in audit[stamp]['views']};hands=[]
            counts['frames']+=1;counts['hands']+=len(row['hands'])
            for h in row['hands']:
                selected=[views[(v['camera'],v['source_id'])] for v in h['source_views'] if (v['camera'],v['source_id']) in views]
                if world is None or len(selected)!=2:hands.append(h);continue
                counts['stereo_fits']+=1;xyz=landmarks(b,h);start=time.perf_counter()
                parameters,debug=fit_wrist(xyz,h['wrist_world_mm'],cameras,selected)
                fit_ms=(time.perf_counter()-start)*1000;reasons[debug['reason']]=reasons.get(debug['reason'],0)+1
                adjusted=apply_correction(h,parameters) if parameters is not None else h
                corrected=transform(xyz,np.asarray(h['wrist_world_mm'])[:3,3],parameters) if parameters is not None else xyz
                before,_=verify_hand(verifier,b.hand,h,b.cameras,images,world)
                after,_=verify_hand(verifier,b.hand,adjusted,b.cameras,images,world) if parameters is not None else (before,None)
                detail=dict(stamp=stamp,seconds=seconds,side=h['side'],source_id=h['source_id'],accepted_hypothesis=parameters is not None,fit_ms=fit_ms,fit=debug,frozen=frozen_comparison(xyz,corrected,cameras,before),baseline_verifier=before,corrected_verifier=after)
                trace.write(json.dumps(detail,allow_nan=False)+'\n');all_rows.append(detail);hands.append(adjusted)
                if parameters is not None:counts['corrected']+=1
            result=dict(row,seconds=seconds,hands=hands,wrist_refinement_hypothesis=True)
            output.write(json.dumps(result,allow_nan=False)+'\n')
            for index,(x,y) in enumerate(a.interval):
                if index not in previewed and (x+y)/2<=seconds<(x+y)/2+.08 and row['capture_pose']['valid']:
                    original=render(images,audit[stamp],row,b.cameras,b.triangles,seconds)
                    candidate=render(images,audit[stamp],result,b.cameras,b.triangles,seconds)
                    panel=np.vstack((original,candidate[-512:]))
                    cv2.putText(panel,'BOTTOM: bounded wrist hypothesis',(10,panel.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                    cv2.imwrite(str(a.output/f'review-{seconds:.3f}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,94]);previewed.add(index)
    report=dict(scope=__doc__+' Fixed-camera verifier landmarks are model agreement, not independent physical ground truth. Output processing_ms remains baseline metadata, not candidate latency.',counts=counts,reasons=reasons,fit_ms_p50_p95_max=np.percentile([r['fit_ms'] for r in all_rows],[50,95,100]).tolist() if all_rows else None,runtime=dict(scipy=scipy.__version__,torch=torch.__version__,ncnn=verifier.version),args={k:str(v) for k,v in vars(a).items()},source_sha256={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in snapshot.glob('*.py')})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['counts','reasons','fit_ms_p50_p95_max']}))

if __name__=='__main__':main()
