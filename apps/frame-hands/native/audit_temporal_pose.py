"""Audit an isolated complete-pose temporal preference; no live model or driver changes."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_native_lengths import stats,sha
from frame_data import read_audit,capture_poses,load_calibration,Recording
from compare_replay import SEGMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','native','session','upstream'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source',type=Path,help='Frozen experimental native source directory, required after reverting the prototype')
 a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
 control=np.loadtxt(a.run/'recorded-control.txt');candidate=np.loadtxt(a.run/'recorded-temporal.txt');old=np.loadtxt(a.native/'whole-pose-history-v1/output.txt')
 assert candidate.shape==control.shape==old.shape==(2940,191)
 for z in (control,candidate):assert np.isfinite(z).all()
 np.testing.assert_array_equal(control[:,:5],old[:,:5]);np.testing.assert_array_equal(control[:,6:],old[:,6:])
 np.testing.assert_array_equal(candidate[:,:4],control[:,:4]);np.testing.assert_array_equal(candidate[:,7:99],control[:,7:99])
 for side in (0,1):
  bounds=np.loadtxt(a.native/f'palm-profile-v1/side{side}.fhrig',skiprows=1,max_rows=20)[:,6:8]
  angles=candidate[candidate[:,2]==side,108:128];assert np.all(angles>=bounds[:,0]-1e-8) and np.all(angles<=bounds[:,1]+1e-8)
 disabled=candidate[:,3]==0;np.testing.assert_array_equal(candidate[disabled,99:],control[disabled,99:])
 seq=[np.loadtxt(a.run/name) for name in ('sequences-control.txt','sequences-temporal.txt')]
 np.testing.assert_array_equal(seq[0],np.loadtxt(a.native/'whole-pose-history-v1/sequences.txt'))
 sequences={}
 for mode,name in enumerate(('clean_fast_gestures','noisy_landmarks','flipped_initialization','loss_and_reacquisition')):
  groups=[z[(z[:,1]==mode)&(z[:,3]==1)] for z in seq]
  sequences[name]=dict(rows=len(groups[0]),pose_rmse_mm=[stats(g[:,9]) for g in groups],motion_delta_error_mm=[stats(g[:,11]) for g in groups],max_accuracy_regression_mm=float((groups[1][:,9]-groups[0][:,9]).max()))
 for side in (0,1):
  group=seq[1][(seq[1][:,1]==3)&(seq[1][:,0]==side)]
  for f in (45,46,47):assert group[f,3]==0
  for f in (48,75,76,80,90):assert group[f,4]==0
  for f in (49,77,81,91):assert group[f,4]==1
 audit=read_audit(a.session/'frame/live-views.txt');heads=capture_poses(a.session)
 records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
 metadata=json.loads((a.native/'whole-pose-seed-v1/metadata.json').read_text());continuity=[];prev={}
 for i,r in enumerate(records):
  h=heads.get(audit[r['raw_ns']]['mono_ns'])
  if h is None:prev.pop(r['side'],None);continue
  rotations=[Rotation.from_matrix(h[:3,:3])*Rotation.from_quat(z[i,101:105][[1,2,3,0]]) for z in (control,candidate)]
  clouds=[z[i,128:191].reshape(21,3)@h[:3,:3].T+h[:3,3] for z in (control,candidate)]
  previous=prev.get(r['side']);prev[r['side']]=(i,rotations,clouds)
  if previous is None or not metadata[i]['continuous']:continue
  j,pr,pc=previous
  assert records[j]['source_id']==r['source_id'] and 0<r['raw_ns']-records[j]['raw_ns']<=100000000
  continuity.append(dict(index=i,previous=j,wrist_degrees=[float((pr[k].inv()*rotations[k]).magnitude()*180/np.pi) for k in (0,1)],max_point_step_mm=[float(np.linalg.norm(clouds[k]-pc[k],axis=1).max()) for k in (0,1)],max_angle_step_rad=[float(np.max(abs(z[i,108:128]-z[j,108:128]))) for z in (control,candidate)]))
 metrics=dict(rows=len(candidate),eligible=int(candidate[:,3].sum()),image_rmse_px=[stats(z[:,100]) for z in (control,candidate)],short_continuous_pairs=len(continuity),wrist_steps_degrees=[stats([r['wrist_degrees'][k] for r in continuity]) for k in (0,1)],wrist_steps_over45=[sum(r['wrist_degrees'][k]>45 for r in continuity) for k in (0,1)],max_point_steps_mm=[stats([r['max_point_step_mm'][k] for r in continuity]) for k in (0,1)],max_angle_steps_rad=[stats([r['max_angle_step_rad'][k] for r in continuity]) for k in (0,1)],host_additional_refiner_us=[stats(z[z[:,3]==1,5]) for z in (control,candidate)])
 # Inspect extremes, including the immediately preceding frame of the largest jump.
 worst=max(continuity,key=lambda r:r['wrist_degrees'][1]);reg=max(continuity,key=lambda r:r['wrist_degrees'][1]-r['wrist_degrees'][0]);gain=max(continuity,key=lambda r:r['wrist_degrees'][0]-r['wrist_degrees'][1])
 selected=[('largest-candidate-jump-before',worst['previous']),('largest-candidate-jump',worst['index']),('largest-jump-regression',reg['index']),('largest-jump-improvement',gain['index']),('largest-image-regression',int(np.argmax(candidate[:,100]-control[:,100])))]
 import cv2
 sys.path.insert(0,str(a.upstream));cams=load_calibration(a.session/'frame/snapshot/device/persist');rec=Recording(a.session/'frame/sets.bin');review=[]
 for label,index in selected:
  r=records[index];stamp=r['raw_ns'];sec=(stamp-rec.index[0][0])/1e9;frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==stamp
  canvas=np.full((740,1200,3),24,np.uint8)
  cv2.putText(canvas,f"{label} | {sec:.3f}s side{r['side']} | image RMSE {control[index,100]:.2f}/{candidate[index,100]:.2f}px",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
  for v,name in enumerate(('slam_left','slam_right')):
   image=frames[0][2][name][0];cam=cams[name];view=next(x for x in audit[stamp]['views'] if x['camera']==name and x['id']==r['source_id'])
   uv=[cam.eye_to_window(cam.world_to_eye(z[index,128:191].reshape(21,3))) for z in (control,candidate)]
   alluv=np.vstack([view['points'],*uv]);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(alluv,axis=0).max()+100)));x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col,kind in enumerate(('image','detector','whole-seed','temporal')):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if kind=='detector':
     for point in view['points']-[x,y]:cv2.circle(picture,tuple(np.rint(point).astype(int)),2,(210,160,255),-1)
    elif col>=2:
     points=uv[col-2]-[x,y]
     for pi,qi,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(points[pi]).astype(int)),tuple(np.rint(points[qi]).astype(int)),(240,220,40) if col==2 else (70,230,60),2,cv2.LINE_AA)
    canvas[65+v*335:365+v*335,col*300:(col+1)*300]=cv2.resize(picture,(300,300));cv2.putText(canvas,name+' / '+kind,(col*300+3,53+v*335),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,index=index,seconds=sec,source=r['source_id'],side=r['side'],image_rmse_px=[float(z[index,100]) for z in (control,candidate)]))
 src=a.run/'source';src.mkdir(exist_ok=True);base=a.source or Path(__file__).parent
 for name in ('rig_temporal_prior.h','fit_rig_pixels.h','whole_pose_fitter.h','probe_whole_pose_seed.cpp','probe_whole_pose_sequence.cpp','audit_temporal_pose.py','whole_pose_seed.h','rig_stage.h','causal_rig_fit.h','fit_hand_rig.h','hand_rig.h','rig_bound_step.h'):
  origin=Path(__file__) if name=='audit_temporal_pose.py' else base/name
  if origin.resolve()!=(src/name).resolve():shutil.copy2(origin,src/name)
 report=dict(metrics=metrics,sequences=sequences,review=review,source_sha256={p.name:sha(p) for p in src.iterdir()},input_sha256={str(p):sha(p) for p in [a.native/'whole-pose-seed-v1/heads.txt',a.native/'whole-pose-seed-v1/metadata.json',a.native/'whole-pose-history-v1/output.txt',a.native/'rig-stage-v2/recorded-input.txt',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig',a.native/'rig-generic-v1/generic.fhrig',a.native/'rig-pixels-viewer-v2/cameras.txt',a.session/'frame/live-views.txt',a.session/'relay-trace.bin',a.session/'sampler-trace.bin']},output_sha256={p.name:sha(p) for p in a.run.glob('*.txt')},limitations=['Isolated experiment with one prespecified35e-2rad/33ms orientation-and-angle scale, linearly widened with age, joint23dimensional Huber3prior. No translation prior; current stereo supplies placement.', 'Two completed fits compared under the same augmented objective. Costs are not comparable to the unregularized control; pixel RMSE excludes prior.', 'Detector, camera and identity observations frozen. Closed-loop fitted pose history. Latest143912Zholdout untouched.', 'Synthetic sequences share anatomy and cameras; lower numerical noise is not wearer acceptance. No collision or finger visibility model.', 'Host timings are additional refiner only, excluding baseline fits and neural inference. No Frame timing or live trial.'])
 (a.run/'continuity.json').write_text(json.dumps(continuity)+'\n');(a.run/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(metrics=metrics,sequences=sequences,review=review),indent=2))
if __name__=='__main__':main()
