"""Audit fixed-palm-root fitting against the whole-pose seed and extra-iteration control."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses,load_calibration,Recording
from compare_replay import SEGMENTS
from audit_native_lengths import stats,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('native','run','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--source',type=Path)
 a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Holdout excluded')
 z=np.loadtxt(a.run/'output-final.txt');old=np.loadtxt(a.native/'whole-pose-history-v1/output.txt');meta=json.loads((a.native/'whole-pose-seed-v1/metadata.json').read_text());heads_rows=np.loadtxt(a.native/'whole-pose-seed-v1/heads.txt')
 assert z.shape==(2940,285) and np.isfinite(z).all();np.testing.assert_array_equal(z[:,9:99],old[:,101:191]);np.testing.assert_allclose(z[:,6],old[:,100],atol=1e-12,rtol=1e-12);assert np.all(z[:,5]<=z[:,4])
 previous={};continuity=[];offsets=(6,99,192)
 for i,r in enumerate(meta):
  if not heads_rows[i,3]:previous.pop(r['side'],None);continue
  head=Rotation.from_quat(heads_rows[i,5:9][[1,2,3,0]]);rotations=[head*Rotation.from_quat(z[i,o+3:o+7][[1,2,3,0]]) for o in offsets];clouds=[head.apply(z[i,o+30:o+93].reshape(21,3))+heads_rows[i,9:12] for o in offsets]
  previous_row=previous.get(r['side']);previous[r['side']]=(i,rotations,clouds)
  if previous_row and r['continuous']:
   j,pr,pc=previous_row;assert meta[j]['source']==r['source'] and 0<r['raw_ns']-meta[j]['raw_ns']<=100000000
   continuity.append(dict(index=i,previous=j,wrist_degrees=[float((pr[k].inv()*rotations[k]).magnitude()*180/np.pi) for k in range(3)],max_point_step_mm=[float(np.linalg.norm(clouds[k]-pc[k],axis=1).max()) for k in range(3)]))
 for side in (0,1):
  bounds=np.loadtxt(a.native/f'palm-profile-v1/side{side}.fhrig',skiprows=1,max_rows=20)[:,6:8]
  for o in offsets:
   angles=z[old[:,2]==side,o+10:o+30];assert np.all(angles>=bounds[:,0]-1e-8) and np.all(angles<=bounds[:,1]+1e-8)
 metrics=dict(rows=len(z),variants=['saved_whole_pose','continued_full_fit','palm_then_locked_root'],all_rmse_px=[stats(z[:,o]) for o in offsets],palm_rmse_px=[stats(z[:,o+1]) for o in offsets],finger_rmse_px=[stats(z[:,o+2]) for o in offsets],wrist_steps_degrees=[stats([r['wrist_degrees'][k] for r in continuity]) for k in range(3)],wrist_steps_over45=[sum(r['wrist_degrees'][k]>45 for r in continuity) for k in range(3)],max_point_steps_mm=[stats([r['max_point_step_mm'][k] for r in continuity]) for k in range(3)],host_extra_microseconds=dict(palm=stats(z[:,1]),locked_articulation=stats(z[:,2]),anchored_total=stats(z[:,1]+z[:,2]),continued_full=stats(z[:,3])))
 synthetic=np.loadtxt(a.run/'synthetic.txt');assert synthetic.shape==(96,10) and np.isfinite(synthetic).all();synth={}
 for mode,name in enumerate(('clean','palm_noise_2px','finger_bias_20_minus15px','all_noise_2px')):
  rows=synthetic[synthetic[:,3]==mode];synth[name]=dict(rows=len(rows),root_translation_error_mm=[stats(rows[:,i]) for i in (4,7)],root_rotation_error_degrees=[stats(rows[:,i]) for i in (5,8)],pose_rmse_mm=[stats(rows[:,i]) for i in (6,9)])
 worst=max(continuity,key=lambda r:r['wrist_degrees'][2]-r['wrist_degrees'][0]);gain=int(np.argmax(z[:,7]-z[:,193]));delta=z[:,192]-z[:,6]
 selected=[('largest-jump-regression-before',worst['previous']),('largest-jump-regression',worst['index']),('largest-palm-gain',gain),('largest-all-image-regression',int(np.argmax(delta)))]
 records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
 # Include a previously reviewed visible closed-fist case without optimizing its selection.
 origin=Recording(a.session/'frame/sets.bin').index[0][0]
 selected.append(('previous-visible-fist',min((i for i,r in enumerate(records) if r['side']==0),key=lambda i:abs((records[i]['raw_ns']-origin)/1e9-38.066665))))
 import cv2
 sys.path.insert(0,str(a.upstream));cams=load_calibration(a.session/'frame/snapshot/device/persist');rec=Recording(a.session/'frame/sets.bin');review=[]
 for label,i in selected:
  r=records[i];raw=r['raw_ns'];sec=(raw-rec.index[0][0])/1e9
  if label=='previous-visible-fist':assert abs(sec-38.066665)<.07
  frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==raw
  canvas=np.full((740,1200,3),24,np.uint8);cv2.putText(canvas,f"{label} | {sec:.3f}s side{r['side']} | palm {z[i,7]:.1f}->{z[i,193]:.1f}px | whole {z[i,6]:.1f}->{z[i,192]:.1f}px",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.49,(240,240,240),1,cv2.LINE_AA)
  for v,name in enumerate(('slam_left','slam_right')):
   image=frames[0][2][name][0];cam=cams[name];uv=[cam.eye_to_window(cam.world_to_eye(z[i,o+30:o+93].reshape(21,3))) for o in offsets];alluv=np.vstack(uv);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(alluv,axis=0).max()+100)));x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col in range(4):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if col:
     pp=uv[col-1]-[x,y]
     for pi,qi,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(pp[pi]).astype(int)),tuple(np.rint(pp[qi]).astype(int)),((240,220,40),(180,120,240),(70,230,60))[col-1],2,cv2.LINE_AA)
    canvas[65+v*335:365+v*335,col*300:(col+1)*300]=cv2.resize(picture,(300,300));cv2.putText(canvas,name+' / '+('image','whole-seed','continued','palm-anchored')[col],(col*300+3,53+v*335),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,index=i,seconds=sec,side=r['side'],source=r['source_id'],all_rmse_px=[float(z[i,o]) for o in offsets],palm_rmse_px=[float(z[i,o+1]) for o in offsets],image_sha256=sha(a.run/(label+'.png'))))
 source=a.run/'source';source.mkdir(exist_ok=True);base=a.source or BASE
 names=['fit_rig_pixels.h','fit_palm_root.h','probe_palm_anchor.cpp','test_palm_anchor.cpp','rig_fit_quality.h','hand_rig.h','fit_hand_rig.h','rig_bound_step.h','audit_palm_anchor.py']
 for name in names:
  origin=Path(__file__) if name=='audit_palm_anchor.py' else base/name
  if origin.resolve()!=(source/name).resolve():shutil.copy2(origin,source/name)
 report=dict(metrics=metrics,synthetic=synth,review=review,source_sha256={name:sha(source/name) for name in names},input_sha256={str(p):sha(p) for p in [a.native/'quality-admission-v2/quality-input.txt',a.native/'whole-pose-history-v1/output.txt',a.native/'whole-pose-seed-v1/heads.txt',a.native/'whole-pose-seed-v1/metadata.json',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig',a.native/'rig-pixels-viewer-v2/cameras.txt']},output_sha256={name:sha(a.run/name) for name in ('output-final.txt','synthetic.txt')},limitations=['Independent per-frame refinement of frozen whole-pose estimates, not a closed-loop native image tracker. No visibility/identity/gesture policy changes.', 'Palm-root has20iterations over6wrist parameters and5stereo landmarks. It freezes20angles. Articulation then has30iterations over20angles with wrist bit-exact. Full continuation has30iterations over26parameters, same seed-angle reference.', 'Smaller training residual is not physical accuracy. Five palm landmarks may be noisy or geometrically ambiguous.', 'Synthetic cases share anatomy/cameras and assumed corruption; they test mechanism, not real-world quality.', 'Host timings exclude original whole-pose fit and neural inference. No Frame timing or wearer acceptance; latest143912Zholdout excluded.'])
 (a.run/'continuity.json').write_text(json.dumps(continuity)+'\n',encoding='utf-8');(a.run/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(metrics=metrics,review=review),indent=2))
if __name__=='__main__':main()
