"""Prepare and audit a causal whole-pose initializer on older saved stereo input."""
import argparse,bisect,json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_native_lengths import sha,stats
from frame_data import read_audit,capture_poses,load_calibration,Recording
from compare_replay import SEGMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','audit'])
 for n in ('native','session','run'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--upstream',type=Path);a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
 records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
 audit=read_audit(a.session/'frame/live-views.txt');times=sorted(audit);heads=capture_poses(a.session)
 if a.action=='prepare':
  a.run.mkdir(parents=True,exist_ok=False);previous={};metadata=[]
  with (a.run/'heads.txt').open('x') as output:
   for r in records:
    raw=r['raw_ns'];source=r['source_id'];side=r['side'];head=heads.get(audit[raw]['mono_ns']);valid=head is not None
    old=previous.get(side);continuous=bool(old and old['source_id']==source and 0<raw-old['raw_ns']<=100000000 and valid)
    if continuous:
     for stamp in times[bisect.bisect_right(times,old['raw_ns']):bisect.bisect_right(times,raw)]:
      hs=audit[stamp]['hands'];hand=hs.get(source);same_side=[k for k,h in hs.items() if int(h['right']>=.5)==side]
      if not hand or not hand['stereo'] or same_side!=[source]:continuous=False;break
    previous[side]=r
    transform=np.eye(4) if head is None else head
    q=Rotation.from_matrix(transform[:3,:3]).as_quat()[[3,0,1,2]]
    output.write(' '.join(map(str,[raw,source,side,int(valid),int(continuous),*q,*transform[:3,3]]))+'\n')
    metadata.append(dict(raw_ns=raw,source=source,side=side,head_valid=valid,continuous=continuous))
  (a.run/'metadata.json').write_text(json.dumps(metadata)+'\n');print(json.dumps(dict(rows=len(records),valid=sum(x['head_valid'] for x in metadata),continuous=sum(x['continuous'] for x in metadata))))
  return
 import cv2
 if a.upstream is None:raise ValueError('Upstream camera implementation required')
 sys.path.insert(0,str(a.upstream));data=np.loadtxt(a.run/'output.txt');old=np.loadtxt(a.native/'rig-stage-v2/recorded-output.txt');meta=json.loads((a.run/'metadata.json').read_text())
 assert data.shape==(len(records),191) and np.isfinite(data).all()
 np.testing.assert_array_equal(data[:,7],old[:,3]);np.testing.assert_array_equal(data[:,36:99],old[:,41:104]);assert np.all(data[:,99]<=data[:,7])
 for side in (0,1):
  bounds=np.loadtxt(a.native/f'palm-profile-v1/side{side}.fhrig',skiprows=1,max_rows=20)[:,6:8]
  angles=data[data[:,2]==side,108:128];assert np.all(angles>=bounds[:,0]-1e-8) and np.all(angles<=bounds[:,1]+1e-8)
 continuity=[];previous={}
 for i,r in enumerate(records):
  h=heads.get(audit[r['raw_ns']]['mono_ns'])
  if h is None:previous.pop(r['side'],None);continue
  seed=Rotation.from_matrix(h[:3,:3])*Rotation.from_quat(old[i,7:11][[1,2,3,0]])
  rot=[Rotation.from_matrix(h[:3,:3])*Rotation.from_quat(data[i,o+2:o+6][[1,2,3,0]]) for o in (7,99)]
  cloud=[data[i,o+29:o+92].reshape(21,3)@h[:3,:3].T+h[:3,3] for o in (7,99)]
  oldrow=previous.get(r['side']);previous[r['side']]=(i,seed,rot,cloud)
  if oldrow is None:continue
  j,ps,pr,pc=oldrow
  if records[j]['source_id']!=r['source_id'] or r['raw_ns']-records[j]['raw_ns']>100000000:continue
  continuity.append(dict(index=i,previous=j,continuous=meta[i]['continuous'],seed_degrees=float((ps.inv()*seed).magnitude()*180/np.pi),
   wrist_degrees=[float((pr[k].inv()*rot[k]).magnitude()*180/np.pi) for k in (0,1)],
   max_point_step_mm=[float(np.linalg.norm(cloud[k]-pc[k],axis=1).max()) for k in (0,1)]))
 expected=[];previous_meta={}
 for x in meta:
  prev=previous_meta.get(x['side'])
  expected.append(bool(x['head_valid'] and x['continuous'] and prev and prev['head_valid'] and x['source']==prev['source'] and 0<x['raw_ns']-prev['raw_ns']<=100000000))
  if 1-x['side'] in previous_meta and previous_meta[1-x['side']]['source']==x['source']:previous_meta.pop(1-x['side'])
  previous_meta[x['side']]=x
 np.testing.assert_array_equal(data[:,3].astype(bool),expected)
 np.testing.assert_array_equal(data[data[:,4]==0,99:],data[data[:,4]==0,7:99])
 continued=np.loadtxt(a.native/'rig-bound-step-v1/output.txt');np.testing.assert_array_equal(continued[:,3],data[:,7])
 versus=data[:,99]-continued[:,179]
 continuation_control=dict(better_over1=int((versus< -1).sum()),worse_over1=int((versus>1).sum()),median_cost_difference=float(np.median(versus)),continued_rmse_px=stats(continued[:,180]),scope='Existing baseline plus30pixel iterations and reset damping; same extra iteration budget, different seed/history path. Not a native performance comparison.')
 delta=data[:,99]-data[:,7];difference=np.linalg.norm(data[:,128:191].reshape(-1,21,3)-data[:,36:99].reshape(-1,21,3),axis=2).max(axis=1)
 first=times[0];eligible=data[:,3].astype(bool)
 metrics=dict(cases=len(records),eligible=int(eligible.sum()),used=int(data[:,4].sum()),improved_over1=int((delta< -1).sum()),worst_cost_delta=float(delta.max()),
  rmse_px=[stats(data[:,o+1]) for o in (7,99)],extra_host_us=stats(data[eligible,5]),max_pose_difference_mm=stats(difference),
  short_same_source_pairs=len(continuity),wrist_steps_deg=[stats([x['wrist_degrees'][k] for x in continuity]) for k in (0,1)],
  wrist_steps_over45=[sum(x['wrist_degrees'][k]>45 for x in continuity) for k in (0,1)],
  max_point_steps_mm=[stats([x['max_point_step_mm'][k] for x in continuity]) for k in (0,1)],
  palm_seed_steps_over45=sum(x['seed_degrees']>45 for x in continuity),
  final_step_over45_seed_under15=sum(x['wrist_degrees'][0]>45 and x['seed_degrees']<15 for x in continuity))
 selected=[('largest-cost-gain',int(np.argmin(delta))),('largest-pose-change',int(np.argmax(difference))),('largest-candidate-step',max(continuity,key=lambda x:x['wrist_degrees'][1])['index'])]
 for t in (1.8,1.9,1.933333):selected.append((f'failure-{t:.3f}',min((i for i,r in enumerate(records) if r['side']==0),key=lambda i:abs((records[i]['raw_ns']-first)/1e9-t))))
 cams=load_calibration(a.session/'frame/snapshot/device/persist');rec=Recording(a.session/'frame/sets.bin');review=[]
 for label,index in selected:
  r=records[index];stamp=r['raw_ns'];sec=(stamp-rec.index[0][0])/1e9;frames=list(rec.frames(sec-1e-7,sec+1e-7));assert len(frames)==1 and frames[0][1]==stamp
  canvas=np.full((740,1200,3),24,np.uint8)
  cv2.putText(canvas,f"{label} | {sec:.3f}s side{r['side']} | used {int(data[index,4])} delta cost {delta[index]:.1f}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
  for v,name in enumerate(('slam_left','slam_right')):
   image=frames[0][2][name][0];cam=cams[name];view=next(x for x in audit[stamp]['views'] if x['camera']==name and x['id']==r['source_id'])
   uv=[cam.eye_to_window(cam.world_to_eye(data[index,o+29:o+92].reshape(21,3))) for o in (7,99)]
   alluv=np.vstack([view['points'],*uv]);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(alluv,axis=0).max()+100)));x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col,kind in enumerate(('image','detector','control','candidate')):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if kind=='detector':
     for point in view['points']-[x,y]:cv2.circle(picture,tuple(np.rint(point).astype(int)),2,(210,160,255),-1)
    elif col>=2:
     points=uv[col-2]-[x,y]
     for p,q,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(points[p]).astype(int)),tuple(np.rint(points[q]).astype(int)),(240,220,40) if col==2 else (70,230,60),2,cv2.LINE_AA)
    canvas[65+v*335:365+v*335,col*300:(col+1)*300]=cv2.resize(picture,(300,300));cv2.putText(canvas,name+' / '+kind,(col*300+3,53+v*335),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,index=index,seconds=sec,source=r['source_id'],side=r['side'],eligible=bool(data[index,3]),used=bool(data[index,4]),cost_delta=float(delta[index]),max_difference_mm=float(difference[index])))
 src=a.run/'source';src.mkdir(exist_ok=True);base=Path(__file__).parent
 for name in ('whole_pose_seed.h','whole_pose_fitter.h','probe_whole_pose_seed.cpp','test_whole_pose_seed.cpp','audit_whole_pose_seed.py','rig_stage.h','causal_rig_fit.h','fit_hand_rig.h','fit_rig_pixels.h','rig_bound_step.h','hand_rig.h'):shutil.copy2(base/name,src/name)
 (a.run/'continuity.json').write_text(json.dumps(continuity)+'\n')
 report=dict(metrics=metrics,continuation_control=continuation_control,eligibility_and_exact_fallback_verified=True,review=review,source_sha256={p.name:sha(p) for p in src.iterdir()},input_sha256={str(p):sha(p) for p in [a.run/'heads.txt',a.native/'rig-stage-v2/recorded-input.txt',a.native/'rig-stage-v2/recorded-output.txt',a.native/'palm-profile-viewer-v2/poses.jsonl',a.native/'rig-pixels-viewer-v2/cameras.txt',a.native/'rig-generic-v1/generic.fhrig',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig',a.session/'frame/live-views.txt',a.session/'relay-trace.bin',a.session/'sampler-trace.bin',a.native/'rig-bound-step-v1/output.txt']},output_sha256=sha(a.run/'output.txt'),
  limitations=['Closed-loop candidate history but frozen detector/association observations. Baseline exactly matches earlier stage fixture.', 'Current palm centroid adjusts translation; previous orientation and all joint angles are head compensated. Fresh stereo must continue through intervening captures;100ms maximum seed age.', 'Same final pixel objective and baseline fresh-angle prior select between complete fits. Lower cost is not proof of plausible pose.', 'Single host CPU batch, additional fit cost excludes capture/neural/rendering and has not run on Frame.', 'No visibility, driver, input confidence or default-path changes; latest143912Zexcluded.'])
 (a.run/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(metrics=metrics,review=review),indent=2))
if __name__=='__main__':main()
