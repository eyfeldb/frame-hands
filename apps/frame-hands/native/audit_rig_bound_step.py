"""Audit opt-in joint-bound step against frozen synthetic and recorded controls."""
import argparse,json,sys,shutil
from pathlib import Path
import cv2
import numpy as np
from probe_monocular_bridge import BASE,RI,MI,NAMES
from frame_data import Recording,read_audit,load_calibration
from compare_replay import SEGMENTS
from audit_native_lengths import stats,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','native','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream))
 if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
 result=np.loadtxt(a.run/'output.txt');old=np.loadtxt(a.native/'rig-stage-v2/recorded-output.txt')
 records=[json.loads(x) for x in (a.native/'palm-profile-viewer-v2/poses.jsonl').read_text().splitlines()]
 if result.shape!=(len(records),267) or len(old)!=len(records) or not np.isfinite(result).all():raise ValueError('Incomplete result')
 assert np.array_equal(result[:,3],old[:,3]);assert np.array_equal(result[:,28:91],old[:,41:104])
 modes={'control':3,'bounded':91,'continued':179};metrics={}
 for name,offset in modes.items():
  delta=result[:,offset]-result[:,3]
  metrics[name]=dict(cost=stats(result[:,offset]),rmse_px=stats(result[:,offset+1]),host_us=stats(result[:,offset+4]),
                    objective_improved_over1=int((delta< -1).sum()),objective_worsened_over1=int((delta>1).sum()),
                    minimum_cost_delta=float(delta.min()),maximum_cost_delta=float(delta.max()))
  for side in (0,1):
   limits=np.loadtxt(a.native/f'palm-profile-v1/side{side}.fhrig',skiprows=1,max_rows=20)[:,6:8]
   angles=result[result[:,2]==side,offset+5:offset+25]
   assert ((angles>=limits[:,0]-1e-8)&(angles<=limits[:,1]+1e-8)).all()
 synthetic={};directory=a.native/'rig-pixels-synthetic-v7';groups=np.array(json.loads((directory/'groups.json').read_text()));truth=np.load(directory/'reference.npz')['truth']
 for method,control in [('direct','fitted.txt'),('staged','staged-fit.txt')]:
  candidate=np.loadtxt(a.run/f'synthetic-{method}.txt');baseline=np.loadtxt(directory/control)
  assert candidate.shape==baseline.shape==(450,99)
  assert np.all(candidate[:,5]<=candidate[:,4]+1e-9)
  assert np.all(candidate[-2:,0]==0)
  np.testing.assert_array_equal(candidate[groups=='missing_thumb',8:],candidate[groups=='missing_dummy',8:])
  synthetic[method]={}
  for group in dict.fromkeys(groups):
   selected=groups==group;values={}
   for label,out in [('control',baseline),('bounded',candidate)]:
    error=np.linalg.norm(out[selected,36:].reshape(-1,21,3)-truth[selected],axis=2)
    values[label]=dict(pose_rmse_mm=float(np.sqrt(np.mean(error**2))),point_error_p95_mm=float(np.percentile(error,95)))
   synthetic[method][group]=values
  assert synthetic[method]['clean']['bounded']['point_error_p95_mm']<.1
  if method=='staged':assert synthetic[method]['cold_clean']['bounded']['point_error_p95_mm']<.1
 displacement=np.max(np.linalg.norm(result[:,116:179].reshape(-1,21,3)-result[:,28:91].reshape(-1,21,3),axis=2),axis=1)
 selected=[('largest-cost-gain',int(np.argmin(result[:,91]-result[:,3]))),('largest-cost-regression',int(np.argmax(result[:,91]-result[:,3]))),('largest-pose-change',int(np.argmax(displacement)))]
 previous=json.loads((a.native/'actual-monocular-flow-v1/review.json').read_text())
 for label in ('first-supported','largest-step','largest-observed-error'):
  row=next(x['case'] for x in previous if x['label']==label)
  index=next(i for i,r in enumerate(records) if (r['raw_ns'],r['source_id'])==(row['prior_raw_ns'],row['source']))
  selected.append(('seed-'+label,index))
 audit=read_audit(a.session/'frame/live-views.txt');cameras=load_calibration(a.session/'frame/snapshot/device/persist');recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0];review=[]
 for label,index in selected:
  row=records[index];stamp=row['raw_ns'];seconds=(stamp-first)/1e9;frames=list(recording.frames(seconds-1e-7,seconds+1e-7));assert len(frames)==1 and frames[0][1]==stamp;images=frames[0][2]
  canvas=np.full((740,1500,3),24,np.uint8);cv2.putText(canvas,f"{label} | {seconds:.3f}s | side{row['side']} | max change {displacement[index]:.1f}mm",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
  for v,name in enumerate(NAMES):
   camera=cameras[name];image=images[name][0];view=next(x for x in audit[stamp]['views'] if x['id']==row['source_id'] and x['camera']==name)
   points={m:result[index,o+25:o+88].reshape(21,3) for m,o in modes.items()}
   uv={m:camera.eye_to_window(camera.world_to_eye(cloud)) for m,cloud in points.items()}
   combined=np.vstack([view['points'],*uv.values()]);center=(combined.min(0)+combined.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(combined,axis=0).max()+100)))
   x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col,mode in enumerate(('image','detector','control','bounded','continued')):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if mode=='detector':
     for point in view['points']-[x,y]:cv2.circle(picture,tuple(np.rint(point).astype(int)),2,(210,160,255),-1)
    elif mode in uv:
     projection=uv[mode]-[x,y];color={'control':(240,220,40),'bounded':(70,230,60),'continued':(40,150,255)}[mode]
     for i,j,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(projection[i]).astype(int)),tuple(np.rint(projection[j]).astype(int)),color,2,cv2.LINE_AA)
    canvas[60+v*340:360+v*340,col*300:(col+1)*300]=cv2.resize(picture,(300,300))
    cv2.putText(canvas,name+' / '+mode,(col*300+4,50+v*340),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,index=index,seconds=seconds,source=row['source_id'],side=row['side'],max_change_mm=float(displacement[index]),costs={m:float(result[index,o]) for m,o in modes.items()}))
 (a.run/'review.json').write_text(json.dumps(review,indent=2)+'\n')
 source=a.run/'source';source.mkdir()
 names=['rig_bound_step.h','fit_rig_pixels.h','fit_hand_rig.h','hand_rig.h','rig_stage.h','causal_rig_fit.h','test_rig_bound_step.cpp','probe_rig_bound_step.cpp','audit_rig_bound_step.py']
 for name in names:shutil.copy2(BASE/name,source/name)
 inputs=[a.native/'rig-stage-v2/recorded-input.txt',a.native/'rig-stage-v2/recorded-output.txt',a.native/'palm-profile-viewer-v2/poses.jsonl',a.native/'rig-generic-v1/generic.fhrig',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig',a.native/'rig-pixels-viewer-v2/cameras.txt',directory/'staged-input.txt',directory/'observations.txt',directory/'reference.npz']
 report=dict(scope=__doc__,cases=len(records),recorded=metrics,synthetic=synthetic,default_cost_and_landmarks_exact=True,
 maximum_candidate_displacement_mm=float(displacement.max()),review=review,source_sha256={name:sha(BASE/name) for name in names},input_sha256={str(p):sha(p) for p in inputs},
 executable_sha256={p.name:sha(p) for p in (a.run/'probe.exe',a.run/'probe-aarch64',a.run/'test.exe',a.run/'evaluate-bounds.exe')},
 limitations=['Frozen baseline angle history/selected seeds; this is not a closed-loop candidate replay.',
 'Candidate changes only local handling of already-active angle bounds. It is not SciPy TRF.',
 'Continuation control adds30iterations and resets damping; not a matched compute budget.',
 'Both-camera fitting residual is not independent physical accuracy. Synthetic shared anatomy is easier than real hands.',
 'Same-input 30iteration candidate has44objective regressions; lower cost alone is not a presentation/gesture admission gate.',
 'Host single sequential timing batch; ARM64 build only, no Frame execution or wearer acceptance.'])
 (a.run/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(recorded=metrics,review=review,synthetic_checks='clean/staged cold/absent/missing-dummy passed'),indent=2))


if __name__=='__main__':main()
