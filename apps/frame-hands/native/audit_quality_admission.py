"""Audit quality-gated display motion, lost visibility and raw-camera comparisons."""
import argparse,bisect,json,sys,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,capture_poses,load_calibration,Recording
from compare_replay import SEGMENTS
from audit_native_lengths import stats,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','native','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if '143912Z' in str(a.session):raise ValueError('Holdout excluded')
 sys.path.insert(0,str(a.upstream));data=json.loads((a.run/'events.json').read_text());events=data['events'];times=[e['now'] for e in events]
 frames=[json.loads(x) for x in (a.run/'enabled.json.frames.jsonl').read_text().splitlines()];disabled=[json.loads(x) for x in (a.run/'disabled.json.frames.jsonl').read_text().splitlines()]
 assert len(frames)==len(disabled)
 # Check actual capture lineage, not just the report's count of skipped proposals.
 good={(round(e['capture'],7),h['side'],h['source_id']) for e in events if e['valid'] for h in e['hands'] if h['fit_rmse_px']<=9}
 continuity=[];last={};intervals=[];active={};both=0;lost=0;gained=0;opaque_lost=0;rows=[[],[]];keys=[[],[]]
 for i,(f,d) in enumerate(zip(frames,disabled)):
  assert f['now']==d['now'] and f['side']==d['side'];assert d['values'][0]==d['values'][1];assert f['values'][0]==d['values'][0]
  side=f['side'];x,y=f['values'];now=f['now'];lost+=bool(x) and not y;gained+=bool(y) and not x;both+=bool(x) and bool(y);opaque_lost+=bool(x) and not y and x['opacity']>.9
  if y:
   assert (round(y['capture'],7),side,y['id']) in good
   assert 0<=y['age']<.170+1e-6 and abs(y['age']-(now-y['capture']))<1e-6
  if x and not y:
   if side not in active:active[side]=[now,now,i,i]
   else:active[side][1]=now;active[side][3]=i
  elif side in active:intervals.append(dict(side=side,start=active[side][0],end=active[side][1],first=active[side][2],last=active[side][3]));active.pop(side)
  for k,v in enumerate((x,y)):
   key=(side,k);prev=last.get(key)
   if v:
    if prev and prev[1]['id']==v['id']:
     j,old=prev;dq=Rotation.from_quat(np.array(old['orientation'])[[1,2,3,0]]).inv()*Rotation.from_quat(np.array(v['orientation'])[[1,2,3,0]])
     continuity.append(dict(index=i,previous=j,variant=k,wrist_step_deg=float(dq.magnitude()*180/np.pi),position_step_mm=float(np.linalg.norm(np.array(v['position'])-old['position'])*1000),angle_step_rad=float(np.max(abs(np.array(v['angles'])-old['angles'])))))
    last[key]=(i,v)
    q=Rotation.from_quat(np.array(v['orientation'])[[1,2,3,0]]).as_quat()[[3,0,1,2]];rows[side].append([side,*q,*(np.array(v['position'])*1000),*v['angles']]);keys[side].append((i,k))
   else:last.pop(key,None)
 for side,v in active.items():intervals.append(dict(side=side,start=v[0],end=v[1],first=v[2],last=v[3]))
 for x in intervals:x['duration']=x['end']-x['start']+1/90
 world={};evaluator=(a.run/'evaluate-rig.exe').resolve()
 for side in (0,1):
  inp=a.run/f'side{side}-poses.txt';out=a.run/f'side{side}-landmarks.txt';np.savetxt(inp,rows[side],fmt='%.17g')
  subprocess.run([str(evaluator),str((a.native/f'palm-profile-v1/side{side}.fhrig').resolve()),str(inp.resolve()),str(out.resolve())],check=True)
  points=np.loadtxt(out)[:,1:].reshape(-1,21,3)
  for key,cloud in zip(keys[side],points):world[key]=cloud
 for r in continuity:r['max_tip_step_mm']=float(np.linalg.norm(world[(r['index'],r['variant'])][:5]-world[(r['previous'],r['variant'])][:5],axis=1).max())
 delta={}
 for r in continuity:delta.setdefault(r['index'],{})[r['variant']]=r
 paired=[(i,r) for i,r in delta.items() if len(r)==2 and r[0]['previous']==r[1]['previous']]
 worst_improvement=max(paired,key=lambda x:x[1][0]['wrist_step_deg']-x[1][1]['wrist_step_deg'])
 worst_difference=max((i for i,f in enumerate(frames) if all(f['values'])),key=lambda i:np.linalg.norm(world[i,1]-world[i,0],axis=1).max())
 selected=[]
 for rank,gap in enumerate(sorted(intervals,key=lambda x:x['duration'],reverse=True)[:3],1):selected.append((f'loss-{rank}',gap['first']+2*((gap['last']-gap['first'])//4)))
 selected += [('largest-wrist-jump-reduction',worst_improvement[0]),('largest-pose-difference',worst_difference)]
 audit=read_audit(a.session/'frame/live-views.txt');heads=capture_poses(a.session);cams=load_calibration(a.session/'frame/snapshot/device/persist');rec=Recording(a.session/'frame/sets.bin')
 import cv2
 review=[]
 for label,index in selected:
  if label.startswith('loss-'):assert frames[index]['values'][0] and not frames[index]['values'][1]
  f=frames[index];ei=bisect.bisect_right(times,f['now'])-1;event=events[ei];raw=event['raw_ns'];h=heads[audit[raw]['mono_ns']];sec=(raw-rec.index[0][0])/1e9;images=list(rec.frames(sec-1e-7,sec+1e-7));assert len(images)==1 and images[0][1]==raw
  points=[None if not v else (world[index,k]-h[:3,3])@h[:3,:3] for k,v in enumerate(f['values'])]
  canvas=np.full((760,1200,3),24,np.uint8);cv2.putText(canvas,f"{label} | capture {sec:.3f}s side{f['side']} | visible {bool(f['values'][0])}/{bool(f['values'][1])}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
  for vi,name in enumerate(('slam_left','slam_right')):
   image=images[0][2][name][0];cam=cams[name];uv=[None if cloud is None else cam.eye_to_window(cam.world_to_eye(cloud)) for cloud in points];alluv=np.vstack([x for x in uv if x is not None]);center=(alluv.min(0)+alluv.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(alluv,axis=0).max()+100)));x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col in range(3):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if col and uv[col-1] is not None:
     pp=uv[col-1]-[x,y]
     for pi,qi,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(pp[pi]).astype(int)),tuple(np.rint(pp[qi]).astype(int)),(240,220,40) if col==1 else (70,230,60),2,cv2.LINE_AA)
    elif col:cv2.putText(picture,'HIDDEN',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(100,100,255),2)
    canvas[65+vi*340:395+vi*340,col*400:col*400+330]=cv2.resize(picture,(330,330));cv2.putText(canvas,name+' / '+('image','control','quality-gate')[col],(col*400+3,53+vi*340),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,index=index,seconds=sec,display_time=f['now'],side=f['side'],visible=[bool(x) for x in f['values']],opacity=[v.get('opacity') for v in f['values']],source_capture=[v.get('capture') for v in f['values']],delivered_rmse=[x['fit_rmse_px'] for x in event['hands'] if x['side']==f['side']],image_sha256=sha(a.run/(label+'.png'))))
 metrics=dict(total_samples=len(frames),both_visible=both,extra_hidden=lost,extra_visible=gained,extra_hidden_with_control_opacity_over09=opaque_lost,extra_hidden_intervals=len(intervals),longest_extra_hidden_seconds=max(x['duration'] for x in intervals),capture_lineage_verified=True,disabled_exact=True)
 metrics['motion']={key:[stats([r[key] for r in continuity if r['variant']==k]) for k in (0,1)] for key in ('wrist_step_deg','position_step_mm','angle_step_rad','max_tip_step_mm')}
 report=dict(metrics=metrics,review=review,raw_replay=json.loads((a.run/'enabled.json').read_text()),input_sha256={p.name:sha(p) for p in [a.run/'events.json',a.run/'enabled.json.frames.jsonl',a.run/'disabled.json.frames.jsonl',a.run/'manifest.json']},source_sha256=sha(Path(__file__)),evaluator_sha256=sha(evaluator),limitations=['Fit residual is agreement with the detector used for fitting, not independent optical truth or calibrated confidence.', 'Assumed50ms delivery and90Hz display; additional missing render samples are not live dropouts. No new cameras/XR/native timing.', 'Motion statistics include only same-ID consecutive visible samples, but hiding difficult periods changes the evaluated population. Lower motion alone is not improvement.', 'Overlays use exact latest delivered recorded capture and its HMD transform; not true display-time images. Selected extremes diagnose failure, not representative quality.', 'Default tracker/viewer unchanged. Threshold9px is a single diagnostic setting, not calibrated production policy.'])
 (a.run/'intervals.json').write_text(json.dumps(intervals,indent=2)+'\n',encoding='utf-8');(a.run/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(metrics=metrics,review=review),indent=2))
if __name__=='__main__':main()
