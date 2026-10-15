"""Audit finger presentation at90Hz and review selected outputs against saved images."""
import argparse,json,sys,subprocess,shutil
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import Recording,read_audit,capture_poses,load_calibration
from compare_replay import SEGMENTS
from audit_native_lengths import stats,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','native','session','upstream','evaluator'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream))
 if '143912Z' in str(a.session):raise ValueError('Latest holdout excluded')
 data=json.loads((a.run/'native-rig-motion.json').read_text());frames=[json.loads(x) for x in (a.run/'native-rig-motion-result.json.frames.jsonl').read_text().splitlines()]
 audit=read_audit(a.session/'frame/live-views.txt');ordered=sorted(audit);heads=capture_poses(a.session);cameras=load_calibration(a.session/'frame/snapshot/device/persist')
 assert len(ordered)==len(data['events'])
 rows=[];previous={};step_angles=[[],[]];offsets=[];ages=[];event_index=-1
 byside=[[],[]];keys=[[],[]]
 for frame in frames:
  while event_index+1<len(data['events']) and data['events'][event_index+1]['now']<=frame['now']:event_index+=1
  x,y=frame['values'];side=frame['side']
  if not x or not y:previous.pop(side,None);continue
  assert x['position']==y['position'] and x['orientation']==y['orientation'] and x['id']==y['id'] and x['opacity']==y['opacity']
  values=[np.array(v['angles']) for v in (x,y)];offset=float(np.abs(values[1]-values[0]).max());offsets.append(offset)
  hold=max(y.get('finger_hold_ages',{}).values(),default=0.);ages.append(hold)
  item=dict(frame=frame,raw_ns=ordered[event_index],event_index=event_index,offset_rad=offset,hold_seconds=hold)
  prev=previous.get(side)
  if prev is not None and rows[prev]['frame']['values'][0]['id']==x['id']:
   item['previous']=prev
   for variant in (0,1):step_angles[variant].append(float(np.abs(values[variant]-np.array(rows[prev]['frame']['values'][variant]['angles'])).max()))
  previous[side]=len(rows)
  for variant,v in enumerate((x,y)):
   q=np.array(v['orientation']);normalized=Rotation.from_quat(q[[1,2,3,0]]).as_quat()[[3,0,1,2]]
   byside[side].append([side,*normalized,*(np.array(v['position'])*1000),*v['angles']]);keys[side].append((len(rows),variant))
  rows.append(item)
 world=np.zeros((len(rows),2,21,3))
 for side in (0,1):
  inp=a.run/f'side{side}-display-poses.txt';out=a.run/f'side{side}-display-landmarks.txt'
  np.savetxt(inp,byside[side],fmt='%.17g');subprocess.run([str(a.evaluator.resolve()),str((a.native/f'palm-profile-v1/side{side}.fhrig').resolve()),str(inp.resolve()),str(out.resolve())],check=True)
  points=np.loadtxt(out)[:,1:].reshape(-1,21,3)
  for key,cloud in zip(keys[side],points):world[key[0],key[1]]=cloud
 tip_steps=[[],[]]
 for i,row in enumerate(rows):
  row['tip_offset_mm']=float(np.linalg.norm(world[i,1,:5]-world[i,0,:5],axis=1).max())
  if 'previous' in row:
   for variant in (0,1):tip_steps[variant].append(float(np.linalg.norm(world[i,variant,:5]-world[row['previous'],variant,:5],axis=1).max()))
 selected=[('longest-hold',max(range(len(rows)),key=lambda i:rows[i]['hold_seconds'])),('largest-tip-difference',max(range(len(rows)),key=lambda i:rows[i]['tip_offset_mm'])),('largest-angle-difference',max(range(len(rows)),key=lambda i:rows[i]['offset_rad']))]
 recording=Recording(a.session/'frame/sets.bin');first=recording.index[0][0];review=[]
 for label,index in selected:
  row=rows[index];frame=row['frame'];stamp=row['raw_ns'];head=heads[audit[stamp]['mono_ns']];seconds=(stamp-first)/1e9
  frameset=list(recording.frames(seconds-1e-7,seconds+1e-7));assert len(frameset)==1 and frameset[0][1]==stamp;images=frameset[0][2]
  points=(world[index]-head[:3,3])@np.linalg.inv(head[:3,:3]).T
  canvas=np.full((740,1200,3),24,np.uint8);cv2.putText(canvas,f"{label} | capture {seconds:.3f}s | side{frame['side']} | hold {row['hold_seconds']*1000:.0f}ms",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
  for v,name in enumerate(('slam_left','slam_right')):
   camera=cameras[name];image=images[name][0];uv=[camera.eye_to_window(camera.world_to_eye(cloud)) for cloud in points]
   combined=np.vstack(uv);center=(combined.min(0)+combined.max(0))*.5;size=int(min(min(image.shape),max(300,np.ptp(combined,axis=0).max()+100)))
   x=int(np.clip(center[0]-size*.5,0,image.shape[1]-size));y=int(np.clip(center[1]-size*.5,0,image.shape[0]-size))
   for col,variant in enumerate((-1,0,1)):
    picture=cv2.cvtColor(image[y:y+size,x:x+size],cv2.COLOR_GRAY2BGR)
    if variant>=0:
     projected=uv[variant]-[x,y];color=(240,220,40) if variant==0 else (70,230,60)
     for i,j,_,_ in SEGMENTS:cv2.line(picture,tuple(np.rint(projected[i]).astype(int)),tuple(np.rint(projected[j]).astype(int)),color,2,cv2.LINE_AA)
    canvas[65+v*335:395+v*335,col*400:col*400+330]=cv2.resize(picture,(330,330))
    cv2.putText(canvas,name+' / '+['image','control','candidate'][col],(col*400+3,53+v*335),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
  cv2.imwrite(str(a.run/(label+'.png')),canvas)
  review.append(dict(label=label,index=index,raw_ns=stamp,seconds=seconds,display_now=frame['now'],side=frame['side'],source=frame['values'][0]['id'],hold_seconds=row['hold_seconds'],offset_rad=row['offset_rad'],tip_offset_mm=row['tip_offset_mm']))
 report=dict(scope=__doc__,visible_hand_frames=len(rows),changed_angle_frames=sum(x>1e-8 for x in offsets),maximum_joint_angle_offset_rad=stats(offsets),
 same_identity_max_joint_step_rad=[stats(x) for x in step_angles],joint_steps_over_half_radian=[sum(v>.5 for v in x) for x in step_angles],
 same_identity_max_tip_step_mm=[stats(x) for x in tip_steps],max_tip_offset_mm=stats([r['tip_offset_mm'] for r in rows]),
 positive_hold_age_seconds=stats([x for x in ages if x>0]),review=review,
 source_sha256=sha(Path(__file__)),evaluator_sha256=sha(a.evaluator),input_sha256={p.name:sha(p) for p in [a.run/'native-rig-motion.json',a.run/'native-rig-motion-result.json.frames.jsonl',a.native/'palm-profile-v1/side0.fhrig',a.native/'palm-profile-v1/side1.fhrig']},
 limitations=['Assumed50ms delivery/90Hz display; replay timestamps are not measured headset latency.',
 'Overlays compare rendered approximation to latest delivered capture, transformed with recorded head pose; not physical display ground truth.',
 'Reduced motion can be lag or freezing; neither lower step size nor joint bounds alone establishes plausibility.',
 'All C++ FK calls validate fixed-profile joint ranges; no visibility, identity or gesture confidence added.'])
 (a.run/'review.json').write_text(json.dumps(review,indent=2)+'\n');(a.run/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
