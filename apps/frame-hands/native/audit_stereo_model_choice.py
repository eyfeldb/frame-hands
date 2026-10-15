"""Independent saved-pixel choice checks and pose-transition review, no live tracking."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from run_tracker_rig_check import sha,stats
from audit_hand_scale_rig import SELECTION
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration,unproject
from compare_replay import SEGMENTS

def inputs(path):
 result=[];stamp=None
 for line in path.read_text().splitlines():
  f=line.split()
  if f[0]=='set':stamp=int(f[1])
  else:result.append((stamp,int(f[1]),int(f[2]),np.array(f[3:],float).reshape(21,2,3)))
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','rig','models','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));meta=json.loads((a.run/'result.json').read_text());models=json.loads((a.models/'result.json').read_text());assert sha(a.rig/'result.json')==meta['rig_result_sha256']
 cams=load_calibration(a.session/'frame/snapshot/device/persist');names=['slam_left','slam_right'];origins=[cams[n].camera_to_world_xf[:3,3] for n in names];baseline=origins[1]-origins[0]
 def geometry(pixels):
  if not np.all(pixels[:20,:,0]):return [-1.,-1.]
  points=[pixels[:20,v,1:] for v in (0,1)];rays=[unproject(cams[n],p)@cams[n].camera_to_world_xf[:3,:3].T for n,p in zip(names,points)];dist=[]
  for v in (0,1):
   normal=np.cross(baseline,rays[1-v]);normal/=np.linalg.norm(normal,axis=1)[:,None];planar=rays[v]-normal*np.sum(normal*rays[v],axis=1)[:,None]
   projected=cams[names[v]].eye_to_window(cams[names[v]].world_to_eye(origins[v]+planar*1000));dist.append(np.linalg.norm(projected-points[v],axis=1))
  z2=np.mean(np.array(dist)**2,axis=0)/18;chains=[[5,6,7,0],[8,9,10,1],[11,12,13,2],[14,15,16,3],[17,18,19,4]]
  return [float(np.sqrt(z2.mean())),float(max(np.sqrt(z2[c].mean()) for c in chains))]
 full=inputs(a.rig/'full-input.txt');lite=inputs(a.rig/'lite-input.txt');selected=inputs(a.run/'first-input.txt');decisions=[f.split() for f in (a.run/'first-decisions.txt').read_text().splitlines()];presence=np.loadtxt(a.run/'presence.txt')[:,2:]
 records=meta['records'];assert len(full)==len(lite)==len(selected)==len(decisions)==len(records);max_geometry_error=0.;choices=[]
 for i,(f,l,s,d,m) in enumerate(zip(full,lite,selected,decisions,records)):
  assert f[:3]==l[:3]==s[:3]==(m['raw_ns'],m['source'],m['side']);assert list(map(int,d[:2]))==list(f[:2]);choice=int(d[2]);choices.append(choice);reported=np.array(d[3:],float).reshape(4,3);expected=[]
  for option in range(4):
   pixels=f[3].copy();ps=presence[i,:2].copy()
   for v in (0,1):
    if option & (2 if v==0 else 1):pixels[:,v]=l[3][:,v];ps[v]=presence[i,2+v]
   g=geometry(pixels);max_geometry_error=max(max_geometry_error,float(np.max(abs(np.array(g)-reported[option,:2]))));assert np.allclose(g,reported[option,:2],atol=1e-4,rtol=0)
   eligible=bool(g[1]>=0 and g[1]<=3 and min(ps)>=.8);assert eligible==bool(reported[option,2]);expected.append((g[0],eligible))
   if option==choice:np.testing.assert_array_equal(pixels,s[3])
  wanted=0
  if not 0<=reported[0,1]<=3:
   candidates=[o for o in range(1,4) if expected[o][1]]
   if candidates:wanted=min(candidates,key=lambda o:expected[o][0])
  assert choice==wanted
 b=np.loadtxt(a.rig/'full-first.txt');c=np.loadtxt(a.run/'first-fitted.txt');assert b.shape==c.shape==(len(records),198);np.testing.assert_array_equal(b[:,:4],c[:,:4])
 if meta.get('independent_history'):
  inactive=np.array(choices)==0;np.testing.assert_array_equal(b[inactive],c[inactive])
  assert (a.run/'first-baseline.txt').read_bytes()==(a.rig/'full-first.txt').read_bytes()
 def local_shape(fit):return Rotation.from_quat(fit[2:6][[1,2,3,0]]).inv().apply(fit[29:].reshape(21,3)[:20]-fit[6:9])
 metrics=defaultdict(dict);steps={};record_lookup={m['case']:i for i,m in enumerate(records)}
 for name,data in [('full',b),('fallback',c)]:
  previous={};window_values=defaultdict(lambda:defaultdict(list));allsteps={}
  for i,m in enumerate(records):
   w=window_values[m['window']];w['selected'].append(choices[i]!=0);w['initialized'].append(bool(data[i,3]));key=(m['window'],m['source'],m['side']);old=previous.get(key);fit=data[i,14:106]
   if not data[i,3]:previous.pop(key,None);continue
   w['residual_px'].append(fit[1])
   if old and 0<m['raw_ns']-old[0]<=100000000:
    angle=float(np.max(abs(fit[9:29]-old[1][9:29])));wrist=float(np.linalg.norm(fit[6:9]-old[1][6:9]));w['angle_step'].append(angle);w['wrist_step'].append(wrist);shape=float(np.max(np.linalg.norm(local_shape(fit)-local_shape(old[1]),axis=1)));w['local_shape_step_mm'].append(shape);allsteps[i]=(angle,wrist,old[2],shape)
   previous[key]=(m['raw_ns'],fit,i)
  steps[name]=allsteps
  for window,v in window_values.items():metrics[window][name]=dict(initialized=sum(v['initialized']),selected=sum(v['selected']),large_angles=sum(x>.5 for x in v['angle_step']),**{k:stats(v[k]) for k in ('residual_px','angle_step','wrist_step','local_shape_step_mm')})
 ranked=sorted([dict(index=i,case=records[i]['case'],previous_case=records[v[2]]['case'],full_angle=steps['full'][i][0],fallback_angle=v[0],extra_angle=v[0]-steps['full'][i][0],full_wrist=steps['full'][i][1],fallback_wrist=v[1],full_shape_mm=steps['full'][i][3],fallback_shape_mm=v[3],extra_shape_mm=v[3]-steps['full'][i][3]) for i,v in steps['fallback'].items()],key=lambda x:x['extra_angle'],reverse=True)
 selection=list(SELECTION)
 shape_ranked=sorted(ranked,key=lambda r:r['extra_shape_mm'],reverse=True)
 for row in ranked[:2]+shape_ranked[:1]:
  for case in (row['previous_case'],row['case']):
   if case not in selection:selection.append(case)
 outdir=a.run/'review';outdir.mkdir(exist_ok=False);reviews=[]
 for case in selection:
  i=record_lookup[case];rows=[r for r in models['records'] if r['case']==case];canvas=np.full((750,960,3),24,np.uint8);cv2.putText(canvas,case+f' / choice {choices[i]}',(5,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
  for v,r in enumerate(rows):
   gray=np.fromfile(a.models/'inputs'/(str(r['index'])+'.gray'),np.uint8).reshape(224,224);inv=cv2.invertAffineTransform(np.array(r['matrix']));cam=cams[r['camera']]
   for col,(label,data) in enumerate([('image',None),('Full',b),('fallback',c)]):
    pic=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);title=label
    if data is not None and data[i,3]:
     fit=data[i,14:106];xyz=fit[29:].reshape(21,3);uv=cam.eye_to_window(cam.world_to_eye(xyz));pts=np.c_[uv,np.ones(21)]@inv.T
     for x,y,_,_ in SEGMENTS:cv2.line(pic,tuple(np.rint(pts[x]).astype(int)),tuple(np.rint(pts[y]).astype(int)),(60,230,70) if col==2 else (230,170,60),1,cv2.LINE_AA)
     for j in (6,8,11,14,17):cv2.line(pic,tuple(np.rint(pts[5]).astype(int)),tuple(np.rint(pts[j]).astype(int)),(130,130,130),1,cv2.LINE_AA)
     for tag,j in zip('IMRL',(1,2,3,4)):cv2.putText(pic,tag,tuple(np.rint(pts[j]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
     title+=f' {fit[1]:.1f}px'
    canvas[85+330*v:385+330*v,col*320:col*320+300]=cv2.resize(pic,(300,300));cv2.putText(canvas,r['camera']+' '+title,(col*320+3,72+330*v),cv2.FONT_HERSHEY_SIMPLEX,.44,(240,240,240),1,cv2.LINE_AA)
  path=outdir/(case+'.png');assert cv2.imwrite(str(path),canvas);reviews.append(dict(case=case,choice=choices[i],image_sha256=sha(path)))
 report=dict(scope=__doc__,hands=len(records),selected=sum(x!=0 for x in choices),choices={str(i):choices.count(i) for i in range(4)},geometry_independent_max_error=max_geometry_error,all_choices_and_pixels_verified=True,coverage_unchanged=True,inactive_poses_exact=meta.get('independent_history',False),metrics=metrics,worst_extra_angle=ranked[:8],worst_extra_shape=shape_ranked[:8],reviews=reviews,result_sha256=sha(a.run/'result.json'),auditor_sha256=sha(Path(__file__)),limitations=meta['limitations']+['Same anatomy but different observations: smaller residual is not independently measured pose accuracy.','Raw angle and wrist steps include real motion and fitting changes; output smoothing/whole-wrist history excluded.'])
 (a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(selected=report['selected'],geometry_error=max_geometry_error,metrics=metrics,worst=ranked[:3]),indent=2))
if __name__=='__main__':main()
