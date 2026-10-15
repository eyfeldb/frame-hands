"""Independently check stereo uncertainty factors against saved camera geometry."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from audit_native_lengths import sha,stats
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration,unproject
MP_TO_RIG=[5,-1,6,7,0,8,9,10,1,11,12,13,2,14,15,16,3,17,18,19,4]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','session','upstream'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));cams=load_calibration(a.session/'frame/snapshot/device/persist');views=read_audit(a.run/'1-candidate/views.txt');rows=[];difference=[];stamp=None
 for line in (a.run/'1-candidate/rig.txt').read_text().splitlines():
  x=line.split()
  if x[0]=='frame':stamp=int(x[1])
  if x[0]!='uncertainty':continue
  assert len(x)==23;source=int(x[1]);actual=np.array(x[2:],float);assert np.isfinite(actual).all() and np.all((actual>=1)&(actual<=4));observed={v['camera']:v for v in views[stamp]['views'] if v['id']==source};assert set(observed)>={'slam_left','slam_right'}
  names=('slam_left','slam_right');origins=[cams[n].camera_to_world_xf[:3,3] for n in names];rays=[unproject(cams[n],observed[n]['points'])@cams[n].camera_to_world_xf[:3,:3].T for n in names];rays=[r/np.linalg.norm(r,axis=1)[:,None] for r in rays];baseline=origins[1]-origins[0];errors=[]
  for v,n in enumerate(names):
   normal=np.cross(baseline,rays[1-v]);normal/=np.linalg.norm(normal,axis=1)[:,None];planar=rays[v]-normal*np.sum(normal*rays[v],axis=1)[:,None];points=origins[v]+planar*1000;projected=cams[n].eye_to_window(cams[n].world_to_eye(points));errors.append(np.linalg.norm(projected-observed[n]['points'],axis=1))
  rms=np.sqrt((errors[0]**2+errors[1]**2)/2);factors=np.clip(rms/(3*np.hypot(3,3)),1,4);expected=np.ones(21)
  for m,r in enumerate(MP_TO_RIG):
   if r>=0:expected[r]=factors[m]
  difference.append(float(abs(expected-actual).max()));rows.append(dict(raw_ns=stamp,source=source,factors=actual.tolist()))
 assert len(rows)==3057 and max(difference)<1e-4
 values=np.array([r['factors'] for r in rows]);report=dict(rows=len(rows),hands_with_changed_weight=int(np.any(values>1,axis=1).sum()),points_with_changed_weight=int((values>1).sum()),points_at_cap=int((values==4).sum()),per_rig_joint_changes=(values>1).sum(axis=0).tolist(),factor_stats=stats(values.ravel()),max_python_native_factor_difference=max(difference),candidate_trace_sha256=sha(a.run/'1-candidate/rig.txt'),auditor_sha256=sha(Path(__file__)),limitations=['Factors describe measurement disagreement, not calibrated uncertainty or accuracy.','Raw learned2D coordinates rounded in audit; comparison tolerance1e-4 covers this plus inverse-camera numeric differences.'])
 (a.run/'uncertainty-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
