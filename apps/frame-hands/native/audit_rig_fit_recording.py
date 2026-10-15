"""Cold-start generic-rig audit on saved stereo observations. Not physical accuracy or calibration."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import read_audit,load_calibration
from compare_replay import source_hands,COMMON,SEGMENTS
from stereo_uncertainty import sensitivity
from audit_native_lengths import sha,stats

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('session','upstream','profile','executable','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False)
 if '143912Z' in str(a.session):raise ValueError('Accepted holdout excluded')
 raw=source_hands(a.session);audits=read_audit(a.session/'frame/live-views.txt');cams=load_calibration(a.session/'frame/snapshot/device/persist')
 rest=np.loadtxt(a.profile,skiprows=21)[:,:3];items=[];rows=[];skips={}
 ri=np.array([x for x,y in COMMON]);mi=np.array([y for x,y in COMMON]);palm_r=[5,8,11,14,17];palm_m=[0,5,9,13,17]
 for stamp,ob in sorted(audits.items()):
  for ident,h in raw.get(ob['mono_ns'],{}).items():
   views=[v for v in ob['views'] if v['id']==ident and v['camera'] in cams]
   if not h['stereo'] or len(views)!=2 or len({v['camera'] for v in views})!=2:continue
   try:x,cov,info=sensitivity(cams,views)
   except ValueError as e:skips[str(e)]=skips.get(str(e),0)+1;continue
   # Proper rigid alignment using only current wrist and MCP observations, no truth pose.
   anatomy=rest.copy();anatomy[:,0]*=(-1 if h['side'] else 1);X=anatomy[palm_r];Y=x[palm_m];xc=X.mean(0);yc=Y.mean(0)
   U,S,Vt=np.linalg.svd((X-xc).T@(Y-yc))
   if S[1]<1e-3:skips['degenerate_palm']=skips.get('degenerate_palm',0)+1;continue
   R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T;t=yc-R@xc;q=Rotation.from_matrix(R).as_quat()
   row=[h['side'],q[3],*q[:3],*t,*np.zeros(20),30,3.,.01]
   points=np.zeros((21,3));points[ri]=x[mi];lower=np.tile(np.eye(3),(21,1,1));lower[ri]=np.linalg.cholesky(cov[mi]*9.)
   for j in range(21):row += [int(j in ri),*points[j],*lower[j][np.tril_indices(3)]]
   rows.append(row);items.append(dict(stamp=stamp,id=ident,side=h['side'],x=x,views=views))
 source=a.output/'observations.txt';np.savetxt(source,rows,fmt='%.17g')
 subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str(source.resolve()),str((a.output/'fitted.txt').resolve())],check=True,capture_output=True,text=True,timeout=90)
 out=np.loadtxt(a.output/'fitted.txt')
 if out.shape!=(len(items),99) or not np.isfinite(out).all() or not np.all(out[:,0]==1):raise ValueError('Incomplete fits')
 fitted=out[:,36:].reshape(-1,21,3);metrics={k:[] for k in ('raw_tip_reprojection_px','fit_tip_reprojection_px','tip_reprojection_change_px','max_correction_mm','observed_rmse_mm','fit_us','palm_width_ratio')};better=worse=guard_reject=behind=0;ratios=[]
 with (a.output/'poses.jsonl').open('x') as poses:
  for i,item in enumerate(items):
   x=item['x'];y=fitted[i];es=[[],[]]
   for view in item['views']:
    cam=cams[view['camera']]
    for k,points in enumerate((x[mi],y[ri])):
     eye=cam.world_to_eye(points)
     if np.any(eye[:,2]<=0):es[k]=None;break
     es[k].append(np.linalg.norm(cam.eye_to_window(eye)-view['points'][mi],axis=1))
    if any(v is None for v in es):break
   change=None;correction=float(np.max(np.linalg.norm(x[mi]-y[ri],axis=1)));metrics['max_correction_mm'].append(correction);metrics['observed_rmse_mm'].append(out[i,6]);metrics['fit_us'].append(out[i,7])
   metrics['palm_width_ratio'].append(float(np.linalg.norm(x[5]-x[17])/np.linalg.norm(rest[8]-rest[17])))
   ratios.append([np.linalg.norm(x[c]-x[d])/np.linalg.norm(rest[a]-rest[b]) for a,b,c,d in SEGMENTS])
   if any(v is None for v in es):behind+=1
   else:
    es=np.array(es);tips=np.isin(mi,[4,8,12,16,20]);e0,e1=es[:,:,tips].mean(axis=(1,2));change=float(e1-e0)
    metrics['raw_tip_reprojection_px'].append(float(e0));metrics['fit_tip_reprojection_px'].append(float(e1));metrics['tip_reprojection_change_px'].append(change);better+=change< -5;worse+=change>5
    guard_reject+=int(not(np.all(es[1].mean(1)<=es[0].mean(1)+2) and correction<=50))
   poses.write(json.dumps(dict(raw_ns=item['stamp'],source_id=item['id'],side=item['side'],raw_head_mm=x.tolist(),fit_rig_head_mm=y.tolist(),tip_reprojection_change_px=change,max_correction_mm=correction))+'\n')
 report=dict(scope=__doc__,session=a.session.name,hands=len(items),skips=skips,behind_camera=behind,tip_reprojection_better_over_5px=int(better),tip_reprojection_worse_over_5px=int(worse),would_fail_previous_reprojection_correction_guard=int(guard_reject),metrics={k:stats(v) for k,v in metrics.items()},median_raw_to_generic_length_ratio_per_segment=np.median(ratios,axis=0).tolist(),parameters=dict(pixel_sigma_assumed=3,iterations=30,huber=3,angle_prior=.01,initialization='Current wrist and four MCP rigid Kabsch; zero finger angles, no truth or temporal seed'),provenance=dict(profile_sha256=sha(a.profile),executable_sha256=sha(a.executable),source_sha256={n:sha(BASE/n) for n in ('fit_hand_rig.h','hand_rig.h','audit_rig_fit_recording.py','stereo_uncertainty.py')},inputs={n:sha(a.session/n) for n in ('relay-trace.bin','frame/live-views.txt')}),limitations=['Uses generic anatomy, not personal calibration; all frames independently cold-started.','Detector-derived reprojection is agreement with fitting inputs, not independent physical accuracy.','Source identity retained; no visibility, crossings, holds, filtering or integration changes.','Rejected cases are only audit counters; no new output or fallback policy implemented.','Host run; no camera, new recording or wearer test.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
