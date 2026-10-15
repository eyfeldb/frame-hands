"""Independent saved-reference recovery stress test; not camera tracking ground truth."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('reference','profile','executable','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 truth=np.load(a.reference/'reference.npz')['expected'][4:68]
 poses=np.loadtxt(a.reference/'poses.txt')[4:68];limits=np.loadtxt(a.profile,skiprows=1,max_rows=20)[:,6:8]
 rest=np.loadtxt(a.profile,skiprows=21)[:,:3]
 rng=np.random.default_rng(50108);seeds=poses.copy();seeds[:,8:]=np.clip(seeds[:,8:]+rng.normal(0,.15,(64,20)),*limits.T)
 seeds[:,5:8]+=rng.normal(0,5,(64,3))
 r=Rotation.from_rotvec(rng.normal(0,.05,(64,3)))*Rotation.from_quat(poses[:,[2,3,4,1]])
 seeds[:,1:5]=r.as_quat()[:,[3,0,1,2]]
 rows=[];metadata=[];targets=[];measurements=[];mask_all=[]
 def add(name,i,points,mask,chol,huber=3.):
  row=[*seeds[i],30,huber,.01]
  for j in range(21):row += [int(mask[j]),*points[j],*chol[j][np.tril_indices(3)]]
  rows.append(row);metadata.append(dict(group=name,reference_index=i+4));targets.append(truth[i]);measurements.append(points);mask_all.append(mask)
 for i in range(64):
  mask=np.ones(21,bool);mask[20]=False # Weighted palm center has no original MediaPipe observation.
  chol=np.tile(np.eye(3)*3.,(21,1,1));clean=truth[i]
  add('clean',i,clean,mask,chol)
  noisy=clean+rng.normal(0,3,(21,3));add('noise3',i,noisy,mask,chol)
  corrupt=noisy.copy();corrupt[i%5]+=rng.normal(0,45,3)
  add('outlier_robust',i,corrupt,mask,chol);add('outlier_squared',i,corrupt,mask,chol,1e6)
  missing=mask.copy();missing[[0,6,7]]=False # Thumb: all remaining thumb observations absent.
  add('missing_thumb',i,noisy,missing,chol)
  alt=noisy.copy();alt[~missing]=1e6;add('missing_thumb_dummy',i,alt,missing,chol)
  covariance=np.diag([1.,1.,144.]);rot=Rotation.random(random_state=rng).as_matrix();L=np.linalg.cholesky(rot@covariance@rot.T)
  anisotropic=clean+rng.normal(size=(21,3))@L.T
  add('anisotropic',i,anisotropic,mask,np.tile(L,(21,1,1)))
  # Cold-start with current wrist/MCP data only, separately from near-truth tests.
  previous=seeds[i].copy();X=rest[[5,8,11,14,17]].copy();X[:,0]*=(-1 if poses[i,0] else 1);Y=clean[[5,8,11,14,17]]
  xc=X.mean(0);yc=Y.mean(0);U,S,Vt=np.linalg.svd((X-xc).T@(Y-yc));R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T
  q=Rotation.from_matrix(R).as_quat();seeds[i,1:5]=q[[3,0,1,2]];seeds[i,5:8]=yc-R@xc;seeds[i,8:]=0
  add('cold_start_clean',i,clean,mask,chol);seeds[i]=previous
 # Missing data must not synthesize a detection; deliberately keep a valid supplied seed.
 add('absent',0,truth[0],np.zeros(21,bool),np.tile(np.eye(3),(21,1,1)))
 add('five_points',0,truth[0],np.arange(21)<5,np.tile(np.eye(3),(21,1,1)))
 source=a.output/'observations.txt';np.savetxt(source,rows,fmt='%.17g')
 result=subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str(source.resolve()),str((a.output/'fitted.txt').resolve())],capture_output=True,text=True,timeout=90)
 if result.returncode:raise RuntimeError(result.stderr)
 out=np.loadtxt(a.output/'fitted.txt');targets=np.array(targets);measurements=np.array(measurements);mask_all=np.array(mask_all)
 if out.shape!=(len(rows),99) or not np.isfinite(out).all():raise ValueError('Incomplete or nonfinite fitter output')
 fitted=out[:,36:].reshape(-1,21,3)
 if not (out[:,5]<=out[:,4]+1e-9).all():raise ValueError('Fit increased objective')
 if not ((out[:,16:36]>=limits[:,0]-1e-8)&(out[:,16:36]<=limits[:,1]+1e-8)).all():raise ValueError('Joint range violated')
 stats={};groups=np.array([m['group'] for m in metadata])
 for group in dict.fromkeys(groups):
  select=groups==group;error=np.linalg.norm(fitted[select]-targets[select],axis=2);raw=np.linalg.norm(measurements[select]-targets[select],axis=2);visible=mask_all[select]
  stats[group]=dict(cases=int(select.sum()),pose_rmse_mm=float(np.sqrt(np.mean(error**2))),point_error_p95_mm=float(np.percentile(error,95)),observed_pose_rmse_mm=float(np.sqrt(np.mean(error[visible]**2))) if visible.any() else None,observation_rmse_mm=float(np.sqrt(np.mean(raw[visible]**2))) if visible.any() else None,iterations_median=float(np.median(out[select,2])),runtime_us_median=float(np.median(out[select,7])))
 if stats['cold_start_clean']['point_error_p95_mm']>.1:raise ValueError('Cold-start clean recovery failed')
 if stats['clean']['point_error_p95_mm']>.1:raise ValueError('Known-pose clean recovery failed: '+str(stats['clean']))
 if stats['outlier_robust']['pose_rmse_mm']>=stats['outlier_squared']['pose_rmse_mm']:raise ValueError('Robust loss did not improve aggregate outlier error')
 if stats['noise3']['observed_pose_rmse_mm']>=stats['noise3']['observation_rmse_mm']:raise ValueError('Fit did not improve aggregate noisy visible points')
 np.testing.assert_allclose(out[groups=='missing_thumb',8:],out[groups=='missing_thumb_dummy',8:],atol=0,rtol=0)
 if np.any(out[-2:,0]!=0) or np.any(out[-2:,2:4]!=0):raise ValueError('Insufficient observations treated as fitted')
 np.testing.assert_allclose(out[-2:,8:],np.concatenate((np.tile(seeds[0],(2,1)),out[-2:,36:]),axis=1),atol=1e-12)
 # Genuine invalid covariance and nonrigid seeds must be rejected, not silently sanitized.
 rejected=[]
 for name,index,value in [('negative_covariance',35,-1),('nonunit_seed',1,2),('invalid_side',0,2),('invalid_iterations',28,101)]:
  row=np.array(rows[0]);row[index]=value;input_path=a.output/(name+'.txt');np.savetxt(input_path,row[None],fmt='%.17g')
  proc=subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str(input_path.resolve()),str((a.output/(name+'.out')).resolve())],capture_output=True,text=True)
  if proc.returncode==0:raise ValueError('Invalid input accepted: '+name)
  rejected.append(dict(case=name,exit_code=proc.returncode,message=proc.stderr.strip()))
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 report=dict(scope=__doc__,seed=50108,cases=len(rows),groups=stats,invalid_cases=rejected,source_sha256={n:sha(BASE/n) for n in ['hand_rig.h','fit_hand_rig.h','evaluate_rig_fit.cpp','test_rig_fit.py']},profile_sha256=sha(a.profile),executable_sha256=sha(a.executable),reference_sha256=sha(a.reference/'reference.npz'),limitations=['Corruption tests use perturbed near-truth seeds; cold_start_clean uses rigid palm alignment and zero angles. No temporal tracking or reacquisition validation.','Truth and observations share generic anatomy; personal anatomical mismatch is untested.','Synthetic covariance and corruption; no camera or wearer accuracy claim.','Missing-thumb angles cannot be inferred from other fingers; output is seed/prior dependent.','Host timing only.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'cases.json').write_text(json.dumps(metadata)+'\n');np.savez_compressed(a.output/'reference.npz',truth=targets,measurement=measurements,mask=mask_all)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
