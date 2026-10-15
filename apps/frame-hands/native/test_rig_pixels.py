"""Calibrated projection equivalence and synthetic stereo pose recovery; no wearer accuracy claim."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE.parent/'exploratory'))
from frame_data import load_calibration
from rig_pixel_data import write_cameras,pixel_row,NAMES
from audit_native_lengths import sha
from stereo_uncertainty import sensitivity

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('reference','persist','upstream','profile','executable','projector','three-d-executable','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.upstream));a.output.mkdir(parents=True,exist_ok=False);cams=load_calibration(a.persist);write_cameras(cams,a.output/'cameras.txt')
 rng=np.random.default_rng(50109);left=cams[NAMES[0]];eye=rng.uniform([-400,-300,250],[400,300,900],(4096,3));eye[:3]=[[0,0,500],[1e-14,0,500],[0,1e-14,500]];head=eye@left.camera_to_world_xf[:3,:3].T+left.camera_to_world_xf[:3,3]
 head=head[np.logical_and.reduce([cams[n].world_to_eye(head)[:,2]>1 for n in NAMES])][:1024]
 np.savetxt(a.output/'head.txt',head,fmt='%.17g');subprocess.run([str(a.projector.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/'head.txt').resolve()),str((a.output/'pixels.txt').resolve())],check=True)
 expected=np.concatenate([cams[n].eye_to_window(cams[n].world_to_eye(head)) for n in NAMES],axis=1);actual=np.loadtxt(a.output/'pixels.txt');projection_error=float(np.max(np.abs(actual-expected)))
 if actual.shape!=(1024,4) or not np.isfinite(actual).all() or projection_error>1e-8:raise ValueError('Projection disagrees with calibrated reference')
 poses=np.loadtxt(a.reference/'poses.txt')[4:68];truth=np.load(a.reference/'reference.npz')['expected'][4:68].copy();forward=sum(cams[n].camera_to_world_xf[:3,2] for n in NAMES);forward/=np.linalg.norm(forward);origin=np.mean([cams[n].camera_to_world_xf[:3,3] for n in NAMES],axis=0);centers=origin+forward*550+rng.uniform([-40,-40,-50],[40,40,50],(64,3));truth+=centers[:,None]-poses[:,None,5:8];poses[:,5:8]=centers
 limits=np.loadtxt(a.profile,skiprows=1,max_rows=20)[:,6:8];rest=np.loadtxt(a.profile,skiprows=21)[:,:3];seeds=poses.copy();seeds[:,8:]=np.clip(seeds[:,8:]+rng.normal(0,.15,(64,20)),*limits.T);seeds[:,5:8]+=rng.normal(0,5,(64,3));q=(Rotation.from_rotvec(rng.normal(0,.05,(64,3)))*Rotation.from_quat(poses[:,[2,3,4,1]])).as_quat();seeds[:,1:5]=q[:,[3,0,1,2]]
 rows=[];groups=[];targets=[]
 def add(name,i,uv,mask,seed=None,huber=3.):rows.append(pixel_row(seeds[i] if seed is None else seed,uv,mask,huber=huber));groups.append(name);targets.append(truth[i])
 for i in range(64):
  uv=np.array([cams[n].eye_to_window(cams[n].world_to_eye(truth[i])) for n in NAMES]);mask=np.ones((2,21),bool);mask[:,20]=False;add('clean',i,uv,mask)
  noisy=uv+rng.normal(0,3,uv.shape);add('noise3px',i,noisy,mask);bad=noisy.copy();bad[0,i%5]+=rng.normal(0,45,2);add('outlier_robust',i,bad,mask);add('outlier_squared',i,bad,mask,huber=1e6)
  missing=mask.copy();missing[:,[0,6,7]]=False;add('missing_thumb',i,noisy,missing);dummy=noisy.copy();dummy[~missing]=1e6;add('missing_dummy',i,dummy,missing)
  seed=poses[i].copy();X=rest[[5,8,11,14,17]].copy();X[:,0]*=(-1 if seed[0] else 1);Y=truth[i,[5,8,11,14,17]];xc=X.mean(0);yc=Y.mean(0);U,S,Vt=np.linalg.svd((X-xc).T@(Y-yc));R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T;q=Rotation.from_matrix(R).as_quat();seed[1:5]=q[[3,0,1,2]];seed[5:8]=yc-R@xc;seed[8:]=0;add('cold_clean',i,uv,mask,seed)
 add('absent',0,np.zeros((2,21,2)),np.zeros((2,21),bool));add('single_view',0,np.zeros((2,21,2)),np.array([[True]*21,[False]*21]))
 np.savetxt(a.output/'observations.txt',rows,fmt='%.17g');subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/'observations.txt').resolve()),str((a.output/'fitted.txt').resolve())],check=True,timeout=90)
 out=np.loadtxt(a.output/'fitted.txt');targets=np.array(targets);groups=np.array(groups)
 if out.shape!=(len(rows),99) or not np.isfinite(out).all() or np.any(out[:,5]>out[:,4]+1e-9):raise ValueError('Invalid fit output/cost')
 if not ((out[:,16:36]>=limits[:,0]-1e-8)&(out[:,16:36]<=limits[:,1]+1e-8)).all():raise ValueError('Joint bounds violated')
 stats={}
 for name in dict.fromkeys(groups):
  k=groups==name;err=np.linalg.norm(out[k,36:].reshape(-1,21,3)-targets[k],axis=2);stats[name]=dict(cases=int(k.sum()),pose_rmse_mm=float(np.sqrt(np.mean(err**2))),point_error_p95_mm=float(np.percentile(err,95)),runtime_us_median=float(np.median(out[k,7])))
 # Independently triangulate the SAME pixels, then use the existing3D solver as initialization.
 rows3d=[];skipped_points=0
 for row in rows:
  packed=np.array(row[31:]).reshape(21,2,4);r3d=list(row[:31])
  for j in range(21):
   present=bool(np.all(packed[j,:,0]));point=np.zeros(3);L=np.eye(3)
   if present:
    try:
     x,cov,_=sensitivity(cams,[dict(camera=n,points=packed[j,v,1:3][None]) for v,n in enumerate(NAMES)]);point=x[0];L=np.eye(3)*3. # Isotropic initialization; uncertainty enters final image fit.
    except ValueError:present=False;skipped_points+=1
   r3d.extend([int(present),*point,*L[np.tril_indices(3)]])
  rows3d.append(r3d)
 np.savetxt(a.output/'triangulated.txt',rows3d,fmt='%.17g')
 subprocess.run([str(a.three_d_executable.resolve()),str(a.profile.resolve()),str((a.output/'triangulated.txt').resolve()),str((a.output/'three-d-fit.txt').resolve())],check=True,timeout=90)
 initial=np.loadtxt(a.output/'three-d-fit.txt');staged_rows=np.array(rows);staged_rows[:,:28]=initial[:,8:36];np.savetxt(a.output/'staged-input.txt',staged_rows,fmt='%.17g')
 subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/'staged-input.txt').resolve()),str((a.output/'staged-fit.txt').resolve())],check=True,timeout=90)
 staged=np.loadtxt(a.output/'staged-fit.txt');staged_stats={}
 if staged.shape!=out.shape or not np.isfinite(staged).all() or np.any(staged[:,5]>staged[:,4]+1e-9):raise ValueError('Invalid staged output')
 for name in dict.fromkeys(groups):
  k=groups==name;err=np.linalg.norm(staged[k,36:].reshape(-1,21,3)-targets[k],axis=2);staged_stats[name]=dict(cases=int(k.sum()),pose_rmse_mm=float(np.sqrt(np.mean(err**2))),point_error_p95_mm=float(np.percentile(err,95)),runtime_us_median=float(np.median(staged[k,7]+initial[k,7])))
 if stats['clean']['point_error_p95_mm']>.1:raise ValueError('Direct nearby-seed clean recovery failed')
 for name in ('clean','cold_clean'):
  if staged_stats[name]['point_error_p95_mm']>.1:raise ValueError('Staged clean recovery failed: '+name+' '+str(staged_stats[name]))
 if stats['outlier_robust']['pose_rmse_mm']>=stats['outlier_squared']['pose_rmse_mm']:raise ValueError('Robust loss failed aggregate outlier comparison')
 np.testing.assert_allclose(out[groups=='missing_thumb',8:],out[groups=='missing_dummy',8:],atol=0,rtol=0)
 np.testing.assert_allclose(staged[groups=='missing_thumb',8:],staged[groups=='missing_dummy',8:],atol=0,rtol=0)
 if not ((staged[:,16:36]>=limits[:,0]-1e-8)&(staged[:,16:36]<=limits[:,1]+1e-8)).all():raise ValueError('Staged joint bounds violated')
 if np.any(staged[-2:,0]!=0) or np.any(staged[-2:,2:4]!=0):raise ValueError('Staged missing stereo treated as fitted')
 if staged_stats['outlier_robust']['pose_rmse_mm']>=staged_stats['outlier_squared']['pose_rmse_mm']:raise ValueError('Staged robust loss failed outlier comparison')
 if np.any(out[-2:,0]!=0) or np.any(out[-2:,2:4]!=0):raise ValueError('Missing stereo treated as fitted')
 rejects=[]
 for name,index,value in [('negative_sigma',34,-1),('invalid_side',0,2),('nonunit_wrist',1,2)]:
  row=np.array(rows[0]);row[index]=value;inp=a.output/(name+'.txt');np.savetxt(inp,row[None],fmt='%.17g');r=subprocess.run([str(a.executable.resolve()),str(a.profile.resolve()),str((a.output/'cameras.txt').resolve()),str(inp.resolve()),str((a.output/(name+'.out')).resolve())],capture_output=True,text=True)
  if r.returncode==0:raise ValueError('Invalid case accepted: '+name)
  rejects.append(dict(case=name,exit_code=r.returncode,message=r.stderr.strip()))
 np.savetxt(a.output/'behind.txt',(left.camera_to_world_xf[:3,3]-left.camera_to_world_xf[:3,2]*100)[None]);r=subprocess.run([str(a.projector.resolve()),str((a.output/'cameras.txt').resolve()),str((a.output/'behind.txt').resolve()),str((a.output/'behind.out').resolve())],capture_output=True,text=True)
 if r.returncode==0:raise ValueError('Behind camera accepted')
 rejects.append(dict(case='behind_camera',exit_code=r.returncode,message=r.stderr.strip()))
 report=dict(scope=__doc__,seed=50109,projection_cases=1024,projection_max_difference_px=projection_error,fit_cases=len(rows),groups=stats,staged_groups=staged_stats,skipped_triangulated_points=skipped_points,direct_cold_start_passes=stats["cold_clean"]["point_error_p95_mm"]<=.1,invalid_cases=rejects,provenance=dict(reference_sha256=sha(a.reference/'reference.npz'),profile_sha256=sha(a.profile),camera_sha256=sha(a.output/'cameras.txt'),executable_sha256=sha(a.executable),projector_sha256=sha(a.projector),three_d_executable_sha256=sha(a.three_d_executable),source_sha256={n:sha(BASE/n) for n in ('fit_rig_pixels.h','evaluate_rig_pixels.cpp','evaluate_rig_camera.cpp','rig_pixel_data.py','test_rig_pixels.py')}),limitations=['Synthetic shared generic anatomy; no personal shape accuracy.','Corruption tests use nearby initial poses; clean cold starts use exact current palm positions.','Single-view poses deliberately not fitted; no temporal/identity/visibility policy.','Robust loss groups both image residuals per landmark.','Direct cold-start quality gate remains failed; staged recovery separately validated.','Staged uses isotropic3mm3D initialization then image fitting; twice the maximum iterations.','Host tests only; no native timing or live acceptance.'])
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'reference.npz',truth=targets);(a.output/'groups.json').write_text(json.dumps(groups.tolist())+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
