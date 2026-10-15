"""Experimental shared palm dimensions with per-frame poses. Offline calibration only.
Two dimensions are global; camera observations never change runtime anatomy per frame.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
PALM_RIG=np.array([5,8,11,14,17]);PALM_MP=np.array([0,5,9,13,17]);NAMES=('slam_left','slam_right')

def palm_points(rest,shape,side):
 p=np.array(rest,float)[PALM_RIG].copy();p[:,0]*=shape[0]*(-1 if side else 1);p[:,2]*=shape[1];return p

def calibrate(rest,side,cameras,pixels,stereo,fix_shape=False):
 uv=np.asarray(pixels,float);xyz=np.asarray(stereo,float);n=len(uv)
 if side not in (0,1) or n<3 or n>32 or uv.shape!=(n,2,5,2) or xyz.shape!=(n,5,3) or not np.isfinite(uv).all() or not np.isfinite(xyz).all():raise ValueError('Need3..32 finite stereo palm observations')
 X=palm_points(rest,[1,1],side);xc=X.mean(0);pose=[]
 for Y in xyz:
  yc=Y.mean(0);U,S,Vt=np.linalg.svd((X-xc).T@(Y-yc))
  if S[1]<1e-3:raise ValueError('Degenerate palm seed')
  R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T;pose.append([*Rotation.from_matrix(R).as_rotvec(),*(yc-R@xc)])
 initial=np.array(pose).reshape(-1);offset=0 if fix_shape else 2
 if not fix_shape:initial=np.r_[1.,1.,initial]
 def unpack(x):return (np.ones(2) if fix_shape else x[:2]),x[offset:].reshape(n,6)
 def predicted(x):
  shape,poses=unpack(x);X=palm_points(rest,shape,side);R=Rotation.from_rotvec(poses[:,:3]).as_matrix();points=np.einsum('nij,kj->nki',R,X)+poses[:,None,3:];result=[]
  for name in NAMES:
   eye=cameras[name].world_to_eye(points.reshape(-1,3))
   if np.any(eye[:,2]<=1):return None
   result.append(cameras[name].eye_to_window(eye).reshape(n,5,2))
  return np.stack(result,axis=1)
 def residual(x):
  pred=predicted(x)
  if pred is None:return np.full(n*20+offset,1e6)
  image=((pred-uv)/3).reshape(-1)
  return image if fix_shape else np.r_[image,np.log(x[:2])/.25]
 lower=initial.copy();upper=initial.copy()
 if offset:lower[:2]=[.65,.65];upper[:2]=[1.3,1.3]
 bounds=np.tile([1.,1.,1.,200.,200.,200.],n);lower[offset:]-=bounds;upper[offset:]+=bounds
 fit=least_squares(residual,initial,bounds=(lower,upper),jac='3-point',tr_solver='exact',x_scale='jac',loss='huber',f_scale=3,max_nfev=200,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 pred=predicted(fit.x)
 if pred is None:raise ValueError('Final palm behind camera')
 shape,poses=unpack(fit.x);errors=np.linalg.norm(pred-uv,axis=3)
 return dict(shape=shape.tolist(),poses=poses.tolist(),success=bool(fit.success),nfev=fit.nfev,cost=float(fit.cost),optimality=float(fit.optimality),residual_px=dict(median=float(np.median(errors)),p95=float(np.percentile(errors,95)),wrist_median=float(np.median(errors[:,:,0])),knuckle_median=float(np.median(errors[:,:,1:]))),near_shape_bound=bool(np.any(shape<.66)|np.any(shape>1.29)),profile_status='Experimental only; not personal calibration acceptance')
