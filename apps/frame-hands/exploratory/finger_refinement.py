"""Offline bounded finger-angle hypotheses; fixed wrist, skeleton and identity.

Two current camera observations constrain 15 articulated landmarks. Image/model
agreement does not establish hidden anatomy or true pose. No live caller.
"""
import numpy as np
import torch
from scipy.optimize import least_squares
from replay_umetrack import COMMON

ARTICULATED_NATIVE=np.array([0,1,2,3,4,6,7,9,10,12,13,15,16,18,19])
MP_BY_NATIVE=dict(COMMON)
ARTICULATED_MP=np.array([MP_BY_NATIVE[i] for i in ARTICULATED_NATIVE])
PARAMETERS=dict(max_angle_change_rad=.35,angle_prior_sigma_rad=.15,
                pixel_sigma=3.,jacobian_step_rad=.0001,max_nfev=30,
                min_mean_improvement_fraction=.1,min_view_presence=.5)


class LandmarkBatch:
    """Pinned native forward kinematics in float64 for stable finite differences."""
    def __init__(self, hand_model, wrist, side):
        self.hand=hand_model._replace(**{k:v.double() for k,v in hand_model._asdict().items()
            if isinstance(v,torch.Tensor) and v.is_floating_point()})
        self.wrist=torch.as_tensor(wrist,dtype=torch.float64).clone()
        if self.wrist.shape!=(4,4) or not torch.isfinite(self.wrist).all() or side not in (0,1):
            raise ValueError('Invalid fixed wrist or side')
        if side:self.wrist[:,0]*=-1
        self.cache={}

    def __call__(self, angles):
        from lib.common.hand_skinning import skin_landmarks
        angles=np.asarray(angles,float)
        if angles.ndim!=2 or angles.shape[1]!=22 or not np.isfinite(angles).all():
            raise ValueError('Expected finite batched native joint angles')
        batch=len(angles)
        if batch not in self.cache:
            self.cache[batch]=self.hand._replace(**{k:v.unsqueeze(0).expand(batch,*v.shape)
                for k,v in self.hand._asdict().items() if isinstance(v,torch.Tensor)})
        with torch.inference_mode():
            return skin_landmarks(self.cache[batch],torch.as_tensor(angles,dtype=torch.float64),
                self.wrist.unsqueeze(0).expand(batch,4,4)).numpy()


def fit_fingers(hand_model, hand, cameras, observations):
    original=np.asarray(hand['joint_angles'],float)
    if original.shape!=(22,) or not np.isfinite(original).all():raise ValueError('Invalid native angles')
    limits=np.asarray(hand_model.joint_limits,dtype=float)
    if limits.shape!=(22,2) or not np.isfinite(limits).all():raise ValueError('Missing native joint limits')
    if len(observations)!=2 or len({v['camera'] for v in observations})!=2:
        return None,dict(reason='Requires two distinct current views')
    cams=[];targets=[]
    for view in observations:
        points=np.asarray(view['model_points'],float)
        if view['camera'] not in cameras or points.shape!=(21,2) or not np.isfinite(points).all() or not .5<=view['presence']<=1:
            return None,dict(reason='Invalid current observation')
        cams.append(cameras[view['camera']]);targets.append(points[ARTICULATED_MP])
    targets=np.asarray(targets)
    skin=LandmarkBatch(hand_model,hand['wrist_world_mm'],hand['side'])
    lo=np.maximum(original[:20]-.35,limits[:20,0]);hi=np.minimum(original[:20]+.35,limits[:20,1])
    if np.any(hi-lo<1e-8):return None,dict(reason='Native limits incompatible with correction bound')
    def errors(parameters):
        parameters=np.atleast_2d(parameters)
        angles=np.broadcast_to(original,(len(parameters),22)).copy();angles[:,:20]=parameters
        xyz=skin(angles)[:,ARTICULATED_NATIVE];rows=[]
        for cam,target in zip(cams,targets):
            eye=cam.world_to_eye(xyz)
            if not np.isfinite(eye).all() or np.any(eye[...,2]<=1):raise ValueError('Invalid fitted projection')
            rows.append(cam.eye_to_window(eye)-target)
        return np.stack(rows,axis=1)
    initial=errors(original[:20])[0]
    def residual_batch(parameters):
        parameters=np.atleast_2d(parameters)
        e=errors(parameters).reshape(len(parameters),-1)/3.
        return np.concatenate((e,(parameters-original[:20])/.15),axis=1)
    def residual(x):return residual_batch(x)[0]
    def jacobian(x):
        delta=np.eye(20)*.0001
        both=residual_batch(np.vstack((x+delta,x-delta)))
        return ((both[:20]-both[20:])/.0002).T
    start=np.clip(original[:20],lo+1e-10,hi-1e-10)
    solved=least_squares(residual,start,jac=jacobian,bounds=(lo,hi),loss='soft_l1',f_scale=1.,max_nfev=30)
    before=np.linalg.norm(initial,axis=2);after=np.linalg.norm(errors(solved.x)[0],axis=2)
    angles=original.copy();angles[:20]=solved.x
    debug=dict(success=bool(solved.success),nfev=solved.nfev,njev=solved.njev,
        before_mean_px=before.mean(1).tolist(),after_mean_px=after.mean(1).tolist(),
        before_tip_mean_px=before[:,:5].mean(1).tolist(),after_tip_mean_px=after[:,:5].mean(1).tolist(),
        max_angle_change_rad=float(np.max(np.abs(angles-original))),
        baseline_outside_native_limit_count=int(np.sum((original[:20]<limits[:20,0])|(original[:20]>limits[:20,1]))))
    if not solved.success:return None,dict(debug,reason='Optimizer did not converge within bound')
    if after.mean()>=before.mean()*.9:return None,dict(debug,reason='Less than ten percent articulated improvement')
    if np.any(after.mean(1)>before.mean(1)+1e-6):return None,dict(debug,reason='One camera worsened')
    return angles,dict(debug,reason='Hypothesis only')


def apply_angles(backend, hand, angles):
    return dict(hand,joint_angles=np.asarray(angles).tolist(),
        vertices_world_mm=backend.mesh(angles,hand['wrist_world_mm'],hand['side']).tolist(),
        hypothesis_only=True,finger_angles_refined=True)
