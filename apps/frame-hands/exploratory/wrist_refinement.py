"""Offline bounded rigid wrist correction; fixed scale, bones and finger angles.

Current stereo palm observations constrain six parameters. Reprojection agreement
is not proof of hand presence, ownership or physical 3D accuracy. No live caller.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PALM_NATIVE=np.array([5,8,11,14,17])
PALM_DETECTOR=np.array([0,5,9,13,17])


def transform(points,center,parameters):
    rotation=Rotation.from_rotvec(np.asarray(parameters[3:])*.1).as_matrix()
    return (np.asarray(points)-center)@rotation.T+center+np.asarray(parameters[:3])*10


def fit_wrist(xyz,wrist,cameras,observations):
    xyz=np.asarray(xyz,float);wrist=np.asarray(wrist,float)
    if xyz.shape!=(21,3) or wrist.shape!=(4,4) or not np.isfinite(xyz).all() or not np.isfinite(wrist).all():
        raise ValueError('Invalid hand geometry')
    if len(observations)!=2 or len({v['camera'] for v in observations})!=2:
        return None,dict(reason='Requires two distinct current views')
    targets=[];cams=[]
    for view in observations:
        points=np.asarray(view['points'],float)
        if view['camera'] not in cameras or points.shape!=(21,2) or not np.isfinite(points).all() or not .5<=view['presence']<=1:
            return None,dict(reason='Invalid current observation')
        cams.append(cameras[view['camera']]);targets.append(points[PALM_DETECTOR])
    center=wrist[:3,3];points=xyz[PALM_NATIVE]
    def errors(x):
        corrected=transform(points,center,x);rows=[]
        for cam,target in zip(cams,targets):
            eye=cam.world_to_eye(corrected)
            if not np.isfinite(eye).all() or np.any(eye[:,2]<=1):return np.full((2,5,2),1e6)
            rows.append(cam.eye_to_window(eye)-target)
        return np.asarray(rows)
    initial=errors(np.zeros(6))
    if np.max(np.abs(initial))>=1e6:return None,dict(reason='Invalid projection')
    def residual(x):
        # Soft-L1 reduces a bad landmark's influence. Explicit priors keep this
        # a correction of the learned pose, never unconstrained triangulation.
        return np.r_[errors(x).reshape(-1)/3.,x[:3]*(10/15),x[3:]*(.1/.12)]
    solved=least_squares(residual,np.zeros(6),bounds=([-2.5]*3+[-1.5]*3,[2.5]*3+[1.5]*3),loss='soft_l1',f_scale=1.,max_nfev=20)
    final=errors(solved.x);before=np.linalg.norm(initial,axis=2);after=np.linalg.norm(final,axis=2)
    translation=float(np.linalg.norm(solved.x[:3]*10));angle=float(np.linalg.norm(solved.x[3:]*.1))
    debug=dict(nfev=solved.nfev,success=bool(solved.success),translation_mm=translation,rotation_rad=angle,
               before_mean_px=before.mean(1).tolist(),after_mean_px=after.mean(1).tolist(),parameters=solved.x.tolist())
    reason=None
    if not solved.success:reason='Optimizer did not converge within bound'
    elif translation>25 or angle>.15:reason='Correction norm exceeds bound'
    elif after.mean()>=before.mean()*.9:reason='Less than ten percent palm improvement'
    elif np.any(after.mean(1)>before.mean(1)+1e-6):reason='One camera worsened'
    if reason:return None,dict(debug,reason=reason)
    return solved.x,dict(debug,reason='Hypothesis only')


def apply_correction(hand,parameters):
    wrist=np.asarray(hand['wrist_world_mm'],float);center=wrist[:3,3].copy()
    rotation=Rotation.from_rotvec(np.asarray(parameters[3:])*.1).as_matrix()
    adjusted=wrist.copy();adjusted[:3,:3]=rotation@wrist[:3,:3];adjusted[:3,3]+=np.asarray(parameters[:3])*10
    return dict(hand,wrist_world_mm=adjusted.tolist(),
                vertices_world_mm=transform(hand['vertices_world_mm'],center,parameters).tolist(),
                hypothesis_only=True,wrist_correction_only=True)
