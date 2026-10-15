"""Independent, stateless batched image evidence for per-view chirality.

Two model hypotheses share a recorded crop, never temporal memory. Agreement of
pose reprojection and relative predicted sigma supplies an uncalibrated side
score; disagreement abstains. The pose backend's own learned memory is separate.
"""
import time
import numpy as np
import torch
from live_backend import Backend
from replay_umetrack import COMMON

class ChiralityEvidence:
    def __init__(self,upstream,cameras,optimized=False,device='cpu'):
        if device!='cpu' and not optimized:raise ValueError('CUDA requires the isolated optimized path')
        self.backend=Backend(upstream,cameras)
        self.optimized=optimized;self.features=None
        if optimized:
            from feature_reuse import PreparedRegressor
            self.regressor=PreparedRegressor(self.backend.model,self.backend.skeleton,device=device)

    @staticmethod
    def combine(errors,sigmas):
        if not np.isfinite(errors+sigmas).all():raise ValueError('Nonfinite hypothesis output')
        pose_choice=int(errors[1]<errors[0]);sigma_choice=int(sigmas[1]<sigmas[0])
        agreed=pose_choice==sigma_choice
        q=float(np.clip(errors[0]/max(1e-6,sum(errors)),.02,.98)) if agreed else .5
        return q,agreed,pose_choice,sigma_choice

    def score(self,observation,world,prepared):
        started=time.perf_counter();b=self.backend;self.features=None
        views=[v for v in observation['views'] if (v['camera'],v['id']) in prepared and v['presence']>=.5]
        if len(views)>8:raise ValueError('Unbounded hypothesis batch')
        if not views:return dict(observation,views=[]),[],0.
        images=[];intrinsics=[];extrinsics=[];sides=[]
        for v in views:
            image,meta=prepared[v['camera'],v['id']]
            if image is None or image.shape!=(96,96):raise ValueError('Invalid hypothesis crop')
            for side in (0,1):
                images.append(image[:,::-1].copy() if side else image)
                intrinsics.append(meta['intrinsics'])
                xf=world@np.asarray(meta['camera_to_head_mm'])
                if side:xf[:,0]*=-1
                xf=np.linalg.inv(xf);xf[:3,3]*=.001;extrinsics.append(xf);sides.append(side)
        count=len(images);indices=torch.arange(count)
        frame=b.Frame(torch.tensor(np.stack(images)/255,dtype=torch.float32),torch.tensor(np.stack(intrinsics),dtype=torch.float32),torch.tensor(np.stack(extrinsics),dtype=torch.float32))
        desc=b.Desc(torch.stack((indices,indices+1),dim=1),indices,torch.zeros(count,dtype=torch.bool),torch.tensor(sides))
        with torch.inference_mode():
            if self.optimized:
                from feature_reuse import FrameFeatures
                features=self.regressor.extract(frame.left_images)
                inferred=self.regressor.regress(frame,desc,features)
                self.features=FrameFeatures(observation['raw_ns'],[(v['camera'],v['id'],side) for v in views for side in (0,1)],frame.left_images,features)
                wrists=inferred.wrist_xfs.clone();wrists[:,:3,3]*=1000;wrists[1::2,:,0]*=-1
                hand=b.hand._replace(**{key:value.unsqueeze(0).expand(count,*value.shape) for key,value in b.hand._asdict().items() if isinstance(value,torch.Tensor)})
                all_xyz=b.landmarks(hand,inferred.joint_angles,wrists).numpy()
            else:inferred=b.model.regress_pose_use_skeleton(frame,desc,b.skeleton)
        output=[];debug=[];model_indices=[i for i,j in COMMON];detector_indices=[j for i,j in COMMON]
        for n,v in enumerate(views):
            errors=[];sigmas=[]
            cam=b.cameras[v['camera']].copy(camera_to_world_xf=world@b.cameras[v['camera']].camera_to_world_xf)
            for side in (0,1):
                i=n*2+side
                if self.optimized:xyz=all_xyz[i]
                else:
                    wrist=inferred.wrist_xfs[i].clone();wrist[:3,3]*=1000
                    if side:wrist[:,0]*=-1
                    with torch.inference_mode():xyz=b.landmarks(b.hand,inferred.joint_angles[i],wrist).numpy()
                uv=cam.eye_to_window(cam.world_to_eye(xyz))
                errors.append(float(np.median(np.linalg.norm(uv[model_indices]-np.asarray(v['points'])[detector_indices],axis=1))))
                sigmas.append(float(inferred.landmark_uncertainty_sigmas[i].mean()))
            q,agreed,pose_choice,sigma_choice=self.combine(errors,sigmas)
            output.append(dict(v,raw_right=v['right'],right=q))
            debug.append(dict(camera=v['camera'],source_id=v['id'],raw_right=v['right'],right=q,agreement=agreed,pose_choice=pose_choice,sigma_choice=sigma_choice,reprojection_px=errors,mean_raw_sigmas=sigmas))
        return dict(observation,views=output),debug,(time.perf_counter()-started)*1000
