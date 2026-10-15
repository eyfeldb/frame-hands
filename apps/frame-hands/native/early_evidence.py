"""Opt-in two-stage identity evidence with a fixed first-hypothesis exit gate.

Only the raw detector prior chooses the first side. Its own fit/sigma may skip
an alternate; ambiguous views still evaluate both. Final pose can request a
missing opposite feature, which is then computed from this capture's pixels.
"""
import time
import numpy as np
import torch
from chirality_evidence import ChiralityEvidence
from feature_reuse import FrameFeatures
from replay_umetrack import COMMON
from early_hypothesis import early_score,first_side


class CompletingFeatures(FrameFeatures):
    def __init__(self,stamp,keys,images,features,regressor):
        super().__init__(stamp,keys,images,features)
        self.regressor=regressor;self.late_passes=0

    def select(self,stamp,keys,images):
        if stamp!=self.stamp:raise ValueError('Feature cache belongs to another capture')
        if len(keys)!=len(images) or len(set(keys))!=len(keys):raise ValueError('Invalid selected keys')
        missing=[]
        for i,key in enumerate(keys):
            if key in self.keys:
                if not torch.equal(self.images[self.keys.index(key)],images[i]):raise ValueError('Feature cache pixels changed')
            else:missing.append(i)
        if missing:
            new=self.regressor.extract(images[missing])
            self.keys.extend(keys[i] for i in missing)
            self.images=torch.cat((self.images,images[missing]))
            self.features=torch.cat((self.features,new))
            self.late_passes+=len(missing)
        return super().select(stamp,keys,images)


class EarlyHypothesisEvidence(ChiralityEvidence):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.optimized:raise ValueError('Early hypothesis requires feature reuse')
        self.exit_count=0;self.view_count=0;self.completed_views=0;self.completed_frames=0
        self.require_stereo_exit=False;self.ownership=None

    def score(self,observation,world,prepared):
        started=time.perf_counter();b=self.backend;self.features=None
        views=[v for v in observation['views'] if (v['camera'],v['id']) in prepared and v['presence']>=.5]
        if len(views)>8:raise ValueError('Unbounded hypothesis batch')
        if not views:return dict(observation,views=[]),[],0.
        model_indices=[i for i,j in COMMON];detector_indices=[j for i,j in COMMON]
        evaluated={};cache_keys=[];cache_images=[];cache_features=[]

        def infer(requests):
            if not requests:return
            images=[];intrinsics=[];extrinsics=[];sides=[]
            for n,side in requests:
                v=views[n];image,meta=prepared[v['camera'],v['id']]
                if image is None or image.shape!=(96,96):raise ValueError('Invalid hypothesis crop')
                images.append(image[:,::-1].copy() if side else image)
                intrinsics.append(meta['intrinsics'])
                xf=world@np.asarray(meta['camera_to_head_mm'])
                if side:xf[:,0]*=-1
                xf=np.linalg.inv(xf);xf[:3,3]*=.001;extrinsics.append(xf);sides.append(side)
            count=len(images);indices=torch.arange(count)
            frame=b.Frame(torch.tensor(np.stack(images)/255,dtype=torch.float32),torch.tensor(np.stack(intrinsics),dtype=torch.float32),torch.tensor(np.stack(extrinsics),dtype=torch.float32))
            desc=b.Desc(torch.stack((indices,indices+1),dim=1),indices,torch.zeros(count,dtype=torch.bool),torch.tensor(sides))
            with torch.inference_mode():
                features=self.regressor.extract(frame.left_images)
                inferred=self.regressor.regress(frame,desc,features)
                wrists=inferred.wrist_xfs.clone();wrists[:,:3,3]*=1000;wrists[torch.tensor(sides,dtype=torch.bool),:,0]*=-1
                hand=b.hand._replace(**{key:value.unsqueeze(0).expand(count,*value.shape) for key,value in b.hand._asdict().items() if isinstance(value,torch.Tensor)})
                xyz=b.landmarks(hand,inferred.joint_angles,wrists).numpy()
            for i,(n,side) in enumerate(requests):
                v=views[n];cam=b.cameras[v['camera']].copy(camera_to_world_xf=world@b.cameras[v['camera']].camera_to_world_xf)
                uv=cam.eye_to_window(cam.world_to_eye(xyz[i]))
                error=float(np.median(np.linalg.norm(uv[model_indices]-np.asarray(v['points'])[detector_indices],axis=1)))
                sigma=float(inferred.landmark_uncertainty_sigmas[i].mean())
                if not np.isfinite([error,sigma]).all():raise ValueError('Nonfinite hypothesis output')
                evaluated[n,side]=(error,sigma)
                cache_keys.append((v['camera'],v['id'],side))
            cache_images.append(frame.left_images);cache_features.append(features)

        first=[first_side(v['right']) for v in views]
        infer(list(enumerate(first)))
        scores=[early_score(v['right'],v['presence'],*evaluated[n,first[n]]) for n,v in enumerate(views)]
        infer([(n,1-first[n]) for n,q in enumerate(scores) if q is None])
        qualified=[q is not None for q in scores];complete=False
        if self.require_stereo_exit and any(qualified):
            if self.ownership is None:raise ValueError('Stereo exit requires pre-frame ownership state')
            from stereo_exit import needs_completion
            provisional=[]
            for n,v in enumerate(views):
                q=scores[n]
                if q is None:
                    q=self.combine([evaluated[n,side][0] for side in (0,1)],[evaluated[n,side][1] for side in (0,1)])[0]
                provisional.append(dict(v,right=q))
            complete=needs_completion(self.ownership,dict(observation,views=provisional),world)
            if complete:
                infer([(n,1-first[n]) for n,q in enumerate(scores) if q is not None])
                self.completed_views+=sum(qualified);self.completed_frames+=1;scores=[None]*len(views)
        self.features=CompletingFeatures(observation['raw_ns'],cache_keys,torch.cat(cache_images),torch.cat(cache_features),self.regressor)
        output=[];debug=[]
        for n,v in enumerate(views):
            early=scores[n] is not None
            errors=[evaluated[n,side][0] if (n,side) in evaluated else None for side in (0,1)]
            sigmas=[evaluated[n,side][1] if (n,side) in evaluated else None for side in (0,1)]
            if early:q=scores[n];agreed=pose_choice=sigma_choice=None
            else:q,agreed,pose_choice,sigma_choice=self.combine(errors,sigmas)
            output.append(dict(v,raw_right=v['right'],right=q))
            debug.append(dict(camera=v['camera'],source_id=v['id'],raw_right=v['right'],right=q,early_exit=early,completed=complete and qualified[n],first_side=first[n],agreement=agreed,pose_choice=pose_choice,sigma_choice=sigma_choice,reprojection_px=errors,mean_raw_sigmas=sigmas))
        self.exit_count+=sum(q is not None for q in scores);self.view_count+=len(views)
        return dict(observation,views=output),debug,(time.perf_counter()-started)*1000
