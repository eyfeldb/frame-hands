"""Opt-in adapter for the pinned UmeTrack model; reuse same-frame image features.

No cross-frame image cache or shared recurrent state. This composes the pinned
upstream layers, whose license remains CC-BY-NC 4.0; weights/source are not copied.
"""
from dataclasses import dataclass
import torch

@dataclass
class FrameFeatures:
    stamp: int
    keys: list
    images: torch.Tensor
    features: torch.Tensor

    def select(self, stamp, keys, images):
        if stamp != self.stamp:raise ValueError('Feature cache belongs to another capture')
        positions=[self.keys.index(k) for k in keys]
        if not torch.equal(self.images[positions],images):raise ValueError('Feature cache pixels changed')
        return self.features[positions]

class PreparedRegressor:
    def __init__(self,model,skeleton,device='cpu'):
        self.device=torch.device(device)
        if model._temporal._mem_features.numel() and model._temporal._mem_features.device!=self.device:
            raise ValueError('Cannot change device with active recurrent state')
        self.model=model.to(self.device)
        with torch.inference_mode():
            self.skeleton_features=model._skeleton_enc(skeleton.joint_rotation_axes.to(self.device),skeleton.joint_rest_positions.to(self.device))

    def warmup(self):
        """Initialize supported kernel shapes on synthetic pixels, then erase memory."""
        from lib.models.umetrack_model import InputFrameData,InputFrameDesc
        generator=torch.Generator().manual_seed(941)
        try:
            with torch.inference_mode():
                for lengths in ([1]*n for n in (2,4,6,8,10,12,14,16)):
                    self._warm_shape(lengths,InputFrameData,InputFrameDesc,generator)
                for lengths in ([1],[2],[2,1],[2,2]):
                    self._warm_shape(lengths,InputFrameData,InputFrameDesc,generator)
        finally:
            self.model._temporal._mem_features=torch.empty(0,device=self.device)
            self.model._temporal._prev_extrinsics=torch.empty(0,device=self.device)

    def _warm_shape(self,lengths,Frame,Desc,generator):
        count=sum(lengths)
        images=torch.rand(count,96,96,generator=generator)
        intrinsics=torch.tensor([[100.,0,48],[0,100.,48],[0,0,1.]]).repeat(count,1,1)
        extrinsics=torch.eye(4).repeat(count,1,1);extrinsics[1::2,0,3]=.06
        ends=torch.tensor(lengths).cumsum(0);starts=torch.cat((torch.zeros(1,dtype=torch.long),ends[:-1]))
        indices=torch.arange(len(lengths))
        desc=Desc(torch.stack((starts,ends),dim=1),indices,torch.zeros(len(lengths),dtype=torch.bool),indices%2)
        self.regress(Frame(images,intrinsics,extrinsics),desc,self.extract(images))

    def extract(self,images):
        return self.model._feature_extractor._image_backbone(images.to(self.device).unsqueeze(1))

    def regress(self,data,desc,features):
        from lib.models.model_utils import compute_singlev_xfs
        from lib.models.umetrack_model import _recover_wrist_xfs_in_world
        model=self.model;extractor=model._feature_extractor
        transforms=compute_singlev_xfs(data.intrinsics.to(self.device))
        extrinsics=data.extrinsics_xf.to(self.device)
        ranges=desc.sample_range
        if torch.all(ranges[:,1]-ranges[:,0]==1):
            selected=ranges[:,0]
            fused=extractor.compute_singlev_features(features[selected],transforms[selected])
        elif len(ranges)*2==len(features) and torch.all(ranges[:,1]-ranges[:,0]==2):
            fused=extractor.compute_multiv_features(features.reshape(-1,2,*features.shape[1:]),transforms.reshape(-1,2,4,4),extrinsics.reshape(-1,2,4,4))
        else:
            fused=[]
            for start,end in ranges.tolist():
                if end-start==1:
                    part=extractor.compute_singlev_features(features[start:end],transforms[start:end])
                elif end-start==2:
                    part=extractor.compute_multiv_features(features[start:end].unsqueeze(0),transforms[start:end].unsqueeze(0),extrinsics[start:end].unsqueeze(0))
                else:raise ValueError('Expected one or two selected views')
                fused.append(part)
            fused=torch.cat(fused)
        cam0=extrinsics[ranges[:,0]]
        temporal=model._temporal.forward_temporal_features(fused,cam0,desc.memory_idx,desc.use_memory)
        skeleton=self.skeleton_features.expand(len(ranges),*self.skeleton_features.shape[1:])
        result=model._regressor_k.regress_poses(torch.cat((temporal,skeleton),dim=1))
        result.wrist_xfs=_recover_wrist_xfs_in_world(desc.hand_idx,cam0,result.wrist_xfs)
        # CPU outputs synchronize CUDA before timing and preserve the existing
        # calibrated projection/skinning path; cached image features stay on GPU.
        for key in ('wrist_xfs','joint_angles','landmark_uncertainty_sigmas'):
            setattr(result,key,getattr(result,key).cpu())
        return result
