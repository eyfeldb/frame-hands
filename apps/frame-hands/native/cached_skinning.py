"""Opt-in fixed generic hand skinning with cached weights and matrix contraction.

Calls the pinned upstream joint transforms unchanged; replaces only the final
weighted point contraction. No image model, association or temporal changes.
"""
import numpy as np
import torch


class FixedHandSkinner:
    def __init__(self,hand):
        from lib.common.hand_skinning import _get_skinning_weights
        from lib.common.hand import NUM_JOINT_FRAMES
        self.hand=hand
        self.fields=('joint_rest_positions','joint_rotation_axes','landmark_rest_positions','landmark_rest_bone_indices','landmark_rest_bone_weights')
        self.pointers={key:getattr(hand,key).data_ptr() for key in self.fields}
        weights=_get_skinning_weights(hand.landmark_rest_bone_indices.unsqueeze(0),hand.landmark_rest_bone_weights.unsqueeze(0),NUM_JOINT_FRAMES)[0]
        self.landmark_points=self.prepare(hand.landmark_rest_positions,weights)
        self.mesh_points=self.prepare(hand.mesh_vertices,hand.dense_bone_weights)

    @staticmethod
    def prepare(points,weights):
        if points.device.type!='cpu' or points.dtype!=torch.float32:raise ValueError('Expected pinned CPU FP32 hand')
        homogeneous=torch.cat((points,torch.ones(len(points),1)),dim=-1)
        return (homogeneous[:,None,:]*weights[:,:,None]).flatten(1).contiguous()

    def points(self,angles,wrist,weighted):
        from lib.common.hand_skinning import _hand_skinning_transform
        leading=angles.shape[:-1];batch=angles.numel()//angles.shape[-1]
        rest=self.hand.joint_rest_positions.unsqueeze(0).expand(batch,-1,-1)
        axes=self.hand.joint_rotation_axes.unsqueeze(0).expand(batch,-1,-1)
        transforms=_hand_skinning_transform(axes,rest,angles.reshape(batch,-1),wrist.reshape(batch,4,4))
        # Sum weights * transformed homogeneous points without materializing
        # a [batch,vertices,bones,4,1] result. Only summation order changes.
        matrix=transforms[:,:,:3,:].permute(0,2,1,3).reshape(batch,3,-1).transpose(1,2)
        result=torch.matmul(weighted,matrix).reshape(*leading,len(weighted),3)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite cached skinning')
        return result

    def landmarks(self,hand_model,angles,wrist):
        if any(getattr(hand_model,key).data_ptr()!=self.pointers[key] for key in self.fields):raise ValueError('Fixed hand changed')
        return self.points(angles,wrist,self.landmark_points)

    def mesh(self,angles,wrist,side):
        wrist=torch.as_tensor(wrist,dtype=torch.float32).clone()
        if side:wrist[:,0]*=-1
        with torch.inference_mode():return self.points(torch.as_tensor(angles,dtype=torch.float32),wrist,self.mesh_points).numpy()


def bind_cached_skinning(pipeline):
    evidence=FixedHandSkinner(pipeline.evidence.backend.hand)
    pose=FixedHandSkinner(pipeline.backend.hand)
    pipeline.evidence.backend.landmarks=evidence.landmarks
    pipeline.backend.mesh=pose.mesh
