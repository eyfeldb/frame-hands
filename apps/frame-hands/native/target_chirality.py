"""Tiny chirality classifier conditioned on existing target hand landmarks."""
import torch
from torch import nn
from tiny_chirality import TinyChirality
ARCH='tiny-target-chirality-v1'


class TargetChirality(TinyChirality):
    def __init__(self):
        super().__init__()
        self.features[0]=nn.Conv2d(3,16,5,stride=2,padding=2,bias=False)


def condition(images,points,generator=None):
    if images.ndim!=4 or images.shape[1:]!=(1,96,96) or points.shape!=(len(images),20,2):raise ValueError('Unexpected conditioning shape')
    if not torch.isfinite(points).all():raise ValueError('Invalid target points')
    if generator is not None:points=points+torch.randn(points.shape,device=points.device,generator=generator)*2
    axis=torch.arange(96,device=images.device,dtype=images.dtype)
    squared=(axis[None,None,None,:]-points[:,:,0,None,None])**2+(axis[None,None,:,None]-points[:,:,1,None,None])**2
    maps=torch.exp(-squared/18)
    if generator is not None:maps=maps*(torch.rand((len(images),20,1,1),device=images.device,generator=generator)>=.1)
    return torch.cat((images,maps[:,:1],maps.amax(1,keepdim=True)),dim=1)


def load_target(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['architecture']!=ARCH:raise ValueError('Unknown architecture')
    model=TargetChirality();model.load_state_dict(saved['state_dict'],strict=True)
    return model.eval().to(device)
