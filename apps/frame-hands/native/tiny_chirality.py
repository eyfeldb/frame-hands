"""Experimental tiny image chirality classifier; no live/runtime integration."""
import torch
from torch import nn
ARCH='tiny-chirality-v1'


class TinyChirality(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(1,16,5,stride=2,padding=2,bias=False),nn.BatchNorm2d(16),nn.ReLU()]
        for source,target,stride in ((16,32,2),(32,64,2),(64,96,2),(96,96,1),(96,96,1)):
            layers += [nn.Conv2d(source,source,3,stride=stride,padding=1,groups=source,bias=False),nn.BatchNorm2d(source),nn.ReLU(),
                       nn.Conv2d(source,target,1,bias=False),nn.BatchNorm2d(target),nn.ReLU()]
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(96,1))
    def forward(self,x):return self.head(self.features(x)).flatten()


def load_classifier(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['architecture']!=ARCH:raise ValueError('Unknown classifier')
    model=TinyChirality();model.load_state_dict(saved['state_dict'],strict=True)
    return model.eval().to(device)
