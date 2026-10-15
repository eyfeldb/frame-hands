"""Smaller research encoder retaining the accepted 72x6x6 feature interface."""
import torch

ARCH = 'resnet_layers_2352-f16'


def make_student(upstream, teacher=None):
    import sys
    sys.path.insert(0, str(upstream))
    from lib.models.model_utils import create_backbone
    student, _ = create_backbone(ARCH, (96, 96), 72)
    if teacher is not None:
        original = teacher.state_dict()
        sliced = {}
        for name, value in student.state_dict().items():
            source = original[name]
            if source.ndim != value.ndim or any(s < d for s, d in zip(source.shape, value.shape)):
                raise ValueError(f'Incompatible teacher tensor: {name}')
            sliced[name] = source[tuple(slice(0, d) for d in value.shape)].clone()
        student.load_state_dict(sliced, strict=True)
    return student


def load_student(upstream, checkpoint, device='cuda'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCH:
        raise ValueError('Unknown student architecture')
    student = make_student(upstream)
    student.load_state_dict(saved['state_dict'], strict=True)
    return student.eval().to(device)
