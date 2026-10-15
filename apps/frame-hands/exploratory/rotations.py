"""Independent Rodrigues exponential; avoids PyTorch3D's compiled dependency."""
import torch


def so3_exp_map(v):
    if v.shape[-1] != 3:
        raise ValueError('Expected axis-angle vectors')
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(v, dim=-1)[..., None, None]
    a = torch.sinc(theta / torch.pi)
    b = 0.5 * torch.sinc(theta / (2 * torch.pi)).square()
    return torch.eye(3, dtype=v.dtype, device=v.device) + a*k + b*(k@k)
