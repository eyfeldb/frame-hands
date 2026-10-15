"""Load the OpenCV Zoo MediaPipe hand models as trainable PyTorch modules (NHWC input).

onnx2torch converts the palm detector's two Resize nodes with a constant output size baked for
batch 1 ([1,256,12,12] / [1,256,24,24]); both are bilinear half-pixel 2x upsamples, so they are
replaced with scale-factor upsampling, which is identical numerically, batches correctly, and
exports back to an ONNX Resize that pnnx maps to ncnn Interp.
"""
from pathlib import Path


def load_model(onnx_path):
    import onnx
    import torch
    import torch.nn.functional as F
    from onnx2torch import convert

    class Upsample2x(torch.nn.Module):
        def forward(self, x, *unused):
            return F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)

    class BatchReshape(torch.nn.Module):
        """ONNX Reshape with a constant target whose leading dim is the batch (1 in the source graph):
        keep the trailing dims, infer the second from the data, batch from the input."""
        def forward(self, x, shape):
            dims = [int(v) for v in shape.reshape(-1).tolist()]
            if len(dims) >= 2:
                dims = [x.shape[0], -1] + dims[2:]
            return x.reshape(*dims)

    model = convert(onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path))))
    replaced = 0
    for name, mod in list(model.named_modules()):
        kind = type(mod).__name__
        if kind == 'OnnxResize' or kind == 'OnnxReshape':
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, Upsample2x() if kind == 'OnnxResize' else BatchReshape())
            replaced += 1
    model.graph.lint()
    model.recompile()
    model.replaced_resizes = replaced
    return model


def load_state(model, checkpoint):
    import torch
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'))
    return model
