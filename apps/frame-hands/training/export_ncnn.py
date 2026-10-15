"""Export a (fine-tuned) PyTorch copy of a MediaPipe hand model back to ncnn, the way the tracker loads it.

PyTorch (onnx2torch GraphModule, NHWC input) -> ONNX -> pnnx (fp16 weights) -> the upstream
param patch (channel Pad -> Padding, drop the input Permute) -> <out>/<short>.ncnn.{param,bin}.
Also verifies the exported ncnn model against the PyTorch module on a given input.

Library; see check_export_roundtrip.py for the self-test.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SIZES = dict(palm=192, hand=224)
OUTPUT_NAMES = dict(palm=['Identity', 'Identity_1'], hand=['Identity', 'Identity_1', 'Identity_2', 'Identity_3'])


def patch_param(param_text, pnnx_param_text):
    """tools/convert_models.py patch(): ncnn has no Pad layer; drop the NHWC->NCHW input Permute."""
    lines = param_text.splitlines()
    assert lines[0] == '7767517'
    nlayers, nblobs = map(int, lines[1].split())
    body = lines[2:]
    pads = dict(re.findall(r'^Pad\s+(\S+)\s.*pads=\(0,0,0,0,0,(\d+),0,0\)', pnnx_param_text, re.M))
    for i, line in enumerate(body):
        f = line.split()
        if f[0] == 'Pad':
            body[i] = 'Padding %s %s %s %s %s 0=0 1=0 2=0 3=0 4=0 5=0.000000e+00 7=0 8=%s' % (f[1], f[2], f[3], f[4], f[5], pads[f[1]])
    perm = next(i for i, line in enumerate(body) if line.split()[0] == 'Permute')
    f = body[perm].split()
    assert f[4] == 'in0' and f[6] == '0=4', body[perm]
    blob = f[5]
    del body[perm]
    for i, line in enumerate(body):
        f = line.split()
        if f[0] == 'Input':
            continue
        nin = int(f[2])
        ins = ['in0' if b == blob else b for b in f[4:4 + nin]]
        body[i] = ' '.join(f[:4] + ins + f[4 + nin:])
    return '\n'.join(['7767517', '%d %d' % (nlayers - 1, nblobs - 1)] + body) + '\n'


def find_pnnx():
    import pnnx as pnnx_pkg
    root = Path(pnnx_pkg.__file__).parent
    c = [p for p in glob.glob(str(root / '**' / 'pnnx*'), recursive=True) if os.path.isfile(p) and (p.endswith('.exe') or not os.path.splitext(p)[1])]
    if not c:
        raise RuntimeError('pnnx executable not found')
    return c[0]


def export(module, short, out_dir, work_dir, opset=13):
    """module: torch nn.Module taking NHWC float (1,n,n,3). Writes <out_dir>/<short>.ncnn.{param,bin}."""
    import torch
    n = SIZES[short]
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    name = short + '_export'
    onnx_path = work_dir / (name + '.onnx')
    module = module.eval().cpu()
    dummy = torch.zeros(1, n, n, 3)
    torch.onnx.export(module, dummy, str(onnx_path), input_names=['input_1'], output_names=OUTPUT_NAMES[short],
                      opset_version=opset, dynamo=False, do_constant_folding=True)
    r = subprocess.run([find_pnnx(), name + '.onnx', 'inputshape=[1,%d,%d,3]' % (n, n), 'fp16=1'], cwd=work_dir, capture_output=True)
    if r.returncode != 0 or not (work_dir / (name + '.ncnn.param')).exists():
        raise RuntimeError('pnnx failed: ' + (r.stderr or b'').decode('utf-8', 'replace')[-2000:])
    param = patch_param((work_dir / (name + '.ncnn.param')).read_text(), (work_dir / (name + '.pnnx.param')).read_text())
    (out_dir / (short + '.ncnn.param')).write_text(param)
    shutil.copy(work_dir / (name + '.ncnn.bin'), out_dir / (short + '.ncnn.bin'))
    return out_dir / (short + '.ncnn.param'), out_dir / (short + '.ncnn.bin')


def run_ncnn(param, binf, chw, outputs, fp16=True):
    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = fp16
    if net.load_param(str(param)) or net.load_model(str(binf)):
        raise RuntimeError('cannot load exported model')
    ex = net.create_extractor()
    ex.input('in0', ncnn.Mat(np.ascontiguousarray(chw, dtype=np.float32)))
    res = []
    for name in outputs:
        ret, m = ex.extract(name)
        if ret != 0:
            raise RuntimeError('extract failed ' + name)
        res.append(np.array(m).reshape(-1).copy())
    return res
