"""Export the GRU refiner to ncnn and verify frame-by-frame stateful inference matches PyTorch.

The live tracker will call the network once per frame with the previous hidden state, so the exported
graph takes (x: (1,FEAT) one frame, h: (1,HIDDEN)) and returns (out: (1,OUT), h2: (1,HIDDEN)).
Path: TorchScript -> pnnx -> ncnn (pnnx maps nn.GRU to ncnn GRU with hidden in/out).

usage: python export_refiner.py --checkpoint <best.pt> --out <dir> [--hidden 128] [--data <refiner-data>/<rec>.npz for a real sequence]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import refiner_model as rm  # noqa: E402


def find_pnnx():
    import pnnx as pnnx_pkg
    root = Path(pnnx_pkg.__file__).parent
    c = [p for p in glob.glob(str(root / '**' / 'pnnx*'), recursive=True) if os.path.isfile(p) and (p.endswith('.exe') or not os.path.splitext(p)[1])]
    return c[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--data')
    a = ap.parse_args()
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = rm.make_model(a.hidden)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu'))
    model.eval()

    class Step(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x, h):
            # x (1,FEAT) -> (1,1,FEAT); h (1,HIDDEN) -> (1,1,HIDDEN)
            y, h2 = self.m(x.unsqueeze(0), h.unsqueeze(0))
            return y[0], h2[0]
    step = Step(model).eval()
    x0 = torch.zeros(1, rm.FEAT); h0 = torch.zeros(1, a.hidden)
    ts = torch.jit.trace(step, (x0, h0))
    work = out / 'work'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    ts.save(str(work / 'refiner.pt'))
    r = subprocess.run([find_pnnx(), 'refiner.pt', f'inputshape=[1,{rm.FEAT}]f32,[1,{a.hidden}]f32', 'fp16=0'], cwd=work, capture_output=True)
    if r.returncode or not (work / 'refiner.ncnn.param').exists():
        print((r.stderr or b'').decode('utf-8', 'replace')[-3000:]); sys.exit('pnnx failed')
    shutil.copy(work / 'refiner.ncnn.param', out / 'refiner.ncnn.param'); shutil.copy(work / 'refiner.ncnn.bin', out / 'refiner.ncnn.bin')
    param = (out / 'refiner.ncnn.param').read_text()
    print(param[:1500])
    # parity: step through a sequence with ncnn, carrying h, vs torch full-sequence
    if a.data:
        import refiner_data as rd
        tr = rd.load_sequences(a.data)[0]
        x = torch.tensor(rm.features(tr, tr['side'])[:120])
    else:
        x = torch.randn(120, rm.FEAT)
    with torch.no_grad():
        ref, _ = model(x[None])
    ref = ref[0].numpy()
    import ncnn
    net = ncnn.Net(); net.opt.use_vulkan_compute = False; net.opt.num_threads = 1
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = False
    if net.load_param(str(out / 'refiner.ncnn.param')) or net.load_model(str(out / 'refiner.ncnn.bin')):
        sys.exit('cannot load exported refiner')
    inputs = [l.split()[1] for l in param.splitlines()[2:] if l.startswith('Input')]
    outputs = []
    for l in param.splitlines()[2:]:
        f = l.split()
        nin, nout = int(f[2]), int(f[3])
        outputs.extend(f[4 + nin:4 + nin + nout])
    consumed = set()
    for l in param.splitlines()[2:]:
        f = l.split(); nin = int(f[2]); consumed.update(f[4:4 + nin])
    out_names = [o for o in outputs if o not in consumed]
    print('inputs', inputs, 'outputs', out_names)
    h = np.zeros((1, a.hidden), np.float32)
    got = np.zeros_like(ref)
    for i in range(x.shape[0]):
        ex = net.create_extractor()
        ex.input(inputs[0], ncnn.Mat(np.ascontiguousarray(x[i:i + 1].numpy())))
        ex.input(inputs[1], ncnn.Mat(np.ascontiguousarray(h)))
        _, o = ex.extract(out_names[0]); _, h2 = ex.extract(out_names[1])
        got[i] = np.array(o).reshape(-1); h = np.array(h2).reshape(1, a.hidden).astype(np.float32)
    err = np.abs(got - ref).max()
    print(json.dumps(dict(max_abs_diff=float(err), ref_abs_max=float(np.abs(ref).max()), params=sum(p.numel() for p in model.parameters()),
                          param_bytes=(out / 'refiner.ncnn.bin').stat().st_size, inputs=inputs, outputs=out_names)))
    (out / 'export.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint), hidden=a.hidden, feat=rm.FEAT, out=rm.OUT, inputs=inputs, outputs=out_names, max_abs_diff=float(err)), indent=1))


if __name__ == '__main__':
    main()
