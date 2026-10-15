"""Export a history-conditioned hand checkpoint to ncnn and verify parity on dev crops with the history channel.

usage: python finalize_hist.py --source <handpose.onnx> --checkpoint <best.pt> --palm <palm models dir> --dev <shard-dir> --out <models-out-dir>
Writes <out>/hand.ncnn.{param,bin} (new), copies palm.ncnn.* from --palm, writes manifest.json with hashes and parity numbers.
The ncnn input stays a 3-channel 224x224 planar float Mat: channel 0 and 1 = grey/255, channel 2 = history map.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import export_ncnn as ex  # noqa: E402
import train_hist as th  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--palm', required=True)
    ap.add_argument('--dev', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from palm_model import load_model
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.source)
    th.stem_surgery(model)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu'))
    model.eval()
    param, binf = ex.export(model, 'hand', out, out / 'work-export')
    for name in ('palm.ncnn.param', 'palm.ncnn.bin'):
        shutil.copy(Path(a.palm) / name, out / name)
    # parity on dev samples (with history given)
    dev = th.Shards([a.dev], augment=False, cache=out / 'cache', limit=200)
    errs = {'screen': [], 'pres': [], 'right': [], 'world': []}
    with torch.no_grad():
        for j in range(0, len(dev), 1):
            x, hist, hkind, *_ = dev[j]
            nhwc = th.assemble(x[None], hist[None], hkind.view(1), 'cpu')
            t_out = model(nhwc); t_out = t_out if isinstance(t_out, (tuple, list)) else [t_out]
            chw = np.ascontiguousarray(nhwc[0].permute(2, 0, 1).numpy().astype(np.float32))
            n_out = ex.run_ncnn(param, binf, chw, ('out0', 'out1', 'out2', 'out3'), fp16=True)
            for k, name in enumerate(errs):
                errs[name].append(float(np.max(np.abs(t_out[k].numpy().reshape(-1) - n_out[k]))))
    parity = {k: dict(max=float(np.max(v)), p50=float(np.median(v))) for k, v in errs.items()}
    manifest = dict(source_onnx=sha(a.source), checkpoint=sha(a.checkpoint), checkpoint_path=str(a.checkpoint), history_channel=2,
                    files={p.name: sha(p) for p in out.glob('*.ncnn.*')}, parity_fp16_vs_torch=parity, samples=len(dev))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
