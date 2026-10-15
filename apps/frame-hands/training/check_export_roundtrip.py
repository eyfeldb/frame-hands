"""Self-test of the training/export path: ONNX -> PyTorch (trainable) -> ONNX -> pnnx -> ncnn.

Checks that (1) the converted PyTorch module has trainable parameters and a working backward
pass, (2) the re-exported ncnn model reproduces the installed ncnn model's outputs on a real
recorded crop, for both the palm detector and the hand landmark model.

usage: python check_export_roundtrip.py <session-dir> <installed-models-dir> <source-models-dir> <scratch-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402
import export_ncnn as ex  # noqa: E402


def main():
    session, installed, src, scratch = map(Path, sys.argv[1:5])
    import onnx
    import torch
    from onnx2torch import convert

    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    stamps = rec.stamps()
    i, v = next((i, v) for i in range(len(stamps) // 3, len(stamps)) for v in audit.get(stamps[i], {'views': []})['views'] if v['presence'] > 0.9)
    img = rec.read(i)[v['camera']]
    center, size, rot = fh.roi_from_points(v['pts'])
    nets = fh.NcnnNets(installed)
    report = {}
    for short, onnx_name, make_input, outputs in (
        ('hand', 'handpose_estimation_mediapipe_2023feb.onnx', lambda: fh.model_input(nets.landmarks(img, center, size, rot)['patch']), ('out0', 'out1', 'out2', 'out3')),
        ('palm', 'palm_detection_mediapipe_2023feb.onnx', lambda: fh.model_input(nets.palms(img, center, 0.45 * max(img.shape), rot)[2]), ('out0', 'out1')),
    ):
        chw = make_input()
        from palm_model import load_model
        mod = load_model(src / onnx_name)
        params = [p for p in mod.parameters()]
        n_train = sum(p.numel() for p in params if p.requires_grad)
        # backward smoke test
        mod.train()
        x = torch.from_numpy(np.transpose(chw, (1, 2, 0))[None]).requires_grad_(False)
        outs = mod(x)
        outs = outs if isinstance(outs, (tuple, list)) else [outs]
        loss = sum(o.float().abs().mean() for o in outs)
        loss.backward()
        grad_ok = all(p.grad is not None for p in params if p.requires_grad)
        mod.zero_grad(set_to_none=True)
        mod.eval()
        with torch.no_grad():
            t_out = [o.numpy().reshape(-1) for o in (lambda r: r if isinstance(r, (tuple, list)) else [r])(mod(x))]
            # batched forward must equal the single-sample forward (the Resize replacement)
            xb = torch.cat([x, x * 0.5, x], 0)
            b_out = [o.numpy() for o in (lambda r: r if isinstance(r, (tuple, list)) else [r])(mod(xb))]
            batch_consistency = float(max(np.max(np.abs(b[0].reshape(-1) - t)) for b, t in zip(b_out, t_out)))
        param, binf = ex.export(mod, short, scratch / 'models', scratch / ('work-' + short))
        new_out = ex.run_ncnn(param, binf, chw, outputs)
        old_out = nets._run(nets.hand if short == 'hand' else nets.palm, chw, outputs)
        bn = [l.split()[0] for l in param.read_text().splitlines()[2:]]
        report[short] = dict(
            trainable_params=n_train, backward_ok=grad_ok, replaced_resizes=getattr(mod, 'replaced_resizes', 0),
            batch_consistency_maxabs=batch_consistency,
            reexport_vs_installed_ncnn_maxabs=[float(np.max(np.abs(a - b))) for a, b in zip(new_out, old_out)],
            reexport_vs_torch_maxabs=[float(np.max(np.abs(a - b))) for a, b in zip(new_out, t_out)],
            layer_types=sorted(set(bn)), layers=len(bn), param_bytes=binf.stat().st_size,
            installed_param_bytes=(installed / (short + '.ncnn.bin')).stat().st_size,
        )
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
