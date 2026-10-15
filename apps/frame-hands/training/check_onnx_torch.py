"""Do the source ONNX models (OpenCV Zoo) and their onnx2torch conversion match the deployed ncnn models?

Runs one real recorded crop through ncnn, ONNX Runtime and the PyTorch conversion and
reports max abs differences. Confirms the trainable PyTorch path reproduces what runs on Frame.

usage: python check_onnx_torch.py <session-dir> <models-dir> <source-models-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fh_common as fh  # noqa: E402


def main():
    session, models, src = map(Path, sys.argv[1:4])
    import onnx
    import onnxruntime as ort
    import torch
    from onnx2torch import convert

    rec = fh.Recording(session / 'frame' / 'sets.bin')
    audit = fh.read_audit(session / 'frame' / 'live-views.txt')
    # a tracked view mid-recording
    stamps = rec.stamps()
    i, v = next((i, v) for i in range(len(stamps) // 2, len(stamps)) for v in audit.get(stamps[i], {'views': []})['views'] if v['presence'] > 0.9)
    img = rec.read(i)[v['camera']]
    center, size, rot = fh.roi_from_points(v['pts'])
    nets = fh.NcnnNets(models, fp16=False)
    report = {}

    # hand landmark model
    r = nets.landmarks(img, center, size, rot)
    chw = fh.model_input(r['patch'])
    nhwc = np.transpose(chw, (1, 2, 0))[None]
    m = onnx.load(str(src / 'handpose_estimation_mediapipe_2023feb.onnx'))
    names_in = [i.name for i in m.graph.input]; names_out = [o.name for o in m.graph.output]
    sess = ort.InferenceSession(str(src / 'handpose_estimation_mediapipe_2023feb.onnx'), providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {names_in[0]: nhwc})
    tmod = convert(onnx.shape_inference.infer_shapes(m)).eval()
    with torch.no_grad():
        t_out = tmod(torch.from_numpy(nhwc))
    t_out = [o.numpy() for o in (t_out if isinstance(t_out, (tuple, list)) else [t_out])]
    ncnn_out = [r['crop_pts'].reshape(-1), np.array([r['presence']]), np.array([r['right']]), r['world'].reshape(-1)]
    report['hand'] = dict(
        onnx_inputs=[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
        onnx_outputs=[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
        ort_shapes=[list(o.shape) for o in ort_out],
        ort_vs_ncnn_maxabs=[float(np.max(np.abs(o.reshape(-1) - n.reshape(-1)))) if o.size == n.size else 'shape-mismatch' for o, n in zip(ort_out, ncnn_out)],
        torch_vs_ort_maxabs=[float(np.max(np.abs(t.reshape(-1) - o.reshape(-1)))) for t, o in zip(t_out, ort_out)],
        ncnn_presence=r['presence'], live_presence=v['presence'],
        torch_params=sum(p.numel() for p in tmod.parameters()),
    )

    # palm detector
    palms, tm, patch = nets.palms(img, center, 0.45 * max(img.shape), rot, threshold=0.2)
    chw = fh.model_input(patch)
    nhwc = np.transpose(chw, (1, 2, 0))[None]
    m = onnx.load(str(src / 'palm_detection_mediapipe_2023feb.onnx'))
    sess = ort.InferenceSession(str(src / 'palm_detection_mediapipe_2023feb.onnx'), providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {m.graph.input[0].name: nhwc})
    boxes, logits = nets._run(nets.palm, chw, ('out0', 'out1'))
    tmod = convert(onnx.shape_inference.infer_shapes(m)).eval()
    with torch.no_grad():
        t_out = tmod(torch.from_numpy(nhwc))
    t_out = [o.numpy() for o in (t_out if isinstance(t_out, (tuple, list)) else [t_out])]
    report['palm'] = dict(
        onnx_inputs=[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],
        onnx_outputs=[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output],
        ort_shapes=[list(o.shape) for o in ort_out],
        ort_vs_ncnn_maxabs=[float(np.max(np.abs(ort_out[0].reshape(-1) - boxes))), float(np.max(np.abs(ort_out[1].reshape(-1) - logits)))],
        torch_vs_ort_maxabs=[float(np.max(np.abs(t.reshape(-1) - o.reshape(-1)))) for t, o in zip(t_out, ort_out)],
        detections=len(palms), best_score=max([p['score'] for p in palms], default=0.0),
        torch_params=sum(p.numel() for p in tmod.parameters()),
    )
    print(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
