"""Export an isolated student backbone for native timing/parity, not promotion."""
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exploratory'))
from replay_mesh_trial import packets
from student_encoder import load_student, ARCH
from export_blocks import NamedBlock
from distill_encoder import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'checkpoint', 'development', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    import pnnx, onnxruntime as ort
    a.output.mkdir(parents=True, exist_ok=False)
    model = load_student(a.upstream, a.checkpoint, device='cpu')
    inputs = []
    for meta, blobs in packets(a.development / 'transport.bin'):
        for blob in blobs:
            x = cv2.imdecode(np.frombuffer(blob, np.uint8), 0)
            if x is None or x.shape != (96, 96):
                raise ValueError('Invalid crop')
            for flip in (False, True):
                inputs.append(torch.from_numpy(np.ascontiguousarray(x[:, ::-1] if flip else x))[None, None].float() / 255.)
        if len(inputs) >= 4:
            break
    inputs = inputs[:4]
    if len(inputs) != 4:
        raise ValueError('Insufficient development examples')
    path = a.output / 'backbone'
    with torch.inference_mode():
        torch.onnx.export(model, (inputs[0],), str(path) + '.onnx', dynamo=False, opset_version=17, input_names=['input'], output_names=['output'])
        pnnx.export(NamedBlock(model), path.as_posix() + '.pt', (inputs[0],), fp16=False)
        session = ort.InferenceSession(str(path) + '.onnx', providers=['CPUExecutionProvider'])
        errors = []
        for i, x in enumerate(inputs):
            y = model(x).numpy()
            actual = session.run(None, {'input': x.numpy()})[0]
            np.testing.assert_allclose(y, actual, atol=3e-5, rtol=3e-4)
            errors.append(float(abs(y - actual).max()))
            x.numpy().astype('<f4').tofile(str(path) + f'.input-{i}.bin')
            y.astype('<f4').tofile(str(path) + f'.output-{i}.bin')
    manifest = dict(scope='Unaccepted student, native backbone speed and export agreement only. Development replay failed quality gate.',
        architecture=ARCH, checkpoint_sha256=sha(a.checkpoint), torch=torch.__version__, ort=ort.__version__, pnnx='20260526',
        cases_source=a.development.name, blocks={'backbone': dict(input_shape=[1, 1, 96, 96], output_shape=[1, 72, 6, 6], cases=errors)})
    manifest['sha256'] = {f.name: sha(f) for f in a.output.iterdir() if f.is_file()}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
