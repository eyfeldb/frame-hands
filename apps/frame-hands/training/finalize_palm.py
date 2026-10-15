"""Export a fine-tuned palm checkpoint to ncnn next to the unchanged hand model, ready for eval_palm.py
and for the device.

usage: python finalize_palm.py --source <palm.onnx> --checkpoint <best.pt> --installed <models-dir> --out <models-out-dir>
Writes <out>/palm.ncnn.{param,bin} (new) and copies <out>/hand.ncnn.{param,bin} (unchanged), plus manifest.json with hashes.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import export_ncnn as ex  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--installed', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from palm_model import load_model, load_state
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = load_state(load_model(a.source), a.checkpoint)
    param, binf = ex.export(model, 'palm', out, out / 'work-export')
    for name in ('hand.ncnn.param', 'hand.ncnn.bin'):
        shutil.copy(Path(a.installed) / name, out / name)
    manifest = dict(source_onnx=sha(a.source), checkpoint=sha(a.checkpoint), checkpoint_path=str(a.checkpoint),
                    files={p.name: sha(p) for p in out.glob('*.ncnn.*')}, installed_hand_unchanged=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
