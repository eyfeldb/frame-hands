"""Does the installed pnnx reproduce the deployed ncnn param files from the source ONNX?

Re-runs the upstream conversion (tools/convert_models.py logic, including the Pad and
Permute patches) into a scratch directory and diffs the resulting .param against the
installed hand/palm params. Validates the export path before any retraining.

usage: python check_pnnx_roundtrip.py <source-models-dir> <installed-models-dir> <scratch-dir>
"""
import difflib
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MODELS = [('palm', 'palm_detection_mediapipe_2023feb', 192), ('hand', 'handpose_estimation_mediapipe_2023feb', 224)]


def patch(param_text, pnnx_param_text):
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
        nin, nout = int(f[2]), int(f[3])
        ins = ['in0' if b == blob else b for b in f[4:4 + nin]]
        body[i] = ' '.join(f[:4] + ins + f[4 + nin:])
    return '\n'.join(['7767517', '%d %d' % (nlayers - 1, nblobs - 1)] + body) + '\n'


def find_pnnx():
    import pnnx as pnnx_pkg
    root = Path(pnnx_pkg.__file__).parent
    cands = glob.glob(str(root / '**' / 'pnnx*.exe'), recursive=True) + glob.glob(str(root / '**' / 'pnnx'), recursive=True)
    cands = [c for c in cands if os.path.isfile(c) and not c.endswith('.py')]
    return cands[0] if cands else None


def main():
    src, installed, scratch = map(Path, sys.argv[1:4])
    scratch.mkdir(parents=True, exist_ok=True)
    exe = find_pnnx()
    print('pnnx executable:', exe)
    for short, name, size in MODELS:
        work = scratch / short
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        shutil.copy(src / (name + '.onnx'), work)
        cmd = [exe, name + '.onnx', 'inputshape=[1,%d,%d,3]' % (size, size), 'fp16=1']
        r = subprocess.run(cmd, cwd=work, capture_output=True)
        err = (r.stderr or b'').decode('utf-8', 'replace').strip()
        print(short, 'pnnx exit', r.returncode, '| stderr tail:', err.splitlines()[-2:] if err else '')
        files = sorted(p.name for p in work.iterdir())
        print('  outputs:', files)
        ncnn_param = work / (name + '.ncnn.param')
        pnnx_param = work / (name + '.pnnx.param')
        if not ncnn_param.exists():
            print('  no ncnn param produced'); continue
        patched = patch(ncnn_param.read_text(), pnnx_param.read_text())
        (work / (short + '.ncnn.param')).write_text(patched)
        shutil.copy(work / (name + '.ncnn.bin'), work / (short + '.ncnn.bin'))
        ref = (installed / (short + '.ncnn.param')).read_text()
        same = patched == ref
        print('  param identical to installed:', same)
        if not same:
            diff = list(difflib.unified_diff(ref.splitlines(), patched.splitlines(), 'installed', 'roundtrip', lineterm='', n=0))
            print('  diff lines:', len(diff)); print('\n'.join(diff[:40]))
        bin_ref = (installed / (short + '.ncnn.bin')).stat().st_size
        bin_new = (work / (short + '.ncnn.bin')).stat().st_size
        print('  bin bytes installed/roundtrip:', bin_ref, bin_new, 'identical bytes:', (installed / (short + '.ncnn.bin')).read_bytes() == (work / (short + '.ncnn.bin')).read_bytes())


if __name__ == '__main__':
    main()
