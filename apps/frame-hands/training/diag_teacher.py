"""Diagnose native-fit vs UmeTrack-teacher discrepancy: constant offset, per-frame offset, scale, or shape.

usage: python diag_teacher.py --data <refiner-data> --teacher <refiner-teacher> --profiles <profiles> --recording <UTC>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import refiner_data as rd  # noqa: E402
from train_refiner import load_split, LM_FINGER  # noqa: E402


def procrustes(a, b):
    """Rigid (rotation+translation) alignment of a onto b, both (N,3). Returns residual rms (mm) and rotation angle (deg)."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    aligned = (Rm @ (a - ca).T).T + cb
    ang = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    return float(np.sqrt(((aligned - b) ** 2).sum(1).mean())), float(ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--profiles', required=True); ap.add_argument('--recording', required=True)
    a = ap.parse_args()
    import torch
    rigs = {0: rd.load_fhrig(Path(a.profiles) / 'side0.fhrig'), 1: rd.load_fhrig(Path(a.profiles) / 'side1.fhrig')}
    fks = {s: rd.RigFK(rigs[s]) for s in (0, 1)}
    tracks = load_split(a.data, [a.recording], rigs, a.teacher)
    offs, errs, err_after_offset, err_after_global, proc, wrist_d, scale_ratio, pershape = [], [], [], [], [], [], [], []
    all_off = []
    for tr in tracks:
        n = len(tr['t'])
        with torch.no_grad():
            pts = fks[tr['side']].landmarks(torch.tensor(tr['angles']), torch.tensor(tr['q']), torch.tensor(tr['tr']), tr['side']).numpy()
        tm = tr['teacher_w'].max(1) > 0
        for i in np.nonzero(tm)[0]:
            p, t = pts[i], tr['teacher'][i]
            off = (t - p).mean(0); all_off.append(off)
            errs.append(np.linalg.norm(p - t, axis=1).mean())
            err_after_offset.append(np.linalg.norm(p + off - t, axis=1).mean())
            r, ang = procrustes(p, t); proc.append((r, ang))
            wrist_d.append(np.linalg.norm(p[5] - t[5]))
            # scale: mean distance of fingertips from wrist
            sp = np.linalg.norm(p[[0, 1, 2, 3, 4]] - p[5], axis=1).mean(); st = np.linalg.norm(t[[0, 1, 2, 3, 4]] - t[5], axis=1).mean()
            scale_ratio.append(sp / st)
    all_off = np.array(all_off)
    g = all_off.mean(0)
    for tr in tracks:
        with torch.no_grad():
            pts = fks[tr['side']].landmarks(torch.tensor(tr['angles']), torch.tensor(tr['q']), torch.tensor(tr['tr']), tr['side']).numpy()
        tm = tr['teacher_w'].max(1) > 0
        for i in np.nonzero(tm)[0]:
            err_after_global.append(np.linalg.norm(pts[i] + g - tr['teacher'][i], axis=1).mean())
    proc = np.array(proc)
    report = dict(recording=a.recording, matched_frames=len(errs),
                  mean_error_mm=float(np.mean(errs)), p50=float(np.median(errs)),
                  global_offset_mm=g.tolist(), offset_std_mm=all_off.std(0).tolist(),
                  error_after_global_offset=float(np.mean(err_after_global)),
                  error_after_per_frame_offset=float(np.mean(err_after_offset)),
                  error_after_per_frame_rigid=float(np.mean(proc[:, 0])), rigid_rotation_deg_p50=float(np.median(proc[:, 1])),
                  wrist_distance_mm_p50=float(np.median(wrist_d)), scale_ratio_native_over_teacher_p50=float(np.median(scale_ratio)))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
