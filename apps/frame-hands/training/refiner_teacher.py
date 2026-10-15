"""Teacher targets from the PC UmeTrack path: per recording, per capture stamp and side, the 21 model
landmarks in the capture head frame (mm) plus per-landmark uncertainty sigmas.

Recomputed from predictions.jsonl (joint_angles, wrist_world_mm, capture_pose) exactly as
exploratory/replay_umetrack.py does: skin_landmarks(generic hand model, angles, wrist with x column
mirrored for the right hand), then head_from_world applied. The rig used by the native tracker was
exported from the same generic model, so landmark order matches hand_rig.h.

usage: python refiner_teacher.py --upstream <UmeTrack dir> --sessions <results root> --out <dir> [--recordings ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RECORDINGS = ['20261005T030445Z', '20261005T050930Z', '20261009T032721Z', '20261009T040502Z', '20261005T143912Z']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', required=True); ap.add_argument('--sessions', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--recordings', nargs='*', default=RECORDINGS)
    a = ap.parse_args()
    import torch
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exploratory'))   # local 'rotations' shim used by the pinned UmeTrack edit
    from lib.common.hand import HandModel
    from lib.common.hand_skinning import skin_landmarks
    raw = json.loads((Path(a.upstream) / 'dataset/generic_hand_model.json').read_text())
    hand = HandModel(**{k: torch.tensor(v) if isinstance(v, list) else v for k, v in raw.items()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for rec in a.recordings:
        path = Path(a.sessions) / f'live-mesh-{rec}' / 'predictions.jsonl'
        stamps, sides, lms, sigmas, memory = [], [], [], [], []
        frames = 0; without_pose = 0
        with path.open() as f:
            for line in f:
                o = json.loads(line); frames += 1
                cp = o.get('capture_pose') or {}
                if not cp.get('valid') or not o['hands']:
                    without_pose += bool(o['hands']); continue
                world = np.eye(4); world[:3, :] = np.array(cp['capture'], float).reshape(3, 4); world[:3, 3] *= 1000.0
                head_from_world = np.linalg.inv(world)
                for h in o['hands']:
                    side = int(h['side'])
                    wrist = torch.tensor(np.array(h['wrist_world_mm'], np.float32))
                    if side:
                        wrist[:, 0] *= -1
                    with torch.inference_mode():
                        pts = skin_landmarks(hand, torch.tensor(np.array(h['joint_angles'], np.float32)), wrist).numpy().astype(np.float64)
                    hp = (head_from_world[:3, :3] @ pts.T).T + head_from_world[:3, 3]
                    stamps.append(int(o['capture_raw_ns'])); sides.append(side); lms.append(hp)
                    sigmas.append(np.array(h.get('uncertainty_raw', [0.0] * 21), float)); memory.append(bool(h.get('memory', False)))
        np.savez_compressed(out / f'{rec}.npz', stamps=np.array(stamps, np.int64), sides=np.array(sides, np.int8),
                            landmarks=np.array(lms), sigmas=np.array(sigmas), memory=np.array(memory))
        print(rec, dict(frames=frames, teacher_hands=len(stamps), hands_without_pose=without_pose,
                        left=int(np.sum(np.array(sides) == 0)), right=int(np.sum(np.array(sides) == 1))), flush=True)


if __name__ == '__main__':
    main()
