"""Pull native rig traces from the headset and build per-recording refiner sequences.

usage: python build_refiner_dataset.py --adb <adb.exe> --device-dir <experiments/palm-ft/rig-traces-v2> --local <dir>
                                       --sessions <session-results-root> --out <dir> [--skip-pull]
For each recording <UTC>: pulls rig.txt/views.txt/summary.txt, uses the recording's sanitized calibration from
<sessions>/live-mesh-<UTC>/frame/snapshot/device/persist, writes <out>/<UTC>.npz and a summary.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import refiner_data as rd  # noqa: E402

RECORDINGS = ['20261005T030445Z', '20261005T050930Z', '20261009T032721Z', '20261009T040502Z', '20261005T143912Z']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adb', required=True); ap.add_argument('--device-dir', required=True); ap.add_argument('--local', required=True)
    ap.add_argument('--sessions', required=True); ap.add_argument('--out', required=True); ap.add_argument('--skip-pull', action='store_true')
    ap.add_argument('--recordings', nargs='*', default=RECORDINGS)
    a = ap.parse_args()
    local, out = Path(a.local), Path(a.out)
    local.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for rec in a.recordings:
        d = local / rec
        if not a.skip_pull:
            d.mkdir(parents=True, exist_ok=True)
            for name in ('rig.txt', 'views.txt', 'summary.txt', 'times.txt', 'stderr.txt'):
                subprocess.run([a.adb, 'pull', f'{a.device_dir}/{rec}/{name}', str(d / name)], capture_output=True)
        persist = Path(a.sessions) / f'live-mesh-{rec}' / 'frame' / 'snapshot' / 'device' / 'persist'
        tracks = rd.build_sequences(d / 'rig.txt', d / 'views.txt', persist)
        rd.save_sequences(tracks, out / f'{rec}.npz')
        frames = rd.parse_rig_trace(d / 'rig.txt')
        fitted = sum(1 for _, hs in frames for h in hs if h['status'] == 'fitted')
        statuses = {}
        for _, hs in frames:
            for h in hs:
                statuses[h['status']] = statuses.get(h['status'], 0) + 1
        lengths = [len(t['t']) for t in tracks]
        summary[rec] = dict(frames=len(frames), fitted_hands=fitted, statuses=statuses, tracks=len(tracks),
                            track_frames=int(sum(lengths)), track_len_p50=float(np.median(lengths)) if lengths else 0,
                            unobserved_filled=int(sum((~t['observed']).sum() for t in tracks)),
                            stderr_tail=(d / 'stderr.txt').read_text(errors='replace').strip().splitlines()[-1:] if (d / 'stderr.txt').exists() else [])
        print(rec, json.dumps(summary[rec]))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
