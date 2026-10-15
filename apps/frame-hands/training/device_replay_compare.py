"""Compare candidate vs installed models with the real native replay binary on the headset.

Pushes a candidate models directory to a new private experiment folder on Frame, then runs
the existing `ft-handreplay-rig` (tracker-replay-v9) over a recording that is already on the
device, once per model set, with the live environment flags and `--oracle` findability checks.
Pulls the summaries and audits back and prints a side-by-side of the acquisition statistics.

Safety: refuses to run while frame-hands-native.service is active (the replay pins CPUs 5-7
like the live tracker). It opens no cameras, starts no XR scene, installs nothing and leaves the
installed tracker untouched. Device paths are under /home/steamos/devkit-game/FrameHands/experiments/.

usage: python device_replay_compare.py --adb <adb.exe> --candidate <models-dir> --label <name>
                                       --recording <UTC id> [--seconds 60] [--oracle 30] [--scheduled]
                                       [--out <local-dir>] [--force]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEVICE_ROOT = '/home/steamos/devkit-game'
REPLAY = DEVICE_ROOT + '/FrameHandsNativeLitert-20261008/tracker-replay-v9/ft-handreplay-rig'
BASE_MODELS = DEVICE_ROOT + '/FrameHandsNativeLitert-20261008/live-v9/source/models/ncnn'
DEVICE_CFG = DEVICE_ROOT + '/FrameHandsNativeLitert-20261008/live-v9/source/device'
FLAGS = ('PAIR_CONTINUITY', 'OPEN_SIDE', 'IDENTITY_LATCH', 'ROTATION_SEARCH', 'FULL_RATE', 'FILTER_GUARD', 'OUTPUT_RECOVERY', 'JOINT_IDENTITY')


def sh(adb, cmd, timeout=600):
    r = subprocess.run([adb, 'shell', cmd], capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
    return r.returncode, r.stdout, r.stderr


def parse_summary(text):
    out = {}
    m = re.search(r'processed (\d+) \(([\d.]+)/s\), ([\d.]+) ms per step', text)
    if m:
        out.update(processed=int(m.group(1)), per_s=float(m.group(2)), ms_per_step=float(m.group(3)))
    m = re.search(r'hands per processed set: 0 (\d+)%, 1 (\d+)%, 2 (\d+)%;\s+a hand on the left (\d+)%, right (\d+)%, both (\d+)%', text)
    if m:
        out.update(sets_0hands_pct=int(m.group(1)), sets_1hand_pct=int(m.group(2)), sets_2hands_pct=int(m.group(3)), both_pct=int(m.group(6)))
    for side in ('left', 'right'):
        m = re.search(side + r'\s+tracks: (\d+), median ([\d.]+) s, total (\d+) s', text)
        if m:
            out[side + '_tracks'] = int(m.group(1)); out[side + '_track_median_s'] = float(m.group(2))
    m = re.search(r'views lost (\d+), handoff misses (\d+), dups (\d+), splits (\d+);\s+hands new (\d+), merged (\d+), forgotten (\d+)', text)
    if m:
        out.update(views_lost=int(m.group(1)), handoff_misses=int(m.group(2)), dups=int(m.group(3)), splits=int(m.group(4)), hands_new=int(m.group(5)), forgotten=int(m.group(7)))
    m = re.search(r'model calls: palm (\d+) \(([\d.]+)/s\), hand (\d+) \(([\d.]+)/s\)', text)
    if m:
        out.update(palm_calls=int(m.group(1)), hand_calls=int(m.group(3)))
    m = re.search(r'oracle, (\d+) sets: a left hand findable in (\d+), the tracker had it in (\d+) \((\d+)%\); right (\d+), had (\d+) \((\d+)%\)', text)
    if m:
        out.update(oracle_sets=int(m.group(1)), oracle_left_findable=int(m.group(2)), oracle_left_had_pct=int(m.group(4)), oracle_right_findable=int(m.group(5)), oracle_right_had_pct=int(m.group(7)))
    m = re.search(r'full search didn.t find: left (\d+), right (\d+)', text)
    if m:
        out.update(oracle_extra_left=int(m.group(1)), oracle_extra_right=int(m.group(2)))
    m = re.search(r'palm jitter.*?measured median ([\d.]+) mm, 90% ([\d.]+) mm', text)
    if m:
        out.update(jitter_median_mm=float(m.group(1)), jitter_p90_mm=float(m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adb', required=True); ap.add_argument('--candidate', required=True); ap.add_argument('--label', required=True)
    ap.add_argument('--recording', required=True, help='UTC id under FrameHands/private/')
    ap.add_argument('--seconds', type=float, default=60); ap.add_argument('--oracle', type=int, default=30)
    ap.add_argument('--scheduled', action='store_true', help='live busy scheduling instead of --all-sets')
    ap.add_argument('--threshold', default='0.2'); ap.add_argument('--out'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--replay-bin', default=REPLAY, help='replay binary on the device (default: tracker-replay-v9)')
    ap.add_argument('--baseline-models', default=BASE_MODELS)
    ap.add_argument('--candidate-env', default='', help="extra env for the candidate run, e.g. 'FRAME_HANDS_HISTORY=1'")
    ap.add_argument('--baseline-env', default='')
    ap.add_argument('--baseline-bin', default=None, help='replay binary for the baseline run (default: same as --replay-bin)')
    a = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.label):
        sys.exit('label must be a simple name')
    rc, out, _ = sh(a.adb, 'systemctl --user is-active frame-hands-native.service')
    if out.strip() == 'active' and not a.force:
        sys.exit('refusing: frame-hands-native.service is active on the headset (live hands would be disturbed); stop it first or pass --force')
    rec = f'{DEVICE_ROOT}/FrameHands/private/{a.recording}'
    rc, out, _ = sh(a.adb, f'test -f {rec}/sets.bin && echo ok')
    if 'ok' not in out:
        sys.exit('recording not found on device: ' + rec)
    exp = f'{DEVICE_ROOT}/FrameHands/experiments/palm-ft/{a.label}'
    rc, out, _ = sh(a.adb, f'test -e {exp} && echo exists')
    if 'exists' in out and not a.force:
        sys.exit('experiment folder exists on device: ' + exp)
    sh(a.adb, f'mkdir -p {exp}/models')
    for f in ('palm.ncnn.param', 'palm.ncnn.bin', 'hand.ncnn.param', 'hand.ncnn.bin'):
        subprocess.run([a.adb, 'push', str(Path(a.candidate) / f), f'{exp}/models/{f}'], check=True, capture_output=True)
    env = ' '.join(f'FRAME_HANDS_{k}=1' for k in FLAGS) + f' FRAME_HANDS_PALM_THRESHOLD={a.threshold} FRAME_JOB_DEVICE_ROOT={DEVICE_CFG}'
    results = {}
    runs = ([] if a.skip_baseline else [('baseline', a.baseline_models, a.baseline_env)]) + [('candidate', f'{exp}/models', a.candidate_env)]
    for name, models, extra in runs:
        folder = f'{exp}/{name}'
        sh(a.adb, f'mkdir -p {folder}')
        sched = '' if a.scheduled else ' --all-sets'
        binary = (a.baseline_bin or a.replay_bin) if name == 'baseline' else a.replay_bin
        cmd = (f'cd {folder} && {env} {extra} FRAME_HANDS_AUDIT={folder}/views.txt nice -n 5 {binary} {rec} --threads 3 --models {models} '
               f'--to {a.seconds} --oracle {a.oracle} --timeline {folder}/timeline.txt --step-times {folder}/times.txt{sched} > {folder}/summary.txt 2> {folder}/stderr.txt; echo rc=$?')
        print(f'[{name}] running replay on device ...', flush=True)
        rc, out, err = sh(a.adb, cmd, timeout=1800)
        print(out.strip(), err.strip())
        _, summary, _ = sh(a.adb, f'cat {folder}/summary.txt')
        _, stderr, _ = sh(a.adb, f'tail -5 {folder}/stderr.txt')
        results[name] = dict(summary=summary, stderr_tail=stderr, parsed=parse_summary(summary))
        print(summary)
    if a.out:
        local = Path(a.out); local.mkdir(parents=True, exist_ok=True)
        for name, _, _ in runs:
            subprocess.run([a.adb, 'pull', f'{exp}/{name}', str(local)], capture_output=True)
        (local / 'comparison.json').write_text(json.dumps(dict(recording=a.recording, seconds=a.seconds, oracle=a.oracle, scheduled=a.scheduled,
                                                                threshold=a.threshold, results={k: v['parsed'] for k, v in results.items()}), indent=1))
    keys = sorted({k for v in results.values() for k in v['parsed']})
    print('\n%-28s' % 'metric' + ''.join('%14s' % n for n, _, _ in runs))
    for k in keys:
        print('%-28s' % k + ''.join('%14s' % results[n]['parsed'].get(k, '') for n, _, _ in runs))


if __name__ == '__main__':
    main()
