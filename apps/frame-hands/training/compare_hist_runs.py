"""Print per-epoch dev metrics of train_hist.py runs side by side.

usage: python compare_hist_runs.py <run-dir> [<run-dir> ...]
"""
import json
import sys
from pathlib import Path

for run in sys.argv[1:]:
    run = Path(run)
    log = json.load(open(run / 'log.json'))
    best = json.load(open(run / 'best.json')) if (run / 'best.json').exists() else None
    print(f'== {run.name}: {len(log) - 1} epochs, best epoch {best["epoch"] if best else "?"}')
    f = lambda v, p=2: 'n/a' if v is None else f'{v:.{p}f}'   # noqa: E731
    for e in log:
        d = e['dev']; g = d['given']; o = d['occluded_given']; n = d['none']
        print(f"ep {e['epoch']:2d} given: err {f(g['tracked_err'])} stereo {f(g['tracked_err_stereo'])} hard {f(g['hard_err'], 1)} "
              f"pres {f(g['tracked_pres'], 3)} right {f(g['right_acc'], 3)} neg {f(g['neg_pres'], 4)} | occluded err {f(o['tracked_err'])} | no-history err {f(n['tracked_err'])}")
