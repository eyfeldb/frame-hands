"""Evaluate pinch rules on a rig trace: FK the fitted poses, measure the thumb-index tip gap, simulate the
driver's DashboardPinch rule and alternatives, list the pinch episodes (start s, duration, min gap).

usage: python pinch_trace_eval.py <rig.txt> <profiles-dir> [--side 0|1|both] [--events out.txt]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from refiner_data import RigFK, load_fhrig, parse_rig_trace  # noqa: E402

RIG_THUMB_TIP, RIG_INDEX_TIP, RIG_INDEX_MCP, RIG_PINKY_MCP, RIG_INDEX_PIP = 0, 1, 8, 17, 9


def fk_gaps(frames, profiles):
    """-> dict side -> list of (stamp_ns, source, gap_m, width_m, index_curl) for fitted hands."""
    import torch
    fks = {s: RigFK(load_fhrig(Path(profiles) / f'side{s}.fhrig')) for s in (0, 1)}
    rows = {0: [], 1: []}
    per_side = {0: [], 1: []}
    for stamp, hands in frames:
        for h in hands:
            if h['status'] == 'fitted':
                per_side[h['side']].append((stamp, h))
    for s in (0, 1):
        if not per_side[s]:
            continue
        a = torch.tensor(np.stack([h['angles'] for _, h in per_side[s]]))
        q = torch.tensor(np.stack([h['q'] for _, h in per_side[s]]))
        t = torch.tensor(np.stack([h['t'] for _, h in per_side[s]]))
        pts = fks[s].landmarks(a, q, t, s).numpy() / 1000.0        # m
        gap = np.linalg.norm(pts[:, RIG_THUMB_TIP] - pts[:, RIG_INDEX_TIP], axis=1)
        width = np.linalg.norm(pts[:, RIG_INDEX_MCP] - pts[:, RIG_PINKY_MCP], axis=1)
        # index tip to index MCP over the finger's extended length proxy (tip-PIP + PIP-MCP): 1 straight, small curled
        ext = np.linalg.norm(pts[:, RIG_INDEX_TIP] - pts[:, RIG_INDEX_PIP], axis=1) + np.linalg.norm(pts[:, RIG_INDEX_PIP] - pts[:, RIG_INDEX_MCP], axis=1)
        curl = np.linalg.norm(pts[:, RIG_INDEX_TIP] - pts[:, RIG_INDEX_MCP], axis=1) / np.maximum(ext, 1e-6)
        for i, (stamp, h) in enumerate(per_side[s]):
            rows[s].append((stamp, h['source'], float(gap[i]), float(width[i]), float(curl[i])))
    return rows


def simulate(rows, close_frac, close_lo, close_hi, open_frac, open_lo, open_hi, hold_ms, keep_timer=False, gap_key=2):
    """DashboardPinch state machine over one side's rows (sorted by stamp). -> list of episodes (start_s, dur_s, min_gap)."""
    armed = False; down = False; began = None; episodes = []; start = None; min_gap = None; prev_src = None
    t0 = rows[0][0] if rows else 0
    for r in rows:
        stamp, src, gap, width = r[0], r[1], r[gap_key], r[3]
        if src != prev_src:
            armed = False; down = False; began = None; prev_src = src
            if start is not None:
                episodes.append(((start - t0) / 1e9, (stamp - start) / 1e9, min_gap, start)); start = None
        close = float(np.clip(width * close_frac, close_lo, close_hi)); open_ = float(np.clip(width * open_frac, open_lo, open_hi))
        if gap >= open_:
            if down and start is not None:
                episodes.append(((start - t0) / 1e9, (stamp - start) / 1e9, min_gap, start)); start = None
            armed = True; down = False; began = None
        elif armed and gap <= close:
            if began is None:
                began = stamp
            elif (stamp - began) >= hold_ms * 1e6 and not down:
                down = True; start = stamp; min_gap = gap
            if down:
                min_gap = min(min_gap, gap)
        elif not down and not keep_timer:
            began = None          # driver today: a frame between close and open restarts the hold
    if start is not None:
        episodes.append(((start - t0) / 1e9, (rows[-1][0] - start) / 1e9, min_gap, start))
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('profiles'); ap.add_argument('--side', default='both'); ap.add_argument('--events')
    a = ap.parse_args()
    frames = parse_rig_trace(a.trace)
    rows = fk_gaps(frames, a.profiles)
    sides = [0, 1] if a.side == 'both' else [int(a.side)]
    out = []
    for s in sides:
        r = sorted(rows[s])
        if not r:
            continue
        gaps = np.array([x[2] for x in r]) * 1000; widths = np.array([x[3] for x in r]) * 1000
        print(f'== side {s} ({"left" if s == 0 else "right"}): {len(r)} fitted updates over {(r[-1][0] - r[0][0]) / 1e9:.0f} s')
        print(f'  hand width mm median {np.median(widths):.0f}; thumb-index gap mm 5% {np.percentile(gaps, 5):.0f} 25% {np.percentile(gaps, 25):.0f} median {np.median(gaps):.0f}')
        print('  gap histogram (mm, share of updates): ' + ', '.join(f'<{b} {100 * np.mean(gaps < b):.1f}%' for b in (10, 15, 20, 24, 30, 40)))
        rules = {
            'current (0.30w 16-30 / 0.52w 30-50, 50 ms)': (0.30, .016, .030, 0.52, .030, .050, 50),
            'A (0.25w 14-22 / 0.50w 30-45, 100 ms keep)': (0.25, .014, .022, 0.50, .030, .045, 100, True),
            'B (0.24w 13-20 / 0.50w 30-45, 100 ms keep)': (0.24, .013, .020, 0.50, .030, .045, 100, True),
            'C (0.22w 12-18 / 0.50w 30-45, 100 ms keep)': (0.22, .012, .018, 0.50, .030, .045, 100, True),
            'D (0.22w 12-18 / 0.50w 30-45, 150 ms keep)': (0.22, .012, .018, 0.50, .030, .045, 150, True),
            'E (0.20w 11-16 / 0.50w 30-45, 120 ms keep)': (0.20, .011, .016, 0.50, .030, .045, 120, True),
            'F (0.18w 10-14 / 0.50w 30-45, 120 ms keep)': (0.18, .010, .014, 0.50, .030, .045, 120, True),
        }
        for name, p in rules.items():
            ep = simulate(r, *p)
            total = sum(e[1] for e in ep)
            print(f'  {name}: {len(ep)} pinches, down {total:.1f} s total; durations s: ' + ' '.join(f'{e[1]:.2f}' for e in ep[:30]))
            if a.events:
                out.append(f'side {s} {name}\n' + '\n'.join(f'  start {e[0]:7.2f}s dur {e[1]:.2f}s min_gap {e[2] * 1000:.0f}mm stamp {e[3]}' for e in ep))
    if a.events:
        Path(a.events).write_text('\n'.join(out)); print('events ->', a.events)


if __name__ == '__main__':
    main()
