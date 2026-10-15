"""Analyse a capture_session CSV: what SteamVR showed the game versus what the relay fed it.

Per hand: availability (mailbox hand present vs SteamVR device connected+valid), lag between the mailbox wrist and
the SteamVR device position (cross-correlation over +-300 ms, now-pose and +40 ms predicted pose), position
agreement, jitter (median absolute second difference per 11 ms) of mailbox vs SteamVR vs predicted, and head motion.
usage: python analyze_capture.py <capture.csv>
"""
import csv, sys
import numpy as np


def load(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f); rows = list(r)
    cols = {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}
    return cols


def lag_ms(t, a, b, max_lag=300):
    """Lag (ms) by which series b trails a, from cross-correlation of resampled 1-D speed signals."""
    grid = np.arange(t[0], t[-1], 4.0)
    ai = np.interp(grid, t, a); bi = np.interp(grid, t, b)
    ai -= ai.mean(); bi -= bi.mean()
    if ai.std() < 1e-6 or bi.std() < 1e-6:
        return None, 0.0
    best, best_c = 0, -2
    for lag in range(-max_lag // 4, max_lag // 4 + 1):
        if lag >= 0:
            x, y = ai[:len(ai) - lag] if lag else ai, bi[lag:]
        else:
            x, y = ai[-lag:], bi[:len(bi) + lag]
        if len(x) < 100:
            continue
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best_c:
            best_c, best = c, lag
    return best * 4, best_c


def jitter(t, p, mask):
    """Median absolute second difference of a position series (mm), only where mask holds for 3 consecutive samples."""
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]; m = mask[2:] & mask[1:-1] & mask[:-2]
    v = np.linalg.norm(d2[m], axis=1) * 1000
    return float(np.median(v)) if len(v) else float('nan'), float(np.percentile(v, 90)) if len(v) else float('nan')


def main():
    c = load(sys.argv[1]); t = c['t_ms']; dur = (t[-1] - t[0]) / 1000
    print(f'{len(t)} samples over {dur:.1f} s ({len(t) / dur:.0f} Hz)')
    hmd = np.stack([c['hmd_x'], c['hmd_y'], c['hmd_z']], 1)
    hv = np.linalg.norm(np.diff(hmd, axis=0), axis=1) / (np.diff(t) / 1000 + 1e-9)
    print(f'head: valid {100 * c["hmd_valid"].mean():.0f}%, speed median {np.median(hv):.2f} m/s, 90% {np.percentile(hv, 90):.2f} m/s')
    print(f'mailbox readable {100 * c["mb_ok"].mean():.0f}%; distinct packets {len(np.unique(c["mb_seq"]))}')
    for s, name in (('l', 'left'), ('r', 'right')):
        mb_on = c[f'mb_{s}_flags'] > 0
        sv_on = (c[f'{s}_conn'] > 0) & (c[f'{s}_valid'] > 0)
        print(f'== {name}: device index {int(c[f"{s}_idx"][-1])}')
        print(f'  present: mailbox {100 * mb_on.mean():.1f}% of samples, SteamVR connected+valid {100 * sv_on.mean():.1f}%, '
              f'SteamVR valid while mailbox present {100 * (sv_on & mb_on).sum() / max(mb_on.sum(), 1):.1f}%, '
              f'SteamVR valid while mailbox absent {100 * (sv_on & ~mb_on).sum() / max((~mb_on).sum(), 1):.1f}% (hold)')
        if mb_on.sum() < 200:
            print('  too little hand data'); continue
        mb = np.stack([c[f'mb_{s}_wx'], c[f'mb_{s}_wy'], c[f'mb_{s}_wz']], 1)
        sv = np.stack([c[f'{s}_x'], c[f'{s}_y'], c[f'{s}_z']], 1)
        svp = np.stack([c[f'{s}p_x'], c[f'{s}p_y'], c[f'{s}p_z']], 1)
        both = mb_on & sv_on
        if both.sum() > 200:
            off = sv[both] - mb[both]
            print(f'  SteamVR now-pose minus mailbox wrist: mean offset {1000 * off.mean(0).round(3)} mm, |diff| median {1000 * np.median(np.linalg.norm(off, axis=1)):.1f} mm, 90% {1000 * np.percentile(np.linalg.norm(off, axis=1), 90):.1f} mm')
            # lag from wrist speed signals
            # Lag from the position along the dominant motion axis (principal component), so a lead shows as a
            # negative lag; speed magnitude would hide the sign.
            tb = t[both]; centred = mb[both] - mb[both].mean(0); axis = np.linalg.svd(centred, full_matrices=False)[2][0]
            mbs = centred @ axis; svs = (sv[both] - mb[both].mean(0)) @ axis; sps = (svp[both] - mb[both].mean(0)) @ axis
            for label, sig in (('SteamVR now-pose', svs), ('SteamVR +40 ms predicted', sps)):
                lag, corr = lag_ms(tb, mbs, sig)
                print(f'  lag of {label} relative to mailbox: {lag:+d} ms (negative = leads; corr {corr:.2f})' if lag is not None else f'  lag of {label}: no motion')
            print(f'  jitter mm (median / 90% second difference): mailbox {jitter(t, mb, mb_on)}, SteamVR now {jitter(t, sv, sv_on)}, SteamVR predicted {jitter(t, svp, sv_on)}')
        curl = c[f'{s}_curl_index']; ok = curl >= 0
        print(f'  skeleton summary available {100 * ok.mean():.0f}% (index curl median {np.median(curl[ok]) if ok.any() else float("nan"):.2f})')
        age = (t - 0) * 0  # placeholder to keep structure simple
        ages = []
        for i in np.where(mb_on)[0][::5]:
            ages.append(c['mb_published'][i] - c[f'mb_{s}_capture'][i])
        if ages:
            ages = np.array(ages); print(f'  mailbox capture age at publish (ms): median {np.median(ages):.0f}, 90% {np.percentile(ages, 90):.0f}, max {ages.max():.0f}')
        del age


if __name__ == '__main__':
    main()
