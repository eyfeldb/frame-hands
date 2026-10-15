"""Side-by-side: what Maestro drew (PC window screenshots, HHMMSS in the file name, PC clock) versus the hand joints
the headset tracked at the same instant, rendered from the headset's own head pose (scenario-recorder chunks:
'raw' rows carry head_world_m, 'shown' rows the 21 world points per hand). Clock: pc = headset + offset_s.
The right pane shows the tracked joints, the palm frame (x red, y green = wrist->middle MCP, z blue = palm normal)
and the direction a sleeve would take if it followed the palm (dashed), so a crooked wrist in Maestro can be
attributed to the joints, to the device-pose convention, or to the game.
usage: python compare_live.py <shots-dir> <chunks-dir> <out-dir> [offset_s=0.8] [max_dt_s=0.1]
"""
import json, math, re, sys
from pathlib import Path
import numpy as np, cv2

shots = Path(sys.argv[1]); chunks = Path(sys.argv[2]); out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
offset = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8; max_dt = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

heads = []; frames = []
for f in sorted(chunks.glob('scn-*.jsonl')):
    m = re.search(r'scn-(\d{2})(\d{2})(\d{2})', f.name); base = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
    for line in open(f, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get('kind') == 'raw' and r.get('context_valid') and r.get('head_world_m'):
            heads.append((base + r['probe_t'], np.array([row[:4] for row in r['head_world_m'][:3]], float)))
        elif r.get('kind') == 'shown' and r.get('hands'):
            frames.append((base + r['probe_t'], {h['side']: np.array(h['points'], float) for h in r['hands']}))
heads.sort(key=lambda x: x[0]); frames.sort(key=lambda x: x[0])
ht = np.array([h[0] for h in heads]); ft = np.array([x[0] for x in frames])
print(f'{len(heads)} head poses, {len(frames)} hand frames, {ht[0] if len(ht) else 0:.0f}..{ht[-1] if len(ht) else 0:.0f} s-of-day')

def nearest(ts, t):
    i = int(np.searchsorted(ts, t)); c = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    return min(c, key=lambda j: abs(ts[j] - t)) if c else None

def to_head(H, p):   # world -> head frame (H = 3x4 head_world_m, rows of rotation + translation)
    R = H[:, :3]; t = H[:, 3]; return R.T @ (p - t)

def render(points_head, size=720):
    img = np.zeros((size, size, 3), np.uint8); f = size / 2
    def proj(p):
        if p[2] >= -0.05: return None
        return (int(size / 2 + f * p[0] / -p[2]), int(size / 2 - f * p[1] / -p[2]))
    for side, pts in points_head.items():
        col = (90, 200, 255) if side == 0 else (120, 255, 120)
        uv = [proj(p) for p in pts]
        for a, b in BONES:
            if uv[a] and uv[b]: cv2.line(img, uv[a], uv[b], col, 2)
        # palm frame
        y = pts[9] - pts[0]; y /= np.linalg.norm(y) + 1e-9
        x = pts[5] - pts[17]; x -= y * (x @ y); x /= np.linalg.norm(x) + 1e-9
        z = np.cross(x, y)
        for axis, c in ((x, (0, 0, 255)), (y, (0, 255, 0)), (z, (255, 80, 0))):
            a = proj(pts[0]); b = proj(pts[0] + axis * 0.05)
            if a and b: cv2.line(img, a, b, c, 2)
        # sleeve if it followed the palm: 20 cm back along -y from the wrist
        a = proj(pts[0]); b = proj(pts[0] - y * 0.20)
        if a and b:
            for k in range(8):
                p0 = (int(a[0] + (b[0] - a[0]) * k / 8), int(a[1] + (b[1] - a[1]) * k / 8)); p1 = (int(a[0] + (b[0] - a[0]) * (k + .5) / 8), int(a[1] + (b[1] - a[1]) * (k + .5) / 8))
                cv2.line(img, p0, p1, col, 1)
        if uv[0]: cv2.putText(img, 'L' if side == 0 else 'R', (uv[0][0] + 6, uv[0][1]), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 1)
    return img

n = 0
for shot in sorted(shots.glob('*.png')):
    m = re.search(r'(\d{2})(\d{2})(\d{2})\.png$', shot.name)
    if not m: continue
    t_pc = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]); t = t_pc - offset
    j = nearest(ft, t); k = nearest(ht, t)
    if j is None or k is None or abs(ft[j] - t) > max_dt or abs(ht[k] - t) > max_dt: continue
    H = heads[k][1]; pts = {s: np.array([to_head(H, p) for p in P]) for s, P in frames[j][1].items()}
    truth = render(pts)
    left = cv2.imread(str(shot))
    if left is None: continue
    h = truth.shape[0]; scale = h / left.shape[0]; left = cv2.resize(left, (int(left.shape[1] * scale), h))
    both = np.hstack([left, truth])
    cv2.putText(both, f'{shot.name}  headset dt {ft[j] - t:+.3f}s', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.imwrite(str(out / ('cmp-' + shot.name)), both); n += 1
print(f'{n} comparisons written to {out}')
