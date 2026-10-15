"""Feed the PC hand driver known hands (recorded from the headset) so that what SteamVR Home / Maestro draw can be
compared against a ground-truth render of the same joints, without the headset tracker in the loop.

Reads a scenario-recorder chunk (scn-*.jsonl: 'shown' rows = FHMODEL2 21 world points per hand at 90 Hz, 'raw'
rows = pose.json packets with the headset head pose at capture), re-places each hand relative to the PC's current
head pose exactly like pc_relay.py (pc_transport.WorldAlignment), and writes FHMODEL2 packets into the registered
driver bundle's mailbox at 30 Hz. The wireless relay must not be running (it would overwrite the mailbox).

modes:
  replay  : play the chunk in real time (loops)
  tour    : hold picked frames for --hold seconds each: open hand, fist, thumbs-up (searched by finger bends),
            then the open hand rotated about the wrist by 45 and 90 deg around each of x, y, z
Every frame change also writes the injected joints (PC raw space and head space) to <out>/truth-<n>.json and a
ground-truth render <out>/truth-<n>.png (head viewpoint, 90 deg fov) for side-by-side comparison with the VR View
screenshot taken at the same moment (vrview_shot.ps1).
usage: python maestro_inject.py <chunk.jsonl> --mode tour|replay [--hold 6] [--out <dir>] [--side 0|1|both]
"""
import argparse, json, math, os, re, struct, subprocess, sys, time
from pathlib import Path
BASE = Path(__file__).resolve().parent; HANDS = BASE.parents[1]; ROOT = HANDS.parents[1]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HANDS / 'pc-input')); sys.path.insert(0, str(HANDS / 'exploratory'))
from pc_transport import HEADER, HAND, WorldAlignment, map_point_world, rigid  # noqa: E402
from relay import publish_atomic  # noqa: E402

FINGERS = {'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12), 'ring': (13, 14, 15, 16), 'pinky': (17, 18, 19, 20)}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def tick(): return time.perf_counter_ns() // 1000000


def ang(a, b, c):
    v1 = [a[i] - b[i] for i in range(3)]; v2 = [c[i] - b[i] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in v1)); n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9: return float('nan')
    return 180 - math.degrees(math.acos(max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))))


def inv_rows(m):
    R = [[m[j][i] for j in range(3)] for i in range(3)]; t = [m[i][3] for i in range(3)]
    return [R[i] + [-sum(R[i][k] * t[k] for k in range(3))] for i in range(3)]


def apply_rows(m, p): return [sum(m[i][k] * p[k] for k in range(3)) + m[i][3] for i in range(3)]


def rot_about(points, pivot, axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg)); x, y, z = axis
    R = [[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s], [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s], [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]]
    out = []
    for p in points:
        d = [p[i] - pivot[i] for i in range(3)]; out.append([sum(R[i][k] * d[k] for k in range(3)) + pivot[i] for i in range(3)])
    return out


class Head:
    """PC head pose through the same pose-service.exe the relay uses."""
    def __init__(self):
        dll = 'C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/openvr_api.dll'
        self.p = subprocess.Popen([str(ROOT / '.local/hand-alternatives/pose-service.exe'), dll], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    def ask(self, age_ms=0):
        self.p.stdin.write((str(age_ms) + '\n').encode()); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())
    def close(self):
        try: self.p.stdin.close(); self.p.wait(timeout=2)
        except Exception: self.p.kill()


def load_chunk(path):
    heads = []; frames = []   # heads: (capture_ns, rows); frames: (capture_ns, {side: points})
    for line in open(path, errors='replace'):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get('kind') == 'raw' and r.get('context_valid') and r.get('head_world_m'):
            heads.append((r['capture_ns'], [row[:4] for row in r['head_world_m'][:3]]))
        elif r.get('kind') == 'shown' and r.get('hands'):
            frames.append((max(h['capture_ns'] for h in r['hands']), {h['side']: h['points'] for h in r['hands']}))
    heads.sort(); return heads, frames


MP_TO_RIG = [5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]


def couple(angles):
    """Presentation couplings of processor/baseline_motion.gd pose_angles(), on the 20 rig angles."""
    a = list(angles)
    for f in range(1, 5):
        pip = a[4 * f + 2]
        if pip > 0.7: a[4 * f + 3] = max(a[4 * f + 3], min(1.43, 0.67 * pip))
    if a[14] > 1.3:
        for k in range(1, 4): a[16 + k] = max(a[16 + k], 0.85 * a[12 + k])
    return a


def load_rig(path, profiles, seconds, do_couple):
    """Fitted frames of a rig trace at the given seconds -> head-space 21 points (metres, mediapipe order).
    A synthetic identity head pose makes --center treat the points as head space directly."""
    sys.path.insert(0, str(HANDS / 'training')); import numpy as np, torch
    from refiner_data import RigFK, load_fhrig, parse_rig_trace
    fks = {s: RigFK(load_fhrig(Path(profiles) / f'side{s}.fhrig')) for s in (0, 1)}
    trace = parse_rig_trace(path); t0 = trace[0][0]; frames = []
    for sec in seconds:
        stamp, hands = min(trace, key=lambda f: abs(f[0] - (t0 + sec * 1e9)))
        out = {}
        for h in hands:
            if h['status'] != 'fitted': continue
            ang = couple(h['angles']) if do_couple else list(h['angles'])
            pts = fks[h['side']].landmarks(torch.tensor(np.array([ang], float)), torch.tensor(np.array([h['q']])), torch.tensor(np.array([h['t']])), h['side']).numpy()[0] / 1000.0
            # mediapipe landmark 1 (thumb CMC) has no rig point: use the wrist as its position (as the processor does)
            out[h['side']] = [list(map(float, pts[r])) if r >= 0 else list(map(float, pts[5])) for r in MP_TO_RIG]
        if out: frames.append((int(sec * 1e9), out))
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    return [(f[0], identity) for f in frames], frames


def nearest_head(heads, capture_ns):
    import bisect
    i = bisect.bisect_left([h[0] for h in heads], capture_ns); c = [heads[j] for j in (i - 1, i) if 0 <= j < len(heads)]
    return min(c, key=lambda h: abs(h[0] - capture_ns))[1] if c else None


def render(points_head, path):
    """Head-frame points (metres, -z forward) -> 640x640 PNG, 90 deg fov, with bones."""
    try: import numpy as np, cv2
    except ImportError: return
    img = np.zeros((640, 640, 3), np.uint8); f = 320.0
    def proj(p):
        if p[2] >= -0.05: return None
        return (int(320 + f * p[0] / -p[2]), int(320 - f * p[1] / -p[2]))
    for side, pts in points_head.items():
        col = (90, 200, 255) if side == 0 else (120, 255, 120)
        uv = [proj(p) for p in pts]
        for a, b in BONES:
            if uv[a] and uv[b]: cv2.line(img, uv[a], uv[b], col, 2)
        if uv[0]: cv2.putText(img, 'L' if side == 0 else 'R', (uv[0][0] + 6, uv[0][1]), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 1)
    cv2.putText(img, 'truth: injected joints from the head viewpoint (90 fov)', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('chunk'); ap.add_argument('--mode', default='tour'); ap.add_argument('--hold', type=float, default=6)
    ap.add_argument('--out', default=None); ap.add_argument('--side', default='both'); ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--center', action='store_true', help='place each wrist at a fixed spot in front of the PC head (L -0.15 / R +0.15 m, 5 cm down, 45 cm ahead), keeping the recorded orientation and articulation')
    ap.add_argument('--mirror-left', action='store_true', help='with --center: the right hand is the recorded LEFT hand mirrored in head space (x -> -x), so both gloves get identical geometry and any left/right difference in the game is the driver/retarget')
    ap.add_argument('--rig', default=None, help='instead of a chunk: a rig trace (rig.txt); frames are forward-kinematics of the fit at --rig-seconds (head space, implies --center)')
    ap.add_argument('--profiles', default=None, help='rig profiles dir for --rig'); ap.add_argument('--rig-seconds', type=float, nargs='*', default=[])
    ap.add_argument('--poses', default=None, help='tour: comma-separated subset/order filter of pose names (open,fist,thumbs,pinch,open_rotx45,...)')
    ap.add_argument('--couple', action='store_true', help='with --rig: apply the presentation couplings (DIP follows PIP, pinky follows ring in a fist) to the fitted angles before FK, to preview that fix')
    a = ap.parse_args()
    out = Path(a.out or (ROOT / '.local/hand-alternatives/results' / ('inject-' + time.strftime('%Y%m%dT%H%M%S')))); out.mkdir(parents=True, exist_ok=True)
    if subprocess.run(['powershell', '-NoProfile', '-Command', "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'pc_relay' } | Measure-Object | Select-Object -ExpandProperty Count"], capture_output=True, text=True).stdout.strip() not in ('', '0'):
        raise SystemExit('the wireless relay is running; stop it first (receiver.stop) or it will overwrite the mailbox')
    dep = json.loads((ROOT / '.local/hand-alternatives/steamvr-model/deployment.json').read_text())
    dest = Path(dep['bundle']) / 'bin/win64/input.bin'; tmp = dest.with_name('inject.tmp')
    if a.rig:
        heads, frames = load_rig(a.rig, a.profiles, a.rig_seconds, a.couple); a.center = True
    else:
        heads, frames = load_chunk(a.chunk)
    if not frames or not heads: raise SystemExit('chunk has no hands or head poses')
    print(f'{len(frames)} hand frames, {len(heads)} head poses; mailbox {dest}')
    sides = [0, 1] if a.side == 'both' else [int(a.side)]

    def bends(pts): return {n: ang(pts[i[0]], pts[i[1]], pts[i[2]]) for n, i in FINGERS.items()}
    def pick(pred):
        best = None
        for cap, hands in frames:
            if not all(s in hands for s in sides): continue
            score = sum(pred(bends(hands[s]), hands[s]) for s in sides)
            if best is None or score > best[0]: best = (score, cap, hands)
        return best[1], best[2]
    script = []
    if a.mode == 'tour':
        open_cap, open_hands = pick(lambda b, p: -sum(b.values()))                                  # straightest fingers
        fist_cap, fist_hands = pick(lambda b, p: min(b['index'], b['middle'], b['ring']))            # most curled
        thumb_cap, thumb_hands = pick(lambda b, p: min(b['index'], b['middle'], b['ring']) - 2 * ang(p[2], p[3], p[4]))  # curled fingers, straight thumb
        def gap(p): return math.dist(p[4], p[8]) / max(math.dist(p[5], p[17]), 1e-6)
        pinch_cap, pinch_hands = pick(lambda b, p: -gap(p))                                        # thumb tip closest to index tip
        script = [('open', open_cap, open_hands), ('fist', fist_cap, fist_hands), ('thumbs', thumb_cap, thumb_hands), ('pinch', pinch_cap, pinch_hands)]
        for axis_name, axis in (('x', (1, 0, 0)), ('y', (0, 1, 0)), ('z', (0, 0, 1))):
            for deg in (45, 90):
                rotated = {s: rot_about(open_hands[s], open_hands[s][0], axis, deg) for s in sides if s in open_hands}
                script.append((f'open_rot{axis_name}{deg}', open_cap, rotated))
        if a.poses: script = [s for s in script if s[0] in a.poses.split(',')]
        for name, cap, hands in script:
            for s in sides:
                if s in hands: print(f'  {name} side {s}: thumb-index gap {gap(hands[s]):.2f} of hand width (driver pinch value {max(0, min(1, (.65 - gap(hands[s])) / .45)):.2f})')
        print('tour: ' + ', '.join(s[0] for s in script))
    # sequence must never go backwards across injector restarts (the driver drops lower sequences until SteamVR restarts)
    head = Head(); align = WorldAlignment(); seq = time.time_ns() // 1000000; n = 0; start = time.monotonic(); last_name = None
    try:
        while time.monotonic() - start < a.seconds:
            if a.mode == 'tour':
                idx = int((time.monotonic() - start) // a.hold) % len(script); name, cap, hands = script[idx]
            else:
                i = int(((time.monotonic() - start) * 90) % len(frames)); cap, hands = frames[i]; name = f'replay{i}'
            native = nearest_head(heads, cap)
            pc = head.ask(0)
            if not pc.get('valid'): time.sleep(.033); continue
            now = tick(); world = align.update(native, rigid(pc['capture']), now)
            output = [HAND.pack(0, 0, 0, 0, *([0.] * 63)) for _ in range(2)]; truth = {}
            for s in sides:
                if s not in hands: continue
                if a.center:
                    # recorded world -> recorded head space, re-anchor the wrist, -> PC world through the PC head
                    inv_native = inv_rows(native); src = hands[0] if (a.mirror_left and s == 1 and 0 in hands) else hands[s]
                    local = [apply_rows(inv_native, p) for p in src]
                    if a.mirror_left and s == 1 and 0 in hands: local = [[-p[0], p[1], p[2]] for p in local]
                    target = [-0.15 if s == 0 else 0.15, -0.05, -0.45]; shift = [target[i] - local[0][i] for i in range(3)]
                    local = [[p[i] + shift[i] for i in range(3)] for p in local]
                    mapped = [apply_rows(rigid(pc['capture']), p) for p in local]
                else:
                    mapped = [map_point_world(p, world) for p in hands[s]]
                capture = now - 20; until = capture + 170
                output[s] = HAND.pack(100 + s, 1, capture, until, *[v for p in mapped for v in p])
                pc_rows = rigid(pc['capture']); inv = inv_rows(pc_rows)
                truth[s] = [apply_rows(inv, p) for p in mapped]
            seq += 1
            publish_atomic(tmp, dest, HEADER.pack(b'FHMODEL2', 2, 592, seq, now, now) + b''.join(output))
            if name != last_name:
                n += 1; last_name = name
                (out / f'truth-{n}-{name}.json').write_text(json.dumps(dict(name=name, head_space=truth, t=time.time())))
                render(truth, out / f'truth-{n}-{name}.png'); print(f'{time.strftime("%H:%M:%S")} injecting {name}', flush=True)
            time.sleep(.033)
    finally:
        head.close()
        try: dest.unlink()
        except OSError: pass
        print('injector stopped; mailbox removed')


if __name__ == '__main__': main()
