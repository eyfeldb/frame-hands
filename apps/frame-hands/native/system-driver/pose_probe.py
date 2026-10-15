"""Live pose read-out from the FHMODEL2 mailbox: per hand, the bend at each finger's PIP and DIP (degrees, from
the 21 published points), the thumb straightness, and the wrist-to-middle-tip direction, printed every 0.5 s.
A proper fist shows PIP ~90-110 and DIP ~60-80; a vertical thumbs-up has the thumb nearly straight (<25) and the
hand direction mostly along +y.
usage: python3 pose_probe.py <input.bin> [seconds]"""
import math, struct, sys, time
from pathlib import Path
p = Path(sys.argv[1]); seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')
FINGERS = {'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12), 'ring': (13, 14, 15, 16), 'pinky': (17, 18, 19, 20)}

def ang(a, b, c):
    v1 = [a[i] - b[i] for i in range(3)]; v2 = [c[i] - b[i] for i in range(3)]
    n1 = math.sqrt(sum(x * x for x in v1)); n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9: return float('nan')
    cosv = max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))
    return 180.0 - math.degrees(math.acos(cosv))   # 0 = straight, 90 = right angle bend

end = time.monotonic() + seconds; last = None
while time.monotonic() < end:
    try: d = p.read_bytes()
    except OSError: time.sleep(.05); continue
    if len(d) != 592: time.sleep(.05); continue
    seq = H.unpack_from(d)[3]
    if seq == last: time.sleep(.02); continue
    last = seq
    out = []
    for s in range(2):
        ident, flags, cap, until, *pts = HAND.unpack_from(d, H.size + s * HAND.size)
        if not flags: out.append(f'{"L" if s == 0 else "R"}: -'); continue
        P = [pts[3 * k:3 * k + 3] for k in range(21)]
        bends = ' '.join(f'{n[0]}{ang(P[i[0]], P[i[1]], P[i[2]]):3.0f}/{ang(P[i[1]], P[i[2]], P[i[3]]):3.0f}' for n, i in FINGERS.items())
        thumb = ang(P[2], P[3], P[4])
        dx, dy, dz = (P[9][k] - P[0][k] for k in range(3)); n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1
        out.append(f'{"L" if s == 0 else "R"}: PIP/DIP {bends} thumbIP {thumb:3.0f} dir({dx / n:+.2f},{dy / n:+.2f},{dz / n:+.2f})')
    print(time.strftime('%H:%M:%S') + '  ' + ' | '.join(out), flush=True)
    time.sleep(0.5)
