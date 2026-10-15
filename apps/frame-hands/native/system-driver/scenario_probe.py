"""Scenario capture: logs every pose.json packet (full hand dicts + head) and every mailbox packet (21 shown points
per hand) for N seconds into a jsonl file, for offline analysis of a scripted scenario (desk still / hover / fist).
usage: python3 scenario_probe.py <session-dir> <input.bin> <seconds> <out.jsonl>"""
import json, struct, sys, time
from pathlib import Path
sess = Path(sys.argv[1]); mb = Path(sys.argv[2]); seconds = float(sys.argv[3]); out = open(sys.argv[4], 'w')
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')
last_cap = None; last_seq = None; t0 = time.monotonic(); n_raw = n_shown = 0
while time.monotonic() - t0 < seconds:
    now = time.monotonic() - t0
    try: d = json.loads((sess / 'pose.json').read_text())
    except Exception: d = None
    if d and d.get('capture_ns') != last_cap:
        last_cap = d.get('capture_ns'); d['probe_t'] = now; d['kind'] = 'raw'; out.write(json.dumps(d) + '\n'); n_raw += 1
    try: b = mb.read_bytes()
    except OSError: b = b''
    if len(b) == 592:
        seq = H.unpack_from(b)[3]
        if seq != last_seq:
            last_seq = seq; hands = []
            for s in range(2):
                ident, flags, cap, until, *pts = HAND.unpack_from(b, H.size + s * HAND.size)
                if flags: hands.append(dict(side=s, id=ident, flags=flags, capture_ns=cap, until_ns=until, points=[pts[3 * k:3 * k + 3] for k in range(21)]))
            out.write(json.dumps(dict(kind='shown', probe_t=now, seq=seq, hands=hands)) + '\n'); n_shown += 1
    time.sleep(0.004)
out.close(); print(f'raw packets {n_raw}, shown packets {n_shown} -> {sys.argv[4]}')
