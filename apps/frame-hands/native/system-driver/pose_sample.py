"""Sample the rig output packet (session pose.json) N times: capture stamp, per hand side, source id, wrist, index
finger angles. Shows whether sides swap or sources alternate upstream of the presentation.
usage: python3 pose_sample.py <session-dir> [count] [interval_s]"""
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1]) / 'pose.json'; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12; dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
last = None
for _ in range(n):
    try: d = json.loads(p.read_text())
    except Exception as e: print('read error', e); time.sleep(dt); continue
    cap = d.get('capture_ns', 0)
    if cap != last:
        last = cap
        hands = []
        for h in d.get('hands', []):
            w = h['wrist_world_m']; a = h['joint_angles']
            hands.append(f"side{h['side']} id{h['source_id']} wrist({w[0][3]:+.3f},{w[1][3]:+.3f},{w[2][3]:+.3f}) idx({a[5]:+.2f},{a[6]:+.2f},{a[7]:+.2f})")
        print(f"{cap // 1000000 % 100000:6d} ctx={d.get('context_valid')} " + ' | '.join(hands), flush=True)
    time.sleep(dt)
