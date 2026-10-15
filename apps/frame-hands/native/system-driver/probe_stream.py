"""Probe the headset's streaming pose export over the wireless pipe exactly as pc_relay.py drives it.
usage: python probe_stream.py <connection.json> [seconds]"""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pc_relay import wireless_command, Rpc, OneWay, tick

cmd = wireless_command(Path(sys.argv[1])); seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3
r = Rpc(cmd)
t0 = tick(); first = r.ask('read'); print('read rtt ms', tick() - t0, 'state', first.get('state'), first.get('reason', ''))
r.send('stream'); ow = OneWay(); arrivals = []; states = {}; delays = []
end = time.monotonic() + seconds
while time.monotonic() < end:
    item = r.latest(timeout=1)
    if item is None:
        continue
    reply, received = item
    arrivals.append(received); states[reply.get('state')] = states.get(reply.get('state'), 0) + 1
    if isinstance(reply.get('native_now_ms'), int):
        delays.append(ow.delay(received, reply['native_now_ms']))
gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
print('lines', len(arrivals), 'states', states)
r.send('quit'); r.close()
if gaps:
    gaps.sort(); print('inter-arrival ms median', gaps[len(gaps) // 2], 'p90', gaps[int(len(gaps) * .9)], 'max', gaps[-1])
if delays:
    d = sorted(delays); print('estimated one-way delay ms median', d[len(d) // 2], 'p90', d[int(len(d) * .9)], 'max', d[-1])
