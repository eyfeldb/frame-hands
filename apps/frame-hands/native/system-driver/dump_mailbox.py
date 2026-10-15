"""Dump an FHMODEL2 mailbox as CSV rows (one per changed packet) for N seconds: t_ms, published, per hand flags,
capture, wrist xyz. Headset and PC runs of this can be compared to see whether motion is added in transit.
usage: python dump_mailbox.py <input.bin> <seconds> <out.csv>"""
import struct, sys, time
from pathlib import Path
p = Path(sys.argv[1]); seconds = float(sys.argv[2]); out = open(sys.argv[3], 'w')
H = struct.Struct('<8sIIQQQ'); HAND = struct.Struct('<IIQQ63f')
now_ms = lambda: time.monotonic_ns() // 1000000 if sys.platform != 'win32' else time.perf_counter_ns() // 1000000
out.write('t_ms,published,l_flags,l_capture,l_x,l_y,l_z,r_flags,r_capture,r_x,r_y,r_z\n')
last = None; end = time.monotonic() + seconds; n = 0
while time.monotonic() < end:
    try: d = p.read_bytes()
    except OSError: time.sleep(.002); continue
    if len(d) != 592: time.sleep(.002); continue
    magic, ver, size, seq, published, source = H.unpack_from(d)
    if seq != last:
        last = seq; row = [now_ms(), published]
        for s in range(2):
            ident, flags, cap, until, *pts = HAND.unpack_from(d, H.size + s * HAND.size)
            row += [flags, cap, round(pts[0], 5), round(pts[1], 5), round(pts[2], 5)]
        out.write(','.join(str(x) for x in row) + '\n'); n += 1
    time.sleep(.002)
out.close(); print('rows', n)
