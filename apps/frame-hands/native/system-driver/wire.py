import math, os, struct, time
HEADER=struct.Struct("<8sIIQQQ")
HAND=struct.Struct("<IIQQ63f")
def pack(value,sequence,now,wall):
    lag=wall-float(value['wall_s'])
    source=int(value['source_ms'])
    if not math.isfinite(lag) or not 0<=lag<.080 or not 0<=now-source<=500:return None
    hands=[HAND.pack(0,0,0,0,*([0.]*63)) for _ in range(2)]
    seen=set();ids=set()
    for h in value['hands']:
        side=int(h['side']);identity=int(h['id']);capture=int(h['capture_ms']);until=int(h['until_ms'])
        points=[float(v) for p in h['points'] for v in p]
        if side not in (0,1) or side in seen or identity in ids or not 0<identity<2**32:raise ValueError('Invalid hand identity')
        if len(points)!=63 or not all(math.isfinite(v) for v in points):raise ValueError('Invalid model joints')
        if not 0<capture<=source or not capture<=until<=capture+350:raise ValueError('Invalid hand expiry')
        seen.add(side);ids.add(identity)
        if now<until:hands[side]=HAND.pack(identity,1,capture,until,*points)
    return HEADER.pack(b'FHMODEL2',2,592,sequence,now-int(lag*1000),source)+b''.join(hands)


def request_mode(root, mode):
    value={'hands':1,'controllers':2}[mode]
    path=root/'mode.bin';temporary=root/'mode.tmp'
    temporary.write_bytes(struct.pack('<8sQII',b'FHMODE1\0',time.monotonic_ns(),value,0))
    os.replace(temporary,path)
