"""Numerical viewer diagnostics; no images or joint coordinates are retained."""
import argparse
import json
import math
from pathlib import Path
import struct
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('frame',type=Path)
parser.add_argument('--seconds',type=int,default=30)
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
if not 1<=args.seconds<=300:parser.error('seconds must be 1..300')
mapping=[1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25]
end=time.monotonic()+args.seconds
last_stamp=None
ids=[set(),set()]
seen={}
side_changes=0
samples=0
raw=[0,0]
steam=[0,0]
candidate=[0,0]
held=[0,0]
gaps=[[],[]]
while time.monotonic()<end:
    time.sleep(.015)
    try:b=args.frame.read_bytes()
    except OSError:continue
    v2=len(b)==1864 and b[:8]==b'FHVIEW2\0'
    if not v2 and (len(b)!=1352 or b[:8]!=b'FHVIEW1\0'):continue
    stamp=struct.unpack_from('<Q',b,8)[0]
    if stamp==last_stamp:continue
    last_stamp=stamp;samples+=1
    for side in range(2):
        offset=16+side*(924 if v2 else 668)
        ident,flags=struct.unpack_from('<II',b,offset)
        if flags&1:
            raw[side]+=1;ids[side].add(ident)
            if ident in seen and seen[ident]!=side:side_changes+=1
            seen[ident]=side
        if flags&2:steam[side]+=1
        if flags&4:candidate[side]+=1
        if flags&8:held[side]+=1
        if flags&3==3:
            distances=[math.dist(struct.unpack_from('<3f',b,offset+8+12*i),struct.unpack_from('<3f',b,offset+260+12*mapping[i])) for i in range(21)]
            gaps[side].append(sum(distances)/21*1000)
def percentile(values,q):
    values=sorted(values)
    return round(values[int((len(values)-1)*q)],3) if values else None
report=dict(samples=samples,raw_samples=raw,steam_samples=steam,source_ids=[sorted(x) for x in ids],within_id_side_changes=side_changes,unpaired_mean_joint_separation_mm_p50=[percentile(x,.5) for x in gaps],unpaired_mean_joint_separation_mm_p95=[percentile(x,.95) for x in gaps],scope='Unpaired source versus smoothed SteamVR output; not physical accuracy. Same-ID side changes do not detect label mistakes assigned on reacquisition.')
report.update(candidate_samples=candidate,held_estimate_samples=held)
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
