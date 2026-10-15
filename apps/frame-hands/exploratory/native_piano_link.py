"""Frame-only PC pose receiver. Authenticated ADB stdin, no listener or recording."""
import argparse
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time

LIB='/opt/steamvr/bin/linuxarm64/libopenvr_api.so'
FEED=Path('/run/user/1000/frametop-hands/pc-piano/pose.json')
POSE=Path('/home/steamos/devkit-game/FrameHands/experiments/piano-native-link-v1/pose-service')

class NativePose:
    def __init__(self, executable=POSE):
        self.child=subprocess.Popen([str(executable),LIB,'0'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr)
    def get(self, capture):
        self.child.stdin.write((str(int(capture))+'\n').encode());self.child.stdin.flush()
        if not select.select([self.child.stdout],[],[],3)[0]:raise TimeoutError('Native head pose query timed out')
        line=self.child.stdout.readline(8192)
        if not line:raise EOFError('Native head pose helper stopped')
        return json.loads(line)
    def close(self):
        try:self.child.stdin.close();self.child.wait(timeout=4)
        except (OSError,subprocess.TimeoutExpired):self.child.terminate();self.child.wait(timeout=3)

def matrix(values):
    if not isinstance(values,list) or len(values)!=12 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):raise ValueError('Invalid head matrix')
    return [values[0:4],values[4:8],values[8:12],[0,0,0,1]]

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,allow_nan=False,separators=(',',':')))
    temp.replace(path)

def expired(now):
    return dict(version=2,sequence=now,capture_ns=now,context_valid=False,hands=[])

def receive_packet(packet, head, now, wall, last_capture):
    capture=packet.get('capture_ns',0)
    if type(capture)!=int or capture<=last_capture or not 0<=now-capture<250_000_000:return None
    if packet.get('version')!=2 or packet.get('context_valid') is not True:return None
    if head.get('valid') is not True:return None
    queried=head.get('queried_mono_ns',0)
    if not 0<=now-queried<=50_000_000:return None
    hands=packet.get('hands')
    if not isinstance(hands,list) or len(hands)>2:return None
    sides=set();ids=set()
    for hand in hands:
        side=hand.get('side');identity=hand.get('source_id')
        if type(side)!=int or side not in (0,1) or side in sides or type(identity)!=int or identity<0 or identity in ids:return None
        sides.add(side);ids.add(identity)
        angles=hand.get('joint_angles',[]);wrist=hand.get('wrist_world_m',[])
        if len(angles)!=22 or not all(type(v) in (int,float) and math.isfinite(v) for v in angles):return None
        if len(wrist)!=4 or any(not isinstance(row,list) or len(row)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in row) for row in wrist):return None
    value=dict(packet)
    value.update(sequence=capture,age_ms=(now-capture)/1e6,published_wall_s=wall,head_world_m=matrix(head['current']))
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    pose=NativePose()
    if a.check:
        try:
            result=pose.get(time.monotonic_ns())
            print(json.dumps(dict(valid=result.get('valid',False),reason=result.get('reason',''))))
            return 0 if result.get('valid') else 1
        finally:pose.close()
    import fcntl
    FEED.parent.mkdir(parents=True,exist_ok=True)
    lock=(FEED.parent/'receiver.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:pose.close();raise RuntimeError('Native hand receiver already active')
    last=0
    atomic(FEED,expired(time.monotonic_ns()))
    try:
        while True:
            if not select.select([sys.stdin.buffer],[],[],10)[0]:break
            line=sys.stdin.buffer.readline(65537)
            if not line or line.strip()==b'quit':break
            if len(line)>65536:raise ValueError('Oversized hand packet')
            packet=json.loads(line)
            if packet == {'kind':'heartbeat'}:
                # Keep the return channel alive while capture is paused. Never
                # renew an old pose or manufacture hands during the gap.
                atomic(FEED,expired(time.monotonic_ns()))
                print(json.dumps(dict(accepted=False,age_ms=None,waiting=True)),flush=True)
                continue
            head=pose.get(time.monotonic_ns())
            now=time.monotonic_ns()
            value=receive_packet(packet,head,now,time.time(),last)
            if value:
                last=value['capture_ns'];atomic(FEED,value)
            else:atomic(FEED,expired(now))
            print(json.dumps(dict(accepted=value is not None,age_ms=value['age_ms'] if value else None)),flush=True)
    finally:
        atomic(FEED,expired(time.monotonic_ns()));pose.close();lock.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
