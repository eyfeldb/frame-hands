"""Read-only latest complete camera/audit exporter. Runs on Frame; never opens cameras.

One response per newline on stdin. Source trial separately owns camera locks/lifetime.
Lossless PNG payloads preserve the exact grayscale model inputs. Partial records wait.
"""
import argparse
import base64
from collections import OrderedDict
import io
import json
from pathlib import Path
import struct
import select
import sys
import time
import cv2
import numpy as np
from frame_data import SET, CAM, NAMES, read_audit, load_calibration, crop_camera, warp


class Tail:
    def __init__(self, directory, continuous=False):
        self.directory = Path(directory)
        self.continuous = continuous
        if continuous:
            from continuous_session import SparseSpool
            self.spool = SparseSpool()
        self.offset = self.audit_offset = 0
        self.records, self.audits = OrderedDict(), OrderedDict()
        self.pending = b''
        self.cameras = None

    def update(self):
        path = self.directory/'sets.bin'
        if path.exists():
            length = path.stat().st_size
            with path.open('rb') as f:
                f.seek(self.offset)
                while length-f.tell() >= SET.size:
                    offset = f.tell()
                    magic,count,size = SET.unpack(f.read(SET.size))
                    if magic != b'FHSET01\0' or not 1 <= count <= 4 or not SET.size+CAM.size <= size <= 70_000_000:
                        raise ValueError('Invalid camera header')
                    if offset+size > length:
                        break
                    rows = [CAM.unpack(f.read(CAM.size)) for _ in range(count)]
                    if any(not 1 <= w <= 4096 or not 1 <= h <= 4096 for _,w,h,_,_ in rows):
                        raise ValueError('Invalid camera dimensions')
                    if size != SET.size+count*CAM.size+sum(w*h for _,w,h,_,_ in rows):
                        raise ValueError('Invalid camera length')
                    stamp = min(r[3] for r in rows)
                    self.records[stamp] = (offset+SET.size+count*CAM.size,rows)
                    self.offset = offset+size
                    f.seek(self.offset)
                    while len(self.records)>120:self.records.popitem(last=False)
        path = self.directory/'live-views.txt'
        if path.exists():
            with path.open('rb') as f:
                f.seek(self.audit_offset)
                data=f.read()
                self.audit_offset=f.tell()
            self.pending = (self.pending+data).replace(b'\r\n',b'\n')
            while b'end\n' in self.pending:
                block,self.pending=self.pending.split(b'end\n',1)
                # Shared strict parser accepts a stream as well as a path.
                for stamp,observation in read_audit(io.StringIO((block+b'end\n').decode())).items():
                    self.audits[stamp]=(block+b'end\n').decode()
                while len(self.audits)>120:self.audits.popitem(last=False)
            if len(self.pending)>1_000_000:raise ValueError('Unbounded incomplete audit')
        if self.continuous:
            # Retain all 120 indexed frames. The exporter never rereads older blocks.
            if self.records:
                first=min(offset-SET.size-len(rows)*CAM.size for offset,rows in self.records.values())
                self.spool.release(self.directory/'sets.bin',first)
            if self.audit_offset:self.spool.release(self.directory/'live-views.txt',self.audit_offset)

    def newest(self,last):
        self.update()
        matches=self.records.keys() & self.audits.keys()
        stamp=max(matches,default=0)
        if stamp<=last:return None
        offset,rows=self.records[stamp]
        if self.cameras is None:
            self.cameras=load_calibration(self.directory/'snapshot/device/persist')
        observation=next(iter(read_audit(io.StringIO(self.audits[stamp])).values()))
        images={}
        with (self.directory/'sets.bin').open('rb') as f:
            f.seek(offset)
            for name,w,h,_,_ in rows:
                pixels=f.read(w*h)
                name=name.rstrip(b'\0').decode('ascii')
                if name in NAMES:images[name]=np.frombuffer(pixels,np.uint8).reshape(h,w)
        blobs=[];views=[]
        for view in observation['views']:
            name=view['camera']
            if name not in images or view['presence']<.5:continue
            try:crop=crop_camera(self.cameras[name],view['points'],0,np.eye(4))
            except ValueError:continue
            pixels=warp(self.cameras[name],crop,images[name])
            ok,encoded=cv2.imencode('.png',pixels,[cv2.IMWRITE_PNG_COMPRESSION,1])
            if not ok:raise ValueError('PNG encode failed')
            blobs.append(encoded.tobytes())
            views.append(dict(camera=name,id=view['id'],intrinsics=crop.uv_to_window_matrix().tolist(),camera_to_head_mm=crop.camera_to_world_xf.tolist()))
        return dict(stamp=stamp,audit=self.audits[stamp],crops=views,lengths=list(map(len,blobs))),blobs



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--seconds',type=int,default=240)
    p.add_argument('--repeat',action='store_true',help='Saved-recording transport benchmark only')
    p.add_argument('--continuous',action='store_true',help='Live sparse spool; no complete recording retained')
    p.add_argument('--native-head',action='store_true')
    a=p.parse_args()
    if a.continuous and a.repeat:p.error('Continuous live spool cannot repeat recordings')
    native_pose=None
    if a.native_head:
        from native_piano_link import NativePose
        native_pose=NativePose()
    tail=Tail(a.directory,a.continuous)
    last=0
    end=float('inf') if a.continuous else time.monotonic()+min(max(a.seconds,5),390)
    while time.monotonic()<end:
        if not select.select([sys.stdin.buffer],[],[],min(1,max(0,end-time.monotonic())))[0]:continue
        command=sys.stdin.buffer.readline().strip()
        if command != b'next':break
        result=tail.newest(0 if a.repeat else last)
        if result:
            meta,blobs=result;last=meta['stamp']
            if native_pose:
                observation=next(iter(read_audit(io.StringIO(meta['audit'])).values()))
                meta['head']=native_pose.get(observation['mono_ns'])
        else:meta,blobs=dict(waiting=True),[]
        meta['source_now_ns']=time.monotonic_ns()
        body=json.dumps(meta,separators=(',',':')).encode()
        sys.stdout.buffer.write(base64.b64encode(struct.pack('<I',len(body))+body+b''.join(blobs))+b'\n')
        sys.stdout.buffer.flush()
    if native_pose:native_pose.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
