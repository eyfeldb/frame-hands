"""Headless accepted pose processing -> isolated SteamVR model driver mailbox."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time
from mesh_packet import publish
BASE=Path(__file__).resolve().parent
HEADER=struct.Struct('<8sIIQQQ')
HAND=struct.Struct('<IIQQ63f')
# Same QPC origin as the model driver; GetTickCount64 is 15.625 ms on this host.
def TICK():return time.perf_counter_ns()//1_000_000

def precise_wall():
    if os.name!='nt':return time.time()
    ticks=ctypes.c_ulonglong()
    ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime(ctypes.byref(ticks))
    return (ticks.value-116444736000000000)/10_000_000

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

def validate_deployment(path):
    value=json.loads(Path(path).read_text())
    if value.get('protocol')!='FHMODEL2' or not value.get('registered'):raise RuntimeError('Model SteamVR driver is not installed')
    binary=Path(value['bundle'])/'bin/win64/driver_fh_skeletal.dll'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=value['files']['bin/win64/driver_fh_skeletal.dll']:raise RuntimeError('SteamVR model driver differs from prepared candidate')
    return value

def verify_live_driver(path):
    value=validate_deployment(path)
    expected=(Path(value['bundle'])/'bin/win64/driver_fh_skeletal.dll').resolve()
    command="Get-Process vrserver -ErrorAction Stop | ForEach-Object { $_.Modules } | Where-Object { $_.ModuleName -eq 'driver_fh_skeletal.dll' } | ForEach-Object { $_.FileName }"
    check=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,timeout=12,creationflags=subprocess.CREATE_NO_WINDOW)
    if check.returncode or not any(Path(line.strip()).resolve()==expected for line in check.stdout.splitlines() if line.strip()):
        raise RuntimeError('Restart SteamVR to load the new model hand driver, then press Start.')
    return value

class SteamSink:
    def __init__(self,godot,directory,deployment,log):
        self.log=log;self.deploy=validate_deployment(deployment);self.error=None;self.count=0;self.valid_hands=0;self.rejected=0;self.write_drops=0;self.max_gap_ms=0.0;self.last_publication=None
        self.view=directory/'view.json'
        self.dest=Path(self.deploy['bundle'])/'bin/win64/input.bin'
        self.temp=self.dest.with_suffix('.tmp')
        project=directory/'processor';project.mkdir()
        for name in ('hand_motion.gd','continuity_motion.gd','gap_motion.gd','game_motion.gd','steam_bridge.gd'):shutil.copyfile(BASE/'mesh-viewer'/name,project/name)
        shutil.copyfile(BASE/'piano-provider/rig_joints.gd',project/'rig_joints.gd')
        (project/'project.godot').write_text('config_version=5\n[application]\nconfig/name="Frame Hands Background Processor"\n[rendering]\nrenderer/rendering_method="gl_compatibility"\n[xr]\nopenxr/enabled=false\n')
        env=dict(os.environ,FRAME_UME_VIEW=str(self.view),FRAME_UME_RIG=str(directory/'rig.json'))
        self.child=subprocess.Popen([str(godot),'--headless','--xr-mode','off','--path',str(project),'--script','res://steam_bridge.gd'],env=env,stdout=subprocess.PIPE,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        # Monotonic across runs; render sequence is independent of capture age.
        self.sequence=max(time.time_ns(),1)
        self.thread=threading.Thread(target=self.pump,daemon=True);self.thread.start()
        sys.path.insert(0,str(BASE.parent/'pc-input'))
        from select_input_mode import request_mode
        request_mode('hands',Path(deployment))
    def pump(self):
        try:
            for line in self.child.stdout:
                if not line.startswith(b'FHMODEL '):
                    try:self.log.write(line)
                    except TypeError:self.log.write(line.decode(errors='replace'))
                    self.log.flush();continue
                value=json.loads(line[8:]);self.sequence+=1
                data=pack(value,self.sequence,TICK(),precise_wall())
                if data is None:self.rejected+=1;continue
                self.temp.write_bytes(data)
                try:os.replace(self.temp,self.dest)
                except PermissionError:self.write_drops+=1;continue
                tick=time.perf_counter()
                if self.last_publication is not None:self.max_gap_ms=max(self.max_gap_ms,(tick-self.last_publication)*1000)
                self.last_publication=tick
                self.count+=1;self.valid_hands+=len(value['hands'])
        except Exception as exc:self.error=exc
    def send(self,packet):
        if self.error:raise self.error
        if self.child.poll() is not None:raise RuntimeError('SteamVR pose processor stopped; see processor log')
        now=TICK();value=dict(packet,pc_sent_ms=now,pc_capture_ms=max(0,now-math.ceil(packet['age_ms'])),published_wall_s=precise_wall())
        publish(self.view,value)
    def close(self):
        if self.child.poll() is None:self.child.terminate()
        self.child.wait(timeout=5);self.thread.join(timeout=2)
        # Do not leave a retained optical role after Stop.
        self.dest.unlink(missing_ok=True)
