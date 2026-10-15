"""Targeted offline bridge check. Temporary unregistered driver mailbox; no XR/cameras."""
import argparse
import json
import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from mesh_packet import pose_packet
from steamvr_sink import SteamSink,pack,TICK,HEADER,HAND

def main():
    p=argparse.ArgumentParser();p.add_argument('--godot',type=Path,required=True);p.add_argument('--rig',type=Path,required=True);p.add_argument('--session',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    observations=[]
    with (a.session/'predictions.jsonl').open() as f:
        for line in f:
            value=json.loads(line)
            if len(value.get('hands',[]))==2:observations.append(value)
            if len(observations)==40:break
    assert len(observations)==40
    # No synthetic packets touch a registered driver or an application feed.
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);bundle=root/'unregistered';binary=bundle/'bin/win64/driver_fh_skeletal.dll';binary.parent.mkdir(parents=True)
        shutil.copyfile(a.build/'driver_fh_skeletal.dll',binary)
        dep=root/'test-deployment.json';dep.write_text(json.dumps(dict(protocol='FHMODEL2',registered=True,input_mode_supported=True,bundle=str(bundle),files={'bin/win64/driver_fh_skeletal.dll':hashlib.sha256(binary.read_bytes()).hexdigest()})))
        shutil.copyfile(a.rig,root/'rig.json')
        with (a.output/'processor.log').open('w') as log:
            sink=SteamSink(a.godot,root,dep,log)
            samples=[]
            try:
                end=time.monotonic()+10;index=0
                while time.monotonic()<end:
                    value=pose_packet(observations[index%40],np.eye(4),index+1,1_000_000_000+index*33_333_333,40,True)
                    sink.send(value);index+=1;time.sleep(1/30)
                    if sink.dest.exists():
                        raw=sink.dest.read_bytes();assert len(raw)==592;samples.append(raw)
                    if len(samples)>=35:break
                assert sink.count>0 and sink.valid_hands>0, (sink.count,sink.valid_hands,(a.output/'processor.log').read_text())
                # Stop new inference: neither display nor envelope can stay fresh.
                time.sleep(.6)
                raw=sink.dest.read_bytes();_,_,_,_,_,source=HEADER.unpack_from(raw)
                assert TICK()-source>500
                before=sink.valid_hands
                for i in range(8):
                    index+=1;sink.send(pose_packet(observations[i],np.eye(4),index+1,1_000_000_000+index*33_333_333,40,True));time.sleep(1/30)
                assert sink.valid_hands>before
            finally:sink.close()
            assert not sink.dest.exists()
        fixture=a.output/'bridge.bin';fixture.write_bytes(b''.join(samples))
        result=subprocess.run([str((a.build/'test-model.exe').resolve()),str(fixture.resolve())],capture_output=True,text=True)
        print(result.stdout);assert result.returncode==0,result.stderr
        (a.output/'result.json').write_text(json.dumps(dict(packets=len(samples),published=sink.count,hand_samples=sink.valid_hands,driver=result.stdout,stop_cleared=True,source_expiry=True,offline_only=True),indent=2))
if __name__=='__main__':main()
