"""Exercise the actual Godot decoder/drawing paths with explicit synthetic frames."""
import os
from pathlib import Path
import struct
import subprocess

root=Path(__file__).resolve().parent
fixture=root/'build/viewer-fixture.bin'
godot=root.parents[2]/'.local/deps/godot-4.7.2/Godot_v4.7.2-stable_win64.exe'
try:
    cases=[(1,False,'all'),(2,False,'all'),(2,True,'all')]
    cases += [(2,held,layer) for held in [False,True] for layer in ['candidate','source','steam']]
    for version,held,layer in cases:
        frame=bytearray(struct.pack('<8sQ',f'FHVIEW{version}\0'.encode(),123))
        for side in range(2):
            flags=12 if held and side else (7 if version==2 else 3)
            frame.extend(struct.pack('<II',7+side if flags&3 else 0,flags))
            for count in ((21,31,3,21) if version==2 else (21,31,3)):
                for i in range(count):frame.extend(struct.pack('<3f',-.2+.4*side+i*.001,0,-.5))
            if version==2:frame.extend(struct.pack('<I',7+side))
        assert len(frame)==(1864 if version==2 else 1352)
        fixture.write_bytes(frame)
        env=dict(os.environ,FRAME_HANDS_PC_VIEW=str(fixture),FRAME_HANDS_STABILITY_PREVIEW='1' if version==2 else '',FRAME_HANDS_VIEW_LAYER=layer)
        command=[str(godot),'--headless','--path',str(root/'viewer'),'--xr-mode','off','--','--check-scene']
        if held:command.append('--check-held')
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=20)
        print(f'version={version} held={held} layer={layer}: {result.stdout}');print(result.stderr)
        assert result.returncode==0 and 'PASS binary frame decoding' in result.stdout and 'SCRIPT ERROR' not in result.stderr
finally:fixture.unlink(missing_ok=True)
