"""Create a private v3 launcher; preserve all existing source/driver entry points."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
base=Path('/home/steamos/devkit-game/FrameHands')
out=base/'experiments/camera-exact-v3'
with ExitStack() as stack:
    for p in (base.parent/'FrameEyes/workflow.lock',base/'trial.lock'):
        f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=out/'launchers'
    shutil.copytree(base/'experiments/ume-mesh-source-v1/launchers',launch,ignore=shutil.ignore_patterns('__pycache__'))
    p=launch/'trial.py';s=p.read_text()
    old='/var/lib/frame-hands/frame-camera-reader-direct-v2'
    oldhash='61b96ff275b2e218d041483a63fd5a3005f44ce11d742f1c87fcbd355711396e'
    assert s.count(old)==1 and s.count(oldhash)==1
    sha=json.loads((out/'build.json').read_text())['binary_sha256']
    p.write_text(s.replace(old,'/var/lib/frame-hands/frame-camera-reader-exact-v3').replace(oldhash,sha))
    p=launch/'recording.py';s=p.read_text()
    assert s.count("tracker.parent/'launchers/")==3
    s=s.replace("tracker.parent/'launchers/trial.py'","Path(__file__).resolve().parent/'trial.py'").replace("tracker.parent/'launchers/pc-input/run_source.py'","Path(__file__).resolve().parent/'pc-input/run_source.py'").replace("tracker.parent/'launchers/recording.py'","Path(__file__).resolve()")
    p.write_text(s)
    result={str(p.relative_to(launch)):hashlib.sha256(p.read_bytes()).hexdigest() for p in launch.rglob('*') if p.is_file()}
    (out/'launcher-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Private v3 launchers staged; no capture or default changes')
