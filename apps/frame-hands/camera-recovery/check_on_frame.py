"""Bounded candidate camera/geometry check. No tracker, XR scene, input or recording."""
from contextlib import ExitStack
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE=Path('/home/steamos/devkit-game/FrameHands')
BINARY=Path('/var/lib/frame-hands/frame-camera-reader-buffer-v1')
EXPECTED='577be3a745a4bca2a08b961112dad26a96d9d2c4435e2ac86eb66f9ab0df353f'
sys.path.insert(0,str(BASE/'probe'))
from readiness import ring_ready


def main():
    global BINARY,EXPECTED
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--accepted',action='store_true',help='Check the preserved accepted reader as a control')
    choice.add_argument('--direct',action='store_true',help='Check the exact kernel-queue reader')
    args=parser.parse_args()
    if args.accepted:
        BINARY=Path('/var/lib/frame-hands/frame-camera-reader')
        EXPECTED='ab07c5a1a5f00cfe3be8701b6a848242c8775880f47606fb18cb114c560fb738'
    if args.direct:
        BINARY=Path('/var/lib/frame-hands/frame-camera-reader-direct-v2')
        EXPECTED='61b96ff275b2e218d041483a63fd5a3005f44ce11d742f1c87fcbd355711396e'
    if not BINARY.is_file() or hashlib.sha256(BINARY.read_bytes()).hexdigest()!=EXPECTED:
        raise RuntimeError('Install the exact reviewed candidate locally with sudo first')
    child=None
    def interrupted(*_):
        raise RuntimeError('Camera check interrupted')
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
        signal.signal(sig,interrupted)
    with ExitStack() as stack:
        for path in [BASE.parent/'FrameEyes/workflow.lock',BASE/'trial.lock']:
            lock=stack.enter_context(path.open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for proc in Path('/proc').glob('[0-9]*'):
            try: name=(proc/'comm').read_text().strip()
            except OSError: continue
            if name in {'ft-camd','ft-hands','ft-ringplay','frame-camera-re'} or name.startswith('ft-hands-'):
                raise RuntimeError(f'Another camera/tracker exists ({proc.name}); leave it unchanged')
        evidence=BASE/'evidence'/('camera-buffer-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
        evidence.mkdir(exist_ok=False)
        result=dict(binary_sha256=EXPECTED,accepted_control=args.accepted,direct_queue=args.direct,recording=False,tracking=False,geometry_verified=False)
        log=stack.enter_context((evidence/'camera.log').open('w'))
        os.sched_setaffinity(0,{7})
        try:
            start=time.monotonic_ns()
            child=subprocess.Popen([str(BINARY),'--sensor','og01a1b','--status','1'],
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if child.poll() is not None: raise RuntimeError(f'Reader exited {child.returncode}')
                if ring_ready(Path(f'/run/user/{os.getuid()}/frametop-hands/cam-ring'),child.pid,start,require_forward=True):
                    result['ready_after_seconds']=round((time.monotonic_ns()-start)/1e9,3)
                    break
                time.sleep(.05)
            else: raise RuntimeError('Both forward cameras did not publish within 30 seconds')
            geometry=subprocess.run([sys.executable,str(BASE/'check_camera_geometry.py'),
                                     '--output',str(evidence/'geometry.json')],
                                    capture_output=True,text=True,timeout=15)
            result['geometry_exit']=geometry.returncode
            result['geometry_verified']=geometry.returncode==0
            if not result['geometry_verified']: raise RuntimeError('Named stereo geometry did not pass')
            time.sleep(5) # bounded additional cadence evidence, no hand trial
            if child.poll() is not None: raise RuntimeError('Reader exited after geometry validation')
        except (RuntimeError,OSError,subprocess.TimeoutExpired) as exc:
            result['error']=str(exc)
        finally:
            for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]: signal.signal(sig,signal.SIG_IGN)
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
            result['owned_process_stopped']=child is None or child.poll() is not None
            (evidence/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(evidence=str(evidence),**result),indent=2))
        return int('error' in result)


if __name__=='__main__': raise SystemExit(main())
