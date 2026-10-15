"""Own one optical trial and stop it on timeout or loss of the PC control pipe."""
import argparse
import fcntl
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=90)
    parser.add_argument('--filter-guard',action='store_true')
    parser.add_argument('--output-recovery',action='store_true')
    parser.add_argument('--joint-identity',action='store_true',help='Opt-in joint stereo identity correction; requires --output-recovery')
    parser.add_argument('--camera-buffer-pool',action='store_true')
    parser.add_argument('--camera-direct',action='store_true')
    parser.add_argument('--record-session',action='store_true')
    parser.add_argument('--session-id')
    args=parser.parse_args()
    if args.joint_identity and not args.output_recovery:parser.error('--joint-identity requires --output-recovery')
    if args.output_recovery and not args.filter_guard:parser.error('--output-recovery requires --filter-guard')
    if args.camera_direct and args.camera_buffer_pool:parser.error('Select only one camera candidate')
    if not 5<=args.seconds<=300:parser.error('seconds must be 5..300')
    trial=Path(__file__).resolve().parent.parent/'trial.py'
    eye_lock=trial.parent.parent/'FrameEyes/workflow.lock'
    eye_guard=None
    if eye_lock.exists():
        eye_guard=eye_lock.open('a')
        try:fcntl.flock(eye_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Frame Eyes is running; finish its trial before starting hands')
    child=None
    def interrupted(*_):raise KeyboardInterrupt
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
    print('SOURCE_LOCKED',flush=True)
    try:
        command=[sys.executable,str(trial),'--pc-session','--seconds',str(args.seconds),'--start-delay','0','--rotation-search','--full-rate','--palm-threshold','0.2','--stable-identity','--identity-latch']
        if args.filter_guard:command.append('--filter-guard')
        if args.output_recovery:command.append('--output-recovery')
        if args.joint_identity:command.append('--joint-identity')
        if args.camera_buffer_pool:command.append('--camera-buffer-pool')
        if args.camera_direct:command.append('--camera-direct')
        if args.record_session:command.append('--record-session')
        if args.session_id:command.extend(['--session-id',args.session_id])
        child=subprocess.Popen(command)
        deadline=time.monotonic()+args.seconds+90
        while child.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('source startup/session timeout')
            ready,_,_=select.select([sys.stdin.buffer],[],[],.2)
            if ready and not sys.stdin.buffer.read(1):break
    except KeyboardInterrupt:
        pass
    finally:
        for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,signal.SIG_IGN)
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                # trial.py owns and stops its workers; a forced kill is an error requiring inspection.
                raise RuntimeError('optical trial did not finish its cleanup')
    return child.returncode if child else 1


if __name__=='__main__':raise SystemExit(main())
