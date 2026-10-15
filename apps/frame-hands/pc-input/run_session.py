"""Start a bounded optical source and PC relay; Ctrl+C disconnects hands and stops the source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=90)
    parser.add_argument('--controls',action='store_true')
    parser.add_argument('--filter-guard',action='store_true')
    parser.add_argument('--output-recovery',action='store_true')
    parser.add_argument('--joint-identity',action='store_true',help='Opt-in joint stereo identity correction; requires --output-recovery')
    parser.add_argument('--camera-buffer-pool',action='store_true')
    parser.add_argument('--camera-direct',action='store_true')
    parser.add_argument('--stop-file',type=Path,help='Graceful stop request from the diagnostic viewer')
    parser.add_argument('--source-ready-file',type=Path,help='Notify viewer only after the headset workflow lock is held')
    parser.add_argument('--record-dir',type=Path,help='Private diagnostic trace directory supplied by run_viewer')
    parser.add_argument('--session-id')
    args=parser.parse_args()
    if args.joint_identity and not args.output_recovery:parser.error('--joint-identity requires --output-recovery')
    import re
    if args.session_id and not re.fullmatch(r'\d{8}T\d{6}Z',args.session_id):parser.error('Invalid session identifier')
    if args.record_dir and (args.controls or not args.session_id or not args.record_dir.is_dir()):parser.error('Recording requires skeleton-only mode, session identifier and an existing private directory')
    if args.output_recovery and (not args.filter_guard or args.controls):parser.error('--output-recovery requires --filter-guard and skeleton-only mode')
    if args.camera_direct and (args.camera_buffer_pool or args.controls):parser.error('Exact-queue candidate requires skeleton-only mode and no other camera candidate')
    if args.filter_guard and args.controls:parser.error('Filter candidate is skeleton-only until worn validation')
    if args.camera_buffer_pool and args.controls:parser.error('Camera candidate is skeleton-only until validation')
    if not 5<=args.seconds<=300:parser.error('seconds must be 5..300')
    root=Path(__file__).resolve().parent
    deploy=json.loads((root/'deployment.json').read_text())
    if not deploy['registered']:parser.error('prepare_driver.py --register first')
    bundle=Path(deploy['bundle'])
    import hashlib
    for name,digest in deploy['files'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:parser.error('registered bundle changed')
    adb=root.parents[2]/'.local/android-tools/platform-tools/adb.exe'
    env=dict(os.environ,ANDROID_USER_HOME=str(Path.home()/'.android'))
    evidence=root/'evidence'/('local-'+(args.session_id or time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())))
    evidence.mkdir(parents=True,exist_ok=False)
    source=relay=None
    print('Keep hands down and head still for camera geometry verification. Inconclusive scene checks retry automatically; tracking starts immediately after confirmation. Source and relay are bounded; Ctrl+C stops both.',flush=True)
    with (evidence/'source.log').open('w') as log:
        try:
            source_command=[str(adb),'-s','frame','shell','-T','python3','/home/steamos/devkit-game/FrameHands/pc-input/run_source.py','--seconds',str(args.seconds)]
            if args.filter_guard:source_command.append('--filter-guard')
            if args.output_recovery:source_command.append('--output-recovery')
            if args.joint_identity:source_command.append('--joint-identity')
            if args.camera_buffer_pool:source_command.append('--camera-buffer-pool')
            if args.camera_direct:source_command.append('--camera-direct')
            if args.session_id:source_command.extend(['--session-id',args.session_id])
            if args.record_dir:source_command.append('--record-session')
            source=subprocess.Popen(source_command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env)
            command=[sys.executable,str(root/'relay.py'),'--driver',str(bundle),'--seconds',str(args.seconds+110),'--report',str(evidence/'transport.json'),'--stop-file',str(evidence/'stop')]
            if args.controls:command.append('--controls')
            if args.record_dir:command.extend(['--record',str(args.record_dir/'relay-trace.bin')])
            relay=subprocess.Popen(command,env=env)
            notified=False
            while source.poll() is None and relay.poll() is None and not (args.stop_file and args.stop_file.exists()):
                if args.source_ready_file and not notified and 'SOURCE_LOCKED' in (evidence/'source.log').read_text():
                    args.source_ready_file.touch()
                    notified=True
                time.sleep(.2)
        except KeyboardInterrupt:pass
        finally:
            if source and source.poll() is None:
                source.stdin.close()
                try:source.wait(timeout=15)
                except subprocess.TimeoutExpired:source.terminate();source.wait(timeout=3)
            if relay and relay.poll() is None:
                (evidence/'stop').touch()
                try:relay.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # A wedged relay is independently covered by the native watchdog.
                    relay.terminate();relay.wait(timeout=3)
                lock=bundle/'bin/win64/input.lock'
                if lock.exists() and lock.read_text()==str(relay.pid):lock.unlink()
    result=dict(evidence=str(evidence),source_exit=None if not source else source.returncode,relay_exit=None if not relay else relay.returncode)
    if args.record_dir:
        (args.record_dir/'transport-exit.json').write_text(json.dumps(result,indent=2)+'\n')
        for name in ('source.log','transport.json'):
            if (evidence/name).is_file():shutil.copyfile(evidence/name,args.record_dir/name)
    print(json.dumps(result))
    return 0 if source and relay and source.returncode==relay.returncode==0 else 1


if __name__=='__main__':raise SystemExit(main())
