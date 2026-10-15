"""Run an isolated, bounded optical-tracking measurement on Frame.

Requires locally authorized camera reader. Raw images are saved only with
--record-test or --record-session. No input driver, service registration or runtime restart.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import math
import re
from recording import snapshot_frame, inspect_camera, save_json

sys.path.insert(0,str(Path(__file__).parent/'probe'))
from readiness import ring_ready, hands_ready

BASE = Path('/home/steamos/devkit-game/FrameHands')
CAMERA = Path('/var/lib/frame-hands/frame-camera-reader')
RUNTIME = Path(f'/run/user/{os.getuid()}/frametop-hands')


def stop(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=2)
        return
    try:
        child.wait(timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--start-delay',type=int,default=0,help='Pause after geometry verification before recording/tracking')
    parser.add_argument('--viewer',action='store_true',help='Show the bounded in-headset joint viewer; another active XR scene must be closed first')
    parser.add_argument('--temporal-viewer',action='store_true',help='Experimental display-rate prediction, identity continuity and short loss fade; requires --viewer')
    parser.add_argument('--smooth-viewer',action='store_true',help='Display-frame bone interpolation without prediction or identity changes; requires --viewer')
    parser.add_argument('--swap-sides', action='store_true')
    parser.add_argument('--rotation-search', action='store_true',help='Experimental multi-angle palm acquisition')
    parser.add_argument('--full-rate', action='store_true',help='Experimental 30 Hz updates and smaller, frequent acquisition batches')
    parser.add_argument('--stable-identity',action='store_true',help='Experimental stereo continuity and open-hand identity retention')
    parser.add_argument('--identity-latch',action='store_true',help='Experimental persistent handedness; uses a separate candidate tracker')
    parser.add_argument('--palm-threshold',type=float,default=.5,choices=[.2,.5],help='Experimental palm proposal threshold; finger validation stays enabled')
    parser.add_argument('--record-test', action='store_true',help='Save private synchronized camera images and live landmarks; requires seconds <= 15')
    parser.add_argument('--record-session',action='store_true',help='Save the entire bounded PC diagnostic trial for offline review')
    parser.add_argument('--session-id',help='Join private Frame and PC recordings by explicit UTC identifier')
    parser.add_argument('--pc-session',action='store_true',help='Accepted optical source for a bounded PC VR session; no local viewer or input bridge')
    parser.add_argument('--filter-guard',action='store_true',help='Opt-in replay-tested smoothing guard for a PC diagnostic trial')
    parser.add_argument('--output-recovery',action='store_true',help='Opt-in fresh monocular recovery; requires --filter-guard and PC diagnostics')
    parser.add_argument('--joint-identity',action='store_true',help='Opt-in joint stereo identity correction; requires --output-recovery')
    parser.add_argument('--camera-buffer-pool',action='store_true',help='Opt-in camera allocation candidate; PC diagnostics only')
    parser.add_argument('--camera-direct',action='store_true',help='Opt-in exact kernel-queue camera reader; PC diagnostics only')
    parser.add_argument('--system-input',choices=['observe','gaze','touch'],help='Bounded system bridge: observe sends no input; gaze/touch require the candidate driver loaded')
    parser.add_argument('--gaze-calibration',type=Path,help='Explicit validated gaze.profile from this eye-tracker session (expires after 15 minutes)')
    args = parser.parse_args()
    if args.joint_identity and not args.output_recovery:parser.error('--joint-identity requires --output-recovery')
    if args.session_id and not re.fullmatch(r'\d{8}T\d{6}Z',args.session_id):parser.error('Invalid session identifier')
    if args.record_session and (not args.pc_session or args.record_test):parser.error('--record-session requires PC diagnostics without --record-test')
    recording_requested=args.record_test or args.record_session
    camera_binary=CAMERA
    if args.camera_direct:
        if not args.pc_session or args.camera_buffer_pool:parser.error('--camera-direct requires PC diagnostics without another camera candidate')
        camera_binary=Path('/var/lib/frame-hands/frame-camera-reader-direct-v2')
        direct_hash='61b96ff275b2e218d041483a63fd5a3005f44ce11d742f1c87fcbd355711396e'
        if not camera_binary.is_file() or hashlib.sha256(camera_binary.read_bytes()).hexdigest()!=direct_hash:
            parser.error('Install the exact reviewed direct-queue reader first')
    if args.camera_buffer_pool:
        if not args.pc_session:parser.error('--camera-buffer-pool requires --pc-session')
        camera_binary=Path('/var/lib/frame-hands/frame-camera-reader-buffer-v1')
        expected_camera='577be3a745a4bca2a08b961112dad26a96d9d2c4435e2ac86eb66f9ab0df353f'
        if not camera_binary.is_file() or hashlib.sha256(camera_binary.read_bytes()).hexdigest()!=expected_camera:
            parser.error('Install the exact reviewed camera-buffer candidate locally first')
    if args.filter_guard and not (args.pc_session and args.identity_latch):
        parser.error('--filter-guard requires the PC diagnostic source and accepted identity flags')
    if args.output_recovery and not args.filter_guard:
        parser.error('--output-recovery requires --filter-guard')
    if args.gaze_calibration and (not args.system_input or not args.gaze_calibration.is_absolute() or not args.gaze_calibration.is_file()):
        parser.error('--gaze-calibration requires system input and an absolute existing profile path')
    if not 5 <= args.seconds <= (300 if args.pc_session else 90):
        parser.error('--seconds must be 5..90, or 5..300 with --pc-session')
    if args.pc_session and (args.viewer or args.system_input or args.record_test or not (args.identity_latch and args.stable_identity and args.full_rate and args.rotation_search and args.palm_threshold==.2)):
        parser.error('--pc-session requires the accepted tracking flags and excludes local viewer, system input and --record-test')
    if not 0 <= args.start_delay <= 30:
        parser.error('--start-delay must be 0..30')
    if args.record_test and args.seconds > 15:
        parser.error('--record-test requires --seconds 5..15')
    if args.temporal_viewer and not args.viewer:
        parser.error('--temporal-viewer requires --viewer')
    if args.temporal_viewer:
        parser.error('Predictive temporal viewer failed physical acceptance; use the baseline or --smooth-viewer')
    if args.smooth_viewer and not args.viewer:
        parser.error('--smooth-viewer requires --viewer')
    if args.identity_latch and not args.stable_identity:
        parser.error('--identity-latch requires --stable-identity')
    if args.system_input:
        if args.viewer or args.record_test:
            parser.error('System input trials use the native dashboard, without the diagnostic scene or camera recording')
        if not (args.identity_latch and args.full_rate and args.rotation_search and args.palm_threshold==.2):
            parser.error('System input requires the accepted --identity-latch --stable-identity --full-rate --rotation-search --palm-threshold 0.2 baseline')
        if args.seconds>60 or not (BASE/'system-input/build/fh-input').is_file():
            parser.error('System input needs its native bridge build and a trial of at most 60 seconds')
        if args.system_input!='observe' and f'@fh_pointer_{os.getuid()}' not in Path('/proc/net/unix').read_text():
            parser.error('The versioned pointer driver is not loaded; use observe first')
    candidate_tracker = BASE/'upstream/build/ft-hands-identity-latch'
    if args.joint_identity:
        candidate_tracker=BASE/'experiments/recovery-v8-joint/ft-hands-joint'
        expected='83b50f1cfed74622eda463ded86e971a87eeaa6eca1e6abece672aafe012fbb8'
        if not candidate_tracker.is_file() or hashlib.sha256(candidate_tracker.read_bytes()).hexdigest()!=expected:
            parser.error('Build the exact isolated joint-identity trial first')
    elif args.output_recovery:
        candidate_tracker=BASE/'experiments/recovery-v5-output/ft-hands-output'
        expected='027b23dcfb01cceea094b9b85969916889a88d14cdb07ca644618d9caa8b824a'
        if not candidate_tracker.is_file() or hashlib.sha256(candidate_tracker.read_bytes()).hexdigest()!=expected:
            parser.error('Build the exact isolated output-recovery trial first')
    elif args.filter_guard:
        candidate_tracker=BASE/'experiments/recovery-v3/ft-hands-filter'
        build_file=candidate_tracker.parent/'trial-build.json'
        if not candidate_tracker.is_file() or not build_file.is_file():parser.error('Build the isolated filter trial first')
        build=json.loads(build_file.read_text())
        if build.get('enabled_flag')!='FRAME_HANDS_FILTER_GUARD' or hashlib.sha256(candidate_tracker.read_bytes()).hexdigest()!=build['binary_sha256']:
            parser.error('Filter candidate failed its build-integrity check')
    if args.identity_latch and not candidate_tracker.is_file():
        parser.error('The separate identity candidate tracker is not built')
    required_bytes=(args.seconds+15)*140_000_000+2_000_000_000 if args.record_session else 2_000_000_000
    if recording_requested and shutil.disk_usage(BASE).free < required_bytes:
        parser.error(f'Recording requires {required_bytes/1e9:.1f} GB free on Frame; recording was not silently disabled')
    if not camera_binary.is_file():
        parser.error('Camera access not installed. Run enable-camera-access.sh locally with sudo first.')
    # Validated on this Frame: shared-buffer learning misses its timing window
    # on the default scheduler. Pin this runner and reader to performance core 7.
    # ft-hands sets its own existing worker affinity to cores 5,6,7.
    os.sched_setaffinity(0,{7})
    camera = tracker = viewer = measurement = system_input = system_surface = None
    evidence = BASE/'evidence'/(args.session_id or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    evidence.mkdir(parents=True, exist_ok=False)
    # Do not replace another owner of the upstream ring or hands file.
    with (BASE/'trial.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('A FrameHands trial is already running.')
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                name=(proc/'comm').read_text().strip()
            except (OSError, UnicodeError):
                continue
            if name in {'ft-camd','ft-hands','ft-ringplay','frame-camera-re'} or name.startswith('ft-hands-'):
                parser.error(f'Another camera/tracker process exists ({proc.name}); leave it unchanged.')
        result = dict(mode='pc-streamed optical source' if args.pc_session else 'standalone', raw_recording_saved=False, input_injected=False,
                      duration_requested=args.seconds, swap_sides=args.swap_sides,
                      reader_cpus=[7],camera_sensor_filter='og01a1b',rotation_search=args.rotation_search,
                      full_rate=args.full_rate)
        result['palm_threshold']=args.palm_threshold
        result['stable_identity']=args.stable_identity
        result['identity_latch']=args.identity_latch
        result['filter_guard']=args.filter_guard
        result['output_recovery']=args.output_recovery
        result['joint_identity']=args.joint_identity
        result['camera_buffer_pool']=args.camera_buffer_pool
        result['camera_direct']=args.camera_direct
        result['camera_binary']=str(camera_binary)
        result['camera_sha256']=hashlib.sha256(camera_binary.read_bytes()).hexdigest()
        tracker_path=candidate_tracker if args.identity_latch else BASE/'upstream/build/ft-hands'
        result['tracker_binary']=tracker_path.name
        result['tracker_sha256']=hashlib.sha256(tracker_path.read_bytes()).hexdigest()
        result['start_delay_seconds']=args.start_delay
        result['viewer_requested']=args.viewer
        result['temporal_viewer']=args.temporal_viewer
        result['smooth_viewer']=args.smooth_viewer
        result['system_input_requested']=args.system_input
        result['pc_session_source']=args.pc_session
        viewer_log = None
        system_log = None
        surface_log = None
        phase_path=evidence/'viewer-state.json'
        def phase(name, remaining=0, detail=''):
            state=dict(phase=name,remaining=max(0,int(remaining)),detail=detail)
            temporary=phase_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state)+'\n')
            temporary.replace(phase_path)
        private = None
        if recording_requested:
            private = BASE/'private'/evidence.name
            private.parent.mkdir(mode=0o700,exist_ok=True)
            private.mkdir(mode=0o700)
            result['raw_recording_requested']=True
            result['private_recording_directory']=str(private)
            result['recording_may_affect_timing']=True
        camera_log = (evidence/'camera.log').open('w')
        tracker_log = (evidence/'tracker.log').open('w')
        try:
            if private:
                snapshot_frame(BASE,private,tracker_path,camera_binary)
                save_json(private/'session.json',dict(session_id=evidence.name,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},versions=result))
            def interrupted(signum, frame):
                raise RuntimeError(f'Trial interrupted by signal {signum}')
            signal.signal(signal.SIGTERM,interrupted)
            signal.signal(signal.SIGINT,interrupted)
            signal.signal(signal.SIGHUP,interrupted)
            phase('checking',detail='Keep hands lowered and head still')
            if args.viewer:
                checked=subprocess.run([str(BASE/'probe/scene-check')],capture_output=True,text=True,timeout=8)
                if checked.returncode:
                    raise RuntimeError('Cannot check XR scene ownership; viewer was not launched')
                owner=json.loads(checked.stdout)
                if not owner.get('available') or owner.get('scene_pid'):
                    raise RuntimeError('Another XR scene is active; close it before starting Frame Hands Lab')
                viewer_environment=os.environ.copy()
                viewer_environment.setdefault('DISPLAY',':0')
                viewer_environment['XR_RUNTIME_JSON']='/opt/steamvr/steamxr_linuxarm64.json'
                viewer_environment['LD_LIBRARY_PATH']='/opt/steamvr/bin/linuxarm64'+(':'+viewer_environment['LD_LIBRARY_PATH'] if viewer_environment.get('LD_LIBRARY_PATH') else '')
                viewer_environment['FRAME_HANDS_TRIAL_STATUS']=str(phase_path)
                viewer_environment['FRAME_HANDS_VIEWER_REPORT']=str(evidence/'viewer-report.json')
                viewer_environment['FRAME_HANDS_VIEWER_TIMEOUT']=str(args.seconds+args.start_delay+65)
                viewer_environment['FRAME_HANDS_TEMPORAL_VIEWER']='1' if args.temporal_viewer else '0'
                viewer_environment['FRAME_HANDS_SMOOTH_VIEWER']='1' if args.smooth_viewer else '0'
                viewer_log=(evidence/'viewer.log').open('w')
                viewer=subprocess.Popen([str(BASE.parent/'FrameCinema/godot'),'--path',str(BASE/'viewer'),'--xr-mode','on'],
                    env=viewer_environment,stdout=viewer_log,stderr=subprocess.STDOUT,start_new_session=True)
            camera_start = time.monotonic_ns()
            camera = subprocess.Popen([str(camera_binary),'--sensor','og01a1b','--status','1'],stdout=camera_log,
                                      stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if viewer is not None and viewer.poll() is not None:
                    raise RuntimeError('Viewer closed before camera readiness')
                if camera.poll() is not None:
                    raise RuntimeError(f'Camera reader exited with {camera.returncode}; inspect camera.log')
                if ring_ready(RUNTIME/'cam-ring',camera.pid,camera_start,require_forward=True):
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('Camera reader did not publish both forward camera streams within 30 seconds')
            expected=3 if args.swap_sides else 0
            # PC sessions keep the same reader and locks while a low-detail view
            # improves. Never retry a confirmed opposite ordering or dead reader.
            geometry_deadline=time.monotonic()+(30 if args.pc_session else 0)
            geometry_attempt=0
            while True:
                geometry_attempt+=1
                geometry=subprocess.run([sys.executable,str(BASE/'check_camera_geometry.py'),
                    '--output',str(evidence/'camera-geometry.json')],timeout=15)
                shutil.copyfile(evidence/'camera-geometry.json',evidence/f'camera-geometry-{geometry_attempt}.json')
                if geometry.returncode==expected:break
                detail=json.loads((evidence/'camera-geometry.json').read_text())
                retryable=(geometry.returncode==2 and detail.get('reason')=='Insufficient consistent geometry; use a well-lit textured scene')
                if not retryable or camera.poll() is not None or time.monotonic()>=geometry_deadline:
                    raise RuntimeError('Camera geometry did not confirm the requested ordering; inspect camera-geometry.json')
                print('Waiting for scene detail; checking camera geometry again automatically.',flush=True)
                time.sleep(.1)
            result['camera_geometry_attempts']=geometry_attempt
            result['camera_geometry_confirmed']=True
            if not args.start_delay:print('Camera geometry passed; starting hand tracking now.',flush=True)
            if args.start_delay:
                print(f'Camera geometry passed; tracking starts in {args.start_delay} seconds.',flush=True)
                deadline=time.monotonic()+args.start_delay
                previous=-1
                while time.monotonic()<deadline:
                    remaining=math.ceil(deadline-time.monotonic())
                    if remaining!=previous:phase('prepare',remaining,'Raise both hands');previous=remaining
                    if viewer is not None and viewer.poll() is not None:
                        raise RuntimeError('Viewer closed before the test')
                    time.sleep(.05)
            command=[str(BASE/'upstream/build/ft-hands'),'--models',str(BASE/'upstream/models/ncnn'),
                     '--cams','mono','--cpus','5,6,7','--threads','3','--status','1',
                     '--seconds',str(args.seconds+8)]
            if args.swap_sides:
                command.append('--swap-sides')
            if private:
                command.extend(['--record',str(private),'--record-for',str(args.seconds+5)])
            tracker_start = time.monotonic_ns()
            tracker_environment=os.environ.copy()
            for flag in list(tracker_environment):
                if not flag.startswith('FRAME_HANDS_'):continue
                tracker_environment.pop(flag,None)
            tracker_environment.pop('FRAME_JOB_DEVICE_ROOT',None)
            if args.stable_identity:
                tracker_environment['FRAME_HANDS_PAIR_CONTINUITY']='1'
                tracker_environment['FRAME_HANDS_OPEN_SIDE']='1'
            if args.identity_latch:
                command[0]=str(candidate_tracker)
                tracker_environment['FRAME_HANDS_IDENTITY_LATCH']='1'
            tracker_environment['FRAME_HANDS_PALM_THRESHOLD']=str(args.palm_threshold)
            if args.rotation_search:tracker_environment['FRAME_HANDS_ROTATION_SEARCH']='1'
            if args.full_rate:tracker_environment['FRAME_HANDS_FULL_RATE']='1'
            if args.filter_guard:
                tracker_environment['FRAME_HANDS_FILTER_GUARD']='1'
                tracker_environment['FRAME_HANDS_RESCUE_AUDIT']='1'
            if args.output_recovery:tracker_environment['FRAME_HANDS_OUTPUT_RECOVERY']='1'
            if args.joint_identity:tracker_environment['FRAME_HANDS_JOINT_IDENTITY']='1'
            if private:tracker_environment['FRAME_HANDS_AUDIT']=str(private/'live-views.txt')
            if private:
                save_json(private/'tracker-options.json',dict(command=command,environment={k:v for k,v in tracker_environment.items() if k.startswith('FRAME_HANDS_')}))
            tracker=subprocess.Popen(command,stdout=tracker_log,stderr=subprocess.STDOUT,
                start_new_session=True,env=tracker_environment)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if tracker.poll() is not None:
                    raise RuntimeError(f'Tracker exited with {tracker.returncode}; inspect tracker.log')
                if hands_ready(RUNTIME/'hands',tracker_start):
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('Tracker did not publish observations from this trial within 5 seconds')
            if args.system_input:
                system_log=(evidence/'system-input.jsonl').open('w')
                input_command=[str(BASE/'system-input/build/fh-input'),'--manifest',str(BASE/'system-input/gaze_actions.json'),'--seconds',str(args.seconds+2)]
                if args.gaze_calibration:
                    input_command+=['--calibration',str(args.gaze_calibration)]
                    result['gaze_calibration_sha256']=hashlib.sha256(args.gaze_calibration.read_bytes()).hexdigest()
                if args.system_input!='observe':input_command.append('--inject')
                if args.system_input=='touch':input_command.append('--touch')
                system_input=subprocess.Popen(input_command,stdout=system_log,stderr=subprocess.STDOUT,start_new_session=True)
                if args.system_input=='observe' and (BASE/'system-input/build/surface-probe').is_file():
                    surface_log=(evidence/'surface-probe.jsonl').open('w')
                    system_surface=subprocess.Popen([str(BASE/'system-input/build/surface-probe'),str(BASE/'system-input/gaze_actions.json')],stdout=surface_log,stderr=subprocess.STDOUT,start_new_session=True)
                result['input_injected']=args.system_input!='observe'
                result['system_bridge_sha256']=hashlib.sha256((BASE/'system-input/build/fh-input').read_bytes()).hexdigest()
            measurement=subprocess.Popen([sys.executable,str(BASE/'probe/measure.py'),'--seconds',str(args.seconds),
                                          '--output',str(evidence/'measurement.json')],start_new_session=True)
            deadline=time.monotonic()+args.seconds
            previous=-1
            while measurement.poll() is None:
                remaining=max(0,math.ceil(deadline-time.monotonic()))
                if remaining!=previous:phase('tracking',remaining,'Open, curl and separate your fingers');previous=remaining
                if time.monotonic()>deadline+5:
                    raise RuntimeError('Measurement timed out')
                if viewer is not None and viewer.poll() is not None:
                    raise RuntimeError('Viewer closed; ending the camera test')
                if camera.poll() is not None or tracker.poll() is not None:
                    raise RuntimeError('A tracking worker stopped during measurement')
                if system_input is not None and system_input.poll() is not None:
                    raise RuntimeError('System input bridge stopped; ending the trial')
                time.sleep(.05)
            result['measurement_returncode']=measurement.returncode
            result['camera_running_at_end']=camera.poll() is None
            result['tracker_running_at_end']=tracker.poll() is None
            if not result['camera_running_at_end'] or not result['tracker_running_at_end']:
                raise RuntimeError('A tracking worker stopped before the measurement completed')
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
            result['error']=str(exc)
        finally:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            stop(system_input)
            stop(system_surface)
            stop(measurement)
            stop(tracker)
            stop(camera)
            try:
                phase('error' if 'error' in result else 'done',detail=result.get('error','Test finished'))
            except OSError as exc:
                result.setdefault('error',f'Cannot write viewer status: {exc}')
            if viewer is not None and viewer.poll() is None:
                time.sleep(1.5)
            stop(viewer)
            if private:
                recording=private/'sets.bin'
                result['raw_recording_saved']=recording.is_file() and recording.stat().st_size>0
                try:
                    result['recording_check']=inspect_camera(private)
                    if not result['recording_check']['all_audited_images_saved']:
                        result.setdefault('error','Recording lost images for live observations; preserve partial evidence')
                except (OSError,ValueError) as exc:
                    result.setdefault('error','Incomplete recording: '+str(exc))
            camera_log.close();tracker_log.close()
            if viewer_log:viewer_log.close()
            if system_log:system_log.close()
            if surface_log:surface_log.close()
            result['owned_processes_stopped']=all(p is None or p.poll() is not None for p in (camera,tracker,viewer,measurement,system_input,system_surface))
            (evidence/'trial.json').write_text(json.dumps(result,indent=2)+'\n')
            if private:save_json(private/'result.json',result)
        print(json.dumps(dict(evidence=str(evidence),**result),indent=2))
        return 1 if 'error' in result or result.get('measurement_returncode',1) else 0


if __name__=='__main__':
    raise SystemExit(main())
