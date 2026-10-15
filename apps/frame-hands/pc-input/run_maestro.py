"""Recorded Maestro trial using the accepted driver; no diagnostic XR scene."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from session_recording import prepare, collect, verify, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=240)
    parser.add_argument('--input-mode',choices=('auto','hands','controllers'),help='Explicit selection; requires the isolated mode-capable driver')
    parser.add_argument('--input-selector',action='store_true',help='Show the desktop Hands/Controllers selector during this session')
    args = parser.parse_args()
    if not 30 <= args.seconds <= 300:
        parser.error('seconds must be 30..300')
    root = Path(__file__).resolve().parent
    steam = Path(r'C:\Program Files (x86)\Steam')
    deploy = json.loads((root/'deployment.json').read_text())
    if args.input_mode is None and deploy.get('input_mode_supported'):
        args.input_mode='hands'
    if args.input_mode and not deploy.get('input_mode_supported'):
        parser.error('The active driver does not support explicit input mode')
    if args.input_selector and not deploy.get('input_mode_supported'):
        parser.error('The active driver does not support the input selector')
    session = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    directory = root/'evidence'/('local-maestro-'+session)
    directory.mkdir(parents=True, exist_ok=False)
    prepare(root, directory, args, deploy)
    meta = json.loads((directory/'session.json').read_text())
    meta.update(session_id=session, mode='PC-streamed Maestro; skeleton-only driver; background observer, no viewer scene')
    save_json(directory/'session.json', meta)
    actions, poses, skeletons = [], [], []
    for side in ('left', 'right'):
        pose = '/actions/main/in/controller'+side
        bone = '/actions/main/in/skeleton'+side
        actions.extend([dict(name=pose, type='pose'), dict(name=bone, type='skeleton', skeleton='/skeleton/hand/'+side)])
        poses.append(dict(output=pose, path=f'/user/hand/{side}/pose/raw'))
        skeletons.append(dict(output=bone, path=f'/user/hand/{side}/input/skeleton/{side}'))
    save_json(directory/'binding.json', dict(controller_type='fh_skeletal', name='Maestro session observer', bindings={'/actions/main': dict(poses=poses, skeleton=skeletons)}))
    save_json(directory/'actions.json', dict(actions=actions, action_sets=[dict(name='/actions/main', usage='leftright')], default_bindings=[dict(controller_type='fh_skeletal', binding_url=str(directory/'binding.json'))]))
    source = sample = selector = None
    error = None
    env = os.environ.copy()
    for name in ('FRAME_HANDS_STABILITY_PREVIEW', 'FRAME_HANDS_REQUIRE_STEAM'):
        env.pop(name, None)
    env['FRAME_HANDS_TRACE'] = str(directory/'sampler-trace.bin')
    try:
        with (directory/'sampler.log').open('w') as log:
            source = subprocess.Popen([sys.executable, str(root/'run_session.py'), '--seconds', str(args.seconds), '--stop-file', str(directory/'stop'), '--source-ready-file', str(directory/'ready'), '--session-id', session, '--record-dir', str(directory), '--camera-direct', '--filter-guard', '--output-recovery', '--joint-identity'])
            deadline = time.monotonic()+20
            while not (directory/'ready').exists():
                if source.poll() is not None:
                    raise RuntimeError('Source failed before obtaining camera workflow locks')
                if time.monotonic() > deadline:
                    raise RuntimeError('Source workflow lock not confirmed')
                time.sleep(.1)
            if args.input_mode:
                subprocess.run([sys.executable,str(root/'select_input_mode.py'),args.input_mode],check=True)
            if args.input_selector:
                selector=subprocess.Popen([sys.executable,str(root/'select_input_mode.py'),'--ui'])
            sample = subprocess.Popen([str(root/'build/sample-viewer.exe'), str(steam/'steamapps/common/SteamVR/bin/win64/openvr_api.dll'), str(directory/'actions.json'), str(Path(deploy['bundle'])/'bin/win64/input.bin'), str(directory/'view.bin'), str(args.seconds+100), str(directory/'sampler.stop')], stdout=log, stderr=subprocess.STDOUT, env=env)
            subprocess.Popen([str(steam/'steam.exe'), '-applaunch', '2701700'])
            print(json.dumps(dict(session=session, evidence=str(directory), seconds=args.seconds, game='Maestro', controls='native skeletons only')), flush=True)
            while source.poll() is None and sample.poll() is None:
                time.sleep(.2)
    except (KeyboardInterrupt, OSError, RuntimeError) as exc:
        error = type(exc).__name__+': '+str(exc)
    finally:
        if selector and selector.poll() is None:
            selector.terminate();selector.wait(timeout=5)
        if sample and sample.poll() is None:
            (directory/'sampler.stop').touch()
            try:
                sample.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sample.terminate(); sample.wait(timeout=5)
        if source and source.poll() is None:
            (directory/'stop').touch()
            source.wait(timeout=30)
        save_json(directory/'exit-status.json', dict(source=None if source is None else source.returncode, sampler=None if sample is None else sample.returncode, error=error))
        if source is not None:
            try:
                collect(root, directory, session)
                if not verify(directory)['complete']:
                    error = error or 'Recording incomplete; inspect recording-check.json'
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                error = error or str(exc)
                save_json(directory/'copy-error.json', dict(error=str(exc), frame_originals_preserved=True))
        print(json.dumps(dict(evidence=str(directory), error=error, game_left_open=True)), flush=True)
    return 1 if error or not source or source.returncode or not sample or sample.returncode else 0


if __name__ == '__main__':
    raise SystemExit(main())
