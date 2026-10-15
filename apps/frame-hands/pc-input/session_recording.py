"""Freeze PC diagnostic inputs, collect Frame files, and verify recordings offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from recording import snapshot, save_json, inspect_camera, inspect_fixed, digest


def prepare(root, directory, args, deploy):
    required=(args.seconds+15)*140_000_000+3_000_000_000
    if shutil.disk_usage(directory).free<required:
        raise RuntimeError(f'Recording needs {required/1e9:.1f} GB free on PC for the full camera copy')
    files={}
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.py','.h','.cpp','.json','.ps1'):
            files['pc-input/'+path.name]=path
    for path in (root/'viewer').iterdir():
        if path.is_file() and path.suffix in ('.gd','.godot','.tscn','.tres'):
            files['pc-input/viewer/'+path.name]=path
    files['pc-input/build/sample-viewer.exe']=root/'build/sample-viewer.exe'
    files['pc-input/build/replay-trace.exe']=root/'build/replay-trace.exe'
    files['recording.py']=root.parent/'recording.py'
    for name in deploy['files']:
        files['driver/'+name]=Path(deploy['bundle'])/name
    snapshot(files,directory/'snapshot')
    steam=Path(r'C:\Program Files (x86)\Steam\steamapps\common\SteamVR')
    godot=root.parents[2]/'.local/deps/godot-4.7.2/Godot_v4.7.2-stable_win64.exe'
    save_json(directory/'runtime.json',dict(steamvr_version=(steam/'bin/version.txt').read_text(),
              openvr_api_sha256=digest(steam/'bin/win64/openvr_api.dll'),godot_sha256=digest(godot)))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    save_json(directory/'session.json',dict(version=1,session_id=directory.name.removeprefix('local-viewer-'),
              git_commit=commit,settings=vars(args),status='prepared',
              snapshot_is_authoritative=True,mode='PC-streamed diagnostic; no synthesized controls',
              cameras='two forward monochrome streams',recording_may_affect_timing=True))


def collect(root, directory, session_id):
    """After owned live processes stop. Retain Frame originals, including on copy failure."""
    adb=root.parents[2]/'.local/android-tools/platform-tools/adb.exe'
    env=dict(os.environ,ANDROID_USER_HOME=str(Path.home()/'.android'))
    destination=directory/'frame'
    if destination.exists():raise RuntimeError('Refusing to overwrite a previous Frame copy')
    print('Tracking stopped. Saving the camera recording to this PC; keep USB connected.',flush=True)
    with (directory/'copy.log').open('w') as log:
        subprocess.run([str(adb),'-s','frame','pull',f'/home/steamos/devkit-game/FrameHands/private/{session_id}',str(destination)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        subprocess.run([str(adb),'-s','frame','pull',f'/home/steamos/devkit-game/FrameHands/evidence/{session_id}',str(directory/'frame-logs')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)


def verify(directory):
    result=dict(version=1,complete=False,errors=[])
    try:
        exits=json.loads((directory/'exit-status.json').read_text())
        transport=json.loads((directory/'transport-exit.json').read_text())
        if exits.get('sampler')!=0 or exits.get('source')!=0 or transport.get('relay_exit')!=0 or transport.get('source_exit')!=0:
            result['errors'].append('A capture process did not exit successfully; inspect exit-status.json and transport-exit.json')
        report=json.loads((directory/'transport.json').read_text())
        if report.get('error'):result['errors'].append('Transport: '+report['error'])
    except (OSError,ValueError) as exc:result['errors'].append('Process completion: '+str(exc))
    for name,path,kind in [('sampler',directory/'sampler-trace.bin','trace'),('relay',directory/'relay-trace.bin','relay')]:
        try:result[name]=inspect_fixed(path,kind)
        except (OSError,ValueError) as exc:result['errors'].append(name+': '+str(exc))
    try:
        result['camera']=inspect_camera(directory/'frame')
        if not result['camera']['all_audited_images_saved']:result['errors'].append('Some live observations have no saved images')
        frame=json.loads((directory/'frame/result.json').read_text())
        if frame.get('error') or frame.get('measurement_returncode')!=0 or not frame.get('owned_processes_stopped'):
            result['errors'].append('Frame trial incomplete; inspect frame/result.json')
        for folder in (directory/'snapshot',directory/'frame/snapshot'):
            for name,meta in json.loads((folder/'files.json').read_text()).items():
                if digest(folder/name)!=meta['sha256']:raise ValueError('Snapshot hash mismatch: '+name)
        for name,expected in json.loads((directory/'frame/snapshot/device/persist/hashes.json').read_text()).items():
            if digest(directory/'frame/snapshot/device/persist'/name)!=expected:raise ValueError('Calibration hash mismatch')
    except (OSError,ValueError,KeyError) as exc:result['errors'].append('Frame/snapshot: '+str(exc))
    result['complete']=not result['errors']
    result['scope']='Saved data integrity and observed-frame coverage; recording overhead and physical accuracy require live validation'
    save_json(directory/'recording-check.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=verify(args.directory)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['complete'] else 1)
