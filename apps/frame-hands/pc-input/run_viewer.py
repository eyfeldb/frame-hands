"""Bounded PC VR skeleton comparison, outside Maestro; no controller synthesis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from session_recording import prepare, collect, verify, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--filter-guard',action='store_true',help='Use the isolated smoothing-guard candidate')
    parser.add_argument('--output-recovery',action='store_true',help='Try bounded fresh monocular geometry recovery; requires --filter-guard')
    parser.add_argument('--joint-identity',action='store_true',help='Opt-in joint stereo identity correction; requires --output-recovery')
    parser.add_argument('--stability-preview',action='store_true',help='Show experimental filtering and bounded gap hold in green/yellow; no driver change')
    parser.add_argument('--no-scene-input',action='store_true',help='Diagnostic hypothesis: disable Godot controller action sets to isolate sampler input')
    parser.add_argument('--view',choices=['candidate','source','steam','all'],help='Visible hand layer; stability trials default to candidate only')
    parser.add_argument('--camera-buffer-pool',action='store_true',help='Use the separately installed camera buffer candidate')
    parser.add_argument('--camera-direct',action='store_true',help='Use the separately installed exact-queue camera reader')
    parser.add_argument('--no-record',action='store_true',help='Explicitly disable the default private offline-review recording')
    args=parser.parse_args()
    if args.joint_identity and not args.output_recovery:parser.error('--joint-identity requires --output-recovery')
    if args.output_recovery and not args.filter_guard:parser.error('--output-recovery requires --filter-guard')
    if args.view=='candidate' and not args.stability_preview:parser.error('--view candidate requires --stability-preview')
    if args.camera_direct and args.camera_buffer_pool:parser.error('Select only one camera candidate')
    if not 5<=args.seconds<=300:parser.error('seconds must be 5..300')
    root=Path(__file__).resolve().parent
    workspace=root.parents[2]
    godot=workspace/'.local/deps/godot-4.7.2/Godot_v4.7.2-stable_win64.exe'
    sampler=root/'build/sample-viewer.exe'
    if not godot.is_file() or not sampler.is_file():parser.error('Build sampler and supply portable Godot 4.7.2 first')
    deploy=json.loads((root/'deployment.json').read_text())
    session_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    evidence=root/'evidence'/('local-viewer-'+session_id)
    evidence.mkdir(parents=True,exist_ok=False)
    if not args.no_record:prepare(root,evidence,args,deploy)
    actions=[];poses=[];skeleton=[]
    for side in ['left','right']:
        pose='/actions/main/in/controller'+side;bone='/actions/main/in/skeleton'+side
        actions.extend([dict(name=pose,type='pose'),dict(name=bone,type='skeleton',skeleton='/skeleton/hand/'+side)])
        poses.append(dict(output=pose,path=f'/user/hand/{side}/pose/raw'))
        skeleton.append(dict(output=bone,path=f'/user/hand/{side}/input/skeleton/{side}'))
    binding=evidence/'binding.json'
    binding.write_text(json.dumps(dict(controller_type='fh_skeletal',name='Skeleton diagnostic only',bindings={'/actions/main':dict(poses=poses,skeleton=skeleton)}),indent=2))
    manifest=evidence/'actions.json'
    manifest.write_text(json.dumps(dict(actions=actions,action_sets=[dict(name='/actions/main',usage='leftright')],default_bindings=[dict(controller_type='fh_skeletal',binding_url=str(binding))]),indent=2))
    env=dict(os.environ,FRAME_HANDS_PC_VIEW=str(evidence/'view.bin'),FRAME_HANDS_PC_SECONDS=str(args.seconds+100))
    env['FRAME_HANDS_VIEW_LAYER']=args.view or ('candidate' if args.stability_preview else 'all')
    env.pop('FRAME_HANDS_NO_SCENE_INPUT',None)
    if args.no_scene_input:env['FRAME_HANDS_NO_SCENE_INPUT']='1'
    env.pop('FRAME_HANDS_REQUIRE_STEAM',None)
    if args.view=='steam':env['FRAME_HANDS_REQUIRE_STEAM']='1'
    env.pop('FRAME_HANDS_STABILITY_PREVIEW',None)
    env.pop('FRAME_HANDS_TRACE',None)
    if not args.no_record:env['FRAME_HANDS_TRACE']=str(evidence/'sampler-trace.bin')
    if args.stability_preview:env['FRAME_HANDS_STABILITY_PREVIEW']='1'
    children=[];source=None;sample=None;recording_error=None
    try:
        with (evidence/'sampler.log').open('w') as samplelog,(evidence/'viewer.log').open('w') as viewlog:
            source_command=[sys.executable,str(root/'run_session.py'),'--seconds',str(args.seconds),'--stop-file',str(evidence/'stop'),'--source-ready-file',str(evidence/'ready'),'--session-id',session_id]
            if not args.no_record:source_command.extend(['--record-dir',str(evidence)])
            if args.filter_guard:source_command.append('--filter-guard')
            if args.output_recovery:source_command.append('--output-recovery')
            if args.joint_identity:source_command.append('--joint-identity')
            if args.camera_buffer_pool:source_command.append('--camera-buffer-pool')
            if args.camera_direct:source_command.append('--camera-direct')
            source=subprocess.Popen(source_command)
            deadline=time.monotonic()+15
            while not (evidence/'ready').exists():
                if source.poll() is not None:return source.returncode or 1
                if time.monotonic()>deadline:raise RuntimeError('Headset source lock was not confirmed; viewer was not opened')
                time.sleep(.1)
            sample=subprocess.Popen([str(sampler),r'C:\Program Files (x86)\Steam\steamapps\common\SteamVR\bin\win64\openvr_api.dll',str(manifest),str(Path(deploy['bundle'])/'bin/win64/input.bin'),str(evidence/'view.bin'),str(args.seconds+100),str(evidence/'sampler.stop')],stdout=samplelog,stderr=subprocess.STDOUT,env=env)
            children.append(sample)
            children.append(subprocess.Popen([str(godot),'--path',str(root/'viewer'),'--xr-mode','on'],env=env,stdout=viewlog,stderr=subprocess.STDOUT))
            time.sleep(2)
            if any(p.poll() is not None for p in children):raise RuntimeError('Viewer startup failed; inspect local viewer logs')
            print(json.dumps(dict(viewer_evidence=str(evidence),seconds=args.seconds)),flush=True)
            while source.poll() is None and all(p.poll() is None for p in children):time.sleep(.2)
    finally:
        if sample and sample.poll() is None:
            (evidence/'sampler.stop').touch()
            try:sample.wait(timeout=3)
            except subprocess.TimeoutExpired:sample.terminate()
        if source and source.poll() is None:
            (evidence/'stop').touch()
            source.wait(timeout=25)
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait(timeout=5)
        save_json(evidence/'exit-status.json',dict(source=None if source is None else source.returncode,
                  sampler=None if sample is None else sample.returncode))
        if not args.no_record and source is not None:
            try:
                collect(root,evidence,session_id)
                check=verify(evidence)
                if not check['complete']:recording_error='Recording is incomplete; inspect recording-check.json'
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                recording_error=str(exc)
                save_json(evidence/'copy-error.json',dict(error=recording_error,frame_originals_preserved=True))
            if recording_error:print('RECORDING ERROR: '+recording_error,flush=True)
        print('Skeleton viewer closed; evidence: '+str(evidence),flush=True)
    return 1 if recording_error or not source or source.returncode or not sample or sample.returncode else 0


if __name__=='__main__':raise SystemExit(main())
