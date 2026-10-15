"""Build the isolated live tracker only from a verified native camera-replay bundle.

No capture, service registration, installation or XR startup. Run locally on Frame.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
from run_tracker_rig_check import sha
from workflow_guard import native_workflow


def verify_inputs(bundle, replay, ncnn):
    bundle=bundle.resolve();replay=replay.resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        path=(bundle/name).resolve()
        if bundle not in path.parents or sha(path)!=digest:
            raise ValueError('Changed tracker bundle: '+name)
    result=json.loads((replay/'result.json').read_text())
    if result.get('scheduling')!='all saved sets' or result.get('original_detector_outputs_identical') is not True:
        raise ValueError('Passing same-input native replay required')
    provenance=result['provenance']
    if provenance['manifest_sha256']!=sha(bundle/'manifest.json'):
        raise ValueError('Replay tested another bundle')
    if provenance['ncnn_library_sha256']!=sha(ncnn/'lib/libncnn.a'):
        raise ValueError('Replay used another inference library')
    if result['environment'].get('machine')!='aarch64':
        raise ValueError('Native ARM64 replay required')
    executable=replay/'ft-handreplay-rig'
    if sha(executable)!=provenance['executable_sha256']:
        raise ValueError('Changed replay executable')
    if result.get('comparison')=='fixed_rig_vs_whole_pose':
        from run_tracker_rig_check import verify_head_evidence,recorded_trace_stats
        head=provenance.get('head_evidence') or {}
        verified=verify_head_evidence(head['path'],provenance['recording_sha256'])
        if verified!=head:raise ValueError('Changed head evidence provenance')
    rows=result.get('results',[])
    if [row.get('name') for row in rows]!=['0-baseline','1-candidate','2-candidate','3-baseline']:
        raise ValueError('Complete ABBA replay required')
    reference=None;fitted=[]
    for index,row in enumerate(rows):
        expected='candidate' if index in (1,2) else 'baseline'
        if row.get('variant')!=expected or row.get('processed_frames',0)<=0:
            raise ValueError('Invalid replay run')
        folder=replay/row['name']
        hashes=tuple(sha(folder/name) for name in ('poses.txt','views.txt'))
        if hashes!=(row['poses_sha256'],row['views_sha256']):
            raise ValueError('Changed replay output')
        if reference is None:reference=hashes
        if hashes!=reference:raise ValueError('Detector output differs')
        if len((folder/'times.txt').read_text().splitlines())!=row['processed_frames']:
            raise ValueError('Missing replay timing rows')
        if result.get('comparison')=='fixed_rig_vs_whole_pose':
            if sha(folder/'rig.txt')!=row.get('rig_sha256'):raise ValueError('Changed whole-pose trace')
            actual=recorded_trace_stats(folder/'rig.txt')
            if any(row.get(key)!=value for key,value in actual.items()):raise ValueError('Whole-pose counters disagree with trace')
            if expected=='candidate' and (row.get('whole_eligible',0)<=0 or row.get('recorded_head_frames')!=row['processed_frames']):raise ValueError('Whole-pose history not exercised')
        if expected=='candidate':
            count=sum(line.startswith('rig ') and ' fitted ' in line for line in (folder/'rig.txt').read_text().splitlines())
            if not count:raise ValueError('No articulated fits exercised')
            fitted.append(count)
    return dict(bundle_manifest_sha256=sha(bundle/'manifest.json'),replay_result_sha256=sha(replay/'result.json'),ncnn_library_sha256=provenance['ncnn_library_sha256'],fitted_outputs=fitted,native_whole_pose_verified=result.get('comparison')=='fixed_rig_vs_whole_pose')


def build_command(source, library, include, executable, capture_head=False):
    return ['c++','-std=c++17','-O2','-fopenmp','-Wno-psabi',*(['-DFRAME_HANDS_WITH_OPENVR'] if capture_head else []),'-I'+str(include),
        *[str(source/'track'/name) for name in ('main.cpp','calib.cpp','nets.cpp','tracker.cpp','io.cpp','record.cpp','pinch.cpp')],
        str(library),'-ljsoncpp','-lpthread',*(['-ldl'] if capture_head else []),'-o',str(executable)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','replay','ncnn','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine()!='aarch64':
        raise ValueError('Build must use the native Frame toolchain')
    if args.output.exists():raise FileExistsError(args.output)
    with native_workflow():
        provenance=verify_inputs(args.bundle,args.replay,args.ncnn)
        args.output.mkdir(parents=True,exist_ok=False)
        source=args.output/'source'
        shutil.copytree(args.bundle,source)
        executable=(args.output/'ft-hands-native-rig').resolve()
        command=build_command(source.resolve(),(args.ncnn/'lib/libncnn.a').resolve(),(args.ncnn/'include/ncnn').resolve(),executable,json.loads((source/'manifest.json').read_text()).get('capture_head_support',False))
        built=subprocess.run(command,capture_output=True,text=True,timeout=180)
        (args.output/'build.stdout').write_text(built.stdout)
        (args.output/'build.stderr').write_text(built.stderr)
        if built.returncode:raise RuntimeError('Native tracker link failed')
        report=dict(scope=__doc__,binary_sha256=sha(executable),provenance=provenance,command=command,
            compiler=subprocess.run(['c++','--version'],capture_output=True,text=True,check=True).stdout,
            runtime_inputs=dict(models='source/models/ncnn',device='source/device',profiles='source/profiles'),
            required_live_flags=['--no-publish','--cams','mono','--cpus','5,6,7','--threads','3'],
            limitations=['Build only; no camera, XR or installed driver changed.','Live main publication/recording and timing still require device validation.','A verified same-input replay is not tracking quality or wearer acceptance.'])
        (args.output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
if __name__=='__main__':main()
