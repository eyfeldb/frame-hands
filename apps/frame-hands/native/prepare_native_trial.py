"""Pin existing native trial artifacts into an explicit local configuration; no startup."""
import argparse
import json
from pathlib import Path
from run_native_trial import BASE, SOURCES, configuration, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('build','viewer','camera','godot','pose-service','openvr','scene-check','geometry','xr-runtime','recording-tool','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--whole-pose',action='store_true',help='Require matching native whole-pose replay evidence')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    value=dict(version=1,whole_pose=args.whole_pose,build=str(args.build.resolve()),viewer=str(args.viewer.resolve()),build_sha256=sha(args.build/'build.json'),viewer_manifest_sha256=sha(args.viewer/'manifest.json'),source_sha256={name:sha(BASE/name) for name in SOURCES},files={})
    for name in ('camera','godot','pose_service','openvr','scene_check','geometry','xr_runtime','recording_tool'):
        path=getattr(args,name).resolve();value['files'][name]=dict(path=str(path),sha256=sha(path))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as output:output.write(json.dumps(value,indent=2)+'\n')
    # A rejected file remains reviewable but will also be refused by the launcher.
    configuration(args.output)
    print(json.dumps(dict(configuration=str(args.output),sha256=sha(args.output),scope='Verified existing artifacts only; fresh readiness still required for launch')))
if __name__=='__main__':main()
