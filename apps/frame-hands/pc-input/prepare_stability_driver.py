"""Freeze the tested candidate beside the accepted bundle; do not register or restart."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=('stability','ownership','input-mode','presentation'),default='stability')
    args=parser.parse_args()
    accepted=json.loads((root/'deployment.json').read_text())
    source=Path(accepted['bundle'])
    build=root/'build'/dict(stability='stability-driver-v1',ownership='ownership-driver-v1',presentation='presentation-driver-v1',**{'input-mode':'input-mode-driver-v1'})[args.candidate]
    manifest=json.loads((build/'manifest.json').read_text())
    binary=build/'driver_fh_skeletal.dll'
    expected=['FRAME_HANDS_DRIVER_STABILITY=1','FRAME_HANDS_DRIVER_ROLE_GRACE=1']
    if args.candidate in ('input-mode','presentation'):expected.append('FRAME_HANDS_DRIVER_INPUT_MODE=1')
    if args.candidate=='presentation':expected.append('FRAME_HANDS_DRIVER_PRESENTATION=1')
    mode_ok=(manifest.get('defines')==expected if args.candidate!='stability' else manifest.get('define')=='FRAME_HANDS_DRIVER_STABILITY=1')
    if not mode_ok or not manifest['skeleton_only']:
        raise ValueError('Unexpected candidate mode')
    if digest(binary)!=manifest['binary_sha256']:
        raise ValueError('Candidate binary changed since testing')
    for name,sha in manifest['sources'].items():
        if digest(root/name)!=sha:raise ValueError('Candidate source changed: '+name)
    files=dict(accepted['files'])
    for name,sha in files.items():
        if digest(source/name)!=sha:raise ValueError('Accepted bundle changed: '+name)
    files['bin/win64/driver_fh_skeletal.dll']=manifest['binary_sha256']
    fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    dest=root.parents[2]/'.local/pc-hands/bundles'/fingerprint/'fh_skeletal'
    if dest.resolve()==source.resolve():raise ValueError('Candidate must be a separate bundle')
    for name,sha in files.items():
        target=dest/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(binary if name=='bin/win64/driver_fh_skeletal.dll' else source/name,target)
        if digest(target)!=sha:raise ValueError('Frozen candidate differs: '+name)
    report=dict(bundle=str(dest),files=files,registered=False,skeleton_only=True,
                accepted_bundle=str(source),build_manifest=manifest,
                next='User-ready trial requires a reversible registration switch while SteamVR is stopped; then verify loaded DLL and actual skeleton output. Do not overwrite deployment.json merely by preparing a candidate.')
    if args.candidate in ('input-mode','presentation'):report['input_mode_supported']=True
    (build/'deployment-candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(bundle=str(dest),registered=False,files=len(files),accepted_preserved=True)))


if __name__=='__main__':main()
