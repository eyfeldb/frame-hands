"""Create and optionally register an immutable, user-local SteamVR hand driver."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steamvr',type=Path,default=Path(r'C:\Program Files (x86)\Steam\steamapps\common\SteamVR'))
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--unregister',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    vrpath=args.steamvr/'bin/win64/vrpathreg.exe'
    if args.unregister:
        target=args.unregister.resolve();assert (target/'driver.vrdrivermanifest').is_file()
        assert json.loads((target/'driver.vrdrivermanifest').read_text())['name']=='fh_skeletal'
        subprocess.run([str(vrpath),'removedriver',str(target)],check=True);return
    source=root/'driver/fh_skeletal'
    files={str(p.relative_to(source)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file() and p.suffix not in {'.pdb','.lib','.exp'} and p.name not in {'input.bin','input.lock','input.tmp'}}
    assert 'bin/win64/driver_fh_skeletal.dll' in files
    fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    dest=root.parents[2]/'.local/pc-hands/bundles'/fingerprint/'fh_skeletal'
    for name,digest in files.items():
        target=dest/name
        if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    current_path=Path.home()/'AppData/Local/openvr/openvrpaths.vrpath'
    current=json.loads(current_path.read_text())
    others=[]
    for path in current.get('external_drivers') or []:
        p=Path(path)/'driver.vrdrivermanifest'
        if p.exists() and json.loads(p.read_text()).get('name')=='fh_skeletal' and Path(path).resolve()!=dest.resolve():others.append(path)
    if args.register and others:raise RuntimeError(f'Another Frame Hands bundle is registered; explicitly unregister it first: {others}')
    report=dict(bundle=str(dest),files=files,registered=False,previous_external_drivers=current.get('external_drivers'),recovery=f'vrpathreg.exe removedriver "{dest}"; close and reopen SteamVR to unload')
    if args.register:
        subprocess.run([str(vrpath),'adddriver',str(dest)],check=True)
        after=json.loads(current_path.read_text());report['registered']=any(Path(x).resolve()==dest.resolve() for x in after.get('external_drivers') or [])
        if not report['registered']:raise RuntimeError('Driver registration could not be verified')
    (root/'deployment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(bundle=str(dest),registered=report['registered'],files=len(files))))


if __name__=='__main__':main()
