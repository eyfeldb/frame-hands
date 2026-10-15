"""Freeze/register the separate model driver; preserves the prior bundle for rollback."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
VR=Path(r'C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/vrpathreg.exe')
REPORT=WORK/'.local/hand-alternatives/steamvr-model/deployment.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    global REPORT
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--rollback',action='store_true');p.add_argument('--output',type=Path,default=REPORT);a=p.parse_args();REPORT=a.output
    paths=Path.home()/'AppData/Local/openvr/openvrpaths.vrpath'
    if a.register or a.rollback:
        running=subprocess.run(['tasklist','/FI','IMAGENAME eq vrserver.exe','/FO','CSV','/NH'],capture_output=True,text=True,check=True)
        if 'vrserver.exe' in running.stdout.lower():raise RuntimeError('Close SteamVR before changing driver registration')
    if a.rollback:
        report=json.loads(REPORT.read_text())
        subprocess.run([str(VR),'removedriver',report['bundle']],check=True)
        for path in report['previous_hand_drivers']:subprocess.run([str(VR),'adddriver',path],check=True)
        report['registered']=False;REPORT.write_text(json.dumps(report,indent=2));print('Previous hand driver registration restored');return
    current=json.loads(paths.read_text())
    original=json.loads((ROOT/'deployment.json').read_text());source=Path(original['bundle'])
    binary=ROOT/'build/model-driver-v2/driver_fh_skeletal.dll'
    files=dict(original['files'])
    for name,digest in files.items():
        if sha(source/name)!=digest:raise RuntimeError('Prior bundle changed: '+name)
    files['bin/win64/driver_fh_skeletal.dll']=sha(binary)
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    dest=WORK/'.local/pc-hands/bundles'/key/'fh_skeletal'
    for name,digest in files.items():
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(binary if name.endswith('.dll') else source/name,target)
        if sha(target)!=digest:raise RuntimeError('Frozen candidate changed: '+name)
    old=[]
    for path in current.get('external_drivers') or []:
        manifest=Path(path)/'driver.vrdrivermanifest'
        if manifest.exists() and json.loads(manifest.read_text()).get('name')=='fh_skeletal' and Path(path).resolve()!=dest.resolve():old.append(path)
    if not old and REPORT.exists():
        previous=json.loads(REPORT.read_text())
        if previous.get('bundle')==str(dest):old=previous.get('previous_hand_drivers',[])
    report=dict(protocol='FHMODEL2',bundle=str(dest),files=files,input_mode_supported=True,skeleton_only=True,registered=False,previous_hand_drivers=old,accepted_bundle=original.get('accepted_bundle',str(source)),live_validated=False)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    if a.register:
        for path in old:subprocess.run([str(VR),'removedriver',path],check=True)
        try:subprocess.run([str(VR),'adddriver',str(dest)],check=True)
        except Exception:
            for path in old:subprocess.run([str(VR),'adddriver',path],check=True)
            raise
        now=json.loads(paths.read_text());report['registered']=any(Path(x).resolve()==dest.resolve() for x in now.get('external_drivers') or [])
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
        if not report['registered']:raise RuntimeError('Registration verification failed')
    print(json.dumps(dict(bundle=str(dest),registered=report['registered'],report=str(REPORT))))
if __name__=='__main__':main()
