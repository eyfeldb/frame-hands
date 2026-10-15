"""Register the frozen display driver with SteamVR closed; preserve accepted deployment."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
VR=Path('C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/vrpathreg.exe')
def main():
    prepared=Path(sys.argv[1]).resolve();report=ROOT/'.local/hand-alternatives/steamvr-model/deployment.json'
    running=subprocess.run(['tasklist','/FI','IMAGENAME eq vrserver.exe','/NH'],capture_output=True,text=True,check=True)
    if 'vrserver.exe' in running.stdout.lower():raise RuntimeError('Close SteamVR before registering this candidate')
    candidate=json.loads((prepared/'pc/display-deployment.json').read_text());source=Path(candidate['bundle'])
    for name,digest in candidate['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Candidate changed: '+name)
    identity=hashlib.sha256(json.dumps(candidate['files'],sort_keys=True).encode()).hexdigest()[:16]
    target=ROOT/'.local/pc-hands/bundles'/identity/'fh_skeletal'
    if not target.exists():shutil.copytree(source,target)
    for name,digest in candidate['files'].items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen driver changed')
    prior=json.loads(report.read_text());backup=prepared/'previous-pc-deployment.json'
    if backup.exists():raise RuntimeError('Already installed; inspect receipt before retrying')
    backup.write_text(json.dumps(prior,indent=2)+'\n')
    subprocess.run([str(VR),'removedriver',prior['bundle']],check=True)
    try:subprocess.run([str(VR),'adddriver',str(target)],check=True)
    except Exception:
        subprocess.run([str(VR),'adddriver',prior['bundle']],check=True);raise
    candidate.update(bundle=str(target),protocol='FHMODEL2',registered=True,previous_hand_drivers=[prior['bundle']])
    report.write_text(json.dumps(candidate,indent=2)+'\n')
    # Controllers remains selected even if an old native Hands epoch exists.
    import struct,time
    (target/'bin/win64/mode.bin').write_bytes(struct.pack('<8sQII',b'FHMODE1',time.time_ns(),2,0))
    app=dict(app_key='local.framehands.handdisplay',launch_type='binary',binary_path_windows=str(prepared/'hand-display.exe'),arguments=subprocess.list2cmdline([str(VR.with_name('openvr_api.dll')),str(prepared/'assets'),str(target/'bin/win64/input.bin'),str(prepared/'display.stop')]),working_directory=str(prepared),is_dashboard_overlay=True,strings={'en_us':{'name':'Frame Hands Display','description':'Animated dashboard gloves; no cameras or recording.'}})
    (prepared/'display.vrmanifest').write_text(json.dumps(dict(source='builtin',applications=[app]),indent=2)+'\n')
    print('Installed driver. Start SteamVR, then register-display install and launch the prepared manifest. Controllers remain selected.')
if __name__=='__main__':main()
