"""Install only the physical-button companion; never restart the tracker or select Hands."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent

def install(base=BASE,start=False):
    if os.name!='posix':raise RuntimeError('Install this helper on Frame')
    if not (base/'controller-watch').is_file():raise FileNotFoundError('Build controller-watch first')
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    runtime=Path('/run/user')/str(os.getuid())/'frame-hands-native';runtime.mkdir(mode=0o700,exist_ok=True)
    (units/'frame-hands-controller-watch.service').write_text('[Unit]\nDescription=Frame Hands physical controller takeover\nPartOf=frame-hands-native.service\n\n[Service]\nExecStart='+str(base/'controller-watch')+' /opt/steamvr/bin/linuxarm64/libopenvr_api.so '+str(runtime/'mode.bin')+'\nUMask=0077\nRestart=on-failure\nRestartSec=2\n')
    drop=units/'frame-hands-native.service.d';drop.mkdir(exist_ok=True)
    (drop/'controller-takeover.conf').write_text('[Unit]\nWants=frame-hands-controller-watch.service\nAfter=frame-hands-controller-watch.service\n')
    manifest=base/'manifest.json'
    if manifest.exists():
        values=json.loads(manifest.read_text());values['controller-watch']=hashlib.sha256((base/'controller-watch').read_bytes()).hexdigest()
        manifest.write_text(json.dumps(values,indent=2)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    if start:subprocess.run(['systemctl','--user','start','frame-hands-controller-watch.service'],check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true');args=parser.parse_args();install(start=args.start)
