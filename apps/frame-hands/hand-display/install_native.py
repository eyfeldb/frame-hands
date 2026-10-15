"""Install the staged hand display after stopping tracking. No runtime restart or Hands selection."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
TRACKER=Path('/home/steamos/devkit-game/FrameHandsNativeSystem-v1')
REG='/opt/steamvr/bin/linuxarm64/vrpathreg'
ENV=dict(os.environ,LD_LIBRARY_PATH='/opt/steamvr/bin/linuxarm64')
def run(*args):return subprocess.run(args,env=ENV,text=True,capture_output=True,check=True)
def main():
    if os.name!='posix':raise RuntimeError('Frame only')
    if subprocess.run(['systemctl','--user','is-active','--quiet','frame-hands-native.service']).returncode==0:raise RuntimeError('Stop native tracking before installing; controllers remain available')
    mode=Path('/run/user')/str(os.getuid())/'frame-hands-native/mode.bin'
    import struct
    if mode.exists() and struct.unpack('<8sQII',mode.read_bytes())[2]!=2:raise RuntimeError('Select Controllers first')
    package=json.loads((BASE/'display-deployment.json').read_text());source=BASE/'fh_skeletal'
    for name,digest in package['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Candidate differs: '+name)
    identity=hashlib.sha256(json.dumps(package['files'],sort_keys=True).encode()).hexdigest()[:16]
    dest=Path.home()/'devkit-game/FrameHands/system-drivers'/('fh_skeletal-native-'+identity)
    if not dest.exists():shutil.copytree(source,dest)
    for name,digest in package['files'].items():
        if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen candidate differs')
    deployment=TRACKER/'deployment.json';previous=json.loads(deployment.read_text())
    backup=BASE/'previous-deployment.json'
    if backup.exists():raise RuntimeError('Already installed; inspect existing receipt before retrying')
    backup.write_text(json.dumps(previous,indent=2)+'\n')
    run(REG,'removedriver',previous['bundle'])
    try:run(REG,'adddriver',str(dest))
    except Exception:
        run(REG,'adddriver',previous['bundle']);raise
    package.update(bundle=str(dest),binary=str(dest/'bin/linuxarm64/driver_fh_skeletal.so'),registered=True)
    deployment.write_text(json.dumps(package,indent=2)+'\n')
    units=Path.home()/'.config/systemd/user';runtime=mode.parent
    (units/'frame-hands-display.service').write_text('[Unit]\nDescription=Frame Hands animated dashboard gloves\nPartOf=frame-hands-native.service\n\n[Service]\nExecStart='+str(BASE/'hand-display-bin')+' /opt/steamvr/bin/linuxarm64/libopenvr_api.so '+str(BASE/'assets')+' '+str(runtime/'input.bin')+' '+str(runtime/'display.stop')+'\nUMask=0077\nRestart=on-failure\nRestartSec=2\n')
    drop=units/'frame-hands-native.service.d';drop.mkdir(exist_ok=True)
    (drop/'hand-display.conf').write_text('[Unit]\nWants=frame-hands-display.service\nAfter=frame-hands-display.service\n')
    run('systemctl','--user','daemon-reload')
    print('Installed. Restart Frame runtime before manual Start. No camera, scene or Hands mode was started.')
if __name__=='__main__':main()
