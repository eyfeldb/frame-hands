"""Native package installation; no camera startup or runtime restart."""
import argparse,ctypes,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
REG='/opt/steamvr/bin/linuxarm64/vrpathreg'
ENV=dict(os.environ,LD_LIBRARY_PATH='/opt/steamvr/bin/linuxarm64')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reg(*args):return subprocess.run([REG,*args],env=ENV,capture_output=True,text=True,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--install',action='store_true');a=p.parse_args()
    source=BASE/'driver/fh_skeletal';binary=source/'bin/linuxarm64/driver_fh_skeletal.so'
    library=ctypes.CDLL(str(binary));factory=library.HmdDriverFactory
    factory.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)];factory.restype=ctypes.c_void_p
    error=ctypes.c_int();assert factory(b'IServerTrackedDeviceProvider_004',ctypes.byref(error))
    assert not factory(b'bad',ctypes.byref(error)) and error.value
    files={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    target=Path.home()/'devkit-game/FrameHands/system-drivers'/('fh_skeletal-native-'+identity)
    if not target.exists():shutil.copytree(source,target)
    if {str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}!=files:raise RuntimeError('Immutable driver differs')
    existing=subprocess.run([REG,'finddriver','fh_skeletal'],env=ENV,capture_output=True,text=True)
    is_registered=existing.returncode==0 and str(target) in existing.stdout
    deployment=dict(bundle=str(target),binary=str(target/'bin/linuxarm64/driver_fh_skeletal.so'),files=files,registered=is_registered)
    (BASE/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    # Generated binaries are pinned after building; self-referential manifests excluded.
    manifest={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and p.name not in ('manifest.json','deployment.json','install-receipt.json') and '__pycache__' not in p.parts and '.godot' not in p.parts}
    (BASE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run([sys.executable,str(BASE/'native_service.py'),'--check'],check=True)
    if not a.install:print(json.dumps(dict(prepared=True,bundle=str(target))));return
    existing=subprocess.run([REG,'finddriver','fh_skeletal'],env=ENV,capture_output=True,text=True)
    if existing.returncode not in (0,1) or (existing.returncode==0 and str(target) not in existing.stdout):raise RuntimeError('Different native skeletal driver already registered; preserve it')
    reg('adddriver',str(target))
    if str(target) not in reg('finddriver','fh_skeletal').stdout:raise RuntimeError('Driver registration did not persist')
    deployment['registered']=True;(BASE/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    unit='[Unit]\nDescription=Frame native optical hands (manual start)\n\n[Service]\nType=notify\nExecStart=/usr/bin/python3 '+str(BASE/'native_service.py')+'\nWorkingDirectory='+str(BASE)+'\nUMask=0077\nKillMode=control-group\nTimeoutStartSec=60\nTimeoutStopSec=25\nRestart=no\n'
    (units/'frame-hands-native.service').write_text(unit)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    from install_controller_watch import install as install_watch
    install_watch(BASE)
    # One entry opens a menu; opening it never starts tracking.
    subprocess.run([sys.executable,str(BASE/'install_menu.py')],check=True)
    receipt=dict(installed=True,driver=deployment,service='frame-hands-native.service',enabled=False,started=False,requires_runtime_restart=True,pc_changed=False,shortcuts=['FrameHands'])
    (BASE/'install-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
