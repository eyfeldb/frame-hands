"""Prepare a content-addressed user-local driver; optionally register/unregister it.

Never restarts SteamVR, never overwrites a driver directory, no root required.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REG=Path('/opt/steamvr/bin/linuxarm64/vrpathreg')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--unregister',action='store_true')
    args=parser.parse_args()
    if args.register and args.unregister:parser.error('Choose one operation')
    source=ROOT/'driver/fh_pointer'
    binary=source/'bin/linuxarm64/driver_fh_pointer.so'
    assert binary.is_file()
    # Load/export check in this helper process; do not initialize a VR provider.
    library=ctypes.CDLL(str(binary))
    factory=library.HmdDriverFactory
    factory.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
    factory.restype=ctypes.c_void_p
    error=ctypes.c_int(0)
    assert factory(b'IServerTrackedDeviceProvider_004',ctypes.byref(error))
    assert not factory(b'InvalidInterface',ctypes.byref(error)) and error.value!=0
    files={str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    for name in ['LICENSE-Frametop','LICENSE-OpenVR']:
        files[name]=digest(ROOT/'vendor'/name)
    identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]
    destination=ROOT.parent/'system-drivers'/f'fh_pointer-{identity}'
    if destination.exists():
        actual={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}
        if actual!=files:raise RuntimeError('Versioned destination differs; refusing to overwrite')
    elif not args.unregister:
        destination.parent.mkdir(exist_ok=True)
        shutil.copytree(source,destination)
        for name in ['LICENSE-Frametop','LICENSE-OpenVR']:shutil.copy2(ROOT/'vendor'/name,destination/name)
    action='prepared'
    if args.register or args.unregister:
        env=dict(os.environ,LD_LIBRARY_PATH='/opt/steamvr/bin/linuxarm64')
        existing=subprocess.run([str(REG),'finddriver','fh_pointer'],env=env,capture_output=True,text=True)
        if existing.returncode not in (0,1):
            raise RuntimeError('Driver lookup failed or multiple copies exist; no registration changed')
        if existing.returncode==0 and str(destination) not in existing.stdout:
            raise RuntimeError('A different fh_pointer bundle is registered; preserve it until explicitly replaced')
        # Only this exact immutable bundle is added/removed. No config copying.
        subprocess.run([str(REG),'removedriver' if args.unregister else 'adddriver',str(destination)],env=env,check=True)
        action='unregistered' if args.unregister else 'registered_for_next_runtime_start'
        verified=subprocess.run([str(REG),'finddriver','fh_pointer'],env=env,capture_output=True,text=True)
        if args.register and (verified.returncode!=0 or str(destination) not in verified.stdout):
            raise RuntimeError('Registration verification failed')
        if args.unregister and verified.returncode!=1:
            raise RuntimeError('Unregistration verification failed')
    report=dict(action=action,path=str(destination),binary_sha256=digest(binary),files=files,
                factory_load_verified=True,runtime_restarted=False,autostart_bridge=False,
                disconnected_by_default=True)
    (ROOT/'driver-preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
