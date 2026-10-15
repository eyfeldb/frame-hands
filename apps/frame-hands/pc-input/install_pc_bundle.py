"""Freeze a newly built Windows hand driver DLL into a content-addressed bundle, register it with PC SteamVR, and
restore the wireless receiver + dashboard display app registrations. SteamVR must be closed.

usage: python install_pc_bundle.py --dll <driver_fh_skeletal.dll> --base <existing bundle dir (fh_skeletal)>
       [--backup <dir with appconfig.json.before and *.vrappconfig>]
"""
import argparse, hashlib, json, shutil, struct, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VR = Path('C:/Program Files (x86)/Steam/steamapps/common/SteamVR/bin/win64/vrpathreg.exe')
STEAM_CONFIG = Path('C:/Program Files (x86)/Steam/config')
DEPLOYMENT = ROOT / '.local/hand-alternatives/steamvr-model/deployment.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dll', required=True); ap.add_argument('--base', required=True); ap.add_argument('--backup')
    ap.add_argument('--while-running', action='store_true', help='register a NEW bundle while SteamVR runs; it loads on the next SteamVR start (the running session keeps the old DLL but loses its mailbox)')
    a = ap.parse_args()
    running = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq vrserver.exe', '/NH'], capture_output=True, text=True, check=True)
    if 'vrserver.exe' in running.stdout.lower() and not a.while_running:
        raise SystemExit('Close SteamVR before registering a driver (or pass --while-running)')
    base = Path(a.base).resolve(); dll = Path(a.dll).resolve()
    files = {}
    for q in base.rglob('*'):
        if q.is_file() and q.name not in ('mode.bin', 'input.bin', 'display.bin', 'dashboard.bin', 'native-relay.tmp'):
            files[q.relative_to(base).as_posix()] = sha(q)
    files['bin/win64/driver_fh_skeletal.dll'] = sha(dll)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    target = ROOT / '.local/pc-hands/bundles' / identity / 'fh_skeletal'
    if not target.exists():
        shutil.copytree(base, target, ignore=shutil.ignore_patterns('mode.bin', 'input.bin', 'display.bin', 'dashboard.bin', 'native-relay.tmp'))
        shutil.copy2(dll, target / 'bin/win64/driver_fh_skeletal.dll')
    for name, digest in files.items():
        if sha(target / name) != digest:
            raise SystemExit('Frozen bundle mismatch: ' + name)
    (target / 'bin/win64/mode.bin').write_bytes(struct.pack('<8sQII', b'FHMODE1', time.time_ns(), 2, 0))   # Controllers selected until Hands is chosen
    prior = json.loads(DEPLOYMENT.read_text()) if DEPLOYMENT.exists() else {}
    show = subprocess.run([str(VR), 'show'], capture_output=True, text=True).stdout
    for line in show.splitlines():
        if 'fh_skeletal :' in line:
            subprocess.run([str(VR), 'removedriver', line.split(':', 1)[1].strip()], check=False)
    subprocess.run([str(VR), 'adddriver', str(target)], check=True)
    report = dict(bundle=str(target), files=files, input_mode_supported=True, skeleton_only=False, registered=True, live_validated=False,
                  protocol='FHMODEL2', gesture_controls='trigger=index pinch, grip=closed hand', render_model='always fh_display_empty',
                  previous_hand_drivers=[prior.get('bundle')] if prior.get('bundle') else [])
    DEPLOYMENT.parent.mkdir(parents=True, exist_ok=True); DEPLOYMENT.write_text(json.dumps(report, indent=2) + '\n')
    # App registrations: wireless receiver (auto-launched by SteamVR) and the dashboard glove display (its argument
    # names the driver mailbox inside the bundle, so it is regenerated for the new bundle path).
    receiver = ROOT / '.local/hand-alternatives/native-wireless/receiver.vrmanifest'
    display_dir = ROOT / '.local/hand-display'
    display = display_dir / 'display.vrmanifest'
    if display.exists():
        m = json.loads(display.read_text()); app = m['applications'][0]
        app['arguments'] = subprocess.list2cmdline([str(VR.with_name('openvr_api.dll')), str(display_dir / 'assets'), str(target / 'bin/win64/input.bin'), str(display_dir / 'display.stop')])
        display.write_text(json.dumps(m, indent=2) + '\n')
    appconfig = STEAM_CONFIG / 'appconfig.json'
    cfg = json.loads(appconfig.read_text())
    for path in (str(receiver), str(display)):
        if Path(path).exists() and path not in cfg['manifest_paths']:
            cfg['manifest_paths'].append(path)
    appconfig.write_text(json.dumps(cfg, indent=3) + '\n')
    if a.backup:
        for f in Path(a.backup).glob('*.vrappconfig'):
            dest = STEAM_CONFIG / 'vrappconfig' / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
    print('bundle', target); print('registered:', subprocess.run([str(VR), 'show'], capture_output=True, text=True).stdout.strip().splitlines()[-1])
    print('manifests:', cfg['manifest_paths'])


if __name__ == '__main__':
    main()
