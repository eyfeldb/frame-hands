"""Keep the Frame's forward cameras in normal exposure while optical hands are in use.

The Frame tracks its controllers' LEDs with short-exposure camera frames (cv driver setting
driver_cv/controllerTracking3 = 'alternate'); while a controller is awake those near-black frames starve the hand
tracker. This follower watches our input mode (mode.bin, FHMODE1: 1 = hands, 2 = controllers) and the hands
service, and sets the controller-tracking mode to the configured 'hands' value only while hands are selected AND
the hands service is active; otherwise it restores the configured 'controllers' value (the Frame's default).
Config: camera-mode.json beside this script: {"hands": "classic", "controllers": "alternate", "poll_s": 0.5}.
Writes nothing but that one SteamVR setting; restores the default on exit.
"""
import json, os, signal, struct, subprocess, sys, time
from pathlib import Path
BASE = Path(__file__).resolve().parent
ROOT = Path('/run/user') / str(os.getuid()) / 'frame-hands-native'
CVTRACK = BASE / 'cvtrack'; OPENVR = '/opt/steamvr/bin/linuxarm64/libopenvr_api.so'
LOG = ROOT / 'camera-mode.log'


def config():
    try:
        c = json.loads((BASE / 'camera-mode.json').read_text())
    except (OSError, ValueError):
        c = {}
    return dict(hands=c.get('hands', 'classic'), controllers=c.get('controllers', 'alternate'), poll_s=float(c.get('poll_s', 0.5)))


def selected_mode():
    try:
        d = (ROOT / 'mode.bin').read_bytes()
        tag, epoch, selection, reserved = struct.unpack('<8sQII', d)
        if tag == b'FHMODE1\0' and selection in (1, 2) and reserved == 0:
            return 'hands' if selection == 1 else 'controllers'
    except (OSError, struct.error):
        pass
    return 'controllers'


def hands_active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'frame-hands-native.service']).returncode == 0


def apply(value):
    r = subprocess.run([str(CVTRACK), OPENVR, 'set', value], capture_output=True, text=True, timeout=20)
    ROOT.mkdir(mode=0o700, exist_ok=True)
    with LOG.open('a') as f:
        f.write(f'{time.strftime("%H:%M:%S")} controllerTracking3 -> {value}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode}\n')
    return r.returncode == 0


def main():
    want = None; stop = False
    def on_signal(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    while not stop:
        c = config()
        target = c['hands'] if (selected_mode() == 'hands' and hands_active()) else c['controllers']
        if target != want:
            if apply(target):
                want = target
        time.sleep(c['poll_s'])
    apply(config()['controllers'])


if __name__ == '__main__':
    main()
