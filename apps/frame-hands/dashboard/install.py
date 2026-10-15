"""Install/remove only the dashboard companion; accepted tracker files are untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from service import validate_build

BASE = Path(__file__).resolve().parent
UNIT = 'frame-hands-dashboard.service'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()
    target = Path.home() / '.config/systemd/user' / UNIT
    if args.remove:
        subprocess.run(['systemctl', '--user', 'disable', '--now', UNIT], check=True)
        if target.exists() and target.read_bytes() == (BASE / UNIT).read_bytes():
            target.unlink()
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
        print('Dashboard companion removed. Hands service selection was not changed.')
        return
    if BASE != Path.home() / 'devkit-game/FrameHandsDashboard-v1':
        raise RuntimeError('Unexpected installation directory')
    validate_build()
    data = (BASE / UNIT).read_bytes()
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError('Another dashboard service definition exists; preserve and inspect it first')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
                for name in ('ui.js', 'service.py', 'cdp.py', UNIT, 'install.py')}
    (BASE / 'installed-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', UNIT], check=True)
    print('Native dashboard companion enabled. Tracking starts only on request.')


if __name__ == '__main__':
    main()
