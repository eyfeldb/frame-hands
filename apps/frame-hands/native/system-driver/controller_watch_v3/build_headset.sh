#!/bin/sh
# Build + install controller-watch v3 on the Frame (run on the headset from /tmp/cw3). Keeps a .v2 backup of the
# binary the service currently runs (FrameHandsDashboard-v1/controller-watch-src, per the systemd drop-in).
set -e
P=/home/steamos/devkit-game/FrameHandsNativeSystem-v1
T=/home/steamos/devkit-game/FrameHandsDashboard-v1/controller-watch-src
cd /tmp/cw3
cp -f $T/controller_takeover.h $T/input_mode.h .
g++ -std=c++17 -O2 -Wall -Wextra -Wno-misleading-indentation -isystem $P/vendor main.cpp -ldl -pthread -o controller-watch 2> build.err; echo build_rc=$?
head -c 1200 build.err
[ -x controller-watch ] || exit 2
[ -f $T/controller-watch.v2 ] || cp -p $T/controller-watch $T/controller-watch.v2
systemctl --user stop frame-hands-controller-watch.service 2>/dev/null || true
cp controller-watch $T/controller-watch && chmod 755 $T/controller-watch
cp takeover_actions.json takeover_frame_controller.json $T/
# drop-in: pass the action manifest as the third argument
cat > /home/steamos/.config/systemd/user/frame-hands-controller-watch.service.d/dashboard-dismissal.conf <<EOC
[Service]
ExecStart=
ExecStart=$T/controller-watch /opt/steamvr/bin/linuxarm64/libopenvr_api.so /run/user/1000/frame-hands-native/mode.bin $T/takeover_actions.json
EOC
systemctl --user daemon-reload
# SteamVR developer setting that enables overlay input overrides
python3 - <<'EOP'
import json, pathlib, shutil
p = pathlib.Path('/home/steamos/.config/openvr/config/steamvr.vrsettings')
bak = p.with_name('steamvr.vrsettings.before-gasp-20261012')
if not bak.exists(): shutil.copy2(p, bak)
d = json.loads(p.read_text()); d.setdefault('steamvr', {})['globalActionSetPriority'] = True
p.write_text(json.dumps(d, indent=3) + '\n'); print('globalActionSetPriority = true')
EOP
echo "native: $(systemctl --user is-active frame-hands-native.service)"
systemctl --user is-active frame-hands-native.service >/dev/null && { systemctl --user start frame-hands-controller-watch.service; sleep 2; systemctl --user is-active frame-hands-controller-watch.service; journalctl --user -u frame-hands-controller-watch --no-pager -n 5 -o cat; } || echo "watcher starts with the next Hands selection"
