#!/usr/bin/env bash
# Restore only the previously approved exact-v3 reader's access; never start capture.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Local sudo authentication required.' >&2; exit 1; }
exec 8>/home/steamos/devkit-game/FrameEyes/workflow.lock
exec 9>/home/steamos/devkit-game/FrameHands/trial.lock
flock -n 8
flock -n 9
reader=/var/lib/frame-hands/frame-camera-reader-exact-v3
[[ ! -L "$reader" && $(stat -c '%u:%g:%a' "$reader") == "0:$(id -g steamos):750" ]]
[[ $(sha256sum "$reader" | cut -d' ' -f1) == cd46259265d7e1df542ea78259c02ac4f42846263127f4606fa19b9725318990 ]]
setcap cap_sys_ptrace,cap_perfmon,cap_dac_read_search+ep "$reader"
getcap "$reader"
echo 'Camera access restored. Tracking has not been started.'
