#!/usr/bin/env bash
# Exact-queue candidate only; preserve both predecessors, start nothing.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run locally on Frame with sudo.' >&2; exit 1; }
expected=61b96ff275b2e218d041483a63fd5a3005f44ce11d742f1c87fcbd355711396e
accepted=ab07c5a1a5f00cfe3be8701b6a848242c8775880f47606fb18cb114c560fb738
source_bin=/home/steamos/devkit-game/FrameHands/experiments/camera-direct-v2/frame-camera-reader-direct-v2
destination=/var/lib/frame-hands/frame-camera-reader-direct-v2
[[ $(sha256sum /var/lib/frame-hands/frame-camera-reader | cut -d' ' -f1) == "$accepted" ]]
[[ $(sha256sum "$source_bin" | cut -d' ' -f1) == "$expected" ]]
if [[ -e "$destination" ]]; then
  [[ ! -L "$destination" && $(stat -c '%u:%g:%a' "$destination") == "0:$(id -g steamos):750" ]]
  [[ $(sha256sum "$destination" | cut -d' ' -f1) == "$expected" ]]
  getcap "$destination"
  exit 0
fi
candidate=$(mktemp /var/lib/frame-hands/.direct-candidate.XXXXXX)
trap 'rm -f -- "$candidate"' EXIT
install -o root -g steamos -m 0750 "$source_bin" "$candidate"
[[ $(sha256sum "$candidate" | cut -d' ' -f1) == "$expected" ]]
setcap cap_sys_ptrace,cap_perfmon,cap_dac_read_search+ep "$candidate"
mv -n -- "$candidate" "$destination"
[[ ! -L "$destination" && $(sha256sum "$destination" | cut -d' ' -f1) == "$expected" ]]
getcap "$destination"
echo 'Exact-queue candidate ready. Accepted reader unchanged; no services started.'
