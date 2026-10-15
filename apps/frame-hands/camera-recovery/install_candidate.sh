#!/usr/bin/env bash
# Install this reviewed candidate alongside the accepted reader; start nothing.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run locally on Frame with sudo.' >&2; exit 1; }
expected=577be3a745a4bca2a08b961112dad26a96d9d2c4435e2ac86eb66f9ab0df353f
accepted=ab07c5a1a5f00cfe3be8701b6a848242c8775880f47606fb18cb114c560fb738
source_bin=/home/steamos/devkit-game/FrameHands/experiments/camera-buffer-v1/frame-camera-reader-buffer-v1
destination=/var/lib/frame-hands/frame-camera-reader-buffer-v1
[[ $(sha256sum /var/lib/frame-hands/frame-camera-reader | cut -d' ' -f1) == "$accepted" ]]
[[ $(sha256sum "$source_bin" | cut -d' ' -f1) == "$expected" ]]
if [[ -e "$destination" ]]; then
  [[ ! -L "$destination" && $(stat -c '%u:%g:%a' "$destination") == "0:$(id -g steamos):750" ]]
  [[ $(sha256sum "$destination" | cut -d' ' -f1) == "$expected" ]]
  getcap "$destination"
  exit 0
fi
candidate=$(mktemp /var/lib/frame-hands/.buffer-candidate.XXXXXX)
trap 'rm -f -- "$candidate"' EXIT
install -o root -g steamos -m 0750 "$source_bin" "$candidate"
[[ $(sha256sum "$candidate" | cut -d' ' -f1) == "$expected" ]]
setcap cap_sys_ptrace,cap_perfmon,cap_dac_read_search+ep "$candidate"
mv -n -- "$candidate" "$destination"
[[ ! -L "$destination" && $(sha256sum "$destination" | cut -d' ' -f1) == "$expected" ]]
getcap "$destination"
echo 'Candidate ready alongside the unchanged accepted reader. No services started.'
