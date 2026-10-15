#!/usr/bin/env bash
# Install only the audited, hash-pinned camera reader. No service is enabled.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run this script with sudo on Frame.' >&2; exit 1; }
source_bin=/home/steamos/devkit-game/FrameHands/upstream/build/ft-camd
expected=ab07c5a1a5f00cfe3be8701b6a848242c8775880f47606fb18cb114c560fb738
actual=$(sha256sum "$source_bin" | cut -d' ' -f1)
[[ $actual == "$expected" ]] || { echo 'Camera reader changed; refusing installation.' >&2; exit 1; }
install -d -o root -g root -m 0755 /var/lib/frame-hands
destination=/var/lib/frame-hands/frame-camera-reader
if [[ -e "$destination" ]]; then
  installed=$(sha256sum "$destination" | cut -d' ' -f1)
  original=6c2d02a6bb287796694b16c4f1cfbbc8213832d2dee3f311b3618d39cd16fb35
  nv12=07c20ae00a8da4a8eed95dcbd6196721de2b8442880fcec5ce5585b19802f45e
  [[ $installed == "$original" || $installed == "$nv12" || $installed == "$expected" ]] || {
    echo 'Installed reader has an unexpected hash; refusing replacement.' >&2; exit 1;
  }
  if [[ $installed != "$expected" ]]; then
    # Preserve the original bytes without granting capabilities to the backup.
    install -o root -g steamos -m 0750 "$destination" "$destination.previous-$installed"
  fi
fi
# Complete and verify the new file before replacing the installed reader.
candidate=$(mktemp /var/lib/frame-hands/.reader.XXXXXX)
trap 'rm -f -- "$candidate"' EXIT
install -o root -g steamos -m 0750 "$source_bin" "$candidate"
[[ $(sha256sum "$candidate" | cut -d' ' -f1) == "$expected" ]]
setcap cap_sys_ptrace,cap_perfmon,cap_dac_read_search+ep "$candidate"
mv -f -- "$candidate" "$destination"
getcap "$destination"
echo 'Camera reader ready. No services started. Remove its access with: sudo setcap -r /var/lib/frame-hands/frame-camera-reader'
