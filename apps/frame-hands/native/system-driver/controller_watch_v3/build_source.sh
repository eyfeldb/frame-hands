#!/bin/sh
# Compile only. No services, settings, camera access or installation.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
hands=$(CDPATH= cd -- "$here/../../.." && pwd)
out="$hands/../../build"
mkdir -p "$out"
c++ -std=c++17 -O2 -Wall -Wextra -Wno-misleading-indentation \
    -isystem "$hands/system-input/vendor" -I "$here/.." \
    "$here/main.cpp" -ldl -pthread -o "$out/controller-watch"
echo "Built controller-watch; nothing installed or started."
