#!/usr/bin/env bash
# Reuse bindings generated for the exact Godot 4.7.2 binary on this Frame.
# Supply another matching binding build explicitly when the engine changes.
set -euo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
bindings=${1:-/home/steamos/devkit-game/FramePoseSyncBuild-5e5vky2r}
test -f "$bindings/build/bin/libgodot-cpp.linux.template_release.arm64.a"
c++ -O2 -std=c++17 -I"$base/probe" "$base/scene_check.cpp" \
  -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 \
  -lopenvr_api -o "$base/probe/scene-check"
c++ -O2 -std=c++17 -fPIC -shared \
  -I"$bindings/godot-cpp/include" -I"$bindings/godot-cpp/gdextension" \
  -I"$bindings/build/godot-cpp/gen/include" -I"$base/upstream/include" \
  "$base/viewer/frame_hands_native.cpp" \
  "$bindings/build/bin/libgodot-cpp.linux.template_release.arm64.a" \
  -lpthread -ldl -o "$base/viewer/libframe_hands.so"
