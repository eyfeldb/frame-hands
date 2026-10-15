#!/usr/bin/env bash
# Dependencies and upstream sources must already be staged at these pinned paths.
# This only builds under FrameHands; never disables SteamOS read-only protection.
set -euo pipefail
base=/home/steamos/devkit-game/FrameHands
python3 "$base/apply_frame_patch.py" "$base/upstream"
cmake -S "$base/deps/ncnn-20260526" -B "$base/deps/ncnn-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$base/deps/ncnn-install" \
  -DCMAKE_INSTALL_LIBDIR=lib -DNCNN_VULKAN=OFF -DNCNN_OPENMP=ON \
  -DNCNN_INT8=ON -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF
nice -n 15 cmake --build "$base/deps/ncnn-build" --target install -j2
cd "$base/upstream"
nice -n 15 make -j2 NCNN="$base/deps/ncnn-install" all tools
cc -O2 -Wall -Wextra -Wno-unused-parameter -I. "$base/probe/test_camera_sampler.c" \
  camd/tp.c camd/xrcams.c -lm -o "$base/probe/test-camera-sampler"
"$base/probe/test-camera-sampler"
c++ -O2 -Wno-psabi -std=c++17 -fopenmp -Itrack -I"$base/deps/ncnn-install/include/ncnn" \
  "$base/probe/test_publication.cpp" track/io.cpp -o "$base/probe/test-publication"
"$base/probe/test-publication"
c++ -O2 -std=c++17 -fopenmp -Itrack -I"$base/deps/ncnn-install/include/ncnn" \
  "$base/probe/inference_bench.cpp" track/nets.cpp \
  "$base/deps/ncnn-install/lib/libncnn.a" -lpthread -o "$base/probe/inference-bench"
c++ -O2 -std=c++17 "$base/probe/probe.cpp" -I"$base/probe" \
  -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 \
  -lopenvr_api -lopenxr_loader -lpthread -ldl -o "$base/probe/capability-probe"
c++ -O2 -std=c++17 "$base/probe/hand_source_probe.cpp" \
  -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 \
  -lopenxr_loader -lpthread -ldl -o "$base/probe/hand-source-probe"
