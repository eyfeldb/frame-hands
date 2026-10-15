#!/bin/bash
set -eu
cd "$(dirname "$0")"
mkdir -p driver/fh_skeletal/bin/linuxarm64
flags="-std=c++17 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation"
g++ $flags -isystem vendor pc-input/test_model_driver.cpp -pthread -o test-model
./test-model
g++ $flags -isystem vendor -shared -fPIC -DFRAME_HANDS_DRIVER_MODEL -DFRAME_HANDS_DRIVER_STABILITY -DFRAME_HANDS_DRIVER_ROLE_GRACE -DFRAME_HANDS_DRIVER_INPUT_MODE pc-input/driver.cpp -pthread -o driver/fh_skeletal/bin/linuxarm64/driver_fh_skeletal.so
g++ $flags -isystem vendor pose_service.cpp -ldl -o pose-service
./pose-service --self-test
g++ -std=c++17 -O2 probe_capabilities.cpp -ldl -o probe-capabilities
python3 build_runtime_tracker.py
g++ -std=c++17 -O2 -Itracker-runtime-source test_seed_rejection.cpp -o test-seed-rejection
./test-seed-rejection tracker-runtime-source/profiles/side0.fhrig tracker-runtime-source/profiles/side1.fhrig
g++ -std=c++17 -O2 -DFRAME_HANDS_WITH_OPENVR -Itracker-runtime-source test_trace_startup.cpp tracker-runtime-source/track/calib.cpp -ljsoncpp -ldl -o test-trace-startup
g++ -std=c++17 -O2 -isystem vendor notify.cpp -ldl -o notify
g++ $flags test_controller_takeover.cpp -o test-controller-takeover
./test-controller-takeover
g++ $flags -isystem vendor controller_watch.cpp -ldl -pthread -o controller-watch
g++ $flags -isystem vendor menu.cpp -ldl -o menu
python3 install.py --prepare
