#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build driver/fh_pointer/bin/linuxarm64
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread test_policy.cpp -o build/test-policy
build/test-policy
c++ -std=c++17 -O2 -Wall -Wextra -Werror test_gaze_math.cpp -o build/test-gaze-math
build/test-gaze-math
python3 test_calibration.py
c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem vendor -pthread -fPIC -fvisibility=hidden -shared driver.cpp -o driver/fh_pointer/bin/linuxarm64/driver_fh_pointer.so
c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem vendor -pthread -Wl,--export-dynamic test_driver.cpp -ldl -o build/test-driver
build/test-driver "$PWD/driver/fh_pointer/bin/linuxarm64/driver_fh_pointer.so"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem vendor -pthread bridge.cpp -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lopenvr_api -o build/fh-input
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -Ivendor gaze_probe.cpp -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lopenvr_api -o build/gaze-probe
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -isystem vendor status.cpp -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lopenvr_api -o build/fh-status
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -isystem vendor surface_probe.cpp -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lopenvr_api -o build/surface-probe
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -isystem vendor gaze_calibrator.cpp -L/opt/steamvr/bin/linuxarm64 -Wl,-rpath,/opt/steamvr/bin/linuxarm64 -lopenvr_api -o build/gaze-calibrator
sha256sum driver/fh_pointer/bin/linuxarm64/driver_fh_pointer.so build/fh-input build/gaze-probe
