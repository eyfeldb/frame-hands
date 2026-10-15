param([string]$Zig, [string]$SteamVR='C:\Program Files (x86)\Steam\steamapps\common\SteamVR')
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$workspace=(Resolve-Path (Join-Path $root '../../..')).Path
if (!$Zig) {$Zig=Join-Path $workspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'}
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/local-cache'
$build=Join-Path $root 'build'
$bin=Join-Path $root 'driver/fh_skeletal/bin/win64'
New-Item -ItemType Directory -Path $build,$bin -Force | Out-Null
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror (Join-Path $root 'test_core.cpp') -o (Join-Path $build 'test-core.exe')
if ($LASTEXITCODE -ne 0) {throw 'Core build failed'}
& (Join-Path $build 'test-core.exe')
if ($LASTEXITCODE -ne 0) {throw 'Core tests failed'}
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -isystem (Join-Path $root '../system-input/vendor') (Join-Path $root 'driver.cpp') -o (Join-Path $bin 'driver_fh_skeletal.dll')
if ($LASTEXITCODE -ne 0) {throw 'Driver build failed'}
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem (Join-Path $root '../system-input/vendor') (Join-Path $root 'test_driver.cpp') -o (Join-Path $build 'test-driver.exe')
if ($LASTEXITCODE -ne 0) {throw 'Driver harness build failed'}
& (Join-Path $build 'test-driver.exe') (Join-Path $bin 'driver_fh_skeletal.dll')
if ($LASTEXITCODE -ne 0) {throw 'Driver harness failed'}
Get-FileHash (Join-Path $bin 'driver_fh_skeletal.dll') -Algorithm SHA256
& $Zig c++ -std=c++17 -O2 -isystem (Join-Path $root '../system-input/vendor') (Join-Path $root 'probe_openvr.cpp') -o (Join-Path $build 'probe-openvr.exe')
if ($LASTEXITCODE -ne 0) {throw 'OpenVR probe build failed'}
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem (Join-Path $root '../system-input/vendor') (Join-Path $root 'sample_viewer.cpp') -o (Join-Path $build 'sample-viewer.exe')
if ($LASTEXITCODE -ne 0) {throw 'Skeleton viewer sampler build failed'}
$xrHeaders=Join-Path $workspace '.local/pc-hands-toolchain/include'
if (Test-Path (Join-Path $xrHeaders 'openxr/openxr.h')) {
    & $Zig c++ -std=c++17 -O2 -isystem $xrHeaders (Join-Path $root 'probe_openxr.cpp') -ld3d11 -ldxgi -o (Join-Path $build 'probe-openxr.exe')
    if ($LASTEXITCODE -ne 0) {throw 'OpenXR probe build failed'}
} else { Write-Warning 'OpenXR probe skipped: provide Khronos headers in .local/pc-hands-toolchain/include/openxr' }
