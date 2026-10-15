param([string]$Zig)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$workspace=(Resolve-Path (Join-Path $root '../../..')).Path
if (!$Zig) {$Zig=Join-Path $workspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'}
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/local-cache'
$build=Join-Path $root 'build'
New-Item -ItemType Directory -Path $build -Force | Out-Null
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror (Join-Path $root 'test_stability.cpp') -o (Join-Path $build 'test-stability.exe')
if ($LASTEXITCODE -ne 0) {throw 'Stability test build failed'}
& (Join-Path $build 'test-stability.exe')
if ($LASTEXITCODE -ne 0) {throw 'Stability tests failed'}
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem (Join-Path $root '../system-input/vendor') (Join-Path $root 'sample_viewer.cpp') -o (Join-Path $build 'sample-viewer.exe')
if ($LASTEXITCODE -ne 0) {throw 'Diagnostic sampler build failed'}
& $Zig c++ -std=c++17 -O2 -Wall -Wextra -Werror (Join-Path $root 'replay_trace.cpp') -o (Join-Path $build 'replay-trace.exe')
if ($LASTEXITCODE -ne 0) {throw 'Offline trace replay build failed'}
# Intentionally does not compile, register or replace a SteamVR driver.
