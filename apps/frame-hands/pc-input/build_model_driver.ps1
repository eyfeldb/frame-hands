param([string]$ToolchainWorkspace=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$zig=Join-Path $ToolchainWorkspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $ToolchainWorkspace '.local/pc-hands-toolchain/cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $ToolchainWorkspace '.local/pc-hands-toolchain/local-cache'
$build=Join-Path $root 'build/model-driver-v2'
New-Item -ItemType Directory -Path $build -Force | Out-Null
$flags=@('-std=c++17','-O2','-Wall','-Wextra','-Werror','-isystem',(Join-Path $root '../system-input/vendor'))
& $zig c++ @flags (Join-Path $root 'test_model_driver.cpp') -o (Join-Path $build 'test-model.exe')
if ($LASTEXITCODE -ne 0) {throw 'Model driver harness build failed'}
& (Join-Path $build 'test-model.exe')
if ($LASTEXITCODE -ne 0) {throw 'Model driver harness failed'}
& $zig c++ @flags -shared -DFRAME_HANDS_DRIVER_MODEL=1 -DFRAME_HANDS_DRIVER_STABILITY=1 -DFRAME_HANDS_DRIVER_ROLE_GRACE=1 -DFRAME_HANDS_DRIVER_INPUT_MODE=1 (Join-Path $root 'driver.cpp') -o (Join-Path $build 'driver_fh_skeletal.dll')
if ($LASTEXITCODE -ne 0) {throw 'Model driver build failed'}
