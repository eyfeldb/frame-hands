param([Parameter(Mandatory=$true)][string]$Workspace)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$zig=Join-Path $Workspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'
$out=Join-Path $root '.local/hand-native'
New-Item -ItemType Directory -Force $out | Out-Null
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $out 'zig-global'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $out 'zig-local'
$source=Join-Path $PSScriptRoot 'pose_service.cpp'
$include=Join-Path $PSScriptRoot '../system-input/vendor'
& $zig c++ -target aarch64-linux-gnu -std=c++17 -O2 -Wall -Wextra -Werror -isystem $include $source -ldl -o (Join-Path $out 'pose-service-aarch64')
if($LASTEXITCODE -ne 0){throw 'ARM64 pose service build failed'}
& $zig c++ -std=c++17 -O2 -Wall -Wextra -Werror -isystem $include $source -o (Join-Path $out 'pose-clock-tests.exe')
if($LASTEXITCODE -ne 0){throw 'Host clock test build failed'}
& (Join-Path $out 'pose-clock-tests.exe') --self-test
if($LASTEXITCODE -ne 0){throw 'Pose clock/validity tests failed'}
