param([Parameter(Mandatory=$true)][string]$Workspace)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$shared=(Resolve-Path $Workspace).Path
$zig=Join-Path $shared '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $root '.local/hand-alternatives/zig-global'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $root '.local/hand-alternatives/zig-local'
$out=Join-Path $root '.local/hand-alternatives/pose-service.exe'
& $zig c++ -std=c++17 -O2 ('-I'+(Join-Path $PSScriptRoot '../system-input/vendor')) (Join-Path $PSScriptRoot 'pose_service.cpp') -o $out
if($LASTEXITCODE -ne 0){throw 'Pose helper build failed'}
& $zig c++ -std=c++17 -O2 ('-I'+(Join-Path $PSScriptRoot '../system-input/vendor')) (Join-Path $PSScriptRoot 'readiness_probe.cpp') -o (Join-Path $root '.local/hand-alternatives/readiness-probe.exe')
if($LASTEXITCODE -ne 0){throw 'Readiness helper build failed'}
& (Join-Path $root '.local/hand-alternatives/venv/Scripts/python.exe') (Join-Path $PSScriptRoot 'test_adapter.py') -v
if($LASTEXITCODE -ne 0){throw 'Adapter tests failed'}
