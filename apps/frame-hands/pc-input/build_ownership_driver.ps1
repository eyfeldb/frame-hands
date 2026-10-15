param([switch]$ExplicitMode)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$workspace=(Resolve-Path (Join-Path $root '../../..')).Path
$zig=Join-Path $workspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/local-cache'
$build=Join-Path $root $(if ($ExplicitMode) {'build/input-mode-driver-v1'} else {'build/ownership-driver-v1'})
New-Item -ItemType Directory -Force $build | Out-Null
$flags=@('-std=c++17','-O2','-Wall','-Wextra','-Werror','-isystem',(Join-Path $root '../system-input/vendor'))
$define=@('-DFRAME_HANDS_DRIVER_ROLE_GRACE=1')
if ($ExplicitMode) {$define+='-DFRAME_HANDS_DRIVER_INPUT_MODE=1'}
$deploy=Get-Content (Join-Path $root 'deployment.json') -Raw | ConvertFrom-Json
$installed=Join-Path $deploy.bundle 'bin/win64/driver_fh_skeletal.dll'
$before=(Get-FileHash $installed -Algorithm SHA256).Hash
$dll=Join-Path $build 'driver_fh_skeletal.dll'
& $zig c++ @flags -DFRAME_HANDS_DRIVER_STABILITY=1 @define -shared (Join-Path $root 'driver.cpp') -o $dll
if ($LASTEXITCODE -ne 0) {throw 'Candidate DLL build failed'}
foreach ($test in @('test_stability_dll','test_driver_stability','test_driver_ownership')) {
    & $zig c++ @flags @define (Join-Path $root ($test+'.cpp')) -o (Join-Path $build ($test+'.exe'))
    if ($LASTEXITCODE -ne 0) {throw "$test build failed"}
}
& (Join-Path $build 'test_stability_dll.exe') $dll
if ($LASTEXITCODE -ne 0) {throw 'Actual DLL tests failed'}
if ($ExplicitMode) {
    & $zig c++ @flags (Join-Path $root 'test_driver_input_mode.cpp') -o (Join-Path $build 'test_driver_input_mode.exe')
    if ($LASTEXITCODE -ne 0) {throw 'Input mode test build failed'}
    & (Join-Path $build 'test_driver_input_mode.exe')
    if ($LASTEXITCODE -ne 0) {throw 'Input mode test failed'}
}
$traces=@('20261004T161941Z','20261004T164332Z') | ForEach-Object {Join-Path $root "evidence/local-viewer-$_/sampler-trace.bin"}
& (Join-Path $build 'test_driver_stability.exe') @traces
if ($LASTEXITCODE -ne 0) {throw 'Accepted pose regression failed'}
& $zig c++ @flags (Join-Path $root 'test_driver_ownership.cpp') -o (Join-Path $build 'baseline.exe')
if ($LASTEXITCODE -ne 0) {throw 'Baseline build failed'}
$trace=Join-Path $root 'evidence/local-maestro-20261004T233925Z/sampler-trace.bin'
$baseline=& (Join-Path $build 'baseline.exe') $trace
if ($LASTEXITCODE -ne 0) {throw 'Baseline replay failed'}
$candidate=& (Join-Path $build 'test_driver_ownership.exe') $trace
if ($LASTEXITCODE -ne 0) {throw 'Candidate replay failed'}
$baseline | Set-Content (Join-Path $build 'baseline.txt')
$candidate | Set-Content (Join-Path $build 'candidate.txt')
$a=[regex]::Match(($baseline -join "`n"),'valid_pose_hash=(\d+)').Groups[1].Value
$b=[regex]::Match(($candidate -join "`n"),'valid_pose_hash=(\d+)').Groups[1].Value
if (!$a -or $a -ne $b) {throw 'Candidate changed valid pose output'}
if ((Get-FileHash $installed -Algorithm SHA256).Hash -ne $before) {throw 'Installed driver changed'}
$sources=[ordered]@{}
foreach ($name in @('driver.cpp','stability.h','skeleton.h','protocol.h','math.h','reference.h','input_mode.h','select_input_mode.py','test_driver_input_mode.cpp','test_driver.cpp','test_driver_stability.cpp','test_stability_dll.cpp','test_driver_ownership.cpp','build_ownership_driver.ps1')) {
    $sources[$name]=(Get-FileHash (Join-Path $root $name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$report=[ordered]@{candidate='ownership grace v1';defines=@('FRAME_HANDS_DRIVER_STABILITY=1','FRAME_HANDS_DRIVER_ROLE_GRACE=1');role_grace_ms=750;pose_hold_ms=80;skeleton_only=$true;binary_sha256=(Get-FileHash $dll -Algorithm SHA256).Hash.ToLowerInvariant();preserved_installed_driver_sha256=$before.ToLowerInvariant();sources=$sources;live_validated=$false;registered=$false;baseline=$baseline;candidate_replay=$candidate;scope='Counterfactual source replay; physical-controller activity and game selection are not replayed'}
if ($ExplicitMode) {$report.candidate='explicit input mode v1';$report.defines+='FRAME_HANDS_DRIVER_INPUT_MODE=1';$report.input_mode_supported=$true}
$report | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $build 'manifest.json')
$report | ConvertTo-Json -Depth 6
