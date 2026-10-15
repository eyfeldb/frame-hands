$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$workspace=(Resolve-Path (Join-Path $root '../../..')).Path
$zig=Join-Path $workspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/local-cache'
$build=Join-Path $root 'build/presentation-driver-v1'
New-Item -ItemType Directory -Force $build | Out-Null
$flags=@('-std=c++17','-O2','-Wall','-Wextra','-Werror','-isystem',(Join-Path $root '../system-input/vendor'))
$defines=@('FRAME_HANDS_DRIVER_STABILITY=1','FRAME_HANDS_DRIVER_ROLE_GRACE=1','FRAME_HANDS_DRIVER_INPUT_MODE=1','FRAME_HANDS_DRIVER_PRESENTATION=1')
$defineFlags=$defines | ForEach-Object {'-D'+$_}
$deploy=Get-Content (Join-Path $root 'deployment.json') -Raw | ConvertFrom-Json
$installed=Join-Path $deploy.bundle 'bin/win64/driver_fh_skeletal.dll'
$before=(Get-FileHash $installed -Algorithm SHA256).Hash
$dll=Join-Path $build 'driver_fh_skeletal.dll'
& $zig c++ @flags @defineFlags -shared (Join-Path $root 'driver.cpp') -o $dll
if ($LASTEXITCODE -ne 0) {throw 'Candidate DLL build failed'}
$results=[ordered]@{}
foreach ($test in @('test_presentation','test_driver_presentation','test_driver_input_mode','test_stability_dll','test_driver_stability')) {
    $extra=@()
    if ($test -eq 'test_stability_dll') {$extra=$defineFlags}
    if ($test -eq 'test_driver_input_mode') {$extra=@('-DFRAME_HANDS_DRIVER_PRESENTATION=1')}
    & $zig c++ @flags @extra (Join-Path $root ($test+'.cpp')) -o (Join-Path $build ($test+'.exe'))
    if ($LASTEXITCODE -ne 0) {throw "$test build failed"}
    $arguments=@()
    if ($test -eq 'test_stability_dll') {$arguments=@($dll)}
    if ($test -eq 'test_driver_presentation') {$arguments=@('local-maestro-20261005T001203Z','local-maestro-20261004T233925Z','local-viewer-20261004T173637Z') | ForEach-Object {Join-Path $root "evidence/$_/sampler-trace.bin"}}
    if ($test -eq 'test_driver_stability') {$arguments=@('20261004T161941Z','20261004T164332Z') | ForEach-Object {Join-Path $root "evidence/local-viewer-$_/sampler-trace.bin"}}
    $result=& (Join-Path $build ($test+'.exe')) @arguments
    if ($LASTEXITCODE -ne 0) {throw "$test failed"}
    $results[$test]=$result
    $result | Set-Content (Join-Path $build ($test+'.txt'))
    $result
}
if ((Get-FileHash $installed -Algorithm SHA256).Hash -ne $before) {throw 'Installed driver changed'}
$sources=[ordered]@{}
foreach ($name in @('driver.cpp','stability.h','skeleton.h','retarget_chain.h','presentation.h','protocol.h','math.h','reference.h','input_mode.h','select_input_mode.py','test_driver_input_mode.cpp','test_driver.cpp','test_driver_stability.cpp','test_stability_dll.cpp','test_presentation.cpp','test_driver_presentation.cpp','build_presentation_driver.ps1')) {
    $sources[$name]=(Get-FileHash (Join-Path $root $name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$report=[ordered]@{candidate='presentation and explicit input mode v1';defines=$defines;pose_hold_ms=80;presentation_response_ms=20;skeleton_only=$true;input_mode_supported=$true;binary_sha256=(Get-FileHash $dll -Algorithm SHA256).Hash.ToLowerInvariant();preserved_installed_driver_sha256=$before.ToLowerInvariant();sources=$sources;tests=$results;live_validated=$false;registered=$false;limitations=@('Additional presentation latency; no extrapolation','No change in source dropout or pose availability','Game controller-popup behavior still requires live validation','Synthetic/replay evidence is not physical accuracy or skinned-game acceptance')}
$report | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $build 'manifest.json')
Write-Output ('candidate_sha256='+$report.binary_sha256)
