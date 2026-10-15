param([string]$Output='build/stability-driver-v1')
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$workspace=(Resolve-Path (Join-Path $root '../../..')).Path
$zig=Join-Path $workspace '.local/pc-hands-toolchain/zig-x86_64-windows-0.15.2/zig.exe'
$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/cache'
$env:ZIG_LOCAL_CACHE_DIR=Join-Path $workspace '.local/pc-hands-toolchain/local-cache'
$build=[IO.Path]::GetFullPath((Join-Path $root $Output))
$allowed=[IO.Path]::GetFullPath((Join-Path $root 'build'))+[IO.Path]::DirectorySeparatorChar
if (!$build.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) {throw 'Candidate output must stay in the isolated build directory'}
New-Item -ItemType Directory -Force $build | Out-Null
$include=Join-Path $root '../system-input/vendor'
$flags=@('-std=c++17','-O2','-Wall','-Wextra','-Werror','-isystem',$include)
$deploy=Get-Content (Join-Path $root 'deployment.json') -Raw | ConvertFrom-Json
$installed=Join-Path $deploy.bundle 'bin/win64/driver_fh_skeletal.dll'
$before=(Get-FileHash $installed -Algorithm SHA256).Hash
$dll=Join-Path $build 'driver_fh_skeletal.dll'
& $zig c++ @flags -DFRAME_HANDS_DRIVER_STABILITY=1 -shared (Join-Path $root 'driver.cpp') -o $dll
if ($LASTEXITCODE -ne 0) {throw 'Candidate DLL build failed'}
foreach ($test in @('test_driver_stability','test_stability_dll')) {
    & $zig c++ @flags (Join-Path $root ($test+'.cpp')) -o (Join-Path $build ($test+'.exe'))
    if ($LASTEXITCODE -ne 0) {throw "$test build failed"}
}
& (Join-Path $build 'test_stability_dll.exe') $dll
if ($LASTEXITCODE -ne 0) {throw 'Actual candidate DLL test failed'}
$traces=@('20261004T161941Z','20261004T164332Z') | ForEach-Object {Join-Path $root "evidence/local-viewer-$_/sampler-trace.bin"}
& (Join-Path $build 'test_driver_stability.exe') @traces
if ($LASTEXITCODE -ne 0) {throw 'Candidate driver replay failed'}
if ((Get-FileHash $installed -Algorithm SHA256).Hash -ne $before) {throw 'Installed driver changed'}
$sources=[ordered]@{}
foreach ($name in @('driver.cpp','stability.h','skeleton.h','protocol.h','math.h','reference.h','test_driver.cpp','test_driver_stability.cpp','test_stability_dll.cpp','build_stability_driver.ps1')) {
    $path=Join-Path $root $name
    if (Test-Path $path) {$sources[$name]=(Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$report=[ordered]@{candidate='observation-time stability driver v1';define='FRAME_HANDS_DRIVER_STABILITY=1';skeleton_only=$true;binary_sha256=(Get-FileHash $dll -Algorithm SHA256).Hash.ToLowerInvariant();preserved_installed_driver_sha256=$before.ToLowerInvariant();sources=$sources;live_validated=$false;registered=$false;replayed_sessions=@('20261004T161941Z','20261004T164332Z')}
$report | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $build 'manifest.json')
$report | ConvertTo-Json -Depth 6
