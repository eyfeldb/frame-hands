# Sweep the PC hand driver's pose.txt knob (device-pose yaw/pitch/roll in palm axes + finger model) while an
# injected hand is held, screenshotting Maestro after each setting. Output: <out>\<yaw>_<pitch>_<roll>_h<hinge>-<HHMMSS>.png
# usage: .\pose_sweep.ps1 <bundle bin\win64 dir> <out-dir> "<yaw pitch roll hinge>;<...>" [settleSeconds=2.5]
param([string]$Bin, [string]$Out, [string]$Settings, [double]$Settle = 2.5)
$Root = Split-Path (Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent) -Parent
$Shot = Join-Path $PSScriptRoot 'window_shot.ps1'
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($s in ($Settings -split ';')) {
    $s = $s.Trim(); if (-not $s) { continue }
    [IO.File]::WriteAllText((Join-Path $Bin 'pose.txt'), "$s`n")
    Start-Sleep -Seconds $Settle
    $tmp = Join-Path $Out 'tmp'; New-Item -ItemType Directory -Force $tmp | Out-Null
    $file = & $Shot $tmp 'Maestro' 1 1 | Select-Object -Last 1
    if ($file -and (Test-Path $file)) {
        $name = ($s -replace '\s+', '_') -replace '^(\S+)_(\S+)_(\S+)_(\S+)$', '$1_$2_$3_h$4'
        $dest = Join-Path $Out ("$name-" + (Get-Date -Format 'HHmmss') + '.png')
        Move-Item $file $dest -Force; "$s -> $(Split-Path $dest -Leaf)"
    } else { "$s -> no Maestro window" }
}
