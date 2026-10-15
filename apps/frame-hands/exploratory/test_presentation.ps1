param([Parameter(Mandatory=$true)][string]$Workspace)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$py=Join-Path $root '.local/hand-alternatives/venv/Scripts/python.exe'
$godot=Join-Path (Resolve-Path $Workspace).Path '.local/deps/godot-4.7.2/Godot_v4.7.2-stable_win64.exe'
& $py (Join-Path $PSScriptRoot 'prepare_motion_replay.py')
if($LASTEXITCODE -ne 0){throw 'Fixture preparation failed'}
$scene=Join-Path $PSScriptRoot 'mesh-viewer'
$derived=Join-Path $root '.local/hand-alternatives/motion-test'
$env:FRAME_UME_TEST=$derived
$env:FRAME_UME_RIG=Join-Path $derived 'rig.json'
foreach($case in @('check','observed','assumed-30hz')) {
 $script='res://test_motion.gd'
 if($case -ne 'check') {
  $script='res://replay_motion.gd'
  $env:FRAME_UME_REPLAY=Join-Path $derived ($case+'.json')
 }
 $p=Start-Process -FilePath $godot -ArgumentList @('--headless','--path',('"'+$scene+'"'),'--xr-mode','off','--script',$script) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $derived ($case+'.log')) -RedirectStandardError (Join-Path $derived ($case+'-error.log'))
 if(-not $p.WaitForExit(30000)){ $p.Kill();throw 'Offline scene check timed out' }
 $errorText=Get-Content (Join-Path $derived ($case+'-error.log')) -Raw
 if($p.ExitCode -ne 0 -or $errorText){throw ('Godot check failed: '+$errorText)}
 Get-Content (Join-Path $derived ($case+'.log'))
}
