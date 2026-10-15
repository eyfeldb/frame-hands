# End-to-end capture of the Frame -> PC -> SteamVR hand chain for a Maestro trial.
# usage:  .\maestro_capture.ps1 start [seconds]   -> ensures the PC receiver runs, starts capture-session.exe,
#                                                    marks the SteamVR log offset, writes a session folder
#         .\maestro_capture.ps1 stop               -> waits for the capture, slices the SteamVR log, copies relay
#                                                    status, pulls the headset recorder chunks, runs the analysis
# Session folders: .local\hand-alternatives\results\maestro-<stamp>\
param([string]$Mode = 'start', [int]$Seconds = 120)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$NW = "$Root\.local\hand-alternatives\native-wireless"
$Py = "$Root\.local\hand-alternatives\venv-cuda\Scripts\python.exe"
$Adb = Join-Path $Root '.local/android-tools/platform-tools/adb.exe'
$Dll = 'C:\Program Files (x86)\Steam\steamapps\common\SteamVR\bin\win64\openvr_api.dll'
$VrLog = 'C:\Program Files (x86)\Steam\logs\vrserver.txt'
$Capture = "$Root\.local\pc-hands\build-20261010b\capture-session.exe"
$Bundle = (Get-Content "$Root\.local\hand-alternatives\steamvr-model\deployment.json" | ConvertFrom-Json).bundle
$Mailbox = "$Bundle\bin\win64\input.bin"
$State = "$Root\.local\hand-alternatives\results\maestro-current.json"

function Relay-Running { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'pc_relay\.py' } | Select-Object -First 1 }

if ($Mode -eq 'start') {
    $stamp = Get-Date -Format 'yyyyMMddTHHmmss'
    $dir = "$Root\.local\hand-alternatives\results\maestro-$stamp"
    New-Item -ItemType Directory -Force $dir | Out-Null
    # 1. receiver: SteamVR normally launches it; start it ourselves when it is not running
    $relay = Relay-Running
    if (-not $relay) {
        # receiver-host.exe (SteamVR's overlay companion) exits when it cannot attach; run the relay directly instead
        Remove-Item "$NW\receiver.stop" -ErrorAction SilentlyContinue
        Start-Process -FilePath $Py -ArgumentList @("`"$Root\apps\frame-hands\native\system-driver\pc_relay.py`"", "--connection", "`"$NW\connection.json`"", "--stop-file", "`"$NW\receiver.stop`"", "--watch") -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput "$dir\relay-stdout.txt" -RedirectStandardError "$dir\relay-stderr.txt"
        Start-Sleep -Seconds 4
        $relay = Relay-Running
    }
    "receiver: " + $(if ($relay) { "running (pid $($relay.ProcessId))" } else { "NOT running" }) | Tee-Object -FilePath "$dir\notes.txt" -Append
    # 2. SteamVR log offset
    $offset = (Get-Item $VrLog).Length
    # 3. headset: make sure the continuous recorder is alive, note the time
    $headset = & $Adb shell 'sh /tmp/j.sh 2>/dev/null | tail -1; date +%H:%M:%S; cat /run/user/1000/frame-hands-native/status.json'
    "headset: $headset" | Tee-Object -FilePath "$dir\notes.txt" -Append
    # 4. PC capture of SteamVR devices + PC mailbox
    $proc = Start-Process -FilePath $Capture -ArgumentList @("`"$Dll`"", "`"$NW\probe-actions.json`"", "`"$Mailbox`"", "`"$dir\capture.csv`"", "$Seconds") -WorkingDirectory $dir -WindowStyle Hidden -PassThru
    @{ dir = $dir; offset = $offset; started = (Get-Date).ToString('o'); seconds = $Seconds; capture_pid = $proc.Id } | ConvertTo-Json | Set-Content $State
    "capture started for $Seconds s -> $dir"
    exit 0
}

if ($Mode -eq 'stop') {
    $s = Get-Content $State | ConvertFrom-Json
    $dir = $s.dir
    $p = Get-Process -Id $s.capture_pid -ErrorAction SilentlyContinue
    if ($p) { "waiting for capture-session to finish (started $($s.started), $($s.seconds) s)"; $p.WaitForExit() }
    # SteamVR log slice since start
    $fs = [IO.File]::Open($VrLog, 'Open', 'Read', 'ReadWrite'); $fs.Seek($s.offset, 'Begin') | Out-Null
    $sr = New-Object IO.StreamReader($fs); $sr.ReadToEnd() | Set-Content "$dir\vrserver-slice.txt"; $sr.Close()
    Copy-Item "$Root\.local\hand-alternatives\native-pc-relay\wireless-status.json" "$dir\relay-status.json" -ErrorAction SilentlyContinue
    Copy-Item "$NW\receiver-lifecycle.log" "$dir\receiver-lifecycle.log" -ErrorAction SilentlyContinue
    # headset recorder chunks written since the start
    $started = [DateTime]::Parse($s.started)
    $chunks = (& $Adb shell 'ls -1 /tmp/scn-*.jsonl') -split "`n" | Where-Object { $_ -match 'scn-(\d{6})' } | ForEach-Object { $_.Trim() }
    foreach ($c in $chunks) {
        if ($c -match 'scn-(\d{2})(\d{2})(\d{2})') { $t = Get-Date -Hour $Matches[1] -Minute $Matches[2] -Second $Matches[3]; if ($t -ge $started.AddMinutes(-2)) { & $Adb pull $c "$dir\$(Split-Path $c -Leaf)" | Out-Null } }
    }
    & $Adb shell 'journalctl --user -u frame-hands-native.service --no-pager -n 300 -o short-iso' | Set-Content "$dir\headset-journal.txt"
    "collected -> $dir"; Get-ChildItem $dir | ForEach-Object { "  $($_.Name)  $([math]::Round($_.Length/1KB)) KB" }
    & $Py "$Root\apps\frame-hands\pc-input\analyze_capture.py" "$dir\capture.csv"
    & $Py "$Root\apps\frame-hands\native\system-driver\maestro_chain_analysis.py" $dir
    exit 0
}
