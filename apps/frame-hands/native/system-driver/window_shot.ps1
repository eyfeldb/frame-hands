# Capture named top-level windows with PrintWindow (works regardless of DPI scaling or overlap).
# usage: .\window_shot.ps1 <out-dir> <titleRegex> [count=1] [intervalSeconds=2]
param([string]$Out, [string]$Title = 'VR View|Headset Window', [int]$Count = 1, [int]$Interval = 2)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public class WS {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
  [StructLayout(LayoutKind.Sequential)] public struct R { public int L, T, Ri, B; }
}
"@
New-Item -ItemType Directory -Force $Out | Out-Null
for ($i = 1; $i -le $Count; $i++) {
    foreach ($w in (Get-Process | Where-Object { $_.MainWindowTitle -match $Title })) {
        $r = New-Object WS+R; [WS]::GetWindowRect($w.MainWindowHandle, [ref]$r) | Out-Null
        $wd = $r.Ri - $r.L; $ht = $r.B - $r.T; if ($wd -lt 10 -or $ht -lt 10) { continue }
        $bmp = New-Object System.Drawing.Bitmap($wd, $ht); $g = [System.Drawing.Graphics]::FromImage($bmp)
        $hdc = $g.GetHdc(); [WS]::PrintWindow($w.MainWindowHandle, $hdc, 2) | Out-Null; $g.ReleaseHdc($hdc); $g.Dispose()
        $safe = ($w.MainWindowTitle -replace '[^A-Za-z0-9]', '_'); $name = "$Out\$safe-$(Get-Date -Format 'HHmmss').png"; $bmp.Save($name); $bmp.Dispose(); $name
    }
    if ($i -lt $Count) { Start-Sleep -Seconds $Interval }
}
