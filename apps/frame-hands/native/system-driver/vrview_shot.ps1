# Screenshot the SteamVR "VR View" mirror window (what the HMD shows) every N seconds into a folder, for
# side-by-side comparison with the injector's truth renders. The window must be open (SteamVR status window ->
# Display VR View) and not minimised.
# usage: .\vrview_shot.ps1 <out-dir> [intervalSeconds=2] [count=30]
param([string]$Out, [int]$Interval = 2, [int]$Count = 30)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public class W { [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r); [StructLayout(LayoutKind.Sequential)] public struct R { public int L, T, Ri, B; } }
"@
New-Item -ItemType Directory -Force $Out | Out-Null
for ($i = 1; $i -le $Count; $i++) {
    $w = Get-Process | Where-Object { $_.MainWindowTitle -match 'VR View' } | Select-Object -First 1
    if (-not $w) { "no 'VR View' window (open it from the SteamVR status window: Display VR View)"; Start-Sleep -Seconds $Interval; continue }
    $r = New-Object W+R; [W]::GetWindowRect($w.MainWindowHandle, [ref]$r) | Out-Null
    $bmp = New-Object System.Drawing.Bitmap(($r.Ri - $r.L), ($r.B - $r.T)); $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size); $g.Dispose()
    $name = "$Out\vrview-$(Get-Date -Format 'HHmmss').png"; $bmp.Save($name); $bmp.Dispose(); $name
    Start-Sleep -Seconds $Interval
}
