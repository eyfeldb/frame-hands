param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$deps = Join-Path $root '.local/hand-alternatives'
$repo = Join-Path $deps 'UmeTrack'
$venv = Join-Path $deps 'venv'
New-Item -ItemType Directory -Force $deps | Out-Null
if (!(Test-Path -LiteralPath $repo)) {
    & git clone https://github.com/facebookresearch/UmeTrack.git $repo
    if ($LASTEXITCODE) { throw 'UmeTrack clone failed' }
    & git -C $repo checkout --detach e263ec0844abf73457468a050a75853444d33d69
    if ($LASTEXITCODE) { throw 'Cannot check out pinned revision' }
}
if (!(Test-Path -LiteralPath (Join-Path $venv 'Scripts/python.exe'))) {
    & $Python -m venv $venv
    if ($LASTEXITCODE) { throw 'Environment creation failed' }
}
$researchPython = Join-Path $venv 'Scripts/python.exe'
& $researchPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE) { throw 'Python dependencies failed' }
& $researchPython -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE) { throw 'Torch CPU installation failed' }
& $researchPython (Join-Path $PSScriptRoot 'prepare_umetrack.py') $repo
if ($LASTEXITCODE) { throw 'Pinned source adaptation failed' }
Write-Output 'Offline research environment ready. No device or driver changes were made.'
