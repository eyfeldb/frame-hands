# Optional PC-only research environment. Never modifies the CPU environment or Frame.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$deps = Join-Path $root '.local/hand-alternatives'
$cpuPython = Join-Path $deps 'venv/Scripts/python.exe'
$cudaRoot = Join-Path $deps 'venv-cuda'
if (!(Test-Path -LiteralPath $cpuPython)) { throw 'Run the standard offline setup first' }
if (!(Test-Path -LiteralPath $cudaRoot)) {
    & $cpuPython -m venv $cudaRoot
    if ($LASTEXITCODE) { throw 'CUDA research environment creation failed' }
}
$cudaPython = Join-Path $cudaRoot 'Scripts/python.exe'
& $cudaPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE) { throw 'CUDA research dependencies failed' }
# Windows CPython 3.12 wheel verified 2026-10-04; dependency pins above are separate.
& $cudaPython -m pip install --no-deps 'https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl#sha256=0a09031e93632d14ef49553acd6211ece3ae9cdc0969a369b8c24e3cc4d15be9'
if ($LASTEXITCODE) { throw 'Pinned CUDA Torch installation failed' }
& $cudaPython -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()'
if ($LASTEXITCODE) { throw 'CUDA unavailable; CPU tracker remains unchanged' }
