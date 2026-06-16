param(
    [string]$EnvName = "voice-clone-dot-tts",
    [switch]$UseCuda
)

$ErrorActionPreference = "Stop"

function Find-Conda {
    $cmd = Get-Command micromamba -ErrorAction SilentlyContinue
    if ($cmd) { return "micromamba" }
    $cmd = Get-Command mamba -ErrorAction SilentlyContinue
    if ($cmd) { return "mamba" }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return "conda" }
    throw "Install Miniforge, Mambaforge, or Anaconda first. Windows uses conda-forge for pynini/OpenFst."
}

$conda = Find-Conda
Write-Host "Using $conda"

& $conda create -y -n $EnvName -c conda-forge python=3.12 pynini openfst pip

$python = @("run", "-n", $EnvName, "python")

& $conda @python -m pip install --upgrade pip setuptools wheel

if ($UseCuda) {
    & $conda @python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
}

& $conda @python -m pip install -e ".[dev,quant]"
& $conda @python -m pytest -q

Write-Host ""
Write-Host "Windows environment ready."
Write-Host "Run the app with:"
Write-Host "  $conda run -n $EnvName python -m voice_clone_dot_tts.main"
