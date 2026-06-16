param(
    [string]$Python = "python",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}

& $Python -m pip install -e ".[dev,quant]"
& $Python -m pytest -q
& $Python -m PyInstaller packaging/voice-clone-dot-tts.spec --clean --noconfirm

$artifact = Join-Path (Get-Location) "dist\Voice Clone dots.tts"
Write-Host "Build complete: $artifact"
