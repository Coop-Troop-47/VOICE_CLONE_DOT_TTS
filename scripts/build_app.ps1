param(
    [string]$Python = "python",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

if ($Clean) {
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}

Invoke-Native -Command $Python -Arguments @("-m", "pip", "install", "-e", ".[dev,quant]")
Invoke-Native -Command $Python -Arguments @("-m", "pytest", "-q")
Invoke-Native -Command $Python -Arguments @("-m", "PyInstaller", "packaging/voice-clone-dot-tts.spec", "--clean", "--noconfirm")

$artifact = Join-Path (Get-Location) "dist\Voice Clone dots.tts"
Write-Host "Build complete: $artifact"
