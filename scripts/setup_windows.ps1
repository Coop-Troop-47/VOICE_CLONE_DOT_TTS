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

function New-LauncherBat {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CondaCommand,
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentName
    )

    $launcherPath = Join-Path (Get-Location) "run_app.bat"
    $resolvedCommand = (Get-Command $CondaCommand -ErrorAction Stop).Source
    $rootPrefixLine = ""
    if ($CondaCommand -eq "micromamba") {
        $rootPrefixLine = 'set "MAMBA_ROOT_PREFIX=%USERPROFILE%\micromamba"'
    }
    $content = @"
@echo off
setlocal
cd /d "%~dp0"
$rootPrefixLine
"$resolvedCommand" run -n $EnvironmentName python main.py
if errorlevel 1 pause
"@
    Set-Content -LiteralPath $launcherPath -Value $content -Encoding ASCII
    return $launcherPath
}

$conda = Find-Conda
Write-Host "Using $conda"

Invoke-Native -Command $conda -Arguments @(
    "create",
    "-y",
    "-n",
    $EnvName,
    "-c",
    "conda-forge",
    "python=3.12",
    "pynini",
    "openfst",
    "pyside6=6.8.3",
    "qt6-multimedia=6.8.3",
    "pip"
)

$python = @("run", "-n", $EnvName, "python")

Invoke-Native -Command $conda -Arguments ($python + @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"))

if ($UseCuda) {
    Invoke-Native -Command $conda -Arguments (
        $python + @("-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128")
    )
}

Invoke-Native -Command $conda -Arguments ($python + @("-m", "pip", "install", "-e", ".[dev,quant]"))
Invoke-Native -Command $conda -Arguments ($python + @("-c", "from PySide6.QtWidgets import QApplication; from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput"))
Invoke-Native -Command $conda -Arguments ($python + @("-m", "pytest", "-q"))
$launcherPath = New-LauncherBat -CondaCommand $conda -EnvironmentName $EnvName

Write-Host ""
Write-Host "Windows environment ready."
Write-Host "Run the app with:"
Write-Host "  .\run_app.bat"
Write-Host "or:"
Write-Host "  $conda run -n $EnvName python main.py"
Write-Host "Launcher written to: $launcherPath"
