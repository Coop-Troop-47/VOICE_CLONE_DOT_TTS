# Windows Setup and Packaging

The app code is Windows-aware through the PyTorch backend. The MLX backend and
MLX int4/int8 model downloads are Apple Silicon macOS only. On Windows, use
PyTorch with CUDA when available. For an RTX 3070 or similar 8 GB consumer GPU,
start with the MeanFlow checkpoint, `rednote-hilab/dots.tts-mf`, in float16
with Auto or Force NVIDIA GPU selected.

The Windows setup installs the optional `quant` extra, which adds PyTorch
torchao. In the app, the PyTorch quantization dropdown exposes int8 and int4
weight-only modes. These are experimental runtime quantization modes for the
official PyTorch checkpoints, not separate downloaded checkpoints. MeanFlow with
CUDA and 4-8 sampler steps is the recommended consumer-hardware path.

The dots.tts dependency stack includes `WeTextProcessing -> pynini -> OpenFst`.
On Windows, use Conda/conda-forge for that native dependency rather than a
plain `.venv`/pip install. Pip falls back to building `pynini` from source on
Windows unless the native OpenFst/MSVC stack is already present; conda-forge
provides working prebuilt packages.

## Recommended Setup

Install Micromamba, Miniforge, Mambaforge, or Anaconda, then from PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

The setup script also creates `run_app.bat` in the repository root. This file is
local to your machine, contains the resolved conda/micromamba command, and is
ignored by Git. The setup also verifies that the Qt media library imports
successfully, because built-in WAV playback is a required app feature.

If you installed Micromamba with winget and the current shell does not know the
new PATH entry yet, restart PowerShell or add the winget package directory for
this session before running the setup script:

```powershell
$env:PATH="$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Mamba.Micromamba_Microsoft.Winget.Source_8wekyb3d8bbwe;$env:PATH"
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

For an NVIDIA GPU build:

```powershell
.\scripts\setup_windows.ps1 -UseCuda
```

The `-UseCuda` option installs PyTorch from the CUDA wheel index before the app
install. If CUDA is not available, the app falls back to CPU.

## Run

After setup, double-click `run_app.bat` or run it from PowerShell:

```powershell
.\run_app.bat
```

To run through the environment manually:

```powershell
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
micromamba run -n voice-clone-dot-tts python main.py
```

If `micromamba` is not on PATH in the current shell yet:

```powershell
$micromamba="$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Mamba.Micromamba_Microsoft.Winget.Source_8wekyb3d8bbwe\micromamba.exe"
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
& $micromamba run -n voice-clone-dot-tts python main.py
```

With Miniforge, Mambaforge, or Anaconda:

```powershell
conda run -n voice-clone-dot-tts python main.py
```

From an activated environment, `python main.py`,
`python -m voice_clone_dot_tts.main`, and `voice-clone-dot-tts` are equivalent.

The app stores models and outputs under:

```text
%LOCALAPPDATA%\Voice Clone dots.tts
```

The model is not bundled. Use the in-app **Download** button to fetch the
selected PyTorch checkpoint from Hugging Face. MeanFlow is the recommended
Windows starting point; SOAR remains available for quality-first runs.

The bottom bar shows model download state while Hugging Face files are being
fetched. The download can be several gigabytes, and exact byte percentages are
not always available through the app, so the progress bar may show a working
state until the local model folder validates.

Reference audio and generated audio each have their own playback controls with
scrubbing. Use these controls to verify the prompt clip before generation and
inspect the generated WAV without opening another media player.

## Package

From an activated Conda environment:

```powershell
.\scripts\build_app.ps1 -Python python -Clean
```

Without activating, pass the environment Python explicitly through `conda run`
or `micromamba run`:

```powershell
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
micromamba run -n voice-clone-dot-tts python -m PyInstaller packaging/voice-clone-dot-tts.spec --clean --noconfirm
```

The output is:

```text
dist\Voice Clone dots.tts\
```

The PyInstaller spec excludes model weights. Confirm before release:

```powershell
Get-ChildItem -Recurse "dist\Voice Clone dots.tts" -Include *.safetensors
```

That command should return no files.

## Windows Validation Checklist

1. Confirm `micromamba run -n voice-clone-dot-tts python -m pytest -q` passes.
2. Launch `.\run_app.bat`.
3. Download `rednote-hilab/dots.tts-mf` in app. Use SOAR after MeanFlow works if the extra quality is needed.
4. Generate a short no-reference sample.
5. Generate prompt-audio-only cloning.
6. Generate prompt-audio plus transcript cloning.
7. Confirm the bottom download progress indicator changes while downloading.
8. Confirm sampler progress shows `Audio patch n/m, sampler x/y` or fallback estimated sampler steps and ETA.
9. Confirm reference and generated audio scrubbers can seek.
10. Test `Experimental torchao int8 weight-only` only after MeanFlow CUDA works.
11. Confirm output folder opens from the app.
12. Build with PyInstaller and launch the packaged executable.
