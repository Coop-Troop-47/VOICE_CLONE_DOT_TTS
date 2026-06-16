# Windows Setup and Packaging

The app code is Windows-aware through the PyTorch backend. The MLX backend and
MLX int4/int8 model downloads are Apple Silicon macOS only. On Windows, use
PyTorch with CUDA when available.

The Windows setup installs the optional `quant` extra, which adds PyTorch
torchao. In the app, the PyTorch quantization dropdown exposes int8 and int4
weight-only modes. These are runtime quantization modes for the official
`rednote-hilab/dots.tts-soar` checkpoint, not separate downloaded checkpoints.

The dots.tts dependency stack includes `WeTextProcessing -> pynini -> OpenFst`.
On Windows, use Conda/conda-forge for that native dependency rather than plain
pip.

## Recommended Setup

Install Miniforge or Mambaforge, then from PowerShell:

```powershell
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

The setup script prints the exact Python path. Run:

```powershell
python -m voice_clone_dot_tts.main
```

The app stores models and outputs under:

```text
%LOCALAPPDATA%\Voice Clone dots.tts
```

The model is not bundled. Use the in-app **Download** button to fetch the fixed
PyTorch SOAR checkpoint, `rednote-hilab/dots.tts-soar`, from Hugging Face.

## Package

From an activated Conda environment:

```powershell
.\scripts\build_app.ps1 -Python python -Clean
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

1. Confirm `python -m pytest -q` passes.
2. Launch `python -m voice_clone_dot_tts.main`.
3. Download `rednote-hilab/dots.tts-soar` in app.
4. Generate a short no-reference sample.
5. Generate prompt-audio-only cloning.
6. Generate prompt-audio plus transcript cloning.
7. Confirm sampler progress shows `Audio patch n/m, sampler x/32` and ETA.
8. Test `PyTorch torchao int8 weight-only` on a short prompt.
9. Confirm output folder opens from the app.
10. Build with PyInstaller and launch the packaged executable.
