# Changelog

## Beta 0.1.0 - 2026-06-17

### Windows Development Stack

- Reworked Windows setup around a conda/micromamba environment named `voice-clone-dot-tts` instead of a plain `.venv` workflow.
- Documented why Windows uses conda-forge: the dots.tts text-normalization stack depends on `WeTextProcessing -> pynini -> OpenFst`, and pip-only installs can fall back to fragile native source builds.
- Updated `scripts/setup_windows.ps1` to find `micromamba`, `mamba`, or `conda`, create the environment with Python, `pynini`, `openfst`, PySide6, and QtMultimedia packages, then install the app editable with dev and quant extras.
- Added native exit-code checking to the Windows setup script so failed conda, pip, Qt import, or pytest steps stop the setup immediately.
- Added optional CUDA PyTorch installation through `scripts/setup_windows.ps1 -UseCuda`.
- Added a generated local `run_app.bat` launcher that runs `main.py` through the conda/micromamba environment, including `MAMBA_ROOT_PREFIX` handling for micromamba.
- Added `/run_app.bat` to `.gitignore` because the launcher is machine-local.
- Updated Windows run instructions for `run_app.bat`, direct `micromamba run`, direct winget-installed micromamba path, and `conda run`.
- Updated Windows test and packaging instructions to run inside the conda/micromamba environment instead of an unsupported `.venv`.
- Switched Windows Qt requirements from PyQt6 to PySide6 because the conda PyQt6 stack did not provide a working QtMultimedia import in this environment.
- Made QtMultimedia a required Windows setup validation step because built-in WAV playback is now required behavior.
- Updated `scripts/build_app.ps1` to work with the restructured Windows environment and PySide6 dependency set.
- Updated `main.py` missing-dependency guidance to point Windows users at the conda/micromamba setup path.

### UI and Runtime

- Added a bottom-bar model download progress indicator so long Hugging Face downloads have visible state.
- Added scrub-able playback controls for both reference audio and the latest generated WAV.
- Added fallback sampler progress for PyTorch checkpoints that do not expose the SOAR ODE hook, including MeanFlow.
- Made audio patch and ETA estimates more conservative so long generations are less likely to be understated.
- Expanded option help text and the built-in guide with clearer layman-first explanations and technical detail.
- Switched the desktop UI to required PySide6 QtMultimedia playback and removed playback-optional behavior.
- Added MeanFlow as the default Windows consumer-GPU checkpoint while keeping SOAR available for quality-first runs.
- Added force GPU / force CPU device choices and exposed experimental PyTorch torchao quantization without switching to MLX.

### Documentation

- Added `docs/MACOS.md` for macOS setup, Apple Silicon MLX guidance, validation, and packaging.
- Expanded `docs/WINDOWS.md` with conda/micromamba setup, CUDA setup, launcher usage, packaging, playback validation, and Windows validation checklist.
- Updated `README.md` to clarify Windows versus macOS setup paths, model download behavior, MeanFlow defaults, playback controls, and progress behavior.
- Updated `docs/IMPLEMENTATION_PLAN.md` to reflect the current PySide6 UI, supported checkpoints, download progress, scrubbing controls, and fallback progress strategy.
