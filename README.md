# Voice Clone dots.tts Desktop

A portable Python/Qt desktop application for the newly released
[`rednote-hilab/dots.tts`](https://github.com/rednote-hilab/dots.tts) model.

The app supports official dots.tts PyTorch checkpoints for Windows/Linux and
converted MLX weights for Apple Silicon. On Windows consumer GPUs such as an
RTX 3070, start with `rednote-hilab/dots.tts-mf` MeanFlow. Use
`rednote-hilab/dots.tts-soar` when quality is more important than speed and
memory pressure. Model weights are not bundled with the app; the in-app
**Download** button fetches the selected backend into the user data directory.
Generation is disabled until a valid local model folder is selected. Use
**Download** for the default location or **Browse** to select an existing local
download.

The app supports these dots.tts voice synthesis modes:

- Continuation voice cloning with prompt audio plus matching transcript.
- X-vector-only voice cloning with prompt audio only.
- No-reference/random voice synthesis for compatible checkpoints.
- Streaming or non-streaming generation from the selected PyTorch checkpoint.
- Language tags, text normalization, seed control, sampling steps, guidance
  scale, speaker scale, ODE method, precision, optimize warmup, and output
  retention. These controls are available inside **Advanced Options**.
- Apple Silicon MLX int4/int8 variants for lower RAM use. MLX is optional and
  macOS arm64 only; Windows keeps using PyTorch/CUDA.
- Experimental PyTorch torchao int8/int4 weight-only quantization options for
  Windows/CUDA or CPU experiments. These quantize the selected official
  PyTorch checkpoint at runtime; they are not separate model forks.
- Patch-aware generation progress with an ETA based on observed sampler step
  speed. MeanFlow and other runtimes that do not expose the same internal
  sampler hook still show fallback step progress.
- Built-in model download progress in the bottom bar.
- Scrub-able playback controls for both reference audio and generated WAVs.

Default settings are consumer-GPU first: MeanFlow PyTorch checkpoint, automatic
PyTorch device selection (`CUDA` > `CPU`), `float16` on CUDA, non-streaming
generation, 8 sampling steps, and unload after each generation to prevent memory
growth across runs. On Apple Silicon, use the MLX backend for GPU acceleration;
PyTorch MPS is disabled by default because it can hard-crash this model inside
Apple's Metal/MPS runtime.

## Current Scope

This repository now includes the functional Python desktop application, UI
tests, real local-model verification, and macOS PyInstaller packaging
scaffolding. Model weights remain outside the app bundle.

## Requirements

- Python 3.10, 3.11, or 3.12.
- Enough disk space for the selected dots.tts checkpoint.
- A supported PyTorch environment. GPU is recommended for practical speed, but
  the upstream runtime falls back to CPU.
- On Windows, use Conda/conda-forge through `scripts/setup_windows.ps1`.
  A plain `.venv`/pip install is not the supported Windows path because
  `dots.tts` pulls in `pynini/OpenFst`, which requires native packages.
- On macOS, `openfst` is required to build `pynini`, which is pulled in by
  dots.tts text normalization.
- On Apple Silicon, the MLX backend is installed from the `dots-tts-mlx` fork
  and can download int4, int8, mf-int4, or mf-int8 converted weights.

## Install

macOS:

```bash
brew install python@3.12 openfst
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CFLAGS="-I/opt/homebrew/include" CXXFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  python -m pip install -e ".[dev]"
```

Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

The Windows script expects `micromamba`, `mamba`, or `conda` on PATH and creates
a `voice-clone-dot-tts` environment with conda-forge `pynini/OpenFst`, PySide6, and
the editable app install. It also writes a local ignored `run_app.bat` launcher
in the repository root. See [docs/WINDOWS.md](docs/WINDOWS.md) for CUDA,
Micromamba, launcher, and packaging notes. See [docs/MACOS.md](docs/MACOS.md)
for Apple Silicon MLX, OpenFst, and macOS packaging notes.

The dots.tts and dots-tts-mlx dependencies download from GitHub. Model weights
are downloaded only through the app's model download flow.

Use Python 3.10, 3.11, or 3.12. The upstream dots.tts package currently declares
`>=3.10,<3.13`, so Python 3.13 is not a supported install target.

## Run

For development and testing from the repository root:

```bash
python main.py
```

On Windows PowerShell:

```powershell
.\run_app.bat
```

The setup script creates `run_app.bat` locally and `.gitignore` excludes it. To
run the same command manually:

```powershell
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
micromamba run -n voice-clone-dot-tts python main.py
```

If Micromamba was just installed with winget and this shell does not recognize
`micromamba` yet, restart PowerShell or call the installed executable directly:

```powershell
$micromamba="$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Mamba.Micromamba_Microsoft.Winget.Source_8wekyb3d8bbwe\micromamba.exe"
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
& $micromamba run -n voice-clone-dot-tts python main.py
```

If you used Miniforge, Mambaforge, or Anaconda instead of Micromamba, run the
same command through that tool, for example:

```powershell
conda run -n voice-clone-dot-tts python main.py
```

After editable install, the console entrypoint is also available:

```bash
voice-clone-dot-tts
```

or:

```bash
python -m voice_clone_dot_tts.main
```

Generated WAV files, downloaded models, and prompt assets are stored in the user
data directory by default:

- macOS: `~/Library/Application Support/Voice Clone dots.tts`
- Windows: `%LOCALAPPDATA%\\Voice Clone dots.tts`
- Linux: `$XDG_DATA_HOME/voice-clone-dot-tts` or `~/.local/share/voice-clone-dot-tts`

Set `VOICE_CLONE_DOT_TTS_DATA_DIR` to override this location.

## Model Download

The model is not bundled. Use **Download** to fetch the selected backend into
the user data directory, or use **Browse** to select an existing compatible
local snapshot. The app does not allow arbitrary checkpoint selection, and
generation stays disabled until the selected folder looks like the expected
PyTorch or MLX dots.tts model.

The bottom bar shows active model download state. Hugging Face snapshot downloads
do not always expose exact byte percentages through the app, so the bar uses an
indeterminate working state until validation completes.

PyTorch folders for MeanFlow or SOAR must contain `config.json`,
`model.safetensors`, `vocoder.safetensors`, and `speaker_encoder.safetensors`.
MLX variant folders must contain `config.json`, `core.safetensors`,
`vocoder.safetensors`, `speaker.safetensors`, `llm_config.json`,
`latent_stats.npz`, and `tokenizer/`.

MLX memory guide:

- `int4`: about 2.4 GB of weights, lowest-memory SOAR option.
- `int8`: about 3.1 GB of weights, conservative quantized SOAR option.
- `mf-int4` and `mf-int8`: MeanFlow variants designed for fewer sampler
  evaluations and faster generation.

PyTorch quantization guide:

- MeanFlow plus CUDA and 4-8 sampler steps is the recommended Windows consumer
  path.
- `torchao-int8wo`: applies PyTorch torchao int8 weight-only quantization to the
  loaded model. This is experimental and depends on torchao, CUDA driver, and
  layer support.
- `torchao-int4wo`: applies PyTorch torchao int4 weight-only quantization. This
  is the most aggressive experimental PyTorch memory option and is more likely
  to hit device or layer support limits.

No compatible drop-in PyTorch int4/int8 MeanFlow or SOAR checkpoint is currently
published in the official RedNote repositories. Community BF16 single-file
checkpoints exist for some ComfyUI workflows, but they do not match this app's
folder-based runtime loader. The app therefore downloads the official PyTorch
folders and exposes torchao as an experimental runtime option.

## Tests

```bash
pytest
```

On Windows, run tests inside the conda environment:

```powershell
$env:MAMBA_ROOT_PREFIX="$env:USERPROFILE\micromamba"
micromamba run -n voice-clone-dot-tts python -m pytest -q
```

The tests mock the heavy dots.tts runtime, so they validate application behavior
without downloading a model. UI tests render the Qt window offscreen at the
minimum supported size with advanced options expanded and assert that scroll
panes do not need horizontal overflow.

Recent local verification rendered the Qt UI in default and expanded states and
validated generation behavior through the mocked runtime test suite.

## Packaging

Packaging scaffolding is included, but model weights are intentionally excluded:

```bash
./scripts/build_app.sh
```

On Windows:

```powershell
.\scripts\build_app.ps1 -Python python -Clean
```

The packaged app keeps the same in-app model download workflow.
