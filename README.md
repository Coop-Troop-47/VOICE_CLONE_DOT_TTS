# Voice Clone dots.tts Desktop

A portable Python/PyQt desktop application for the newly released
[`rednote-hilab/dots.tts`](https://github.com/rednote-hilab/dots.tts) model.

The app is intentionally locked to dots.tts SOAR only. It supports the official
PyTorch checkpoint, `rednote-hilab/dots.tts-soar`, and on Apple Silicon it can
also use the converted MLX weights from `shraey/dots-tts-mlx`. Model weights are
not bundled with the app; the in-app **Download** button fetches the selected
backend into the user data directory. Generation is disabled until a valid local
SOAR folder is selected. Use **Download** for the default location or **Browse**
to select an existing local download.

The app supports the SOAR runtime voice synthesis modes:

- Continuation voice cloning with prompt audio plus matching transcript.
- X-vector-only voice cloning with prompt audio only.
- No-reference/random voice synthesis for compatible checkpoints.
- Streaming or non-streaming generation from the fixed SOAR checkpoint.
- Language tags, text normalization, seed control, sampling steps, guidance
  scale, speaker scale, ODE method, precision, optimize warmup, and output
  retention. These controls are available inside **Advanced Options**.
- Apple Silicon MLX int4/int8 variants for lower RAM use. MLX is optional and
  macOS arm64 only; Windows keeps using PyTorch/CUDA.
- PyTorch torchao int8/int4 weight-only quantization options for Windows/CUDA
  or CPU experiments. These quantize the official SOAR checkpoint at runtime;
  they are not separate model forks.
- Patch-aware generation progress with an ETA based on observed sampler step
  speed.

Default settings are quality-first but lower-memory than the first build: local
SOAR checkpoint only, automatic PyTorch device selection (`CUDA` > `CPU`),
`float16` on CUDA, non-streaming generation, 32 sampling steps, and unload
after each generation to prevent memory growth across runs. On Apple Silicon,
use the MLX backend for GPU acceleration; PyTorch MPS is disabled by default
because it can hard-crash this model inside Apple's Metal/MPS runtime.

## Current Scope

This repository now includes the functional Python desktop application, UI
tests, real local-model verification, and macOS PyInstaller packaging
scaffolding. Model weights remain outside the app bundle.

## Requirements

- Python 3.10, 3.11, or 3.12.
- Enough disk space for the selected dots.tts checkpoint.
- A supported PyTorch environment. GPU is recommended for practical speed, but
  the upstream runtime falls back to CPU.
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

See [docs/WINDOWS.md](docs/WINDOWS.md) for CUDA and packaging notes.

The dots.tts and dots-tts-mlx dependencies download from GitHub. Model weights
are downloaded only through the app's fixed SOAR download flow.

Use Python 3.10, 3.11, or 3.12. The upstream dots.tts package currently declares
`>=3.10,<3.13`, so Python 3.13 is not a supported install target.

## Run

For development and testing from the repository root:

```bash
python main.py
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\main.py
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
the user data directory, or use **Browse** to select an existing local SOAR
snapshot. The app does not allow arbitrary checkpoint selection, and generation
stays disabled until the selected folder looks like the expected PyTorch or MLX
dots.tts SOAR model.

PyTorch folders must contain `config.json`, `model.safetensors`,
`vocoder.safetensors`, and `speaker_encoder.safetensors`. MLX variant folders
must contain `config.json`, `core.safetensors`, `vocoder.safetensors`,
`speaker.safetensors`, `llm_config.json`, `latent_stats.npz`, and `tokenizer/`.

MLX memory guide:

- `int4`: about 2.4 GB of weights, lowest-memory SOAR option.
- `int8`: about 3.1 GB of weights, conservative quantized SOAR option.
- `mf-int4` and `mf-int8`: MeanFlow variants designed for fewer sampler
  evaluations and faster generation.

PyTorch quantization guide:

- `torchao-int8wo`: applies PyTorch torchao int8 weight-only quantization to the
  loaded SOAR model. This is the conservative PyTorch memory option.
- `torchao-int4wo`: applies PyTorch torchao int4 weight-only quantization. This
  is the most aggressive PyTorch memory option and is more likely to hit device
  or layer support limits.

I did not find an official or clearly reputable drop-in PyTorch int4/int8
`dots.tts-soar` checkpoint. The app therefore keeps downloading the official
SOAR model and applies PyTorch's own quantization path when selected.

## Tests

```bash
pytest
```

The tests mock the heavy dots.tts runtime, so they validate application behavior
without downloading a model. UI tests render the PyQt window offscreen at the
minimum supported size with advanced options expanded and assert that scroll
panes do not need horizontal overflow.

Recent local verification also rendered the PyQt UI in default and expanded
states, then generated a real WAV from `/Users/coopermatthews/Downloads/Mum
Reference.mp3` using the local SOAR checkpoint on Apple MPS.

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
