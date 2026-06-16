# macOS Setup and Packaging

The app supports two macOS paths:

- Apple Silicon: use the MLX backend for GPU acceleration and lower-memory int4/int8 converted checkpoints.
- Intel Mac: use the PyTorch backend on CPU. Generation is expected to be slow and memory-heavy.

PyTorch MPS is blocked by default because dots.tts can abort inside Apple's Metal/MPS runtime before Python can show a recoverable error. On Apple Silicon, MLX is the supported GPU route.

## Recommended Setup

Install Python and OpenFst with Homebrew:

```bash
brew install python@3.12 openfst
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CFLAGS="-I/opt/homebrew/include" CXXFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  python -m pip install -e ".[dev]"
```

For Apple Silicon MLX support:

```bash
python -m pip install "git+https://github.com/sb1992/dots-tts-mlx.git"
```

## Run

From the repository root with the environment activated:

```bash
python main.py
```

The console entrypoint is also available after editable install:

```bash
voice-clone-dot-tts
```

The app stores models and outputs under:

```text
~/Library/Application Support/Voice Clone dots.tts
```

## Model Guidance

Use the MLX backend on Apple Silicon when practical. The app can download int4, int8, mf-int4, and mf-int8 converted weights from `shraey/dots-tts-mlx`. These variants reduce memory pressure compared with full PyTorch weights.

The PyTorch MeanFlow and SOAR checkpoints remain available for compatibility. MeanFlow is faster and lower-latency; SOAR is quality-first and heavier.

## Validation Checklist

1. Confirm `python -m pytest -q` passes.
2. Launch `python main.py`.
3. Download an MLX variant on Apple Silicon, or a PyTorch checkpoint on Intel/CPU.
4. Confirm the bottom download progress indicator changes while downloading.
5. Load reference audio and confirm the reference scrubber can seek.
6. Generate a short sample and confirm sampler progress, ETA, and generated-audio scrubbing.
7. Confirm output folder opens from the app.

## Package

Build the macOS app bundle with:

```bash
./scripts/build_app.sh
```

Model weights are excluded from the app bundle and remain downloaded per user.
