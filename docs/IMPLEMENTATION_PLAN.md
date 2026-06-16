# Implementation Plan

## Built in This Phase

The desktop app is structured around three layers:

- `voice_clone_dot_tts.service`: model-runtime orchestration, validation, WAV
  writing, runtime caching, seed control, and output retention.
- `voice_clone_dot_tts.models`: typed request/result dataclasses shared by the
  UI and service.
- `voice_clone_dot_tts.ui`: PySide6 desktop interface with a background synthesis
  worker.

The UI supports the official PyTorch MeanFlow and SOAR checkpoints through the
upstream dots.tts runtime. MeanFlow is the default Windows consumer-GPU path;
SOAR remains available for quality-first generation. Model weights are not
bundled, and generation is disabled until a valid local model folder is
selected. Users can download the selected checkpoint into the default user data
directory or browse to an existing compatible local snapshot.

The UI exposes the supported dots.tts inference surface:

- Continuation cloning: prompt audio plus exact prompt transcript.
- X-vector-only cloning: prompt audio with no transcript.
- No-reference synthesis: no prompt audio.
- Streaming and non-streaming runtime execution.
- Runtime controls: precision, optimize warmup, max generation length, optional
  Hugging Face revision, and optional cache directory.
- Synthesis controls: template, language tag, text normalization, ODE method,
  sampling steps, guidance scale, speaker scale, seed, and profiling. These are
  tucked into a collapsed Advanced Options section so the default workflow stays
  simple.
- Output controls: output directory, retention count, generated WAV playback,
  generated WAV scrubbing, reference audio preview/scrubbing, metrics, logs,
  and runtime unload.
- Packaging-friendly model management: the selected checkpoint downloads into
  the user data directory with a stable local path, existing compatible local
  folders can be selected, and weights are not included in the app bundle.
- Responsive Qt layout: the settings column is scrollable, Advanced Options is
  a regular show/hide section instead of a checkable group box, and smaller
  windows keep controls clickable.
- Clear step counter: the UI shows `Step n/6` progress for validation, runtime
  load, seeding, generation, WAV writing, and completion.
- Patch-aware sampler counter: the UI shows `Audio patch n/m, sampler x/y` and
  an ETA estimated from observed sampler-step speed. MeanFlow and other paths
  that do not call the SOAR ODE hook emit fallback estimated sampler progress.
- Bottom-bar model download progress: Hugging Face downloads show an active
  working state and switch to complete/failed after validation.
- Consumer-GPU defaults: MeanFlow checkpoint, automatic PyTorch CUDA/CPU
  selection, `float16` on CUDA, non-streaming generation, and 8 sampling steps.
- PyTorch quantization options: torchao int8/int4 weight-only modes are exposed
  for Windows/CUDA or CPU experiments. The app downloads official PyTorch
  checkpoints because no compatible official drop-in PyTorch int4/int8
  MeanFlow/SOAR folder checkpoint is currently published.
- Apple Silicon support: MLX is the supported GPU path. PyTorch MPS is blocked
  by default because dots.tts can abort inside Apple's Metal/MPS matmul path
  before Python can show a recoverable error. `Apple MPS` remains available only
  as an unsupported diagnostic override with
  `VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS=1`.
- Windows support: user data defaults to `%LOCALAPPDATA%\Voice Clone dots.tts`,
  the app uses native `os.startfile()` for opening output folders,
  `multiprocessing.freeze_support()` is enabled for packaged execution, and
  PowerShell setup/build scripts are included. The recommended Windows install
  path uses Conda/conda-forge for `pynini/OpenFst`.

## Test Strategy

Automated tests use a fake dots.tts runtime so the service contract can be
validated without downloading multi-gigabyte checkpoints.

Covered now:

- Request validation.
- Prompt transcript and prompt audio consistency.
- Runtime loading and cache reuse.
- Runtime reload on config changes.
- Runtime argument forwarding.
- WAV file writing.
- Output retention cleanup.
- Fixed supported-checkpoint UI behavior.
- Generation blocked until a valid compatible model folder exists.
- Local compatible model folder selection.
- Collapsed Advanced Options behavior.
- Scrollable settings panel behavior.
- Step counter parsing and progress bar updates.
- Download progress state updates.
- Reference and generated audio playback controls.

Runtime verification completed locally in a Python 3.12 environment:

1. Rendered the simplified default Qt UI offscreen.
2. Rendered the expanded Advanced Options Qt UI offscreen.
3. Validated generation request handling, WAV writing, progress parsing, and
   failure handling through mocked runtime tests.
4. Confirmed the Windows environment imports PySide6 QtWidgets and QtMultimedia.
5. On Windows, run `docs/WINDOWS.md` validation on a native Windows host.

## Packaging Notes

- Standalone macOS app creation uses `scripts/build_app.sh`.
- Model/checkpoint distribution is explicit: do not bundle weights; download in
  app.
- Installer/notarization/signing remain future work.
- Platform-specific PyTorch wheel selection.
- Optional first-run dependency and checkpoint bootstrap wizard.
