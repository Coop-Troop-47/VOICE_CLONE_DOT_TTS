from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
import gc
import os
import re
import tempfile

import soundfile as sf

from .constants import (
    DEFAULT_ODE_METHOD,
    ODE_METHOD_CHOICES,
    MLX_QUANTIZATION_VALUES,
    PYTORCH_QUANTIZATION_VALUES,
    SUPPORTED_AUDIO_SUFFIXES,
    TEMPLATE_CHOICES,
)
from .models import RuntimeConfig, SynthesisRequest, SynthesisResult

ProgressCallback = Callable[[str], None]


class DotsTtsUnavailableError(RuntimeError):
    """Raised when the upstream dots.tts package cannot be imported."""


class DotsTtsOperationError(RuntimeError):
    """Raised when a synthesis stage fails after request validation."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class DotsTtsService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtime: Any | None = None
        self._runtime_key: tuple[Any, ...] | None = None

    def unload_runtime(self) -> None:
        with self._lock:
            self._runtime = None
            self._runtime_key = None
        _release_accelerator_memory()

    def synthesize(
        self,
        request: SynthesisRequest,
        progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        _emit(progress, "STEP 1/6: Validating request")
        normalized = normalize_request(request)
        with self._lock:
            backend_label = "MLX dots.tts runtime" if normalized.runtime.backend == "mlx" else "PyTorch dots.tts runtime"
            _emit(progress, f"STEP 2/6: Loading {backend_label}")
            try:
                runtime = self._get_runtime(normalized.runtime)
            except Exception as exc:
                raise DotsTtsOperationError(
                    "Runtime load",
                    f"Failed while loading the {backend_label}. "
                    f"{_diagnose_exception(exc)}",
                ) from exc
            _emit(progress, "STEP 3/6: Seeding and preparing generation")
            try:
                _seed_everything(normalized.seed)
            except Exception as exc:
                raise DotsTtsOperationError(
                    "Seeding",
                    "Failed while seeding the inference runtime. "
                    f"{_diagnose_exception(exc)}",
                ) from exc
            _emit(progress, f"STEP 4/6: Generating audio with {normalized.num_steps} sampling steps")
            started_at = time.time()
            try:
                if normalized.runtime.backend == "mlx":
                    result = self._generate_mlx(runtime, normalized, progress)
                elif normalized.execution_mode == "generate_stream":
                    result = self._generate_stream(runtime, normalized, progress)
                else:
                    with _sampling_progress(
                        normalized.num_steps,
                        normalized.ode_method,
                        progress,
                        estimated_patch_total=_estimate_audio_patch_total(runtime, normalized),
                    ):
                        result = runtime.generate(**_runtime_kwargs(normalized))
                    result["chunk_count"] = 1
            except Exception as exc:
                raise DotsTtsOperationError(
                    "Generation",
                    "Failed while generating audio. "
                    f"{_diagnose_exception(exc)}",
                ) from exc
            elapsed = time.time() - started_at

            try:
                audio = result["audio"]
                sample_rate = int(result["sample_rate"])
                samples = int(audio.shape[-1])
            except Exception as exc:
                raise DotsTtsOperationError(
                    "Runtime output",
                    "dots.tts returned an unexpected result shape. "
                    f"{_diagnose_exception(exc)}",
                ) from exc
            _emit(progress, "STEP 5/6: Writing WAV output")
            try:
                audio_path = _write_audio(
                    audio=audio,
                    sample_rate=sample_rate,
                    output_dir=normalized.output_dir,
                )
                cleanup_outputs(
                    normalized.output_dir,
                    retention_count=normalized.output_retention_count,
                )
            except Exception as exc:
                raise DotsTtsOperationError(
                    "Output write",
                    f"Failed while writing the generated WAV to {normalized.output_dir}. "
                    f"{_diagnose_exception(exc)}",
                ) from exc
            audio_seconds = samples / sample_rate if sample_rate else 0.0
            device = "mlx" if normalized.runtime.backend == "mlx" else _select_device(normalized.runtime.device)
            precision = (
                normalized.runtime.precision
                if normalized.runtime.backend == "mlx"
                else _effective_precision(normalized.runtime.precision, device=device)
            )
            metrics = {
                "request_id": result.get("fid"),
                "model": normalized.runtime.model_name_or_path,
                "device": device,
                "precision": precision,
                "execution_mode": normalized.execution_mode,
                "template_name": normalized.template_name,
                "language": normalized.language,
                "sample_rate": sample_rate,
                "samples": samples,
                "audio_seconds": round(audio_seconds, 3),
                "elapsed_seconds": round(float(result.get("time_used", elapsed)), 3),
                "wall_seconds": round(elapsed, 3),
                "rtf": round(float(result.get("rtf", 0.0)), 4),
                "chunk_count": int(result.get("chunk_count", 1)),
                "seed": normalized.seed,
                "num_steps": normalized.num_steps,
                "guidance_scale": normalized.guidance_scale,
                "speaker_scale": normalized.speaker_scale,
                "output_path": str(audio_path),
            }
            if result.get("profiling") is not None:
                metrics["profiling"] = result["profiling"]
            metrics["backend"] = normalized.runtime.backend
            metrics["quantization"] = normalized.runtime.quantization
            _emit(progress, f"STEP 6/6: Done - wrote {audio_path}")
            if normalized.runtime.unload_after_generation:
                self._runtime = None
                self._runtime_key = None
                _release_accelerator_memory()
            return SynthesisResult(
                audio_path=audio_path,
                sample_rate=sample_rate,
                samples=samples,
                metrics=metrics,
            )

    def _get_runtime(self, config: RuntimeConfig) -> Any:
        if config.backend == "mlx":
            return self._get_mlx_runtime(config)
        return self._get_pytorch_runtime(config)

    def _get_pytorch_runtime(self, config: RuntimeConfig) -> Any:
        device = _select_device(config.device)
        precision = _effective_precision(config.precision, device=device)
        key = (
            "pytorch",
            _resolve_model_path(config.model_name_or_path),
            config.revision or "",
            config.cache_dir or "",
            device,
            precision,
            bool(config.optimize),
            int(config.max_generate_length),
            bool(config.unload_after_generation),
            config.quantization,
        )
        if self._runtime is not None and self._runtime_key == key:
            return self._runtime
        if self._runtime is not None:
            self._runtime = None
            self._runtime_key = None
            _release_accelerator_memory()

        try:
            from dots_tts.runtime import DotsTtsRuntime
        except Exception as exc:  # pragma: no cover - exact import failure varies.
            raise DotsTtsUnavailableError(
                "Failed to import the PyTorch dots.tts runtime. "
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc

        kwargs: dict[str, Any] = {
            "precision": precision,
            "optimize": config.optimize,
            "max_generate_length": config.max_generate_length,
        }
        if config.revision:
            kwargs["revision"] = config.revision
        if config.cache_dir:
            kwargs["cache_dir"] = config.cache_dir
        self._runtime = DotsTtsRuntime.from_pretrained(key[1], **kwargs)
        _move_runtime_to_device(self._runtime, device)
        _apply_pytorch_quantization(self._runtime, config.quantization, device=device)
        self._runtime_key = key
        return self._runtime

    def _get_mlx_runtime(self, config: RuntimeConfig) -> Any:
        if not _is_macos_arm64():
            raise DotsTtsUnavailableError("MLX backend requires Apple Silicon macOS.")
        key = (
            "mlx",
            _resolve_model_path(config.model_name_or_path),
            config.precision,
            config.quantization,
            bool(config.unload_after_generation),
        )
        if self._runtime is not None and self._runtime_key == key:
            return self._runtime
        if self._runtime is not None:
            self._runtime = None
            self._runtime_key = None
            _release_accelerator_memory()
        try:
            import mlx.core as mx
            from dots_tts_mlx.loader import from_pretrained
        except Exception as exc:  # pragma: no cover - exact import failure varies.
            raise DotsTtsUnavailableError(
                "The MLX dots.tts runtime is not installed. Install the macOS MLX extra with "
                "`python -m pip install 'git+https://github.com/sb1992/dots-tts-mlx.git'`."
            ) from exc
        dtype = _mlx_dtype(mx, config.precision)
        loaded = from_pretrained(_resolve_model_path(config.model_name_or_path), dtype=dtype)
        self._runtime = getattr(loaded, "model", loaded)
        self._runtime_key = key
        return self._runtime

    @staticmethod
    def _generate_stream(
        runtime: Any,
        request: SynthesisRequest,
        progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        import torch

        started_at = time.time()
        chunks = []
        with _sampling_progress(
            request.num_steps,
            request.ode_method,
            progress,
            estimated_patch_total=_estimate_audio_patch_total(runtime, request),
        ):
            for index, chunk in enumerate(runtime.generate_stream(**_runtime_kwargs(request)), start=1):
                chunks.append(chunk.detach().float().cpu())
                _emit(progress, f"Received stream chunk {index}")
        if not chunks:
            raise ValueError("Streaming generation returned no audio chunks.")

        audio = torch.cat(chunks, dim=-1)
        elapsed = time.time() - started_at
        sample_rate = int(runtime.sample_rate)
        audio_seconds = int(audio.shape[-1]) / sample_rate if sample_rate else 0.0
        return {
            "fid": _build_stream_request_id(runtime, request),
            "audio": audio,
            "sample_rate": sample_rate,
            "time_used": elapsed,
            "rtf": elapsed / audio_seconds if audio_seconds > 0 else 0.0,
            "chunk_count": len(chunks),
        }

    @staticmethod
    def _generate_mlx(
        runtime: Any,
        request: SynthesisRequest,
        progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        if request.prompt_audio_path is None or not request.prompt_text:
            raise ValueError("MLX backend requires prompt audio and exact prompt transcript.")
        if request.execution_mode == "generate_stream":
            _emit(progress, "MLX backend does not expose streaming through this app; using normal generate.")
        started_at = time.time()
        with _mlx_prompt_audio_path(request.prompt_audio_path, progress) as prompt_audio_path:
            with _mlx_sampling_progress(
                runtime,
                request.num_steps,
                progress,
                estimated_patch_total=_estimate_audio_patch_total(runtime, request),
            ):
                out = runtime.generate(
                    text=request.text,
                    prompt_audio=prompt_audio_path,
                    prompt_text=request.prompt_text,
                    language=request.language,
                    num_steps=request.num_steps,
                    guidance_scale=request.guidance_scale,
                    speaker_scale=request.speaker_scale,
                    seed=request.seed,
                    max_generate_length=request.runtime.max_generate_length,
                )
        _emit(progress, "MLX finalization: runtime returned audio; converting output tensor")

        audio = out["audio"]
        try:
            import mlx.core as mx

            import numpy as np

            audio_array = mx.array(audio).astype(mx.float32)
            mx.eval(audio_array)
            audio = np.array(audio_array)
        except Exception:
            import numpy as np

            audio = np.asarray(audio, dtype=np.float32)
        _emit(progress, "MLX finalization: output tensor ready")
        if audio.ndim == 1:
            import numpy as np

            audio = np.expand_dims(audio, axis=0)
        sample_rate = int(out.get("sample_rate", 48000))
        elapsed = time.time() - started_at
        audio_seconds = int(audio.shape[-1]) / sample_rate if sample_rate else 0.0
        return {
            "fid": out.get("fid"),
            "audio": audio,
            "sample_rate": sample_rate,
            "time_used": elapsed,
            "rtf": elapsed / audio_seconds if audio_seconds > 0 else 0.0,
            "chunk_count": 1,
            "profiling": None,
        }


@contextmanager
def _mlx_prompt_audio_path(prompt_audio_path: str, progress: ProgressCallback | None) -> Iterator[str]:
    source = Path(prompt_audio_path).expanduser()
    if _is_riff_wav(source):
        yield str(source)
        return

    _emit(progress, "Preparing MLX prompt audio: converting reference audio to temporary WAV")
    with tempfile.TemporaryDirectory(prefix="voice-clone-dot-tts-mlx-") as temp_dir:
        target = Path(temp_dir) / "prompt.wav"
        try:
            _convert_audio_to_wav(source, target)
        except Exception as exc:
            raise ValueError(
                "MLX can only read WAV prompt audio directly, and this app could not decode "
                f"{source.name!r} for automatic conversion. Convert the reference clip to a normal PCM WAV and try again. "
                f"{_diagnose_exception(exc)}"
            ) from exc
        yield str(target)


def _is_riff_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"RIFF"
    except Exception:
        return False


def _convert_audio_to_wav(source: Path, target: Path) -> None:
    try:
        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
    except Exception:
        audio, sample_rate = _read_audio_with_torchaudio(source)
    audio = _mono_audio_array(audio)
    if getattr(audio, "size", 0) == 0:
        raise ValueError("Reference audio is empty.")
    sf.write(str(target), audio, int(sample_rate), format="WAV", subtype="PCM_16")


def _read_audio_with_torchaudio(source: Path) -> tuple[Any, int]:
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(source))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.detach().float().cpu()
    return waveform.numpy(), int(sample_rate)


def _mono_audio_array(audio: Any) -> Any:
    import numpy as np

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)
    elif array.ndim > 2:
        array = np.squeeze(array)
        if array.ndim > 1:
            array = array.reshape(-1, array.shape[-1]).mean(axis=0)
    return array


def normalize_request(request: SynthesisRequest) -> SynthesisRequest:
    runtime = normalize_runtime_config(request.runtime)
    text = request.text.strip()
    if not text:
        raise ValueError("Text to synthesize is required.")

    prompt_audio_path = _normalize_optional_path(request.prompt_audio_path)
    if prompt_audio_path is not None:
        suffix = Path(prompt_audio_path).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported prompt audio type: {suffix}")
        if not Path(prompt_audio_path).is_file():
            raise ValueError(f"Prompt audio does not exist: {prompt_audio_path}")

    prompt_text = (request.prompt_text or "").strip() or None
    if prompt_text and prompt_audio_path is None:
        raise ValueError("Prompt transcript requires prompt audio.")
    if runtime.backend == "mlx":
        if prompt_audio_path is None or not prompt_text:
            raise ValueError("MLX backend requires prompt audio and exact prompt transcript.")
        if request.normalize_text:
            raise ValueError("MLX backend does not support in-runtime text normalization; turn Normalize text off.")

    execution_mode = request.execution_mode
    if execution_mode not in ("generate", "generate_stream"):
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    template_name = request.template_name.strip() or "tts"
    supported_templates = {value for _, value in TEMPLATE_CHOICES}
    if template_name not in supported_templates:
        raise ValueError(f"Unsupported template: {template_name}")

    ode_method = request.ode_method.strip() or DEFAULT_ODE_METHOD
    if ode_method not in ODE_METHOD_CHOICES:
        raise ValueError(f"Unsupported ODE method: {ode_method}")

    num_steps = int(request.num_steps)
    if num_steps <= 0:
        raise ValueError("Num steps must be positive.")

    guidance_scale = float(request.guidance_scale)
    if guidance_scale <= 0:
        raise ValueError("Guidance scale must be positive.")

    speaker_scale = float(request.speaker_scale)
    if speaker_scale < 0:
        raise ValueError("Speaker scale cannot be negative.")

    output_dir = Path(request.output_dir).expanduser()
    retention = int(request.output_retention_count)
    if retention < 0:
        raise ValueError("Keep WAVs cannot be negative.")

    language = (request.language or "").strip() or None
    if runtime.backend == "mlx" and language == "auto_detect":
        raise ValueError("MLX backend does not support Auto detect language. Choose an explicit language code.")

    return SynthesisRequest(
        runtime=runtime,
        text=text,
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        execution_mode=execution_mode,
        template_name=template_name,
        language=language,
        ode_method=ode_method,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        speaker_scale=speaker_scale,
        normalize_text=bool(request.normalize_text),
        profile_inference=bool(request.profile_inference),
        seed=int(request.seed),
        output_dir=output_dir,
        output_retention_count=retention,
    )


def normalize_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    backend = (config.backend or "pytorch").strip().lower()
    if backend not in {"pytorch", "mlx"}:
        raise ValueError(f"Unsupported backend: {config.backend}")
    model_name_or_path = config.model_name_or_path.strip()
    if not model_name_or_path:
        raise ValueError("Model name or path is required.")

    max_generate_length = int(config.max_generate_length)
    if max_generate_length <= 0:
        raise ValueError("Max generate length must be positive.")

    precision = config.precision.strip() or "bfloat16"
    if precision not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    quantization = (config.quantization or "none").strip()
    if quantization not in set(PYTORCH_QUANTIZATION_VALUES + MLX_QUANTIZATION_VALUES):
        raise ValueError(f"Unsupported quantization: {quantization}")
    if backend == "pytorch" and quantization not in PYTORCH_QUANTIZATION_VALUES:
        raise ValueError("This quantization option is MLX-only. Use a PyTorch torchao quantization option or None.")
    if backend == "mlx" and quantization not in MLX_QUANTIZATION_VALUES:
        quantization = "int4"
    device = config.device.strip().lower() or "auto"
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {device}")

    cache_dir = _normalize_optional_path(config.cache_dir)
    return RuntimeConfig(
        backend=backend,  # type: ignore[arg-type]
        model_name_or_path=model_name_or_path,
        revision=(config.revision or "").strip() or None,
        cache_dir=cache_dir,
        device=device,
        precision=precision,
        quantization=quantization,
        optimize=bool(config.optimize),
        max_generate_length=max_generate_length,
        unload_after_generation=bool(config.unload_after_generation),
    )


def cleanup_outputs(output_dir: Path, retention_count: int) -> None:
    if retention_count <= 0:
        return
    if not output_dir.is_dir():
        return
    wav_files = sorted(
        output_dir.glob("*.wav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_file in wav_files[retention_count:]:
        stale_file.unlink(missing_ok=True)


def _runtime_kwargs(request: SynthesisRequest) -> dict[str, Any]:
    kwargs = {
        "text": request.text,
        "prompt_audio_path": request.prompt_audio_path,
        "prompt_text": request.prompt_text,
        "template_name": request.template_name,
        "language": request.language,
        "speaker_scale": request.speaker_scale,
        "ode_method": request.ode_method,
        "num_steps": request.num_steps,
        "guidance_scale": request.guidance_scale,
        "normalize_text": request.normalize_text,
        "profile_inference": request.profile_inference,
    }
    return kwargs


def _build_stream_request_id(runtime: Any, request: SynthesisRequest) -> str | None:
    required_attrs = ("_process_text", "_process_prompt_text", "_build_request_id")
    if not all(hasattr(runtime, attr) for attr in required_attrs):
        return None
    try:
        normalized_text, normalized_language = runtime._process_text(
            request.text,
            language=request.language,
            normalize=request.normalize_text,
        )
        normalized_prompt_text = runtime._process_prompt_text(
            request.prompt_text,
            language=normalized_language,
        )
        if normalized_language is not None and not normalized_prompt_text:
            from dots_tts.utils.text import attach_language_tag

            normalized_text = attach_language_tag(normalized_text, normalized_language)
        kwargs = {
            "text": normalized_text,
            "prompt_audio_path": request.prompt_audio_path,
            "prompt_text": normalized_prompt_text,
            "template_name": request.template_name,
        }
        if normalized_language is not None:
            kwargs["language"] = normalized_language
        return runtime._build_request_id(**kwargs)
    except Exception:
        return None


def _estimate_audio_patch_total(runtime: Any, request: SynthesisRequest) -> int:
    model = getattr(runtime, "model", runtime)
    config = getattr(model, "config", None)
    patch_size = int(getattr(config, "patch_size", 8) or 8)
    hop_size = int(getattr(model, "hop_size", 512) or 512)
    sample_rate = int(getattr(runtime, "sample_rate", 48000) or 48000)
    patch_seconds = max(0.04, (patch_size * hop_size) / sample_rate)

    text = request.text.strip()
    word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
    non_space_chars = len(re.sub(r"\s+", "", text))
    if word_count >= 3:
        # Cloned speech is usually slower than audiobook narration once pauses,
        # prompt conditioning, and punctuation are included. Bias high so the UI
        # does not routinely understate patch count.
        estimated_seconds = word_count / 1.85
    else:
        estimated_seconds = non_space_chars / 5.8
    punctuation_padding = min(
        8.0,
        text.count(",") * 0.35
        + text.count(".") * 0.55
        + text.count(";") * 0.45
        + text.count(":") * 0.35
        + text.count("?") * 0.7
        + text.count("!") * 0.65,
    )
    line_padding = max(0, text.count("\n")) * 0.45
    long_text_padding = non_space_chars / 180.0
    estimated_seconds = max(1.4, (estimated_seconds + punctuation_padding + line_padding + long_text_padding) * 1.25)

    estimate = int((estimated_seconds + patch_seconds - 1e-9) // patch_seconds) + 1
    estimate = max(1, estimate)
    return min(int(request.runtime.max_generate_length), estimate)


def _write_audio(audio: Any, sample_rate: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
    if hasattr(audio, "detach"):
        waveform = audio.detach().float().cpu().squeeze().numpy()
    else:
        import numpy as np

        waveform = np.asarray(audio, dtype=np.float32).squeeze()
    if getattr(waveform, "ndim", 0) == 0:
        raise ValueError("Generated audio is empty.")
    sf.write(output_path, waveform, sample_rate)
    return output_path


def _seed_everything(seed: int) -> None:
    try:
        from dots_tts.utils.util import seed_everything

        seed_everything(seed)
    except Exception:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _resolve_model_path(model_name_or_path: str) -> str:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path.resolve())
    return model_name_or_path


def _select_device(preference: str) -> str:
    normalized = preference.strip().lower() or "auto"
    try:
        import torch
    except Exception:
        return "cpu"
    if normalized == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA was selected, but CUDA is not available.")
    if normalized == "mps":
        if _torch_mps_available(torch):
            if _allow_pytorch_mps():
                return "mps"
            raise RuntimeError(
                "Apple MPS was selected for the PyTorch backend, but PyTorch MPS can hard-crash dots.tts "
                "inside Apple's Metal/MPS matmul path. Use the MLX backend for Apple Silicon GPU acceleration. "
                "To force unsupported PyTorch MPS anyway, launch with VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS=1."
            )
        raise RuntimeError("Apple MPS was selected, but MPS is not available.")
    if normalized == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if _torch_mps_available(torch) and _allow_pytorch_mps():
        return "mps"
    return "cpu"


def _torch_mps_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def _allow_pytorch_mps() -> bool:
    return os.environ.get("VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS", "").strip() == "1"


def _effective_precision(precision: str, *, device: str) -> str:
    normalized = precision.strip() or "bfloat16"
    if normalized in {"float32", "fp32", "torch.float32"}:
        return "float32"
    if device in {"cuda", "mps"}:
        return normalized
    return "float32"


def _move_runtime_to_device(runtime: Any, device: str) -> None:
    current_device = getattr(runtime, "device", None)
    if current_device is not None and str(current_device) == device:
        return
    try:
        import torch

        target = torch.device(device)
        runtime.model = runtime.model.to(target).eval()
        runtime.device = target
    except Exception as exc:
        raise RuntimeError(f"Failed to move dots.tts runtime to {device}: {exc}") from exc


def _apply_pytorch_quantization(runtime: Any, quantization: str, *, device: str) -> None:
    normalized = (quantization or "none").strip().lower()
    if normalized == "none":
        return
    try:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    except Exception as exc:
        raise RuntimeError(
            "PyTorch quantization requires torchao. Install it in this environment before using "
            "`PyTorch torchao int8/int4`, or choose `None / full PyTorch weights`."
        ) from exc

    model = getattr(runtime, "model", None)
    if model is None:
        raise RuntimeError("Cannot quantize PyTorch runtime because it does not expose a model attribute.")
    try:
        if normalized == "torchao-int8wo":
            quantize_(model, Int8WeightOnlyConfig(), device=device)
        elif normalized == "torchao-int4wo":
            quantize_(model, Int4WeightOnlyConfig(group_size=128), device=device)
        else:
            raise RuntimeError(f"Unsupported PyTorch quantization mode: {quantization}")
        if hasattr(model, "eval"):
            model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"torchao failed to apply {normalized} to dots.tts. This is an experimental Windows/CUDA/CPU "
            "memory mode and may not support every layer in this architecture. Choose None or MLX on Apple Silicon. "
            f"{_diagnose_exception(exc)}"
        ) from exc


def _diagnose_exception(exc: Exception) -> str:
    cause = exc.__cause__ or exc.__context__
    message = f"{type(exc).__name__}: {exc}"
    if cause is not None:
        message += f" Cause: {type(cause).__name__}: {cause}"
    return message


@contextmanager
def _sampling_progress(
    num_steps: int,
    ode_method: str,
    progress: ProgressCallback | None,
    *,
    estimated_patch_total: int | None = None,
) -> Iterator[None]:
    total_steps = max(1, int(num_steps))
    method = (ode_method or "euler").lower()
    evals_per_step = {"euler": 1, "midpoint": 2, "rk4": 4}.get(method, 1)
    patch_state = {"index": 0}
    estimated_total = max(1, int(estimated_patch_total or 1))
    try:
        import dots_tts.models.dots_tts.core as core_module
    except Exception:
        _emit(progress, f"SAMPLING_PATCH 1/{estimated_total} 0/{total_steps}: sampler running")
        with _fallback_sampling_progress(total_steps, method, progress, estimated_total=estimated_total):
            yield
        _emit(progress, f"SAMPLING_PATCH 1/{estimated_total} {total_steps}/{total_steps}: sampler finished")
        return

    original_odeint = core_module.odeint

    def odeint_with_progress(*args, **kwargs):
        patch_state["index"] += 1
        patch_index = patch_state["index"]
        patch_total = max(estimated_total, patch_index)
        last_step = {"value": -1}
        original_func = kwargs.get("func")
        if original_func is None and args:
            original_func = args[0]
        if original_func is None:
            return original_odeint(*args, **kwargs)
        eval_count = {"value": 0}
        _emit(progress, f"SAMPLING_PATCH {patch_index}/{patch_total} 0/{total_steps}: {method} audio patch starting")

        def wrapped_func(*func_args, **func_kwargs):
            eval_count["value"] += 1
            current_step = min(total_steps, max(0, (eval_count["value"] + evals_per_step - 1) // evals_per_step))
            if current_step != last_step["value"]:
                last_step["value"] = current_step
                _emit(progress, f"SAMPLING_PATCH {patch_index}/{patch_total} {current_step}/{total_steps}: {method} sampler")
            return original_func(*func_args, **func_kwargs)

        try:
            if "func" in kwargs:
                kwargs = dict(kwargs)
                kwargs["func"] = wrapped_func
                return original_odeint(*args, **kwargs)
            args = (wrapped_func, *args[1:])
            return original_odeint(*args, **kwargs)
        finally:
            _emit(progress, f"SAMPLING_PATCH {patch_index}/{patch_total} {total_steps}/{total_steps}: {method} audio patch finished")

    core_module.odeint = odeint_with_progress
    _emit(progress, f"SAMPLING_PATCH 0/{estimated_total} 0/{total_steps}: {method} sampler starting")
    try:
        with _fallback_sampling_progress(
            total_steps,
            method,
            progress,
            estimated_total=estimated_total,
            should_emit=lambda: patch_state["index"] == 0,
        ):
            yield
    finally:
        core_module.odeint = original_odeint
        final_patch = max(1, patch_state["index"])
        final_total = max(estimated_total, final_patch)
        _emit(progress, f"SAMPLING_PATCH {final_patch}/{final_total} {total_steps}/{total_steps}: {method} sampler finished")


@contextmanager
def _fallback_sampling_progress(
    num_steps: int,
    method: str,
    progress: ProgressCallback | None,
    *,
    estimated_total: int,
    should_emit: Callable[[], bool] | None = None,
) -> Iterator[None]:
    if progress is None:
        yield
        return
    total_steps = max(1, int(num_steps))
    stop_event = threading.Event()

    def emit_progress() -> None:
        current = 0
        while not stop_event.wait(0.5):
            if should_emit is not None and not should_emit():
                continue
            current = min(total_steps - 1, current + 1)
            if current <= 0:
                continue
            _emit(progress, f"SAMPLING_PATCH 1/{estimated_total} {current}/{total_steps}: {method} sampler running")
            if current >= total_steps - 1:
                current = max(0, total_steps - 2)

    thread = threading.Thread(target=emit_progress, name="dots-tts-progress-fallback", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


@contextmanager
def _mlx_sampling_progress(
    runtime: Any,
    num_steps: int,
    progress: ProgressCallback | None,
    *,
    estimated_patch_total: int | None = None,
) -> Iterator[None]:
    total_steps = max(1, int(num_steps))
    estimated_total = max(1, int(estimated_patch_total or 1))
    flow_solver = getattr(runtime, "flow_solver", None)
    dit = getattr(flow_solver, "dit", None)
    dit_type = type(dit) if dit is not None else None
    original_call = getattr(dit_type, "__call__", None) if dit_type is not None else None
    original_denoise = getattr(flow_solver, "denoise", None)
    original_meanflow = getattr(flow_solver, "meanflow_sample", None)
    vae = getattr(runtime, "vae", None)
    original_decode = getattr(vae, "decode", None)
    state = {"patch": 0, "step": 0, "last": -1, "active": False}

    def call_with_progress(self, *args, **kwargs):
        result = original_call(self, *args, **kwargs)
        if self is dit and state["active"] and state["step"] < total_steps:
            state["step"] += 1
            if state["step"] != state["last"]:
                state["last"] = state["step"]
                description = "MLX sampler"
                if state["step"] >= total_steps:
                    description = "MLX sampler complete; encoding patch"
                patch_total = max(estimated_total, int(state["patch"]))
                _emit(
                    progress,
                    f"SAMPLING_PATCH {state['patch']}/{patch_total} {state['step']}/{total_steps}: {description}",
                )
        return result

    def solver_with_progress(original_solver, label: str):
        def wrapped_solver(*args, **kwargs):
            state["patch"] += 1
            state["step"] = 0
            state["last"] = -1
            state["active"] = True
            patch_total = max(estimated_total, int(state["patch"]))
            _emit(progress, f"SAMPLING_PATCH {state['patch']}/{patch_total} 0/{total_steps}: MLX {label} patch starting")
            try:
                return original_solver(*args, **kwargs)
            finally:
                state["active"] = False
                _emit(
                    progress,
                    f"SAMPLING_PATCH {state['patch']}/{patch_total} {total_steps}/{total_steps}: MLX {label} patch complete; continuing generation",
                )

        return wrapped_solver

    def decode_with_progress(*args, **kwargs):
        final_patch = max(1, int(state["patch"]))
        final_total = max(estimated_total, final_patch)
        _emit(progress, f"SAMPLING_PATCH {final_patch}/{final_total} {total_steps}/{total_steps}: MLX vocoder decoding audio")
        try:
            return original_decode(*args, **kwargs)
        finally:
            _emit(progress, f"SAMPLING_PATCH {final_patch}/{final_total} {total_steps}/{total_steps}: MLX vocoder decode complete; cleaning audio")

    if dit is not None and dit_type is not None and original_call is not None:
        dit_type.__call__ = call_with_progress
    if flow_solver is not None and callable(original_denoise):
        flow_solver.denoise = solver_with_progress(original_denoise, "flow")  # type: ignore[method-assign]
    if flow_solver is not None and callable(original_meanflow):
        flow_solver.meanflow_sample = solver_with_progress(original_meanflow, "MeanFlow")  # type: ignore[method-assign]
    if vae is not None and callable(original_decode):
        vae.decode = decode_with_progress  # type: ignore[method-assign]
    _emit(progress, f"SAMPLING_PATCH 0/{estimated_total} 0/{total_steps}: MLX generation starting")
    failed = False
    try:
        yield
    except Exception:
        failed = True
        final_patch = max(1, int(state["patch"]))
        final_total = max(estimated_total, final_patch)
        _emit(progress, f"SAMPLING_PATCH {final_patch}/{final_total} {total_steps}/{total_steps}: MLX generation failed before output")
        raise
    finally:
        if dit is not None and dit_type is not None and original_call is not None:
            dit_type.__call__ = original_call
        if flow_solver is not None and callable(original_denoise):
            flow_solver.denoise = original_denoise  # type: ignore[method-assign]
        if flow_solver is not None and callable(original_meanflow):
            flow_solver.meanflow_sample = original_meanflow  # type: ignore[method-assign]
        if vae is not None and callable(original_decode):
            vae.decode = original_decode  # type: ignore[method-assign]
        if not failed:
            final_patch = max(1, int(state["patch"]))
            final_total = max(estimated_total, final_patch)
            _emit(progress, f"SAMPLING_PATCH {final_patch}/{final_total} {total_steps}/{total_steps}: MLX generation finished; finalizing output")


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


def _is_macos_arm64() -> bool:
    import platform
    import sys

    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _mlx_dtype(mx: Any, precision: str) -> Any:
    normalized = (precision or "bfloat16").lower()
    if normalized in {"float16", "fp16", "torch.float16"}:
        return mx.float16
    if normalized in {"float32", "fp32", "torch.float32"}:
        return mx.float32
    return mx.bfloat16


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return str(Path(stripped).expanduser())


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
