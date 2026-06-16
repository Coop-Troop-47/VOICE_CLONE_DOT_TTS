from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_clone_dot_tts.models import RuntimeConfig, SynthesisRequest
from voice_clone_dot_tts.service import (
    DotsTtsOperationError,
    DotsTtsService,
    _select_device,
    _mlx_sampling_progress,
    cleanup_outputs,
    normalize_request,
    normalize_runtime_config,
)


class FakeAudio:
    def __init__(self, samples: int = 16) -> None:
        self._array = np.linspace(-0.1, 0.1, samples, dtype=np.float32).reshape(1, -1)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def ndim(self) -> int:
        return self._array.ndim

    def detach(self) -> "FakeAudio":
        return self

    def float(self) -> "FakeAudio":
        return self

    def cpu(self) -> "FakeAudio":
        return self

    def squeeze(self) -> "FakeAudio":
        squeezed = FakeAudio(1)
        squeezed._array = np.squeeze(self._array)
        return squeezed

    def numpy(self) -> np.ndarray:
        return self._array


class FakeModel:
    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        return self


class FakeRuntime:
    load_calls: list[tuple[str, dict]] = []
    generate_calls: list[dict] = []
    sample_rate = 48000

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        cls.load_calls.append((model_name_or_path, kwargs))
        return cls()

    def __init__(self) -> None:
        self.model = FakeModel()
        self.device = "cpu"

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return {
            "fid": "fake-request",
            "audio": FakeAudio(),
            "sample_rate": self.sample_rate,
            "time_used": 0.25,
            "rtf": 0.5,
            "profiling": None,
        }


@pytest.fixture(autouse=True)
def fake_dots_tts_modules(monkeypatch):
    FakeRuntime.load_calls = []
    FakeRuntime.generate_calls = []

    dots_module = types.ModuleType("dots_tts")
    runtime_module = types.ModuleType("dots_tts.runtime")
    runtime_module.DotsTtsRuntime = FakeRuntime
    utils_module = types.ModuleType("dots_tts.utils")
    util_module = types.ModuleType("dots_tts.utils.util")
    util_module.seed_everything = lambda seed: None

    monkeypatch.setitem(sys.modules, "dots_tts", dots_module)
    monkeypatch.setitem(sys.modules, "dots_tts.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "dots_tts.utils", utils_module)
    monkeypatch.setitem(sys.modules, "dots_tts.utils.util", util_module)


def test_normalize_request_requires_text() -> None:
    with pytest.raises(ValueError, match="Text to synthesize"):
        normalize_request(SynthesisRequest(text=" "))


def test_normalize_request_requires_audio_for_prompt_text() -> None:
    with pytest.raises(ValueError, match="Prompt transcript requires prompt audio"):
        normalize_request(SynthesisRequest(text="hello", prompt_text="reference words"))


def test_pytorch_runtime_rejects_quantization() -> None:
    with pytest.raises(ValueError, match="MLX-only"):
        normalize_runtime_config(RuntimeConfig(backend="pytorch", quantization="int8"))


def test_pytorch_runtime_accepts_torchao_quantization() -> None:
    config = normalize_runtime_config(RuntimeConfig(backend="pytorch", quantization="torchao-int8wo"))

    assert config.quantization == "torchao-int8wo"


def test_auto_device_skips_pytorch_mps_by_default(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS", raising=False)

    assert _select_device("auto") == "cpu"


def test_explicit_pytorch_mps_requires_override(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS", raising=False)

    with pytest.raises(RuntimeError, match="Use the MLX backend"):
        _select_device("mps")


def test_explicit_pytorch_mps_override(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS", "1")

    assert _select_device("mps") == "mps"


def test_mlx_runtime_requires_prompt_audio_and_transcript(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"fake")

    with pytest.raises(ValueError, match="MLX backend requires prompt audio"):
        normalize_request(SynthesisRequest(runtime=RuntimeConfig(backend="mlx"), text="hello"))

    with pytest.raises(ValueError, match="exact prompt transcript"):
        normalize_request(
            SynthesisRequest(runtime=RuntimeConfig(backend="mlx"), text="hello", prompt_audio_path=str(prompt))
        )

    request = normalize_request(
        SynthesisRequest(
            runtime=RuntimeConfig(backend="mlx", model_name_or_path="local-mlx"),
            text="hello",
            prompt_audio_path=str(prompt),
            prompt_text="reference words",
            language="EN",
        )
    )

    assert request.runtime.backend == "mlx"
    assert request.prompt_audio_path == str(prompt)


def test_mlx_runtime_rejects_unsupported_text_features(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"fake")
    base = {
        "runtime": RuntimeConfig(backend="mlx", model_name_or_path="local-mlx"),
        "text": "hello",
        "prompt_audio_path": str(prompt),
        "prompt_text": "reference words",
    }

    with pytest.raises(ValueError, match="text normalization"):
        normalize_request(SynthesisRequest(**base, normalize_text=True))

    with pytest.raises(ValueError, match="Auto detect language"):
        normalize_request(SynthesisRequest(**base, language="auto_detect"))


def test_mlx_generation_converts_non_wav_prompt_audio(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.mp3"
    sf.write(
        prompt,
        np.linspace(-0.1, 0.1, 800, dtype=np.float32),
        16000,
        format="WAV",
        subtype="PCM_16",
    )

    class FakeMlxRuntime:
        def __init__(self) -> None:
            self.prompt_audio: str | None = None

        def generate(self, **kwargs):
            self.prompt_audio = kwargs["prompt_audio"]
            prompt_path = Path(self.prompt_audio)
            assert prompt_path.suffix == ".wav"
            assert prompt_path.read_bytes().startswith(b"RIFF")
            return {
                "audio": np.linspace(-0.1, 0.1, 16, dtype=np.float32),
                "sample_rate": 48000,
                "fid": "mlx-fake",
            }

    runtime = FakeMlxRuntime()
    messages: list[str] = []
    result = DotsTtsService._generate_mlx(
        runtime,
        SynthesisRequest(
            runtime=RuntimeConfig(backend="mlx", model_name_or_path="local-mlx"),
            text="hello",
            prompt_audio_path=str(prompt),
            prompt_text="reference words",
            output_dir=tmp_path,
        ),
        messages.append,
    )

    assert result["fid"] == "mlx-fake"
    assert runtime.prompt_audio is not None
    assert not Path(runtime.prompt_audio).exists()
    assert any("temporary WAV" in message for message in messages)


def test_mlx_sampler_progress_uses_patch_format() -> None:
    messages: list[str] = []

    with _mlx_sampling_progress(object(), 32, messages.append):
        pass

    assert messages[0] == "SAMPLING_PATCH 0/1 0/32: MLX generation starting"
    assert messages[-1] == "SAMPLING_PATCH 1/1 32/32: MLX generation finished; finalizing output"


def test_mlx_sampler_progress_tracks_patch_and_vocoder() -> None:
    messages: list[str] = []

    class FakeDit:
        def __call__(self):
            return "dit"

    class FakeFlowSolver:
        def __init__(self) -> None:
            self.dit = FakeDit()

        def denoise(self):
            for _ in range(3):
                self.dit()
            return "patch"

        def meanflow_sample(self):
            return "meanflow"

    class FakeVae:
        def decode(self):
            return "wav"

    class FakeMlxRuntime:
        def __init__(self) -> None:
            self.flow_solver = FakeFlowSolver()
            self.vae = FakeVae()

    runtime = FakeMlxRuntime()
    with _mlx_sampling_progress(runtime, 3, messages.append, estimated_patch_total=2):
        runtime.flow_solver.denoise()
        runtime.vae.decode()

    assert "SAMPLING_PATCH 1/2 0/3: MLX flow patch starting" in messages
    assert "SAMPLING_PATCH 1/2 3/3: MLX sampler complete; encoding patch" in messages
    assert "SAMPLING_PATCH 1/2 3/3: MLX vocoder decoding audio" in messages
    assert messages[-1] == "SAMPLING_PATCH 1/2 3/3: MLX generation finished; finalizing output"


def test_synthesize_writes_wav_and_passes_runtime_kwargs(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"placeholder")
    service = DotsTtsService()
    request = SynthesisRequest(
        runtime=RuntimeConfig(model_name_or_path="rednote-hilab/dots.tts-soar"),
        text="hello world",
        prompt_audio_path=str(prompt),
        prompt_text="reference words",
        execution_mode="generate",
        language="EN",
        output_dir=tmp_path,
    )

    result = service.synthesize(request)

    assert result.audio_path.is_file()
    assert result.sample_rate == 48000
    assert result.metrics["request_id"] == "fake-request"
    assert result.metrics["device"] in {"cpu", "mps", "cuda"}
    assert FakeRuntime.load_calls[0][0] == "rednote-hilab/dots.tts-soar"
    assert FakeRuntime.load_calls[0][1]["precision"] in {"float16", "float32"}
    assert FakeRuntime.load_calls[0][1]["optimize"] is False
    assert FakeRuntime.load_calls[0][1]["max_generate_length"] == 500
    assert FakeRuntime.generate_calls[-1]["text"] == "hello world"
    assert FakeRuntime.generate_calls[-1]["prompt_audio_path"] == str(prompt)
    assert FakeRuntime.generate_calls[-1]["prompt_text"] == "reference words"
    assert FakeRuntime.generate_calls[-1]["language"] == "EN"


def test_synthesize_unloads_runtime_after_generation(tmp_path: Path) -> None:
    service = DotsTtsService()

    service.synthesize(
        SynthesisRequest(
            runtime=RuntimeConfig(unload_after_generation=True),
            text="hello",
            execution_mode="generate",
            output_dir=tmp_path,
        )
    )

    assert service._runtime is None
    assert service._runtime_key is None


def test_synthesize_reports_sampler_progress(monkeypatch, tmp_path: Path) -> None:
    core_module = types.ModuleType("dots_tts.models.dots_tts.core")

    def fake_odeint(func, *args, **kwargs):
        for _ in range(32):
            func(None)
        return None

    core_module.odeint = fake_odeint
    monkeypatch.setitem(sys.modules, "dots_tts.models", types.ModuleType("dots_tts.models"))
    monkeypatch.setitem(sys.modules, "dots_tts.models.dots_tts", types.ModuleType("dots_tts.models.dots_tts"))
    monkeypatch.setitem(sys.modules, "dots_tts.models.dots_tts.core", core_module)
    original_generate = FakeRuntime.generate

    def generate_with_sampler(self, **kwargs):
        import importlib

        core = importlib.import_module("dots_tts.models.dots_tts.core")
        core.odeint(lambda _x: None)
        return original_generate(self, **kwargs)

    monkeypatch.setattr(FakeRuntime, "generate", generate_with_sampler)
    messages: list[str] = []

    DotsTtsService().synthesize(
        SynthesisRequest(text="hello", execution_mode="generate", output_dir=tmp_path),
        progress=messages.append,
    )

    assert any(message.startswith("SAMPLING_PATCH 0/") and "euler sampler starting" in message for message in messages)
    assert any(" 32/32: euler sampler" in message for message in messages)
    assert any(message.startswith("SAMPLING_PATCH ") and "euler sampler finished" in message for message in messages)


def test_runtime_cache_reuses_matching_config(tmp_path: Path) -> None:
    service = DotsTtsService()
    request = SynthesisRequest(
        runtime=RuntimeConfig(unload_after_generation=False),
        text="hello",
        execution_mode="generate",
        output_dir=tmp_path,
    )

    service.synthesize(request)
    service.synthesize(request)

    assert len(FakeRuntime.load_calls) == 1


def test_runtime_cache_reloads_when_config_changes(tmp_path: Path) -> None:
    service = DotsTtsService()
    service.synthesize(
        SynthesisRequest(
            runtime=RuntimeConfig(precision="bfloat16", unload_after_generation=False),
            text="hello",
            execution_mode="generate",
            output_dir=tmp_path,
        )
    )
    service.synthesize(
        SynthesisRequest(
            runtime=RuntimeConfig(max_generate_length=256, unload_after_generation=False),
            text="hello",
            execution_mode="generate",
            output_dir=tmp_path,
        )
    )

    assert len(FakeRuntime.load_calls) == 2


def test_synthesize_reports_generation_stage_errors(monkeypatch, tmp_path: Path) -> None:
    def fail_generate(self, **kwargs):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(FakeRuntime, "generate", fail_generate)
    service = DotsTtsService()

    with pytest.raises(DotsTtsOperationError) as exc_info:
        service.synthesize(SynthesisRequest(text="hello", output_dir=tmp_path))

    assert exc_info.value.stage == "Generation"
    assert "simulated model failure" in str(exc_info.value)


def test_cleanup_outputs_keeps_newest_files(tmp_path: Path) -> None:
    files = []
    for index in range(4):
        path = tmp_path / f"{index}.wav"
        path.write_bytes(b"wav")
        timestamp = time.time() + index
        path.touch()
        path.chmod(0o644)
        # Some filesystems coalesce timestamps, so set them explicitly.
        import os

        os.utime(path, (timestamp, timestamp))
        files.append(path)

    cleanup_outputs(tmp_path, retention_count=2)

    remaining = sorted(path.name for path in tmp_path.glob("*.wav"))
    assert remaining == ["2.wav", "3.wav"]
