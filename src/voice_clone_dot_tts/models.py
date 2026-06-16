from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .constants import (
    DEFAULT_DEVICE,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_NUM_STEPS,
    DEFAULT_ODE_METHOD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PRECISION,
    DEFAULT_BACKEND,
    DEFAULT_QUANTIZATION,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
    DEFAULT_TEMPLATE_NAME,
    DEFAULT_UNLOAD_AFTER_GENERATION,
)

ExecutionMode = Literal["generate", "generate_stream"]
Backend = Literal["pytorch", "mlx"]


@dataclass(frozen=True)
class RuntimeConfig:
    backend: Backend = DEFAULT_BACKEND
    model_name_or_path: str = DEFAULT_MODEL
    revision: str | None = None
    cache_dir: str | None = None
    device: str = DEFAULT_DEVICE
    precision: str = DEFAULT_PRECISION
    quantization: str = DEFAULT_QUANTIZATION
    optimize: bool = False
    max_generate_length: int = DEFAULT_MAX_GENERATE_LENGTH
    unload_after_generation: bool = DEFAULT_UNLOAD_AFTER_GENERATION


@dataclass(frozen=True)
class SynthesisRequest:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    text: str = ""
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    execution_mode: ExecutionMode = DEFAULT_EXECUTION_MODE
    template_name: str = DEFAULT_TEMPLATE_NAME
    language: str | None = None
    ode_method: str = DEFAULT_ODE_METHOD
    num_steps: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    speaker_scale: float = DEFAULT_SPEAKER_SCALE
    normalize_text: bool = False
    profile_inference: bool = False
    seed: int = DEFAULT_SEED
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_retention_count: int = DEFAULT_OUTPUT_RETENTION


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: Path
    sample_rate: int
    samples: int
    metrics: dict[str, Any]
