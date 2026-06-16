from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Voice Clone dots.tts"


def _default_data_dir() -> Path:
    override = os.environ.get("VOICE_CLONE_DOT_TTS_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "voice-clone-dot-tts"


DATA_DIR = _default_data_dir()
DEFAULT_MODEL_DIR = DATA_DIR / "models"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs"
DEFAULT_PROMPTS_DIR = DATA_DIR / "prompts"

DEFAULT_SOAR_MODEL = "rednote-hilab/dots.tts-soar"
DEFAULT_MF_MODEL = "rednote-hilab/dots.tts-mf"
DEFAULT_MODEL = DEFAULT_MF_MODEL
DEFAULT_MODEL_LABEL = "dots.tts MeanFlow - PyTorch (consumer GPU)"
DEFAULT_MLX_MODEL = "shraey/dots-tts-mlx"
DEFAULT_MLX_MODEL_LABEL = "dots.tts SOAR - MLX int4"
DEFAULT_BACKEND = "pytorch"
DEFAULT_DEVICE = "auto"
DEFAULT_PRECISION = "float16"
DEFAULT_QUANTIZATION = "none"
DEFAULT_EXECUTION_MODE = "generate"
DEFAULT_TEMPLATE_NAME = "tts"
DEFAULT_ODE_METHOD = "euler"
DEFAULT_NUM_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 1.2
DEFAULT_SPEAKER_SCALE = 1.5
DEFAULT_MAX_GENERATE_LENGTH = 500
DEFAULT_SEED = 42
DEFAULT_OUTPUT_RETENTION = 20
DEFAULT_UNLOAD_AFTER_GENERATION = True

MODEL_CHOICES = (
    (DEFAULT_MODEL_LABEL, DEFAULT_MF_MODEL),
    ("dots.tts SOAR - PyTorch quality", DEFAULT_SOAR_MODEL),
)

PYTORCH_MODEL_CHOICES = (
    (DEFAULT_MODEL_LABEL, DEFAULT_MF_MODEL),
    ("dots.tts SOAR - PyTorch quality", DEFAULT_SOAR_MODEL),
)

MODEL_DESCRIPTIONS = {
    DEFAULT_MF_MODEL: "Official MeanFlow checkpoint distilled from SOAR. Recommended first choice for Windows consumer GPUs.",
    DEFAULT_SOAR_MODEL: "Highest-quality PyTorch dots.tts voice cloning checkpoint. Recommended when enough VRAM/RAM is available.",
}

TEMPLATE_CHOICES = (
    ("TTS", "tts"),
    ("Instruction TTS", "instruction_tts"),
    ("Text to Audio", "text_to_audio"),
    ("Interleaved TTS", "tts_interleave"),
)

PRECISION_CHOICES = ("bfloat16", "float16", "float32")
BACKEND_CHOICES = (
    ("PyTorch / Windows NVIDIA or CPU", "pytorch"),
    ("MLX / Apple Silicon only", "mlx"),
)
PYTORCH_QUANTIZATION_CHOICES = (
    ("None / recommended for MeanFlow", "none"),
    ("Experimental torchao int8 weight-only", "torchao-int8wo"),
    ("Experimental torchao int4 weight-only", "torchao-int4wo"),
)
MLX_QUANTIZATION_CHOICES = (
    ("MLX int4 - lowest memory", "int4"),
    ("MLX int8 - conservative quantized", "int8"),
    ("MLX mf-int4 - faster MeanFlow", "mf-int4"),
    ("MLX mf-int8 - faster conservative", "mf-int8"),
)
QUANTIZATION_CHOICES = PYTORCH_QUANTIZATION_CHOICES + MLX_QUANTIZATION_CHOICES
PYTORCH_QUANTIZATION_VALUES = tuple(value for _, value in PYTORCH_QUANTIZATION_CHOICES)
MLX_QUANTIZATION_VALUES = tuple(value for _, value in MLX_QUANTIZATION_CHOICES)
DEVICE_CHOICES = (
    ("Auto - use NVIDIA GPU when available", "auto"),
    ("Force NVIDIA GPU (CUDA)", "cuda"),
    ("Apple MPS (unsafe override)", "mps"),
    ("Force CPU", "cpu"),
)
ODE_METHOD_CHOICES = ("euler", "midpoint", "rk4")

LANGUAGE_CHOICES = (
    ("None", ""),
    ("Auto detect", "auto_detect"),
    ("English", "EN"),
    ("Mandarin", "ZH"),
    ("Cantonese", "Cantonese"),
    ("Spanish", "ES"),
    ("French", "FR"),
    ("German", "DE"),
    ("Japanese", "JA"),
    ("Korean", "KO"),
    ("Portuguese", "PT"),
    ("Russian", "RU"),
    ("Italian", "IT"),
    ("Arabic", "AR"),
    ("Hindi", "HI"),
)

SUPPORTED_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
