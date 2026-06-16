from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .constants import DEFAULT_MODEL_DIR, DEFAULT_MLX_MODEL, MODEL_CHOICES, MODEL_DESCRIPTIONS

ProgressCallback = Callable[[str], None]

REQUIRED_LOCAL_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "vocoder.safetensors",
    "speaker_encoder.safetensors",
)

REQUIRED_MLX_MODEL_FILES = (
    "config.json",
    "core.safetensors",
    "vocoder.safetensors",
    "speaker.safetensors",
    "llm_config.json",
    "latent_stats.npz",
)

MLX_MODEL_VARIANTS = ("int4", "int8", "mf-int4", "mf-int8")


@dataclass(frozen=True)
class ManagedModel:
    label: str
    repo_id: str
    description: str
    local_path: Path
    is_downloaded: bool
    size_bytes: int | None = None

    @property
    def display_size(self) -> str:
        if self.size_bytes is None:
            return "unknown size"
        return format_bytes(self.size_bytes)


def repo_id_to_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def local_model_path(repo_id: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    return Path(model_dir).expanduser() / repo_id_to_dir_name(repo_id)


def list_managed_models(model_dir: Path = DEFAULT_MODEL_DIR) -> list[ManagedModel]:
    return [
        ManagedModel(
            label=label,
            repo_id=repo_id,
            description=MODEL_DESCRIPTIONS.get(repo_id, ""),
            local_path=local_model_path(repo_id, model_dir=model_dir),
            is_downloaded=is_model_downloaded(repo_id, model_dir=model_dir),
        )
        for label, repo_id in MODEL_CHOICES
    ]


def is_model_downloaded(repo_id: str, model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
    return is_local_model_path_valid(local_model_path(repo_id, model_dir=model_dir), backend="pytorch")


def has_model_config(path: Path) -> bool:
    return (path / "config.json").is_file() or (path / "model_config.json").is_file()


def local_mlx_model_path(variant: str = "int4", model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    normalized = normalize_mlx_variant(variant)
    return local_model_path(DEFAULT_MLX_MODEL, model_dir=model_dir) / normalized


def normalize_mlx_variant(variant: str | None) -> str:
    normalized = (variant or "int4").strip()
    if normalized in {"none", ""}:
        return "int4"
    if normalized not in MLX_MODEL_VARIANTS:
        return "int4"
    return normalized


def is_local_model_path_valid(path: str | Path, *, backend: str = "pytorch") -> bool:
    model_path = Path(path).expanduser()
    if not model_path.is_dir():
        return False
    if backend == "mlx":
        return local_mlx_model_validation_error(model_path) is None
    if not _has_required_model_files(model_path):
        return False
    return _has_dots_tts_config(model_path)


def local_model_validation_error(path: str | Path, *, backend: str = "pytorch") -> str | None:
    model_path = Path(path).expanduser()
    if backend == "mlx":
        return local_mlx_model_validation_error(model_path)
    if not model_path.exists():
        return f"Folder does not exist: {model_path}"
    if not model_path.is_dir():
        return f"Path is not a folder: {model_path}"
    missing = [name for name in REQUIRED_LOCAL_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        return f"Missing dots.tts model files: {', '.join(missing)}"
    if not _has_dots_tts_config(model_path):
        return "config.json is not a dots.tts PyTorch model config."
    return None


def local_mlx_model_validation_error(path: str | Path) -> str | None:
    model_path = Path(path).expanduser()
    if not model_path.exists():
        return f"Folder does not exist: {model_path}"
    if not model_path.is_dir():
        return f"Path is not a folder: {model_path}"
    missing = [name for name in REQUIRED_MLX_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        return f"Missing MLX model files: {', '.join(missing)}"
    if not (model_path / "tokenizer").is_dir():
        return "MLX model folder must contain the tokenizer directory."
    return None


def _has_required_model_files(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_LOCAL_MODEL_FILES)


def _has_dots_tts_config(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("model_type") == "dots_tts" or "DotsTTSForConditionalGeneration" in data.get("architectures", [])


def get_remote_model_size(repo_id: str) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception:
        return None
    total = 0
    found_size = False
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        if size is not None:
            total += int(size)
            found_size = True
    return total if found_size else None


def download_model(
    repo_id: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    revision: str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    target_dir = local_model_path(repo_id, model_dir=model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"Downloading {repo_id} to {target_dir}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - import failures vary by environment.
        raise RuntimeError(
            "huggingface_hub is not installed. Install the application dependencies before downloading models."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
    )
    if not has_model_config(target_dir):
        raise RuntimeError(f"Downloaded snapshot does not look like a dots.tts model: {target_dir}")
    _emit(progress, f"Model ready: {target_dir}")
    return target_dir


def download_mlx_model(
    variant: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    revision: str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    normalized = normalize_mlx_variant(variant)
    target_root = local_model_path(DEFAULT_MLX_MODEL, model_dir=model_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"Downloading MLX {normalized} weights from {DEFAULT_MLX_MODEL} to {target_root}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - import failures vary by environment.
        raise RuntimeError(
            "huggingface_hub is not installed. Install the application dependencies before downloading MLX weights."
        ) from exc

    snapshot_download(
        repo_id=DEFAULT_MLX_MODEL,
        revision=revision,
        local_dir=str(target_root),
        allow_patterns=[f"{normalized}/*", f"{normalized}/**/*"],
    )
    model_path = target_root / normalized
    validation_error = local_mlx_model_validation_error(model_path)
    if validation_error is not None:
        raise RuntimeError(f"Downloaded MLX snapshot is incomplete: {validation_error}")
    _emit(progress, f"MLX model ready: {model_path}")
    return model_path


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
