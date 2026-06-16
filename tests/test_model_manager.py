from __future__ import annotations

from pathlib import Path

from voice_clone_dot_tts.constants import DEFAULT_MODEL
from voice_clone_dot_tts.model_manager import (
    format_bytes,
    is_model_downloaded,
    is_local_model_path_valid,
    list_managed_models,
    local_mlx_model_path,
    local_model_path,
    local_model_validation_error,
    normalize_mlx_variant,
    repo_id_to_dir_name,
)


def test_repo_id_to_dir_name_is_path_safe() -> None:
    assert repo_id_to_dir_name("rednote-hilab/dots.tts-mf") == "rednote-hilab__dots.tts-mf"


def test_local_model_path_uses_configured_model_dir(tmp_path: Path) -> None:
    assert local_model_path(DEFAULT_MODEL, model_dir=tmp_path) == (tmp_path / "rednote-hilab__dots.tts-mf")


def test_is_model_downloaded_requires_config_file(tmp_path: Path) -> None:
    model_path = local_model_path(DEFAULT_MODEL, model_dir=tmp_path)
    model_path.mkdir(parents=True)
    (model_path / "weights.safetensors").write_bytes(b"fake")

    assert not is_model_downloaded(DEFAULT_MODEL, model_dir=tmp_path)

    (model_path / "config.json").write_text('{"model_type": "dots_tts"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"fake")
    (model_path / "vocoder.safetensors").write_bytes(b"fake")
    (model_path / "speaker_encoder.safetensors").write_bytes(b"fake")
    assert is_model_downloaded(DEFAULT_MODEL, model_dir=tmp_path)


def test_local_model_validation_reports_missing_files(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "dots_tts"}', encoding="utf-8")

    assert not is_local_model_path_valid(model_path)
    assert "Missing dots.tts model files" in str(local_model_validation_error(model_path))


def test_local_mlx_model_validation_requires_converted_variant_files(tmp_path: Path) -> None:
    model_path = local_mlx_model_path("int8", model_dir=tmp_path)
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    assert not is_local_model_path_valid(model_path, backend="mlx")
    assert "Missing MLX model files" in str(local_model_validation_error(model_path, backend="mlx"))

    for name in ("core.safetensors", "vocoder.safetensors", "speaker.safetensors", "llm_config.json"):
        (model_path / name).write_bytes(b"fake")
    (model_path / "latent_stats.npz").write_bytes(b"fake")

    assert not is_local_model_path_valid(model_path, backend="mlx")
    assert "tokenizer directory" in str(local_model_validation_error(model_path, backend="mlx"))

    (model_path / "tokenizer").mkdir()
    assert is_local_model_path_valid(model_path, backend="mlx")


def test_normalize_mlx_variant_defaults_to_int4() -> None:
    assert normalize_mlx_variant(None) == "int4"
    assert normalize_mlx_variant("none") == "int4"
    assert normalize_mlx_variant("unknown") == "int4"
    assert normalize_mlx_variant("mf-int8") == "mf-int8"


def test_list_managed_models_reports_known_models(tmp_path: Path) -> None:
    models = list_managed_models(model_dir=tmp_path)

    assert [model.repo_id for model in models] == ["rednote-hilab/dots.tts-mf", "rednote-hilab/dots.tts-soar"]
    assert all(str(tmp_path) in str(model.local_path) for model in models)


def test_format_bytes() -> None:
    assert format_bytes(12) == "12 B"
    assert format_bytes(1536) == "1.5 KB"
