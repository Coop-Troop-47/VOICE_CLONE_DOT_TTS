from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QScrollArea, QToolButton

from voice_clone_dot_tts.constants import DEFAULT_MODEL
from voice_clone_dot_tts.ui import MainWindow

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def _make_valid_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "dots_tts"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fake")
    (path / "vocoder.safetensors").write_bytes(b"fake")
    (path / "speaker_encoder.safetensors").write_bytes(b"fake")
    return path


def _make_valid_mlx_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "core.safetensors", "vocoder.safetensors", "speaker.safetensors", "llm_config.json"):
        (path / name).write_bytes(b"fake")
    (path / "latent_stats.npz").write_bytes(b"fake")
    (path / "tokenizer").mkdir()
    return path


def test_ui_defaults_to_fixed_downloaded_soar_model(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        assert DEFAULT_MODEL in window.model_name_label.text()
        assert window.advanced_group.isHidden()
        window.advanced_toggle_button.setChecked(True)
        assert not window.advanced_group.isHidden()
        assert window.num_steps_spin.isEnabled()
        assert window.generate_button.isEnabled()
        assert window.findChild(QScrollArea) is not None

        window.text_edit.setPlainText("hello")
        request = window._build_request()

        assert request.runtime.model_name_or_path == str(fixed_path)
        assert request.runtime.model_name_or_path != DEFAULT_MODEL
    finally:
        window.close()


def test_ui_blocks_generation_until_fixed_model_is_downloaded(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = tmp_path / "rednote-hilab__dots.tts-soar"
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        assert not window.generate_button.isEnabled()
        assert "Not ready" in window.model_status_label.text()
    finally:
        window.close()


def test_ui_can_use_user_selected_local_model(monkeypatch, tmp_path: Path) -> None:
    _app()
    default_path = tmp_path / "rednote-hilab__dots.tts-soar"
    selected_path = _make_valid_model_dir(tmp_path / "other-download" / "dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: default_path)

    window = MainWindow()
    try:
        assert not window.generate_button.isEnabled()
        window.model_path_edit.setText(str(selected_path))
        assert window.generate_button.isEnabled()

        request = window._build_request()

        assert request.runtime.model_name_or_path == str(selected_path)
    finally:
        window.close()


def test_ui_step_counter_tracks_runtime_messages(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        window._append_log("STEP 4/6: Generating audio with 32 sampling steps")

        assert window.step_counter_label.text() == "Step 4/6: Generating audio with 32 sampling steps"
        assert window.step_progress.value() == 66

        window._append_log("SAMPLING 16/32: euler sampler")

        assert window.sampler_counter_label.text() == "Sampler 16/32: euler sampler"
        assert window.sampler_progress.value() == 50

        window._append_log("SAMPLING_PATCH 3/8 8/32: euler sampler")

        assert window.sampler_counter_label.text() == "Audio patch 3/8 estimated, sampler 8/32: euler sampler"
        assert window.sampler_progress.value() == 25
        assert window.eta_label.text().startswith("ETA:")

        window._append_log("SAMPLING 16/32: MLX sampler")

        assert window.sampler_counter_label.text() == "Audio patch 1/1 estimated, sampler 16/32: MLX sampler"
        assert "waiting for sampler progress" not in window.eta_label.text()
    finally:
        window.close()


def test_ui_switches_to_mlx_quantized_model(monkeypatch, tmp_path: Path) -> None:
    _app()
    pytorch_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    mlx_root = tmp_path / "shraey__dots-tts-mlx"
    _make_valid_mlx_model_dir(mlx_root / "int4")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: pytorch_path)
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_mlx_model_path", lambda variant: mlx_root / variant)

    window = MainWindow()
    try:
        window.backend_combo.setCurrentIndex(window.backend_combo.findData("mlx"))

        assert window.quantization_combo.isEnabled()
        assert window.quantization_combo.currentData() == "int4"
        assert not window.device_combo.isEnabled()
        assert not window.optimize_check.isEnabled()
        assert not window.streaming_check.isEnabled()
        assert window.model_path_edit.text() == str(mlx_root / "int4")
        assert "MLX" in window.model_name_label.text()
        assert "Ready" in window.model_status_label.text()

        request = window._build_request()

        assert request.runtime.backend == "mlx"
        assert request.runtime.quantization == "int4"
        assert request.runtime.model_name_or_path == str(mlx_root / "int4")
    finally:
        window.close()


def test_ui_exposes_pytorch_torchao_quantization(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        assert window.backend_combo.currentData() == "pytorch"
        assert window.quantization_combo.findData("torchao-int8wo") >= 0
        assert window.quantization_combo.findData("torchao-int4wo") >= 0

        window.quantization_combo.setCurrentIndex(window.quantization_combo.findData("torchao-int8wo"))
        request = window._build_request()

        assert request.runtime.quantization == "torchao-int8wo"
    finally:
        window.close()


def test_ui_has_visible_help_for_options(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        window.advanced_toggle_button.setChecked(True)
        help_buttons = [
            button
            for button in window.findChildren(QToolButton)
            if button.text() == "?" and button.toolTip().strip()
        ]

        assert len(help_buttons) >= 25
        assert window.model_path_edit.toolTip()
        assert window.prompt_audio_edit.toolTip()
        assert window.text_edit.toolTip()
        assert window.generate_button.toolTip()
    finally:
        window.close()


def test_ui_layout_does_not_overflow_horizontally_when_small(monkeypatch, tmp_path: Path) -> None:
    app = _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        window.resize(760, 560)
        window.advanced_toggle_button.setChecked(True)
        window.show()
        app.processEvents()

        for scroll in window.findChildren(QScrollArea):
            assert scroll.horizontalScrollBar().maximum() == 0

        for button in window.findChildren(QToolButton):
            if not button.isVisible():
                continue
            top_left = button.mapTo(window, button.rect().topLeft())
            bottom_right = button.mapTo(window, button.rect().bottomRight())
            assert top_left.x() >= 0
            assert bottom_right.x() <= window.width()
    finally:
        window.close()


def test_ui_guide_tab_and_menu_action(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        assert window.tabs.indexOf(window.guide_edit) >= 0
        assert "rednote-hilab/dots.tts-soar" in window.guide_edit.toPlainText()

        window.tabs.setCurrentWidget(window.log_edit)
        window._open_guide()

        assert window.tabs.currentWidget() is window.guide_edit
    finally:
        window.close()


def test_ui_diagnostics_log_local_state(monkeypatch, tmp_path: Path) -> None:
    _app()
    fixed_path = _make_valid_model_dir(tmp_path / "rednote-hilab__dots.tts-soar")
    monkeypatch.setattr("voice_clone_dot_tts.ui.local_model_path", lambda repo_id: fixed_path)

    window = MainWindow()
    try:
        window._append_diagnostics()
        log = window.log_edit.toPlainText()

        assert "Diagnostics" in log
        assert "Model ready: True" in log
        assert "Output parent writable:" in log
        assert "Python:" in log
    finally:
        window.close()
