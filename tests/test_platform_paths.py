from __future__ import annotations

from pathlib import Path

from voice_clone_dot_tts import constants


def test_default_data_dir_uses_windows_localappdata(monkeypatch) -> None:
    monkeypatch.delenv("VOICE_CLONE_DOT_TTS_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")
    monkeypatch.setattr(constants.sys, "platform", "win32")

    assert constants._default_data_dir() == Path(r"C:\Users\Example\AppData\Local") / constants.APP_NAME


def test_default_data_dir_override_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOICE_CLONE_DOT_TTS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(constants.sys, "platform", "win32")

    assert constants._default_data_dir() == tmp_path
