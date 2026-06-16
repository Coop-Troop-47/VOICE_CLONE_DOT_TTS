from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
import traceback
from html import escape
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QUrl, Signal
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    BACKEND_CHOICES,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_MAX_GENERATE_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_MODEL_LABEL,
    DEFAULT_MLX_MODEL,
    DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_RETENTION,
    DEFAULT_PRECISION,
    DEFAULT_BACKEND,
    DEFAULT_QUANTIZATION,
    DEFAULT_SEED,
    DEFAULT_SPEAKER_SCALE,
    DEFAULT_UNLOAD_AFTER_GENERATION,
    DEVICE_CHOICES,
    LANGUAGE_CHOICES,
    MLX_QUANTIZATION_CHOICES,
    MLX_QUANTIZATION_VALUES,
    ODE_METHOD_CHOICES,
    PRECISION_CHOICES,
    PYTORCH_MODEL_CHOICES,
    PYTORCH_QUANTIZATION_CHOICES,
    PYTORCH_QUANTIZATION_VALUES,
    TEMPLATE_CHOICES,
)
from .model_manager import (
    download_model,
    download_mlx_model,
    is_local_model_path_valid,
    local_mlx_model_path,
    local_model_path,
    local_model_validation_error,
)
from .models import RuntimeConfig, SynthesisRequest, SynthesisResult
from .service import DotsTtsService


HELP_TEXT = {
    "fixed_model": (
        "The selected checkpoint controls the speed, memory, and quality profile. MeanFlow is the practical "
        "Windows starting point because it is distilled to produce useful speech with far fewer sampler steps. "
        "That reduces wait time and peak working memory on cards such as an RTX 3070. SOAR remains available "
        "for quality-first runs, but it normally needs more steps and more temporary memory during generation."
    ),
    "pytorch_model": (
        "Choose which official PyTorch checkpoint to run. MeanFlow is the consumer-hardware option: it trades a "
        "small amount of maximum quality headroom for much lower latency and lower sampler-step counts. SOAR is "
        "the full quality model and is better for final renders when the machine has enough VRAM/RAM and longer "
        "generation time is acceptable."
    ),
    "model_folder": (
        "Local folder containing the downloaded dots.tts files. PyTorch folders must include config.json, "
        "model.safetensors, vocoder.safetensors, and speaker_encoder.safetensors. MLX folders use a converted "
        "variant directory with core/vocoder/speaker weights and a tokenizer. Use Download for the managed path "
        "or Browse when a compatible Hugging Face snapshot already exists on disk."
    ),
    "download_model": (
        "Downloads the selected checkpoint into this app's user data directory. The app package never contains "
        "multi-gigabyte model weights, so release builds stay small and users can replace or delete model folders "
        "without reinstalling the app. The bottom progress bar shows the active download state."
    ),
    "default_model_path": "Restores the standard per-user model directory for this app.",
    "refresh_model": "Rechecks whether the selected local model folder is complete and valid.",
    "model_status": (
        "Shows whether generation is currently allowed. If the folder is incomplete, this tells which required "
        "file or directory is missing so the fix is clear before loading the model."
    ),
    "backend": (
        "Runtime backend. On Windows, use PyTorch with CUDA for NVIDIA GPUs such as RTX 3070, or force CPU only "
        "when GPU memory is unavailable. MLX is only for Apple Silicon Macs and uses separate converted weights."
    ),
    "quantization": (
        "Memory mode for the selected backend. For Windows PyTorch, the recommended consumer path is MeanFlow with "
        "CUDA and fewer sampler steps. The torchao int8/int4 modes are experimental runtime quantization paths; "
        "they are not separate downloaded checkpoints and may fail on unsupported layers or driver combinations. "
        "For Apple Silicon MLX, int4/int8 are converted low-memory checkpoints."
    ),
    "unload_after_generation": (
        "Lower-memory mode. When enabled, the app unloads the model and clears CUDA/MPS/MLX caches after each output. "
        "This prevents memory growth across runs and swap pressure, but the next generation must reload the model."
    ),
    "memory_estimate": (
        "Memory estimate for the selected setup. The model file is only part of the footprint: generation also needs "
        "the vocoder, speaker encoder, text state, audio latents, and CUDA allocator memory. Shorter text, MeanFlow, "
        "Euler sampling, fewer steps, and unloading after each generation reduce pressure on consumer GPUs."
    ),
    "output_dir": "Folder where generated WAV files are written. The app creates it if it does not exist.",
    "retention": (
        "Maximum number of generated WAV files to keep in the output folder. Set 0 to keep every output. Cleanup "
        "only removes older WAVs created in the output folder; it does not touch model files or reference audio."
    ),
    "revision": (
        "Optional Hugging Face revision, branch, tag, or commit to download. Leave blank for the repository default. "
        "Use this only when testing a pinned model snapshot or reproducing a specific release."
    ),
    "cache_dir": (
        "Optional Hugging Face cache directory. Leave blank for normal use. Set this when a machine has a separate "
        "large drive for model cache data or when several local projects should share the same downloaded blobs."
    ),
    "precision": (
        "Requested runtime precision. PyTorch float16 reduces model-weight memory on CUDA and is the app default for "
        "lower RAM. bfloat16 is often a good CUDA choice and matches the upstream optimized example. float32 is the "
        "compatibility fallback and uses the most memory. CPU generation is forced to float32. PyTorch MPS is disabled "
        "by default because it can hard-crash this model; MLX uses this dtype when loading converted weights, but its "
        "int4/int8 setting is the larger memory lever."
    ),
    "device": (
        "PyTorch device. Auto uses an NVIDIA CUDA GPU when PyTorch can see one, then falls back to CPU. Force NVIDIA "
        "GPU is useful when a CUDA-capable card is expected and CPU fallback would be too slow. Force CPU is useful "
        "for troubleshooting or machines without enough VRAM."
    ),
    "optimize": (
        "Enables torch.compile warmup when supported. Upstream notes that optimize makes first launch slower "
        "but can improve steady-state generation speed."
    ),
    "max_length": "Upper bound for generated token/audio length. Higher values allow longer outputs but use more memory.",
    "execution": (
        "generate is the normal high-quality path. generate_stream yields audio chunks with the same generation "
        "arguments and is mainly useful for lower-latency playback/client streaming."
    ),
    "template": (
        "dots.tts prompt template. TTS is the normal voice cloning path. The other template is exposed for upstream "
        "compatibility and experimentation, but normal cloned-speech generation should stay on TTS."
    ),
    "language": (
        "Optional language tag. Upstream supports none, auto_detect, language codes like EN/ZH, and names like "
        "english/chinese. Useful for multilingual or code-switched text."
    ),
    "normalize": (
        "Runs WeTextProcessing text normalization before inference. This can improve numbers, dates, punctuation, "
        "and mixed-language text, but it adds the pynini/OpenFst dependency on Windows and is not supported by the "
        "MLX backend path."
    ),
    "ode": (
        "PyTorch ODE sampler. Euler is the default and is the best speed/memory baseline: one DiT evaluation per step. "
        "Midpoint evaluates about twice per step, which can smooth the trajectory but costs roughly 2x sampler work. "
        "RK4 evaluates about four times per step and is the heaviest option; it may sound different, but it is usually "
        "not the right first choice on low-memory machines. MLX converted checkpoints use their own Euler/MeanFlow "
        "solver path, so this control is mainly for PyTorch SOAR."
    ),
    "num_steps": (
        "Sampler steps control how much refinement happens before the WAV is written. MeanFlow is designed for low "
        "step counts, so 4-8 is a good starting point. SOAR usually benefits from 16-32 steps but takes longer and "
        "uses more working memory."
    ),
    "guidance": (
        "Classifier-free guidance scale. Upstream default is 1.2 and warns that values above 2 progressively "
        "amplify audio energy. Increase cautiously."
    ),
    "speaker": (
        "Voice cloning strength. Higher values push the generated speech closer to the reference speaker. Very high "
        "values can overfit the prompt and may create artifacts, unstable tone, or less natural pronunciation."
    ),
    "seed": (
        "Deterministic RNG seed. Fixed seed gives repeatable output; changing the seed explores different rhythm "
        "and intonation for the same text/reference."
    ),
    "profiling": "Collects runtime timing details in the Metrics tab. Leave off for normal use.",
    "prompt_audio": (
        "Reference voice audio for cloning. Upstream recommends about 10 seconds; longer audio will not "
        "necessarily improve results. Prefer high sample rate, one speaker, low background noise, no trailing "
        "noise, and natural speech. The PyTorch backend accepts common audio files through the upstream runtime. "
        "The MLX backend reads WAV internally, so the app automatically converts MP3, FLAC, M4A, and OGG references "
        "to a temporary PCM WAV before generation."
    ),
    "prompt_text": (
        "Exact transcript of the reference audio. Upstream recommends prompt audio plus matching prompt text "
        "for best speaker similarity. Transcript mismatches degrade stability and can cause word-level errors."
    ),
    "synthesis_text": (
        "The text to speak in the cloned voice. Short tests are best while tuning model, device, precision, and "
        "steps. Longer text creates more latent audio patches and increases total generation time."
    ),
    "generate": (
        "Starts synthesis after validating the model folder, speech text, optional reference audio, and output "
        "folder. Progress is reported in the step bar, sampler bar, ETA label, and Log tab. If generation fails, "
        "the progress indicators reset and the error remains visible."
    ),
    "sampler_progress": (
        "Shows audio-patch and sampler progress. dots.tts generates one latent audio patch at a time, and each patch "
        "runs its own flow-matching solve, so a single WAV can legitimately show Patch 1 sampler 1/32, Patch 2 sampler "
        "1/32, and so on. The patch total is estimated from text length and model patch duration, then adjusted upward "
        "if generation exceeds the estimate. ETA is calculated from the average time per observed sampler step."
    ),
    "play": "Plays the most recently generated WAV inside the app. The generated-audio scrubber can seek within the output.",
    "audio_scrubber": (
        "Playback and scrubbing for the selected audio file. Use Play/Pause to preview, drag the slider to seek, "
        "and Stop to return to the beginning. The reference control previews input audio; the generated control "
        "previews the latest output WAV."
    ),
    "open_output": "Opens the output folder in Finder, Explorer, or the platform file manager.",
    "guide": "Opens the built-in guide tab with setup, cloning, quality, memory, and troubleshooting notes.",
    "diagnostics": "Writes local app, model, output, and device diagnostics to the Log tab without loading the model.",
}

GUIDE_SECTIONS = (
    (
        "Start",
        (
            "On Windows consumer GPUs, start with the MeanFlow PyTorch checkpoint and Auto or Force NVIDIA GPU.",
            "Download the selected checkpoint or browse to an existing compatible local folder.",
            "Choose a clean reference voice clip and enter its exact transcript when available.",
            "Enter the speech text, keep the default MeanFlow settings, then press Generate.",
        ),
    ),
    (
        "Model Rules",
        (
            "This app supports the official PyTorch dots.tts MeanFlow and SOAR checkpoints, plus Apple Silicon MLX converted weights.",
            "The packaged app does not include model weights; this keeps the app distributable and lets users download the model on demand.",
            "A valid PyTorch model folder must contain config.json, model.safetensors, vocoder.safetensors, and speaker_encoder.safetensors.",
            "A valid MLX model folder must contain the selected int4/int8/mf variant files plus its tokenizer directory.",
            "On Windows, PyTorch CUDA is the supported GPU route. On Mac, MLX is the supported GPU route.",
        ),
    ),
    (
        "Quality Defaults",
        (
            "The default PyTorch path is MeanFlow, non-streaming generation, 8 sampling steps, and float16 on CUDA.",
            "Auto device selection uses CUDA first, then CPU. It skips Apple MPS unless VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS=1 is set.",
            "Increasing sampling steps can improve stability but increases generation time. Use SOAR with 16-32 steps for quality-first runs.",
        ),
    ),
    (
        "Reference Voice",
        (
            "Use one clean speaker, low background noise, and a natural speaking clip.",
            "The prompt transcript should match the reference audio exactly when possible.",
            "Prompt-audio-only cloning is allowed, but a transcript is usually easier to debug.",
        ),
    ),
    (
        "Memory",
        (
            "Runtime memory can exceed the model file size because tensors, caches, text state, latents, and temporary buffers are allocated during generation.",
            "Unload model after each generation is enabled by default to stop memory growth across repeated runs.",
            "Use MLX int4 or int8 on Apple Silicon when PyTorch uses too much RAM or when you need Mac GPU acceleration.",
            "Use Runtime > Unload Runtime to manually release the loaded model from the app process.",
        ),
    ),
    (
        "Troubleshooting",
        (
            "Use Help > Run Diagnostics to log platform, model, output, and device readiness.",
            "If synthesis fails, the dialog shows the short error and the Log tab keeps the traceback.",
            "If a packaged app reports import errors, rebuild and rerun the packaged import check before testing synthesis.",
        ),
    ),
)


def create_app(argv: list[str]) -> QApplication:
    _configure_qt_font_dir()
    app = QApplication(argv)
    app.setApplicationName("Voice Clone dots.tts")
    app.setOrganizationName("Voice Clone dots.tts")
    return app


def _configure_qt_font_dir() -> None:
    if os.environ.get("QT_QPA_FONTDIR"):
        return
    if not sys.platform.startswith("win"):
        return
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    font_dir = windows_dir / "Fonts"
    if font_dir.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(font_dir)


class SynthesisWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, service: DotsTtsService, request: SynthesisRequest) -> None:
        super().__init__()
        self._service = service
        self._request = request

    def run(self) -> None:
        try:
            result = self._service.synthesize(self._request, self.progress.emit)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.succeeded.emit(result)


class ModelDownloadWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str, str)

    def __init__(self, repo_id: str, revision: str | None, *, backend: str = "pytorch", quantization: str = "none") -> None:
        super().__init__()
        self._repo_id = repo_id
        self._revision = revision
        self._backend = backend
        self._quantization = quantization

    def run(self) -> None:
        try:
            if self._backend == "mlx":
                path = download_mlx_model(
                    self._quantization,
                    revision=self._revision,
                    progress=self.progress.emit,
                )
            else:
                path = download_model(
                    self._repo_id,
                    revision=self._revision,
                    progress=self.progress.emit,
                )
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.succeeded.emit(str(path))


class AudioPlaybackControl(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._duration_ms = 0
        self._seeking = False
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.9)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setWordWrap(True)
        attach_help(self.title_label, "audio_scrubber")
        self.play_button = QPushButton("Play")
        attach_help(self.play_button, "audio_scrubber")
        self.play_button.clicked.connect(self._toggle_playback)
        self.stop_button = QPushButton("Stop")
        attach_help(self.stop_button, "audio_scrubber")
        self.stop_button.clicked.connect(self.stop)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        attach_help(self.position_slider, "audio_scrubber")
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(72)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        attach_help(self.time_label, "audio_scrubber")

        layout.addWidget(self.title_label, 0, 0, 1, 3)
        layout.addWidget(self.play_button, 1, 0)
        layout.addWidget(self.stop_button, 1, 1)
        layout.addWidget(self.time_label, 1, 2)
        layout.addWidget(self.position_slider, 2, 0, 1, 3)
        layout.setColumnStretch(2, 1)

        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderReleased.connect(self._finish_seek)
        self.position_slider.sliderMoved.connect(self._preview_seek)
        self._player.durationChanged.connect(self._duration_changed)
        self._player.positionChanged.connect(self._position_changed)
        self._player.playbackStateChanged.connect(self._playback_state_changed)
        self.set_source(None)

    def set_source(self, path: str | Path | None) -> None:
        candidate = Path(path).expanduser() if path else None
        if candidate is None or not candidate.is_file():
            self._path = None
            self._duration_ms = 0
            self._player.stop()
            self._player.setSource(QUrl())
            self.position_slider.setRange(0, 0)
            self.position_slider.setValue(0)
            self.position_slider.setEnabled(False)
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.play_button.setText("Play")
            self.time_label.setText("0:00 / 0:00")
            return
        self._path = candidate
        self._duration_ms = 0
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(candidate)))
        self.position_slider.setRange(0, 0)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.play_button.setText("Play")
        self.time_label.setText("0:00 / 0:00")

    def stop(self) -> None:
        self._player.stop()
        self._player.setPosition(0)
        self.position_slider.setValue(0)

    def play(self) -> None:
        if self._path is not None:
            self._player.play()

    def _toggle_playback(self) -> None:
        if self._path is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _begin_seek(self) -> None:
        self._seeking = True

    def _finish_seek(self) -> None:
        self._seeking = False
        self._player.setPosition(self.position_slider.value())

    def _preview_seek(self, value: int) -> None:
        self.time_label.setText(f"{format_media_time(value)} / {format_media_time(self._duration_ms)}")

    def _duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = max(0, int(duration_ms))
        self.position_slider.setRange(0, self._duration_ms)
        self.position_slider.setEnabled(self._path is not None and self._duration_ms > 0)
        self.time_label.setText(f"{format_media_time(self._player.position())} / {format_media_time(self._duration_ms)}")

    def _position_changed(self, position_ms: int) -> None:
        if not self._seeking:
            self.position_slider.setValue(max(0, int(position_ms)))
        self.time_label.setText(f"{format_media_time(position_ms)} / {format_media_time(self._duration_ms)}")

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.play_button.setText("Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play")


def help_text(key: str) -> str:
    return HELP_TEXT.get(key, "")


def attach_help(widget: QWidget, key: str) -> None:
    text = help_text(key)
    if not text:
        return
    widget.setToolTip(text)
    widget.setAccessibleDescription(text)


def help_button(parent: QWidget, title: str, key: str) -> QToolButton:
    button = QToolButton(parent)
    button.setText("?")
    button.setAutoRaise(True)
    button.setCursor(Qt.CursorShape.WhatsThisCursor)
    button.setFixedSize(20, 20)
    button.setToolTip(f"More info: {title}")
    button.setAccessibleName(f"More info for {title}")
    button.setAccessibleDescription(help_text(key))
    button.setStyleSheet(
        """
        QToolButton {
            border: 0;
            background: transparent;
            color: #4f8cff;
            font-weight: 700;
            padding: 0;
        }
        QToolButton:hover {
            color: #1f6feb;
            text-decoration: underline;
        }
        """
    )
    button.clicked.connect(lambda: QMessageBox.information(parent, title, help_text(key)))
    return button


def option_header(parent: QWidget, title: str, key: str) -> QWidget:
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = QLabel(title, container)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    label.setStyleSheet("font-weight: 600;")
    attach_help(label, key)
    layout.addWidget(label, 0)
    layout.addWidget(help_button(parent, title, key), 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addStretch(1)
    return container


def add_option_row(form: QFormLayout, parent: QWidget, title: str, field, key: str) -> None:
    row = QWidget(parent)
    row_layout = QVBoxLayout(row)
    row_layout.setContentsMargins(0, 4, 0, 8)
    row_layout.setSpacing(5)
    row_layout.addWidget(option_header(parent, title, key))
    attach_help_to_field(field, key)
    if isinstance(field, QWidget):
        row_layout.addWidget(field)
    else:
        row_layout.addLayout(field)
    form.addRow(row)


def attach_help_to_field(field, key: str) -> None:
    if isinstance(field, QWidget):
        attach_help(field, key)
        for child in field.findChildren(QWidget):
            attach_help(child, key)
        return
    for index in range(field.count()):
        item = field.itemAt(index)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            attach_help(widget, key)
        nested = item.layout()
        if nested is not None:
            attach_help_to_field(nested, key)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Voice Clone dots.tts")
        self.resize(1280, 820)
        self.setMinimumSize(760, 560)
        self._service = DotsTtsService()
        self._worker: SynthesisWorker | None = None
        self._download_worker: ModelDownloadWorker | None = None
        self._last_audio_path: Path | None = None
        self._sampler_started_at: float | None = None
        self._last_sampler_completed_units = 0
        self._last_sampler_total_units = 0

        self._build_actions()
        self._build_ui()
        self._set_running(False)

    def _build_actions(self) -> None:
        unload_action = QAction("Unload Runtime", self)
        unload_action.triggered.connect(self._unload_runtime)
        self.menuBar().addMenu("Runtime").addAction(unload_action)

        guide_action = QAction("Open Guide", self)
        guide_action.triggered.connect(self._open_guide)
        diagnostics_action = QAction("Run Diagnostics", self)
        diagnostics_action.triggered.connect(self._append_diagnostics)
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(guide_action)
        help_menu.addAction(diagnostics_action)

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal, root)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_work_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 880])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.download_progress_label = QLabel("Download: idle")
        self.download_progress_label.setMinimumWidth(170)
        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setRange(0, 100)
        self.download_progress_bar.setValue(0)
        self.download_progress_bar.setFormat("%p%")
        self.download_progress_bar.setMaximumWidth(260)
        attach_help(self.download_progress_label, "download_model")
        attach_help(self.download_progress_bar, "download_model")
        bottom_bar.addWidget(self.status_label, 1)
        bottom_bar.addWidget(self.download_progress_label)
        bottom_bar.addWidget(self.download_progress_bar)
        layout.addLayout(bottom_bar)
        self.setCentralWidget(root)

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        runtime_group = QGroupBox("Model")
        runtime_form = QFormLayout(runtime_group)
        runtime_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        runtime_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.backend_combo = QComboBox()
        for label, value in BACKEND_CHOICES:
            self.backend_combo.addItem(label, value)
        self.backend_combo.setCurrentText(next(label for label, value in BACKEND_CHOICES if value == DEFAULT_BACKEND))
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        add_option_row(runtime_form, runtime_group, "Backend", self.backend_combo, "backend")

        self.pytorch_model_combo = QComboBox()
        for label, value in PYTORCH_MODEL_CHOICES:
            self.pytorch_model_combo.addItem(label, value)
        default_model_index = self.pytorch_model_combo.findData(DEFAULT_MODEL)
        self.pytorch_model_combo.setCurrentIndex(default_model_index if default_model_index >= 0 else 0)
        self.pytorch_model_combo.currentIndexChanged.connect(self._pytorch_model_changed)
        add_option_row(runtime_form, runtime_group, "PyTorch Checkpoint", self.pytorch_model_combo, "pytorch_model")

        self.quantization_combo = QComboBox()
        self._populate_quantization_combo(DEFAULT_BACKEND, DEFAULT_QUANTIZATION)
        self.quantization_combo.currentIndexChanged.connect(self._quantization_changed)
        add_option_row(runtime_form, runtime_group, "Quantization", self.quantization_combo, "quantization")

        self.model_name_label = QLabel(f"{DEFAULT_MODEL_LABEL}\n{DEFAULT_MODEL}")
        self.model_name_label.setWordWrap(True)
        self.model_name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        add_option_row(runtime_form, runtime_group, "Fixed Model", self.model_name_label, "fixed_model")

        self.model_path_edit = QLineEdit(str(local_model_path(DEFAULT_MODEL)))
        self.model_path_edit.setMinimumWidth(0)
        self.model_path_edit.setPlaceholderText("Choose a local dots.tts PyTorch model folder")
        self.model_path_edit.textChanged.connect(self._refresh_model_status)
        self.choose_model_button = QPushButton("Browse")
        attach_help(self.choose_model_button, "model_folder")
        self.choose_model_button.clicked.connect(self._choose_model_dir)
        model_path_row = QHBoxLayout()
        model_path_row.addWidget(self.model_path_edit, 1)
        model_path_row.addWidget(self.choose_model_button)
        add_option_row(runtime_form, runtime_group, "Folder", model_path_row, "model_folder")

        model_buttons = QGridLayout()
        self.download_model_button = QPushButton("Download")
        attach_help(self.download_model_button, "download_model")
        self.download_model_button.clicked.connect(self._download_selected_model)
        self.use_default_model_button = QPushButton("Default Path")
        attach_help(self.use_default_model_button, "default_model_path")
        self.use_default_model_button.clicked.connect(self._use_default_model_dir)
        self.refresh_model_button = QPushButton("Refresh")
        attach_help(self.refresh_model_button, "refresh_model")
        self.refresh_model_button.clicked.connect(self._refresh_model_status)
        model_buttons.addWidget(self.download_model_button, 0, 0)
        model_buttons.addWidget(self.use_default_model_button, 0, 1)
        model_buttons.addWidget(self.refresh_model_button, 1, 0, 1, 2)
        model_buttons.setColumnStretch(0, 1)
        model_buttons.setColumnStretch(1, 1)
        add_option_row(runtime_form, runtime_group, "Model Actions", model_buttons, "download_model")

        self.model_status_label = QLabel("")
        self.model_status_label.setWordWrap(True)
        add_option_row(runtime_form, runtime_group, "Status", self.model_status_label, "model_status")

        self.memory_estimate_label = QLabel("")
        self.memory_estimate_label.setWordWrap(True)
        add_option_row(runtime_form, runtime_group, "Estimated RAM", self.memory_estimate_label, "memory_estimate")

        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        output_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.output_dir_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self.output_dir_edit.setMinimumWidth(0)
        self.output_dir_button = QPushButton("Browse")
        attach_help(self.output_dir_button, "output_dir")
        self.output_dir_button.clicked.connect(self._choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(self.output_dir_button)
        add_option_row(output_form, output_group, "Output Dir", output_row, "output_dir")

        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(0, 10000)
        self.retention_spin.setValue(DEFAULT_OUTPUT_RETENTION)
        add_option_row(output_form, output_group, "Keep WAVs", self.retention_spin, "retention")

        self.advanced_toggle_button = QPushButton("Show Advanced Options")
        self.advanced_toggle_button.setCheckable(True)
        attach_help(self.advanced_toggle_button, "guide")
        self.advanced_toggle_button.toggled.connect(self._set_advanced_visible)

        advanced_group = QGroupBox("Advanced Options")
        self.advanced_group = advanced_group
        advanced_layout = QVBoxLayout(advanced_group)
        self.advanced_options_body = QWidget(advanced_group)
        advanced_form = QFormLayout(self.advanced_options_body)
        advanced_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        advanced_layout.addWidget(self.advanced_options_body)

        self.revision_edit = QLineEdit()
        self.revision_edit.setPlaceholderText("Optional Hugging Face revision for the download")
        add_option_row(advanced_form, advanced_group, "Revision", self.revision_edit, "revision")

        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setMinimumWidth(0)
        self.cache_dir_edit.setPlaceholderText("Optional Hugging Face cache directory")
        self.cache_dir_button = QPushButton("Browse")
        attach_help(self.cache_dir_button, "cache_dir")
        self.cache_dir_button.clicked.connect(self._choose_cache_dir)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.cache_dir_edit, 1)
        cache_row.addWidget(self.cache_dir_button)
        add_option_row(advanced_form, advanced_group, "Cache Dir", cache_row, "cache_dir")

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(PRECISION_CHOICES)
        self.precision_combo.setCurrentText(DEFAULT_PRECISION)
        self.precision_combo.currentIndexChanged.connect(self._refresh_model_status)
        add_option_row(advanced_form, advanced_group, "Precision", self.precision_combo, "precision")

        self.device_combo = QComboBox()
        for label, value in DEVICE_CHOICES:
            self.device_combo.addItem(label, value)
        add_option_row(advanced_form, advanced_group, "Device", self.device_combo, "device")

        self.optimize_check = QCheckBox("Enable torch.compile warmup")
        add_option_row(advanced_form, advanced_group, "Optimize", self.optimize_check, "optimize")

        self.unload_runtime_check = QCheckBox("Unload model after each generation")
        self.unload_runtime_check.setChecked(DEFAULT_UNLOAD_AFTER_GENERATION)
        add_option_row(
            advanced_form,
            advanced_group,
            "Memory Mode",
            self.unload_runtime_check,
            "unload_after_generation",
        )

        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1, 10000)
        self.max_length_spin.setValue(DEFAULT_MAX_GENERATE_LENGTH)
        add_option_row(advanced_form, advanced_group, "Max Generate Length", self.max_length_spin, "max_length")

        self.streaming_check = QCheckBox("Use generate_stream")
        self.streaming_check.setChecked(DEFAULT_EXECUTION_MODE == "generate_stream")
        add_option_row(advanced_form, advanced_group, "Execution", self.streaming_check, "execution")

        self.template_combo = QComboBox()
        for label, value in TEMPLATE_CHOICES:
            self.template_combo.addItem(label, value)
        add_option_row(advanced_form, advanced_group, "Template", self.template_combo, "template")

        self.language_combo = QComboBox()
        self.language_combo.setEditable(True)
        for label, value in LANGUAGE_CHOICES:
            self.language_combo.addItem(label, value)
        add_option_row(advanced_form, advanced_group, "Language", self.language_combo, "language")

        self.normalize_check = QCheckBox("Normalize text")
        add_option_row(advanced_form, advanced_group, "Text", self.normalize_check, "normalize")

        self.ode_combo = QComboBox()
        self.ode_combo.addItems(ODE_METHOD_CHOICES)
        add_option_row(advanced_form, advanced_group, "ODE Method", self.ode_combo, "ode")

        self.num_steps_spin = QSpinBox()
        self.num_steps_spin.setRange(1, 128)
        self.num_steps_spin.setValue(DEFAULT_NUM_STEPS)
        add_option_row(advanced_form, advanced_group, "Num Steps", self.num_steps_spin, "num_steps")

        self.guidance_spin = DecimalSpinBox(0.1, 10.0, 0.1, DEFAULT_GUIDANCE_SCALE)
        add_option_row(advanced_form, advanced_group, "Guidance Scale", self.guidance_spin, "guidance")

        self.speaker_spin = DecimalSpinBox(0.0, 10.0, 0.1, DEFAULT_SPEAKER_SCALE)
        add_option_row(advanced_form, advanced_group, "Speaker Scale", self.speaker_spin, "speaker")

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(DEFAULT_SEED)
        add_option_row(advanced_form, advanced_group, "Seed", self.seed_spin, "seed")

        self.profile_check = QCheckBox("Collect runtime profiling")
        add_option_row(advanced_form, advanced_group, "Profiling", self.profile_check, "profiling")

        layout.addWidget(runtime_group)
        layout.addWidget(output_group)
        layout.addWidget(self.advanced_toggle_button)
        layout.addWidget(advanced_group)
        layout.addStretch(1)
        self._set_advanced_visible(False)
        self._backend_changed()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(400)
        return scroll

    def _build_work_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        prompt_group = QGroupBox("Reference Voice")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_form = QFormLayout()
        prompt_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        prompt_layout.addLayout(prompt_form)

        self.prompt_audio_edit = QLineEdit()
        self.prompt_audio_edit.setPlaceholderText(
            "Reference audio for voice cloning. WAV, MP3, FLAC, M4A, and OGG are supported."
        )
        self.prompt_audio_edit.textChanged.connect(self._prompt_audio_changed)
        self.prompt_audio_button = QPushButton("Choose Audio")
        attach_help(self.prompt_audio_button, "prompt_audio")
        self.prompt_audio_button.clicked.connect(self._choose_prompt_audio)
        self.clear_prompt_button = QPushButton("Clear")
        attach_help(self.clear_prompt_button, "prompt_audio")
        self.clear_prompt_button.clicked.connect(self._clear_prompt)
        prompt_audio_row = QGridLayout()
        prompt_audio_row.addWidget(self.prompt_audio_edit, 0, 0, 1, 2)
        prompt_audio_row.addWidget(self.prompt_audio_button, 1, 0)
        prompt_audio_row.addWidget(self.clear_prompt_button, 1, 1)
        prompt_audio_row.setColumnStretch(0, 1)
        prompt_audio_row.setColumnStretch(1, 1)
        add_option_row(prompt_form, prompt_group, "Prompt Audio", prompt_audio_row, "prompt_audio")
        self.prompt_audio_player = AudioPlaybackControl("Reference preview", prompt_group)
        prompt_layout.addWidget(self.prompt_audio_player)

        self.prompt_text_edit = QPlainTextEdit()
        self.prompt_text_edit.setPlaceholderText(
            "Exact transcript of the reference audio. Leave empty only for prompt-audio-only cloning."
        )
        self.prompt_text_edit.setMaximumHeight(120)
        add_option_row(prompt_form, prompt_group, "Prompt Text", self.prompt_text_edit, "prompt_text")

        text_group = QGroupBox("Text to Synthesize")
        text_layout = QVBoxLayout(text_group)
        text_header = QHBoxLayout()
        text_header.addWidget(option_header(text_group, "Speech Text", "synthesis_text"))
        text_header.addStretch(1)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Enter the speech text to generate.")
        self.text_edit.setPlainText("Hello, this is a zero-shot voice cloning demonstration.")
        attach_help(self.text_edit, "synthesis_text")
        text_layout.addLayout(text_header)
        text_layout.addWidget(self.text_edit)

        controls = QGridLayout()
        self.generate_button = QPushButton("Generate")
        attach_help(self.generate_button, "generate")
        self.generate_button.clicked.connect(self._generate)
        self.play_button = QPushButton("Play")
        attach_help(self.play_button, "play")
        self.play_button.clicked.connect(self._play_last)
        self.open_output_button = QPushButton("Open Output Folder")
        attach_help(self.open_output_button, "open_output")
        self.open_output_button.clicked.connect(self._open_output_folder)
        controls.addWidget(self.generate_button, 0, 0)
        controls.addWidget(self.play_button, 0, 1)
        controls.addWidget(self.open_output_button, 1, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self.generated_audio_player = AudioPlaybackControl("Generated audio", panel)

        progress_row = QHBoxLayout()
        self.step_counter_label = QLabel("Step 0/6: Ready")
        self.step_counter_label.setWordWrap(True)
        self.step_progress = QProgressBar()
        self.step_progress.setRange(0, 100)
        self.step_progress.setValue(0)
        progress_row.addWidget(self.step_counter_label)
        progress_row.addWidget(self.step_progress, 1)

        sampler_row = QHBoxLayout()
        self.sampler_counter_label = QLabel(f"Sampler 0/{DEFAULT_NUM_STEPS}: Ready")
        self.sampler_counter_label.setWordWrap(True)
        attach_help(self.sampler_counter_label, "sampler_progress")
        self.sampler_progress = QProgressBar()
        self.sampler_progress.setRange(0, 100)
        self.sampler_progress.setValue(0)
        sampler_row.addWidget(self.sampler_counter_label)
        sampler_row.addWidget(self.sampler_progress, 1)

        eta_row = QHBoxLayout()
        self.eta_label = QLabel("ETA: waiting for sampler progress")
        self.eta_label.setWordWrap(True)
        attach_help(self.eta_label, "sampler_progress")
        eta_row.addWidget(self.eta_label)
        eta_row.addStretch(1)

        self.tabs = QTabWidget()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.metrics_edit = QTextEdit()
        self.metrics_edit.setReadOnly(True)
        self.guide_edit = QTextEdit()
        self.guide_edit.setReadOnly(True)
        self.guide_edit.setHtml(build_guide_html())
        self.tabs.addTab(self.log_edit, "Log")
        self.tabs.addTab(self.metrics_edit, "Metrics")
        self.tabs.addTab(self.guide_edit, "Guide")

        layout.addWidget(prompt_group)
        layout.addWidget(text_group, 1)
        layout.addLayout(controls)
        layout.addWidget(self.generated_audio_player)
        layout.addLayout(progress_row)
        layout.addLayout(sampler_row)
        layout.addLayout(eta_row)
        layout.addWidget(self.tabs, 1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        return scroll

    def _generate(self) -> None:
        if self._worker is not None or self._download_worker is not None:
            return
        try:
            request = self._build_request()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Request", str(exc))
            return
        self.log_edit.clear()
        self.metrics_edit.clear()
        self._reset_generation_eta()
        self._set_step_counter(0, 6, "Starting")
        self._set_sampler_counter(0, self.num_steps_spin.value(), "Ready")
        self._append_log("Starting synthesis")
        self._worker = SynthesisWorker(self._service, request)
        self._worker.progress.connect(self._append_log)
        self._worker.succeeded.connect(self._generation_succeeded)
        self._worker.failed.connect(self._generation_failed)
        self._worker.finished.connect(self._worker_finished)
        self._set_running(True)
        self._worker.start()

    def _build_request(self) -> SynthesisRequest:
        backend = self._selected_backend()
        model_path = self._selected_model_path()
        validation_error = local_model_validation_error(model_path, backend=backend)
        if validation_error is not None:
            raise ValueError(f"Choose a valid local dots.tts model folder first. {validation_error}")
        language = combo_value_or_text(self.language_combo)
        runtime = RuntimeConfig(
            backend=backend,  # type: ignore[arg-type]
            model_name_or_path=str(model_path),
            revision=self.revision_edit.text(),
            cache_dir=self.cache_dir_edit.text(),
            device=str(self.device_combo.currentData() or "auto"),
            precision=self.precision_combo.currentText(),
            quantization=self._selected_quantization(),
            optimize=self.optimize_check.isChecked(),
            max_generate_length=self.max_length_spin.value(),
            unload_after_generation=self.unload_runtime_check.isChecked(),
        )
        execution_mode = "generate_stream" if self.streaming_check.isChecked() else "generate"
        return SynthesisRequest(
            runtime=runtime,
            text=self.text_edit.toPlainText(),
            prompt_audio_path=self.prompt_audio_edit.text(),
            prompt_text=self.prompt_text_edit.toPlainText(),
            execution_mode=execution_mode,
            template_name=str(self.template_combo.currentData() or "tts"),
            language=language,
            ode_method=self.ode_combo.currentText(),
            num_steps=self.num_steps_spin.value(),
            guidance_scale=self.guidance_spin.value(),
            speaker_scale=self.speaker_spin.value(),
            normalize_text=self.normalize_check.isChecked(),
            profile_inference=self.profile_check.isChecked(),
            seed=self.seed_spin.value(),
            output_dir=Path(self.output_dir_edit.text()).expanduser(),
            output_retention_count=self.retention_spin.value(),
        )

    def _generation_succeeded(self, result: SynthesisResult) -> None:
        self._last_audio_path = result.audio_path
        self.generated_audio_player.set_source(result.audio_path)
        self.metrics_edit.setPlainText(json.dumps(result.metrics, indent=2, sort_keys=True))
        self._append_log(f"Done: {result.audio_path}")
        self.status_label.setText(f"Ready: {result.audio_path.name}")

    def _generation_failed(self, message: str, details: str = "") -> None:
        self._append_log(f"Failed: {message}")
        self._set_step_counter(0, 6, f"Error: {message}")
        self._set_sampler_counter(0, self.num_steps_spin.value(), "Stopped after error")
        self._reset_generation_eta()
        self.eta_label.setText("ETA: stopped after error")
        if details:
            self._append_log("")
            self._append_log("Diagnostic traceback:")
            self._append_log(details.rstrip())
        self.status_label.setText("Failed")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Synthesis Failed")
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.generate_button.setEnabled(not running and self._model_ready())
        self.play_button.setEnabled(not running and self._last_audio_path is not None)
        self.download_model_button.setEnabled(not running)
        self.choose_model_button.setEnabled(not running)
        self.use_default_model_button.setEnabled(not running)
        self.refresh_model_button.setEnabled(not running)
        self.backend_combo.setEnabled(not running)
        self.pytorch_model_combo.setEnabled(not running and self._selected_backend() == "pytorch")
        self.quantization_combo.setEnabled(not running)
        self.precision_combo.setEnabled(not running)
        self.unload_runtime_check.setEnabled(not running)
        if running:
            self.status_label.setText("Running")

    def _append_log(self, message: str) -> None:
        self.log_edit.append(f"{message}")
        self.status_label.setText(message)
        self._update_step_counter_from_message(message)

    def _download_progress(self, message: str) -> None:
        self._append_log(message)
        lowered = message.lower()
        if "ready" in lowered or "complete" in lowered:
            self._set_download_progress(False, "Download complete", value=100)
        elif "download" in lowered:
            self._set_download_progress(True, "Downloading model", indeterminate=True)
        else:
            self._set_download_progress(True, "Preparing model download", indeterminate=True)

    def _set_download_progress(
        self,
        active: bool,
        label: str,
        *,
        value: int = 0,
        indeterminate: bool = False,
    ) -> None:
        self.download_progress_label.setText(label)
        if active and indeterminate:
            self.download_progress_bar.setRange(0, 0)
            self.download_progress_bar.setFormat("Working")
            return
        self.download_progress_bar.setRange(0, 100)
        self.download_progress_bar.setValue(max(0, min(100, int(value))))
        self.download_progress_bar.setFormat("%p%" if value else "")

    def _download_selected_model(self) -> None:
        if self._worker is not None or self._download_worker is not None:
            return
        backend = self._selected_backend()
        repo_id = DEFAULT_MLX_MODEL if backend == "mlx" else self._selected_pytorch_model()
        self._append_log(f"Downloading fixed {backend} model: {repo_id}")
        self._set_download_progress(True, f"Downloading {repo_id}", indeterminate=True)
        revision = self.revision_edit.text().strip() or None
        self._download_worker = ModelDownloadWorker(
            repo_id,
            revision,
            backend=backend,
            quantization=self._selected_quantization(),
        )
        self._download_worker.progress.connect(self._download_progress)
        self._download_worker.succeeded.connect(self._model_download_succeeded)
        self._download_worker.failed.connect(self._model_download_failed)
        self._download_worker.finished.connect(self._download_worker_finished)
        self._set_running(True)
        self._download_worker.start()

    def _model_download_succeeded(self, path: str) -> None:
        self.model_path_edit.setText(path)
        self._append_log(f"Model downloaded: {path}")
        self._set_download_progress(False, "Download complete", value=100)
        self._refresh_model_status()

    def _model_download_failed(self, message: str, details: str = "") -> None:
        self._append_log(f"Model download failed: {message}")
        self._set_download_progress(False, "Download failed", value=0)
        if details:
            self._append_log("")
            self._append_log("Diagnostic traceback:")
            self._append_log(details.rstrip())
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Model Download Failed")
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _download_worker_finished(self) -> None:
        if self._download_worker is not None:
            self._download_worker.deleteLater()
        self._download_worker = None
        self._set_running(False)

    def _refresh_model_status(self, *_args) -> None:
        backend = self._selected_backend()
        model_path = self._selected_model_path()
        validation_error = local_model_validation_error(model_path, backend=backend)
        if validation_error is None:
            default_path = self._default_model_path().expanduser()
            if model_path == default_path:
                self.model_status_label.setText(f"Ready. Generation will use the default local {backend} checkpoint.")
            else:
                self.model_status_label.setText(f"Ready. Generation will use this local {backend} folder: {model_path}")
        else:
            self.model_status_label.setText(
                f"Not ready. {validation_error} Download the selected backend weights or choose an existing local folder."
            )
        self.memory_estimate_label.setText(estimate_memory_text(backend, self._selected_quantization(), self.precision_combo.currentText()))
        if hasattr(self, "generate_button"):
            self.generate_button.setEnabled(self._worker is None and self._download_worker is None and validation_error is None)

    def _selected_model_path(self) -> Path:
        return Path(self.model_path_edit.text().strip()).expanduser()

    def _model_ready(self) -> bool:
        return is_local_model_path_valid(self._selected_model_path(), backend=self._selected_backend())

    def _selected_backend(self) -> str:
        return str(self.backend_combo.currentData() or "pytorch")

    def _selected_pytorch_model(self) -> str:
        return str(self.pytorch_model_combo.currentData() or DEFAULT_MODEL)

    def _selected_quantization(self) -> str:
        value = str(self.quantization_combo.currentData() or "none")
        if self._selected_backend() == "pytorch":
            return value if value in PYTORCH_QUANTIZATION_VALUES else "none"
        return value if value in MLX_QUANTIZATION_VALUES else "int4"

    def _default_model_path(self) -> Path:
        if self._selected_backend() == "mlx":
            return local_mlx_model_path(self._selected_quantization())
        return local_model_path(self._selected_pytorch_model())

    def _choose_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Local dots.tts Model Folder",
            str(self._selected_model_path()),
        )
        if path:
            self.model_path_edit.setText(path)
            self._refresh_model_status()

    def _backend_changed(self, *_args) -> None:
        backend = self._selected_backend()
        previous_quantization = str(self.quantization_combo.currentData() or "")
        self._populate_quantization_combo(backend, previous_quantization)
        self.pytorch_model_combo.setEnabled(backend == "pytorch")
        self.quantization_combo.setEnabled(True)
        self.device_combo.setEnabled(backend == "pytorch")
        self.optimize_check.setEnabled(backend == "pytorch")
        self.streaming_check.setEnabled(backend == "pytorch")
        self.model_name_label.setText(
            f"{self.pytorch_model_combo.currentText()}\n{self._selected_pytorch_model()}"
            if backend == "pytorch"
            else f"dots.tts SOAR - MLX\n{DEFAULT_MLX_MODEL}/{self._selected_quantization()}"
        )
        self.model_path_edit.setText(str(self._default_model_path()))
        self._refresh_model_status()

    def _pytorch_model_changed(self, *_args) -> None:
        if self._selected_backend() == "pytorch":
            self.model_name_label.setText(f"{self.pytorch_model_combo.currentText()}\n{self._selected_pytorch_model()}")
            self.model_path_edit.setText(str(self._default_model_path()))
        self._refresh_model_status()

    def _quantization_changed(self, *_args) -> None:
        if self._selected_backend() == "mlx":
            self.model_name_label.setText(f"dots.tts SOAR - MLX\n{DEFAULT_MLX_MODEL}/{self._selected_quantization()}")
            self.model_path_edit.setText(str(self._default_model_path()))
        self._refresh_model_status()

    def _populate_quantization_combo(self, backend: str, preferred: str | None = None) -> None:
        choices = PYTORCH_QUANTIZATION_CHOICES if backend == "pytorch" else MLX_QUANTIZATION_CHOICES
        fallback = "none" if backend == "pytorch" else "int4"
        preferred_value = preferred if preferred in {value for _, value in choices} else fallback
        self.quantization_combo.blockSignals(True)
        self.quantization_combo.clear()
        for label, value in choices:
            self.quantization_combo.addItem(label, value)
        index = self.quantization_combo.findData(preferred_value)
        self.quantization_combo.setCurrentIndex(index if index >= 0 else 0)
        self.quantization_combo.blockSignals(False)

    def _use_default_model_dir(self) -> None:
        self.model_path_edit.setText(str(self._default_model_path()))
        self._refresh_model_status()

    def _choose_cache_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Cache Directory")
        if path:
            self.cache_dir_edit.setText(path)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def _choose_prompt_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Prompt Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)",
        )
        if path:
            self.prompt_audio_edit.setText(path)

    def _prompt_audio_changed(self, text: str) -> None:
        self.prompt_audio_player.set_source(text.strip() or None)

    def _clear_prompt(self) -> None:
        self.prompt_audio_edit.clear()
        self.prompt_text_edit.clear()
        self.prompt_audio_player.set_source(None)

    def _play_last(self) -> None:
        if self._last_audio_path is None:
            return
        self.generated_audio_player.set_source(self._last_audio_path)
        self.generated_audio_player.play()

    def _open_output_folder(self) -> None:
        path = Path(self.output_dir_edit.text()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        import subprocess
        import sys

        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606 - user-selected local folder.
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _unload_runtime(self) -> None:
        self._service.unload_runtime()
        self._append_log("Runtime unloaded")

    def _open_guide(self) -> None:
        if hasattr(self, "tabs") and hasattr(self, "guide_edit"):
            self.tabs.setCurrentWidget(self.guide_edit)

    def _append_diagnostics(self) -> None:
        self._append_log("Diagnostics")
        for line in build_diagnostics(
            model_path=self._selected_model_path(),
            output_dir=Path(self.output_dir_edit.text()).expanduser(),
            backend=self._selected_backend(),
            quantization=self._selected_quantization(),
        ):
            self._append_log(f"  {line}")

    def _set_advanced_visible(self, visible: bool) -> None:
        if hasattr(self, "advanced_group"):
            self.advanced_group.setVisible(visible)
        if hasattr(self, "advanced_options_body"):
            self.advanced_options_body.setVisible(visible)
        if hasattr(self, "advanced_toggle_button"):
            self.advanced_toggle_button.setText("Hide Advanced Options" if visible else "Show Advanced Options")

    def _set_step_counter(self, current: int, total: int, description: str) -> None:
        self.step_counter_label.setText(f"Step {current}/{total}: {description}")
        percent = 0 if total <= 0 else int(max(0, min(100, current / total * 100)))
        self.step_progress.setValue(percent)

    def _reset_generation_eta(self) -> None:
        self._sampler_started_at = None
        self._last_sampler_completed_units = 0
        self._last_sampler_total_units = 0
        if hasattr(self, "eta_label"):
            self.eta_label.setText("ETA: waiting for sampler progress")

    def _set_sampler_counter(
        self,
        current: int,
        total: int,
        description: str,
        *,
        patch_current: int | None = None,
        patch_total: int | None = None,
    ) -> None:
        if patch_current is not None and patch_total is not None:
            patch_current = max(0, int(patch_current))
            patch_total = max(1, int(patch_total), patch_current)
            self.sampler_counter_label.setText(
                f"Audio patch {patch_current}/{patch_total} estimated, sampler {current}/{total}: {description}"
            )
            if any(word in description.lower() for word in ("finalizing", "vocoder", "cleaning", "converting", "output")):
                if hasattr(self, "eta_label"):
                    self.eta_label.setText(f"ETA: sampler complete; {description}")
                percent = 0 if total <= 0 else int(max(0, min(100, current / total * 100)))
                self.sampler_progress.setValue(percent)
                return
            self._update_generation_eta(patch_current, patch_total, current, total)
        else:
            self.sampler_counter_label.setText(f"Sampler {current}/{total}: {description}")
        percent = 0 if total <= 0 else int(max(0, min(100, current / total * 100)))
        self.sampler_progress.setValue(percent)

    def _update_generation_eta(self, patch_current: int, patch_total: int, step_current: int, step_total: int) -> None:
        if step_total <= 0 or patch_current <= 0:
            return
        now = time.monotonic()
        if self._sampler_started_at is None:
            self._sampler_started_at = now
        completed_units = max(0, (patch_current - 1) * step_total + step_current)
        total_units = max(completed_units, patch_total * step_total)
        if completed_units <= 0:
            if hasattr(self, "eta_label"):
                self.eta_label.setText(f"ETA: measuring sampler speed; estimated {patch_total} audio patches")
            return
        elapsed = max(0.001, now - self._sampler_started_at)
        seconds_per_unit = elapsed / completed_units
        remaining_units = max(0, total_units - completed_units)
        eta_safety_factor = 1.35 if completed_units < max(8, step_total) else 1.2
        finalization_padding = 8.0 if remaining_units > 0 else 0.0
        eta_seconds = remaining_units * seconds_per_unit * eta_safety_factor + finalization_padding
        self._last_sampler_completed_units = completed_units
        self._last_sampler_total_units = total_units
        if hasattr(self, "eta_label"):
            self.eta_label.setText(
                f"ETA: ~{format_duration(eta_seconds)} left "
                f"({completed_units}/{total_units} estimated sampler steps, {seconds_per_unit:.2f}s/step observed)"
            )

    def _update_step_counter_from_message(self, message: str) -> None:
        match = re.match(r"STEP\s+(\d+)/(\d+):\s*(.*)", message)
        if match and hasattr(self, "step_counter_label"):
            current = int(match.group(1))
            total = int(match.group(2))
            description = match.group(3).strip()
            self._set_step_counter(current, total, description)
            return
        sampler_match = re.match(r"SAMPLING\s+(\d+)/(\d+):\s*(.*)", message)
        if sampler_match and hasattr(self, "sampler_counter_label"):
            current = int(sampler_match.group(1))
            total = int(sampler_match.group(2))
            description = sampler_match.group(3).strip()
            patch_current = 1 if "mlx" in description.lower() else None
            patch_total = 1 if "mlx" in description.lower() else None
            self._set_sampler_counter(current, total, description, patch_current=patch_current, patch_total=patch_total)
            return
        patch_match = re.match(r"SAMPLING_PATCH\s+(\d+)/(\d+)\s+(\d+)/(\d+):\s*(.*)", message)
        if patch_match and hasattr(self, "sampler_counter_label"):
            patch_current = int(patch_match.group(1))
            patch_total = int(patch_match.group(2))
            current = int(patch_match.group(3))
            total = int(patch_match.group(4))
            description = patch_match.group(5).strip()
            self._set_sampler_counter(
                current,
                total,
                description,
                patch_current=patch_current,
                patch_total=patch_total,
            )


class DecimalSpinBox(QDoubleSpinBox):
    def __init__(self, minimum: float, maximum: float, step: float, value: float) -> None:
        super().__init__()
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setDecimals(2)
        self.setValue(value)


def combo_value_or_text(combo: QComboBox) -> str:
    text = combo.currentText().strip()
    for index in range(combo.count()):
        if combo.itemText(index) == text:
            value = combo.itemData(index)
            return str(value if value is not None else text).strip()
    return text


def build_guide_html() -> str:
    parts = [
        "<h2>Voice Clone dots.tts Guide</h2>",
        "<p>This app supports official dots.tts PyTorch checkpoints for Windows/Linux and MLX converted weights for Apple Silicon.</p>",
    ]
    for title, bullets in GUIDE_SECTIONS:
        parts.append(f"<h3>{escape(title)}</h3>")
        parts.append("<ul>")
        for bullet in bullets:
            parts.append(f"<li>{escape(bullet)}</li>")
        parts.append("</ul>")
    parts.append("<h3>Option Reference</h3>")
    parts.append("<dl>")
    for key, text in HELP_TEXT.items():
        parts.append(f"<dt><b>{escape(key.replace('_', ' ').title())}</b></dt>")
        parts.append(f"<dd>{escape(text)}</dd>")
    parts.append("</dl>")
    return "\n".join(parts)


def estimate_memory_text(backend: str, quantization: str, precision: str) -> str:
    if backend == "mlx":
        estimates = {
            "int4": "MLX int4: weights ~2.4 GB; expected working peak roughly 6-11 GB depending prompt/text length.",
            "int8": "MLX int8: weights ~3.1 GB; expected working peak roughly 7-12 GB depending prompt/text length.",
            "mf-int4": "MLX mf-int4: weights ~2.4 GB; MeanFlow uses fewer decoder steps and should be lower/faster than SOAR.",
            "mf-int8": "MLX mf-int8: weights ~3.1 GB; conservative quantization with faster MeanFlow decoding.",
        }
        return estimates.get(quantization, estimates["int4"])
    if quantization == "torchao-int8wo":
        return "Experimental PyTorch int8 weight-only: can reduce some Linear layer weight memory, but compatibility depends on torchao, GPU driver, and model layers. MeanFlow with CUDA is the recommended consumer setting."
    if quantization == "torchao-int4wo":
        return "Experimental PyTorch int4 weight-only: most aggressive memory option and most likely to hit unsupported layers. Use only after the recommended MeanFlow CUDA path is working."
    if precision == "float16":
        return "PyTorch float16: recommended for NVIDIA GPUs such as RTX 3070. The model loads through system RAM, then runs on CUDA when the selected device is Auto or Force NVIDIA GPU."
    if precision == "bfloat16":
        return "PyTorch bfloat16: good on newer CUDA cards with strong bfloat16 support. RTX 3070 usually favors float16."
    return "PyTorch float32: highest compatibility but largest memory footprint. Use mainly for CPU troubleshooting."


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_media_time(milliseconds: int) -> str:
    total_seconds = max(0, int(milliseconds // 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def build_diagnostics(model_path: Path, output_dir: Path, backend: str = "pytorch", quantization: str = "none") -> list[str]:
    model_error = local_model_validation_error(model_path, backend=backend)
    output_parent = output_dir if output_dir.exists() else output_dir.parent
    lines = [
        f"App: {QApplication.applicationName() or 'Voice Clone dots.tts'}",
        f"Packaged: {bool(getattr(sys, 'frozen', False))}",
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.split()[0]}",
        f"Backend: {backend}",
        f"Quantization: {quantization}",
        f"Model path: {model_path}",
        f"Model ready: {model_error is None}",
        f"Model status: {model_error or 'valid local model folder'}",
        f"Output dir: {output_dir}",
        f"Output parent writable: {output_parent.exists() and os.access(output_parent, os.W_OK)}",
        "Built-in playback available: True",
    ]
    try:
        import torch

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps_backend.is_available()) if mps_backend is not None else False
        lines.extend(
            [
                f"Torch: {getattr(torch, '__version__', 'unknown')}",
                f"CUDA available: {torch.cuda.is_available()}",
                f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
                f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}",
                f"Apple MPS available: {mps_available}",
                f"PyTorch MPS override enabled: {os.environ.get('VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS') == '1'}",
            ]
        )
    except Exception as exc:
        lines.append(f"Torch diagnostics unavailable: {type(exc).__name__}: {exc}")
    try:
        import torchao

        lines.append(f"torchao: {getattr(torchao, '__version__', 'installed')}")
    except Exception as exc:
        lines.append(f"torchao unavailable: {type(exc).__name__}: {exc}")
    return lines
