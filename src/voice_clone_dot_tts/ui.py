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

from PyQt6.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
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
        "This app is intentionally locked to rednote-hilab/dots.tts-soar. Upstream describes SOAR as the "
        "self-corrective-aligned checkpoint with the best voice cloning performance. The app package does not "
        "include model weights; download them in-app or choose an existing local SOAR folder."
    ),
    "model_folder": (
        "Local folder containing the downloaded dots.tts SOAR files. The folder must include config.json, "
        "model.safetensors, vocoder.safetensors, and speaker_encoder.safetensors. You can use the in-app "
        "download or browse to an existing Hugging Face snapshot/local-dir download."
    ),
    "download_model": (
        "Downloads the fixed SOAR checkpoint into this app's model directory. The upstream CLI also accepts a "
        "local model directory or Hugging Face repo id, but this UI deliberately allows SOAR only."
    ),
    "default_model_path": "Restores the standard per-user model directory for this app.",
    "refresh_model": "Rechecks whether the selected local model folder is complete and valid.",
    "model_status": "Shows whether generation is available and explains what is missing if the model is not ready.",
    "backend": (
        "Runtime backend. PyTorch is the cross-platform path for Windows/Linux CUDA and CPU and uses the official "
        "rednote-hilab/dots.tts-soar checkpoint. PyTorch MPS is disabled by default because this model can abort the "
        "whole app inside Apple's Metal/MPS matmul implementation instead of raising a Python error. MLX is the "
        "Apple-Silicon GPU path and uses converted shraey/dots-tts-mlx weights. Choose MLX on Mac for lower memory "
        "and GPU acceleration; choose PyTorch for Windows compatibility, CUDA systems, or CPU fallback."
    ),
    "quantization": (
        "Backend-specific memory mode. For PyTorch, the torchao int8/int4 entries apply reputable PyTorch runtime "
        "weight-only quantization to the official SOAR checkpoint after it loads; these are intended for Windows/CUDA "
        "or CPU experiments and require torchao to be installed. They are not separate Hugging Face model forks. I did "
        "not find an official or clearly reputable drop-in PyTorch int4/int8 SOAR checkpoint. For MLX, int4 is the "
        "lowest-memory Apple Silicon SOAR path; int8 is a conservative larger quantized fallback. mf-int4 and mf-int8 "
        "use the distilled MeanFlow checkpoint; MeanFlow normally works with far fewer flow evaluations, so it is "
        "faster and lighter, but guidance scale is effectively fused into the distilled model instead of acting like "
        "SOAR CFG."
    ),
    "unload_after_generation": (
        "Lower-memory mode. When enabled, the app unloads the model and clears CUDA/MPS/MLX caches after each output. "
        "This prevents memory growth across runs and swap pressure, but the next generation must reload the model."
    ),
    "memory_estimate": (
        "Rough memory expectation for the selected backend. PyTorch SOAR can use far more RAM than the model file size "
        "because the 2B model, vocoder, speaker encoder, text stack, generated-token state, audio latents, and allocator "
        "caches all coexist. On Apple Silicon, PyTorch MPS has shown 12-18+ GB peaks and can hard-crash, so Auto skips "
        "it. MLX int4 uses about 2.4 GB of weights and is expected to peak much lower; MLX int8 uses about 3.1 GB of "
        "weights. Longer text, longer prompt audio, RK4/more steps, and keeping the model loaded all increase memory pressure."
    ),
    "output_dir": "Folder where generated WAV files are written. The app creates it if it does not exist.",
    "retention": "Maximum number of generated WAV files to keep in the output folder. Set 0 to disable cleanup.",
    "revision": "Optional Hugging Face revision, branch, tag, or commit to download. Leave blank for the default release.",
    "cache_dir": "Optional Hugging Face cache directory. Leave blank unless you need downloads stored somewhere specific.",
    "precision": (
        "Requested runtime precision. PyTorch float16 reduces model-weight memory on CUDA and is the app default for "
        "lower RAM. bfloat16 is often a good CUDA choice and matches the upstream optimized example. float32 is the "
        "compatibility fallback and uses the most memory. CPU generation is forced to float32. PyTorch MPS is disabled "
        "by default because it can hard-crash this model; MLX uses this dtype when loading converted weights, but its "
        "int4/int8 setting is the larger memory lever."
    ),
    "device": (
        "PyTorch device. Auto chooses CUDA when available, then CPU. Apple MPS is deliberately skipped because the "
        "attached crash report shows a native Metal/MPS abort in PyTorch matmul; Python cannot catch that kind of "
        "process abort. Use MLX for Apple Silicon GPU acceleration. The Apple MPS entry is only for unsupported "
        "debugging and requires launching with VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS=1."
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
    "template": "dots.tts prompt template. TTS is the normal voice cloning path.",
    "language": (
        "Optional language tag. Upstream supports none, auto_detect, language codes like EN/ZH, and names like "
        "english/chinese. Useful for multilingual or code-switched text."
    ),
    "normalize": "Runs WeTextProcessing text normalization before inference. Useful for numbers, dates, punctuation, and mixed-language text.",
    "ode": (
        "PyTorch ODE sampler. Euler is the default and is the best speed/memory baseline: one DiT evaluation per step. "
        "Midpoint evaluates about twice per step, which can smooth the trajectory but costs roughly 2x sampler work. "
        "RK4 evaluates about four times per step and is the heaviest option; it may sound different, but it is usually "
        "not the right first choice on low-memory machines. MLX converted checkpoints use their own Euler/MeanFlow "
        "solver path, so this control is mainly for PyTorch SOAR."
    ),
    "num_steps": (
        "Flow-matching sampling steps shown as Sampler 0/N through N/N during generation. The app defaults to 32 for "
        "highest SOAR quality. 10-16 is faster and lower memory; 32 is slower but usually cleaner and more stable. "
        "MeanFlow MLX variants are distilled for very low NFE, commonly around 4, so using 32 there trades speed for "
        "little expected benefit."
    ),
    "guidance": (
        "Classifier-free guidance scale. Upstream default is 1.2 and warns that values above 2 progressively "
        "amplify audio energy. Increase cautiously."
    ),
    "speaker": "Voice cloning strength. Higher values push closer to the reference speaker but can increase artifacts.",
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
    "synthesis_text": "The text to speak in the cloned voice.",
    "generate": "Starts synthesis after validating the model folder, text, prompt audio, and output folder.",
    "sampler_progress": (
        "Shows audio-patch and sampler progress. dots.tts generates one latent audio patch at a time, and each patch "
        "runs its own flow-matching solve, so a single WAV can legitimately show Patch 1 sampler 1/32, Patch 2 sampler "
        "1/32, and so on. The patch total is estimated from text length and model patch duration, then adjusted upward "
        "if generation exceeds the estimate. ETA is calculated from the average time per observed sampler step."
    ),
    "play": "Plays the most recently generated WAV inside the app.",
    "open_output": "Opens the output folder in Finder, Explorer, or the platform file manager.",
    "guide": "Opens the built-in guide tab with setup, cloning, quality, memory, and troubleshooting notes.",
    "diagnostics": "Writes local app, model, output, and device diagnostics to the Log tab without loading the model.",
}

GUIDE_SECTIONS = (
    (
        "Start",
        (
            "1. Download the fixed rednote-hilab/dots.tts-soar model or browse to an existing local SOAR folder.",
            "2. Choose a clean reference voice clip and enter its exact transcript when available.",
            "3. Enter the speech text, keep the default quality settings, then press Generate.",
        ),
    ),
    (
        "Model Rules",
        (
            "This app supports only dots.tts SOAR: the official PyTorch rednote-hilab/dots.tts-soar checkpoint and the Apple Silicon MLX conversion of that model.",
            "The packaged app does not include model weights; this keeps the app distributable and lets users download the model on demand.",
            "A valid PyTorch model folder must contain config.json, model.safetensors, vocoder.safetensors, and speaker_encoder.safetensors.",
            "A valid MLX model folder must contain the selected int4/int8/mf variant files plus its tokenizer directory.",
            "On Mac, MLX is the supported GPU route. PyTorch MPS is blocked by default because this model can crash in Metal before Python can handle the error.",
        ),
    ),
    (
        "Quality Defaults",
        (
            "The default PyTorch path is non-streaming generation with 32 sampling steps and float16 on CUDA.",
            "Auto device selection uses CUDA first, then CPU. It skips Apple MPS unless VOICE_CLONE_DOT_TTS_ALLOW_PYTORCH_MPS=1 is set.",
            "Increasing sampling steps can improve stability but increases generation time; RK4 multiplies sampler work and memory pressure.",
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
    app = QApplication(argv)
    app.setApplicationName("Voice Clone dots.tts")
    app.setOrganizationName("Voice Clone dots.tts")
    return app


class SynthesisWorker(QThread):
    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

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
    progress = pyqtSignal(str)
    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str, str)

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
    button.setToolTip(help_text(key))
    button.setAccessibleName(f"Help for {title}")
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
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.9)

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
        splitter.setSizes([460, 820])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
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
        self.model_path_edit.setPlaceholderText("Choose a local rednote-hilab/dots.tts-soar folder")
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

        progress_row = QHBoxLayout()
        self.step_counter_label = QLabel("Step 0/6: Ready")
        self.step_counter_label.setWordWrap(True)
        self.step_progress = QProgressBar()
        self.step_progress.setRange(0, 100)
        self.step_progress.setValue(0)
        progress_row.addWidget(self.step_counter_label)
        progress_row.addWidget(self.step_progress, 1)

        sampler_row = QHBoxLayout()
        self.sampler_counter_label = QLabel("Sampler 0/32: Ready")
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
            raise ValueError(f"Choose a valid local dots.tts SOAR model folder first. {validation_error}")
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
        self.metrics_edit.setPlainText(json.dumps(result.metrics, indent=2, sort_keys=True))
        self._append_log(f"Done: {result.audio_path}")
        self.status_label.setText(f"Ready: {result.audio_path.name}")

    def _generation_failed(self, message: str, details: str = "") -> None:
        self._append_log(f"Failed: {message}")
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
        self.quantization_combo.setEnabled(not running and self._selected_backend() == "mlx")
        self.precision_combo.setEnabled(not running)
        self.unload_runtime_check.setEnabled(not running)
        if running:
            self.status_label.setText("Running")

    def _append_log(self, message: str) -> None:
        self.log_edit.append(f"{message}")
        self.status_label.setText(message)
        self._update_step_counter_from_message(message)

    def _download_selected_model(self) -> None:
        if self._worker is not None or self._download_worker is not None:
            return
        backend = self._selected_backend()
        repo_id = DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_MODEL
        self._append_log(f"Downloading fixed {backend} model: {repo_id}")
        revision = self.revision_edit.text().strip() or None
        self._download_worker = ModelDownloadWorker(
            repo_id,
            revision,
            backend=backend,
            quantization=self._selected_quantization(),
        )
        self._download_worker.progress.connect(self._append_log)
        self._download_worker.succeeded.connect(self._model_download_succeeded)
        self._download_worker.failed.connect(self._model_download_failed)
        self._download_worker.finished.connect(self._download_worker_finished)
        self._set_running(True)
        self._download_worker.start()

    def _model_download_succeeded(self, path: str) -> None:
        self.model_path_edit.setText(path)
        self._append_log(f"Model downloaded: {path}")
        self._refresh_model_status()

    def _model_download_failed(self, message: str, details: str = "") -> None:
        self._append_log(f"Model download failed: {message}")
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

    def _selected_quantization(self) -> str:
        value = str(self.quantization_combo.currentData() or "none")
        if self._selected_backend() == "pytorch":
            return value if value in PYTORCH_QUANTIZATION_VALUES else "none"
        return value if value in MLX_QUANTIZATION_VALUES else "int4"

    def _default_model_path(self) -> Path:
        if self._selected_backend() == "mlx":
            return local_mlx_model_path(self._selected_quantization())
        return local_model_path(DEFAULT_MODEL)

    def _choose_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Local dots.tts SOAR Model Folder",
            str(self._selected_model_path()),
        )
        if path:
            self.model_path_edit.setText(path)
            self._refresh_model_status()

    def _backend_changed(self, *_args) -> None:
        backend = self._selected_backend()
        previous_quantization = str(self.quantization_combo.currentData() or "")
        self._populate_quantization_combo(backend, previous_quantization)
        self.quantization_combo.setEnabled(True)
        self.device_combo.setEnabled(backend == "pytorch")
        self.optimize_check.setEnabled(backend == "pytorch")
        self.streaming_check.setEnabled(backend == "pytorch")
        self.model_name_label.setText(
            f"{DEFAULT_MODEL_LABEL}\n{DEFAULT_MODEL}"
            if backend == "pytorch"
            else f"dots.tts SOAR - MLX\n{DEFAULT_MLX_MODEL}/{self._selected_quantization()}"
        )
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

    def _clear_prompt(self) -> None:
        self.prompt_audio_edit.clear()
        self.prompt_text_edit.clear()

    def _play_last(self) -> None:
        if self._last_audio_path is None:
            return
        self._player.setSource(QUrl.fromLocalFile(str(self._last_audio_path)))
        self._player.play()

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
        eta_seconds = remaining_units * seconds_per_unit
        self._last_sampler_completed_units = completed_units
        self._last_sampler_total_units = total_units
        if hasattr(self, "eta_label"):
            self.eta_label.setText(
                f"ETA: ~{format_duration(eta_seconds)} left "
                f"({completed_units}/{total_units} sampler steps observed, {seconds_per_unit:.2f}s/step)"
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
        "<p>This app is built for <b>rednote-hilab/dots.tts-soar</b> only.</p>",
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
        return "PyTorch torchao int8 weight-only: experimental Windows/CUDA/CPU memory mode; can reduce Linear weight memory but may not accelerate every layer."
    if quantization == "torchao-int4wo":
        return "PyTorch torchao int4 weight-only: most aggressive experimental PyTorch memory mode; support depends on torchao, device, and layer coverage."
    if precision == "float16":
        return "PyTorch float16: lower model weight memory on CUDA. On Mac, Auto skips MPS because it can hard-crash; CPU falls back to float32 and may be slow/high RAM."
    if precision == "bfloat16":
        return "PyTorch bfloat16: good CUDA choice and upstream-friendly. On Mac, use MLX int4/int8 for GPU acceleration instead of PyTorch MPS."
    return "PyTorch float32: highest compatibility but largest memory footprint; CPU peaks can still be well above the weight file size."


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


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
        f"Model status: {model_error or 'valid local SOAR folder'}",
        f"Output dir: {output_dir}",
        f"Output parent writable: {output_parent.exists() and os.access(output_parent, os.W_OK)}",
    ]
    try:
        import torch

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps_available = bool(mps_backend.is_available()) if mps_backend is not None else False
        lines.extend(
            [
                f"Torch: {getattr(torch, '__version__', 'unknown')}",
                f"CUDA available: {torch.cuda.is_available()}",
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
