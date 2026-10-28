# =============================================================================
# 03_mm_hello_multimedia.py — Hello Multimedia: Waveform Synthesiser
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A professional sine tone generator and waveform synthesiser.
#   The user selects a waveform type (sine, square, sawtooth, triangle),
#   controls frequency and gain with sliders, and clicks Play/Stop to start
#   and stop audio output.
#
#   A QPainter canvas displays one cycle of the current waveform live,
#   updated at 30 fps by a QTimer.
#
#   Audio is generated in a sounddevice OutputStream callback using phase
#   accumulation — the professional technique for continuous, glitch-free
#   tone generation. This is the audio callback pattern that all subsequent
#   demos in this seminar build on.
#
#   WEBGL EQUIVALENT:
#     sounddevice.OutputStream  ↔  AudioContext (both represent the output device)
#     The sounddevice callback  ↔  ScriptProcessorNode / AudioWorkletProcessor
#     numpy waveform array      ↔  Float32Array passed to an AudioWorklet
#
# How to run:
#   uv sync                                      # first time only
#   uv run python 03_mm_hello_multimedia.py
#
# Ubuntu system dependencies (install once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0 libportaudio2
#
# Connections to other seminars:
#   Chapter 01 (Web Audio API) — AudioContext and ScriptProcessorNode are the
#     browser equivalents of sounddevice's OutputStream and its callback.
#   Chapter 02 (WebGL 2) — QPainter here plays the same role as a 2D canvas:
#     a low-level drawing API where you control every pixel.
# =============================================================================

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# STEP 1 — Constants
# =============================================================================

WINDOW_TITLE: str = "Hello Multimedia — Multimediaprogrammierung Seminar 03"
WINDOW_MIN_WIDTH: int = 860
WINDOW_MIN_HEIGHT: int = 480

# Audio
SAMPLE_RATE: int = 48_000       # Hz — standard professional sample rate (matches most hardware)
BLOCK_SIZE: int = 512           # frames per callback block — 512/48000 ≈ 10.7 ms latency

# Default parameter values
DEFAULT_FREQUENCY_HZ: float = 440.0    # A4 — the universal tuning reference
DEFAULT_GAIN: float = 0.4              # linear gain 0.0–1.0 (0.4 ≈ −8 dBFS, comfortable level)

# Frequency slider range
FREQ_MIN_HZ: int = 20          # lower limit of human hearing
FREQ_MAX_HZ: int = 2_000       # upper limit for this demo (well within audible range)

# Display refresh
DISPLAY_FPS: int = 30                                   # frames per second for the canvas
DISPLAY_TIMER_MS: int = 1_000 // DISPLAY_FPS            # 33 ms interval

# Waveform types — integer indices matching QComboBox order
WAVEFORM_SINE: int = 0
WAVEFORM_SQUARE: int = 1
WAVEFORM_SAWTOOTH: int = 2
WAVEFORM_TRIANGLE: int = 3
WAVEFORM_NAMES: list[str] = ["Sine", "Square", "Sawtooth", "Triangle"]

# Canvas colours — dark theme consistent with the course HTML design system
CANVAS_BG: str = "#0d0f14"          # --bg from the HTML design system
GRID_COLOR: str = "#1a1f2b"         # subtle grid lines
AXIS_COLOR: str = "#252b38"         # --border
CURVE_COLOR: str = "#4f9cf9"        # --accent blue — the waveform line
LABEL_COLOR: str = "#5a6175"        # --muted — axis labels
ZERO_LINE_COLOR: str = "#252b38"    # horizontal zero line

# AMS logo path — one level up from this script (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"


# =============================================================================
# STEP 2 — WaveformState dataclass (pure Python, no Qt dependencies)
# =============================================================================

@dataclass
class WaveformState:
    """Holds all audio generation parameters.

    This is the audio model layer. It has zero Qt dependencies — it can be
    created, read, and modified from any thread. The audio callback reads it
    on the PortAudio thread; the UI writes it on the Qt main thread.

    The ``phase`` field is the phase accumulator: it carries the waveform's
    current phase angle (in radians) from one callback block to the next,
    ensuring continuity at block boundaries.

    Attributes:
        waveform_type: Index into WAVEFORM_* constants.
        frequency: Tone frequency in Hz.
        gain: Linear output gain in [0.0, 1.0].
        phase: Current phase accumulator in [0, 2π). Mutated by the callback.
    """

    waveform_type: int = WAVEFORM_SINE
    frequency: float = DEFAULT_FREQUENCY_HZ
    gain: float = DEFAULT_GAIN
    phase: float = field(default=0.0)   # mutable — written by the audio callback


# =============================================================================
# STEP 3 — waveform_block(): pure NumPy audio generation
# =============================================================================

def waveform_block(state: WaveformState, frames: int) -> np.ndarray:
    """Generate one block of audio samples using phase accumulation.

    AUDIO CONCEPT — Phase accumulation:
    The naive approach to generating a sine tone is to call np.sin(t) on a
    fresh time array every block. This restarts the waveform at t=0 on every
    block boundary, producing a discontinuity (a click) at each boundary.

    Phase accumulation avoids this. We maintain a running phase angle θ (in
    radians) across blocks. Each block advances θ by exactly
    (2π × frequency / sample_rate) per sample. The last θ of one block
    becomes the starting θ of the next — the waveform is mathematically
    continuous across all block boundaries.

    This is how every professional real-time audio synthesiser works —
    from hardware synths to plugin instruments in DAWs.

    WEBGL EQUIVALENT:
    Phase accumulation here ↔ advancing a uniform ``u_time`` in a WebGL
    vertex shader frame by frame. Both maintain continuity by carrying state.

    Args:
        state: The WaveformState holding type, frequency, gain, and phase.
               ``state.phase`` is mutated in-place.
        frames: Number of audio frames (samples) to generate.

    Returns:
        NumPy array of shape ``(frames, 1)``, dtype ``float32``, values in
        ``[-1.0, 1.0]`` after gain scaling. Shape ``(N, 1)`` is the
        sounddevice convention for mono output with explicit channel axis.
    """
    # Build the phase array for this block.
    # np.arange(frames) gives [0, 1, 2, ..., frames-1].
    # Multiplying by the per-sample phase increment gives the phase offset
    # for each sample relative to state.phase (the block's starting phase).
    phase_increment: float = 2.0 * np.pi * state.frequency / SAMPLE_RATE
    t: np.ndarray = state.phase + np.arange(frames, dtype=np.float64) * phase_increment

    # Advance the accumulator for the next block.
    # We take modulo 2π to keep it in [0, 2π) and prevent float overflow
    # on very long playback sessions.
    state.phase = float(t[-1] % (2.0 * np.pi))

    # Generate the waveform.
    # All four waveforms are derived from the sine. Their Fourier series:
    #   Sine:     fundamental only — the purest tone
    #   Square:   odd harmonics only (1, 3, 5, …) with amplitude 1/n
    #   Sawtooth: all harmonics (1, 2, 3, …) with amplitude 1/n — the richest timbre
    #   Triangle: odd harmonics only with amplitude 1/n² — softer than square
    if state.waveform_type == WAVEFORM_SINE:
        samples = np.sin(t)

    elif state.waveform_type == WAVEFORM_SQUARE:
        # np.sign maps sine values to {-1, 0, +1} — an ideal square wave.
        # Real square waves have sharp edges and strong high-frequency content.
        samples = np.sign(np.sin(t))

    elif state.waveform_type == WAVEFORM_SAWTOOTH:
        # Map phase to [-1, 1] linearly using modulo arithmetic.
        # (t % (2π)) / π − 1 maps [0, 2π) → [-1, 1).
        samples = (t % (2.0 * np.pi)) / np.pi - 1.0

    else:  # WAVEFORM_TRIANGLE
        # The triangle wave is a folded sawtooth: it ramps up in the first half
        # of each cycle and back down in the second half.
        # 2/π × arcsin(sin(t)) is the closed-form definition.
        samples = (2.0 / np.pi) * np.arcsin(np.sin(t))

    # Apply gain and convert to float32.
    # sounddevice expects float32 with values in [-1.0, 1.0].
    # Values outside this range clip at the DAC — always scale before output.
    output: np.ndarray = (samples * state.gain).astype(np.float32)

    # Reshape to (frames, 1) — sounddevice's convention for mono with an
    # explicit channel dimension. This matches the stream's channels=1 setting.
    return output.reshape(frames, 1)


# =============================================================================
# STEP 4 — WaveformCanvas: QPainter visualisation widget
# =============================================================================

class WaveformCanvas(QWidget):
    """A QPainter widget that draws one cycle of the current waveform.

    Receives new waveform data via ``set_waveform()`` and redraws on the next
    Qt paint event. Drawing is triggered by calling ``self.update()``, which
    schedules a repaint through Qt's event loop — never call ``paintEvent``
    directly.

    The canvas has no knowledge of the audio engine. It is a pure view:
    it receives a NumPy array and draws it. This is the model/view pattern.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with a flat zero waveform."""
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Store one cycle of samples for display (1024 points is enough for any waveform)
        self._data: np.ndarray = np.zeros(1_024, dtype=np.float32)

    def set_waveform(self, data: np.ndarray) -> None:
        """Update the waveform data and schedule a repaint.

        Args:
            data: 1-D float32 array of waveform samples, values in [-1, 1].
        """
        self._data = data
        self.update()   # schedules paintEvent via Qt's event loop — never call paintEvent() directly

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw background, grid, zero line, and the waveform path."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad_x, pad_y = 36, 20   # pixels of padding on each side

        # ── Background ────────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        # ── Grid lines ────────────────────────────────────────────────────────
        grid_pen = QPen(QColor(GRID_COLOR), 1)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            # Vertical grid at t = 0.25, 0.50, 0.75
            gx = int(pad_x + (i / 4.0) * (w - pad_x * 2))
            painter.drawLine(gx, pad_y, gx, h - pad_y)
        for i in range(1, 4):
            # Horizontal grid at amplitude = −0.5, 0.0, +0.5 (0.0 drawn separately)
            gy = int(pad_y + (i / 4.0) * (h - pad_y * 2))
            painter.drawLine(pad_x, gy, w - pad_x, gy)

        # ── Zero (centre) line ────────────────────────────────────────────────
        zero_pen = QPen(QColor(ZERO_LINE_COLOR), 1)
        painter.setPen(zero_pen)
        cy = h // 2   # pixel row of the zero-amplitude line
        painter.drawLine(pad_x, cy, w - pad_x, cy)

        # ── Axis labels ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(LABEL_COLOR)))
        painter.setFont(QFont("IBM Plex Mono", 9))
        painter.drawText(pad_x - 32, cy + 4, " 0.0")
        painter.drawText(pad_x - 32, pad_y + 10, "+1.0")
        painter.drawText(pad_x - 32, h - pad_y - 2, "−1.0")

        # ── Waveform path ─────────────────────────────────────────────────────
        # Coordinate mapping from normalised audio space to pixel space:
        #   Audio X: sample index in [0, N-1]  → pixel X in [pad_x, w - pad_x]
        #   Audio Y: amplitude in [-1.0, +1.0] → pixel Y in [h-pad_y, pad_y]
        # Note: pixel Y increases downward, so +1.0 maps to the TOP of the canvas.
        data = self._data
        n = len(data)
        draw_w = w - pad_x * 2   # usable pixel width for the waveform
        draw_h = h - pad_y * 2   # usable pixel height for the waveform

        path = QPainterPath()
        for i, sample in enumerate(data):
            px = int(pad_x + (i / max(n - 1, 1)) * draw_w)
            # Map amplitude [-1, 1] to pixel Y: centre at cy, +1 → top, -1 → bottom
            py = int(pad_y + (1.0 - float(sample)) / 2.0 * draw_h)
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)

        painter.setPen(QPen(QColor(CURVE_COLOR), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.end()


# =============================================================================
# STEP 5 — AudioEngine: manages the sounddevice OutputStream
# =============================================================================

class AudioEngine:
    """Manages a sounddevice OutputStream for real-time waveform synthesis.

    THREAD CONCEPT — two threads, strict rules:
      Qt main thread:  all widget updates, all signal emissions, all UI logic.
      PortAudio audio thread: only the ``_callback`` method runs here.

    The ``_state`` dataclass is shared between the two threads. Python's GIL
    (Global Interpreter Lock) makes individual attribute assignments atomic
    for built-in types (int, float), so the thread-safe setters below are
    safe without an explicit Lock for this use case. If you were passing a
    NumPy array or a complex object, you would need a threading.Lock.
    """

    def __init__(self) -> None:
        """Create the audio state and OutputStream, but do not start playback."""
        self._state: WaveformState = WaveformState()

        # AUDIO CONCEPT — The callback pattern:
        # We create the stream with callback=self._callback. sounddevice
        # (via PortAudio) will call _callback() from a dedicated high-priority
        # OS audio thread at precisely the right time — roughly every
        # BLOCK_SIZE / SAMPLE_RATE seconds. We never call _callback ourselves.
        self._stream: sd.OutputStream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,           # mono output
            blocksize=BLOCK_SIZE,
            dtype="float32",      # sounddevice expects float32 in [-1.0, 1.0]
            callback=self._callback,
        )

    # ── Audio callback — runs on the PortAudio audio thread ──────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill the output buffer with synthesised audio.

        CRITICAL RULES for this function (violations cause audio glitches or crashes):
          1. No Qt calls — never update a widget or emit a signal from here.
          2. No blocking I/O — no file reads, no network, no sleep.
          3. No memory allocation — pre-allocate everything; do not create
             new Python objects (lists, dicts) in the hot path.
          4. No Python exceptions — wrap risky code in try/except if needed.

        These rules exist because PortAudio runs this function in a real-time
        OS thread with elevated scheduling priority. Any delay here causes an
        underflow — the DAC runs out of data and you hear a glitch or dropout.

        Args:
            outdata: Output buffer to fill, shape (frames, channels), float32.
                     Write directly into this array — do not reassign it.
            frames: Number of frames requested (equals BLOCK_SIZE most of the time).
            time_info: PortAudio timing struct (not used in this demo).
            status: Flags indicating stream health — check for underflows.
        """
        if status.output_underflow:
            # An underflow means the callback was called too late — the DAC ran
            # out of samples. This usually means the CPU is overloaded or
            # BLOCK_SIZE is too small. Log it; don't crash.
            print(f"[AudioEngine] output underflow — consider increasing BLOCK_SIZE", flush=True)

        # Generate the audio block and write it into the output buffer.
        # outdata[:] writes into the existing buffer (in-place) rather than
        # creating a new array — this is the zero-copy pattern for callbacks.
        outdata[:] = waveform_block(self._state, frames)

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start audio output. PortAudio begins calling the callback."""
        if not self._stream.active:
            self._stream.start()

    def stop(self) -> None:
        """Stop audio output. The callback will no longer be called."""
        if self._stream.active:
            self._stream.stop()

    def close(self) -> None:
        """Stop and release the audio device. Call this in closeEvent."""
        self._stream.stop()
        self._stream.close()

    # ── Thread-safe parameter setters ─────────────────────────────────────────
    # These are called from the Qt main thread. Python float/int assignment is
    # atomic under the GIL, making these safe without an explicit threading.Lock
    # for scalar values. If you needed to swap a NumPy array, use a Lock.

    def set_frequency(self, hz: float) -> None:
        """Set the tone frequency.

        Args:
            hz: Frequency in Hz, typically in [20, 20000].
        """
        self._state.frequency = hz

    def set_gain(self, gain: float) -> None:
        """Set the output gain.

        Args:
            gain: Linear gain in [0.0, 1.0].
        """
        self._state.gain = gain

    def set_waveform_type(self, wtype: int) -> None:
        """Set the waveform type.

        Args:
            wtype: One of WAVEFORM_SINE, WAVEFORM_SQUARE, etc.
        """
        self._state.waveform_type = wtype

    @property
    def state(self) -> WaveformState:
        """Read-only access to the current audio state (for display use)."""
        return self._state


# =============================================================================
# STEP 6 — ControlPanel: right-side UI widget
# =============================================================================

class ControlPanel(QWidget):
    """Right-side panel with waveform selector, frequency and gain sliders.

    All UI events are emitted as Qt signals so MainWindow can route them to
    both the AudioEngine and the WaveformCanvas without this panel knowing
    about either. This is the signal/slot decoupling pattern: the UI never
    calls DSP functions directly.

    Signals:
        waveform_changed: Emitted when the combo box selection changes.
                          Carries the new waveform type index.
        frequency_changed: Emitted when the frequency slider moves.
                           Carries the new frequency in Hz (float).
        gain_changed: Emitted when the gain slider moves.
                      Carries the new linear gain (float, 0.0–1.0).
    """

    waveform_changed: Signal = Signal(int)
    frequency_changed: Signal = Signal(float)
    gain_changed: Signal = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the control panel UI."""
        super().__init__(parent)
        self.setFixedWidth(240)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all widgets and connect internal signals."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ── AMS logo ──────────────────────────────────────────────────────────
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH)).scaledToWidth(
                80,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(pixmap)
        else:
            # Text fallback if the logo file is not found
            logo_label.setText("AMS Group – TU Ilmenau")
            logo_label.setStyleSheet("color: #4f9cf9; font-size: 10px;")
        layout.addWidget(logo_label)

        # ── Author credit ─────────────────────────────────────────────────────
        author_label = QLabel("Renato Fera P.")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet(
            "color: #5a6175; font-size: 10px; letter-spacing: 0.5px;"
        )
        layout.addWidget(author_label)

        layout.addSpacing(6)

        # ── Waveform selector ─────────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        waveform_label = QLabel("Waveform")
        waveform_label.setStyleSheet("color: #5a6175; font-size: 12px;")
        self._combo = QComboBox()
        self._combo.addItems(WAVEFORM_NAMES)
        self._combo.setCurrentIndex(WAVEFORM_SINE)
        form.addRow(waveform_label, self._combo)

        # ── Frequency slider ──────────────────────────────────────────────────
        # The slider operates in integer steps (1 Hz per step).
        # Range: FREQ_MIN_HZ to FREQ_MAX_HZ.
        freq_label = QLabel("Frequency")
        freq_label.setStyleSheet("color: #5a6175; font-size: 12px;")
        self._freq_slider = QSlider(Qt.Orientation.Horizontal)
        self._freq_slider.setRange(FREQ_MIN_HZ, FREQ_MAX_HZ)
        self._freq_slider.setValue(int(DEFAULT_FREQUENCY_HZ))
        form.addRow(freq_label, self._freq_slider)

        # Live readout label next to the slider
        self._freq_display = QLabel(f"{int(DEFAULT_FREQUENCY_HZ)} Hz")
        self._freq_display.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        form.addRow("", self._freq_display)

        # ── Gain slider ───────────────────────────────────────────────────────
        # Slider range 0–100; divide by 100.0 to get linear gain 0.0–1.0.
        gain_label = QLabel("Gain")
        gain_label.setStyleSheet("color: #5a6175; font-size: 12px;")
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, 100)
        self._gain_slider.setValue(int(DEFAULT_GAIN * 100))
        form.addRow(gain_label, self._gain_slider)

        self._gain_display = QLabel(f"{int(DEFAULT_GAIN * 100)} %")
        self._gain_display.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        form.addRow("", self._gain_display)

        layout.addLayout(form)

        # ── Play / Stop button ────────────────────────────────────────────────
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(79,156,249,0.12);"
            "  color: #4f9cf9;"
            "  border: 1px solid rgba(79,156,249,0.4);"
            "  border-radius: 6px;"
            "  padding: 8px 16px;"
            "  font-size: 13px;"
            "  font-family: 'IBM Plex Mono';"
            "}"
            "QPushButton:checked {"
            "  background: rgba(251,146,60,0.14);"
            "  color: #fb923c;"
            "  border-color: rgba(251,146,60,0.5);"
            "}"
            "QPushButton:hover { background: rgba(79,156,249,0.2); }"
        )
        layout.addWidget(self._play_btn)

        # ── Info box ──────────────────────────────────────────────────────────
        info = QLabel(
            "Waveform harmonics:\n"
            "• Sine:     fundamental only\n"
            "• Square:   odd harmonics (1/n)\n"
            "• Sawtooth: all harmonics (1/n)\n"
            "• Triangle: odd harmonics (1/n²)\n\n"
            "Phase accumulation keeps the\n"
            "waveform continuous across\n"
            "audio block boundaries."
        )
        info.setStyleSheet(
            "color: #5a6175; font-size: 11px; font-family: 'IBM Plex Mono';"
            " padding: 10px; background: #161a22; border-radius: 4px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()

        # ── Internal connections ───────────────────────────────────────────────
        self._combo.currentIndexChanged.connect(self._on_waveform_changed)
        self._freq_slider.valueChanged.connect(self._on_freq_changed)
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        self._play_btn.toggled.connect(self._on_play_toggled)

    # ── Internal slots (translate widget events to typed signals) ─────────────

    def _on_waveform_changed(self, index: int) -> None:
        self.waveform_changed.emit(index)

    def _on_freq_changed(self, value: int) -> None:
        self._freq_display.setText(f"{value} Hz")
        self.frequency_changed.emit(float(value))

    def _on_gain_changed(self, value: int) -> None:
        self._gain_display.setText(f"{value} %")
        self.gain_changed.emit(value / 100.0)

    def _on_play_toggled(self, checked: bool) -> None:
        self._play_btn.setText("■  Stop" if checked else "▶  Play")

    @property
    def is_playing(self) -> bool:
        """True when the Play button is pressed."""
        return self._play_btn.isChecked()

    @property
    def play_button(self) -> QPushButton:
        """Direct access to the play button for MainWindow to connect to."""
        return self._play_btn


# =============================================================================
# STEP 7 — MainWindow: assembly, signals, and the display timer
# =============================================================================

class MainWindow(QMainWindow):
    """Application main window.

    Layout (QSplitter, horizontal):
    ┌─────────────────────────────┬──────────────┐
    │  WaveformCanvas             │ ControlPanel │
    │  (live QPainter display)    │              │
    └─────────────────────────────┴──────────────┘

    Signal flow:
      ControlPanel.waveform_changed  → AudioEngine.set_waveform_type
                                     → (canvas updated by timer)
      ControlPanel.frequency_changed → AudioEngine.set_frequency
      ControlPanel.gain_changed      → AudioEngine.set_gain
      ControlPanel.play_button.toggled → AudioEngine.start / stop
      QTimer.timeout (30 fps)        → _refresh_canvas → WaveformCanvas.set_waveform

    ARCHITECTURE CONCEPT — signal-only inter-component communication:
    The ControlPanel does not hold a reference to the AudioEngine or the
    WaveformCanvas. It emits signals. MainWindow connects those signals to
    the appropriate slots. This means you can replace the ControlPanel UI
    (e.g. with a MIDI controller handler) without touching the AudioEngine,
    and you can replace the WaveformCanvas (e.g. with a spectrogram) without
    touching the ControlPanel.
    """

    def __init__(self) -> None:
        """Create engine, panels, timer, and wire all signals."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Audio engine — created before the UI so it is ready when signals fire
        self._engine = AudioEngine()

        # Display refresh timer — fires on the Qt main thread at 30 fps
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(DISPLAY_TIMER_MS)

        self._build_ui()
        self._connect_signals()
        self._display_timer.start()

    def _build_ui(self) -> None:
        """Create and arrange the canvas and control panel."""
        self._canvas = WaveformCanvas()
        self._controls = ControlPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 1)   # canvas expands
        splitter.setStretchFactor(1, 0)   # controls stay fixed width
        splitter.setSizes([620, 240])

        self.setCentralWidget(splitter)

    def _connect_signals(self) -> None:
        """Connect all signals between UI, engine, and timer.

        Every connect() call is documented with source → destination → purpose.
        """
        # Waveform type → engine + immediate canvas refresh
        self._controls.waveform_changed.connect(self._engine.set_waveform_type)

        # Frequency → engine (canvas refreshes via the timer)
        self._controls.frequency_changed.connect(self._engine.set_frequency)

        # Gain → engine
        self._controls.gain_changed.connect(self._engine.set_gain)

        # Play/Stop toggle → engine start/stop
        self._controls.play_button.toggled.connect(self._on_play_toggled)

        # 30 fps display timer → canvas refresh
        self._display_timer.timeout.connect(self._refresh_canvas)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_play_toggled(self, checked: bool) -> None:
        """Start or stop the audio engine when the Play/Stop button is toggled.

        Args:
            checked: True when the button is in the 'playing' state.
        """
        if checked:
            self._engine.start()
        else:
            self._engine.stop()

    def _refresh_canvas(self) -> None:
        """Generate one cycle of the current waveform and send it to the canvas.

        This runs on the Qt main thread at 30 fps. It reads the current audio
        state (frequency, waveform type, gain) and synthesises exactly one
        cycle of samples for display. This is independent of the audio callback
        — the canvas always shows a clean single cycle regardless of the actual
        audio block boundaries.
        """
        state = self._engine.state
        n_samples = int(SAMPLE_RATE / max(state.frequency, 1.0))   # samples in one cycle
        # Cap to a reasonable display size — very low frequencies would generate
        # too many samples. 4096 is well above what any screen can display.
        n_samples = min(n_samples, 4_096)

        # Create a temporary state for display only (do not disturb the engine's phase)
        display_state = WaveformState(
            waveform_type=state.waveform_type,
            frequency=state.frequency,
            gain=1.0,   # always show the full waveform shape at unit gain
            phase=0.0,  # always start from zero for a clean single-cycle display
        )
        block = waveform_block(display_state, n_samples)  # shape (n_samples, 1)
        self._canvas.set_waveform(block[:, 0])            # pass as 1-D array

    # ── Window lifecycle ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Stop and release the audio device when the window closes.

        RESOURCE MANAGEMENT: always close sounddevice streams in closeEvent.
        An unreleased audio stream holds the OS audio device open, which can
        prevent other applications from using the device until the process ends.
        """
        self._display_timer.stop()
        self._engine.close()
        super().closeEvent(event)


# =============================================================================
# STEP 8 — Entry point
# =============================================================================

def main() -> None:
    """Create the QApplication, show the window, and start the event loop."""
    app = QApplication(sys.argv)

    # Apply the dark theme palette consistent with the course HTML design system
    app.setStyle("Fusion")
    palette = app.palette()
    from PySide6.QtGui import QPalette
    palette.setColor(QPalette.ColorRole.Window, QColor("#0d0f14"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#d8dce6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#161a22"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1f2b"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#161a22"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#d8dce6"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4f9cf9"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d0f14"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 03_mm_hello_multimedia.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
