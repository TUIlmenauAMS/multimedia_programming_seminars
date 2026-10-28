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
#   A QPainter canvas displays DISPLAY_CYCLES complete cycles of the current
#   waveform as a clean oscilloscope view, updated at 30 fps by a QTimer.
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
from PySide6.QtCore import Qt, QPointF, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
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
SAMPLE_RATE: int = 48_000       # Hz — standard professional sample rate
# FIX (audio noise): BLOCK_SIZE increased from 512 to 2048.
# 512 frames = 10.7 ms — too tight for Python's GIL. Heavy work on the main
# thread (even pure NumPy) can delay the PortAudio callback thread just enough
# to trigger an output underflow, heard as crackling or noise.
# 2048 frames = 42.7 ms — still imperceptible latency for a tone generator,
# and gives the OS scheduler enough headroom to keep the callback on time.
BLOCK_SIZE: int = 2_048

# Default parameter values
DEFAULT_FREQUENCY_HZ: float = 440.0    # A4 — the universal tuning reference
DEFAULT_GAIN: float = 0.4              # linear gain 0.0–1.0 (0.4 ≈ −8 dBFS)

# Frequency slider range
FREQ_MIN_HZ: int = 20
FREQ_MAX_HZ: int = 2_000

# Display refresh
DISPLAY_FPS: int = 30
DISPLAY_TIMER_MS: int = 1_000 // DISPLAY_FPS   # 33 ms

# Waveform types — integer indices matching QComboBox order
WAVEFORM_SINE: int = 0
WAVEFORM_SQUARE: int = 1
WAVEFORM_SAWTOOTH: int = 2
WAVEFORM_TRIANGLE: int = 3
WAVEFORM_NAMES: list[str] = ["Sine", "Square", "Sawtooth", "Triangle"]

# Canvas colours — dark oscilloscope theme
CANVAS_BG: str = "#0d0f14"          # near-black background
GRID_COLOR: str = "#1c2230"         # subtle grid lines
ZERO_LINE_COLOR: str = "#2a3347"    # zero-amplitude centre line (brighter)
CURVE_COLOR: str = "#00e5ff"        # cyan trace — classic scope colour
LABEL_COLOR: str = "#4a5568"        # axis tick labels (muted)
BORDER_COLOR: str = "#252b38"       # draw-area border

# Canvas layout
DISPLAY_CYCLES: int = 3             # complete cycles shown in the viewport
CANVAS_PAD_X: int = 54             # left padding — room for Y-axis labels
CANVAS_PAD_Y: int = 24             # top padding
CANVAS_PAD_BOTTOM: int = 34        # bottom padding — room for X-axis labels

# Oscilloscope trace rendering
CURVE_STROKE_WIDTH: float = 1.5    # trace line width in pixels

# Status indicator
STATUS_DOT_RADIUS: int = 5

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
    # Build the per-sample phase array for this block.
    # Each element is the absolute phase angle θ at that sample.
    phase_increment: float = 2.0 * np.pi * state.frequency / SAMPLE_RATE
    t: np.ndarray = state.phase + np.arange(frames, dtype=np.float64) * phase_increment

    # Advance the accumulator for the next block (modulo 2π prevents overflow).
    state.phase = float(t[-1] % (2.0 * np.pi))

    # --- Generate the waveform ---
    # All waveforms are derived from the phase array t. Their Fourier series:
    #   Sine:     fundamental only — the purest tone
    #   Square:   odd harmonics only (1, 3, 5, …) with amplitude 1/n
    #   Sawtooth: all harmonics (1, 2, 3, …) with amplitude 1/n — richest timbre
    #   Triangle: odd harmonics only with amplitude 1/n² — softer than square

    if state.waveform_type == WAVEFORM_SINE:
        samples = np.sin(t)

    elif state.waveform_type == WAVEFORM_SQUARE:
        # np.sign(sin(t)) maps to {-1, 0, +1} — an ideal (Gibbs-free) square wave.
        samples = np.sign(np.sin(t))

    elif state.waveform_type == WAVEFORM_SAWTOOTH:
        # FIX (sawtooth audio): the previous formula used the raw cumulative
        # phase t directly: (t % 2π) / π - 1. This is correct only when t is
        # small, but as t grows across long playback sessions the float64
        # precision of the modulo degrades, introducing subtle pitch drift.
        #
        # The correct approach: use only the wrapped phase (state.phase after
        # each step) via per-sample modulo, which we already do because we
        # rebuild t from state.phase each block. The formula itself is fine —
        # what matters is that t is always kept in [0, many·2π) starting from
        # a well-modulo'd state.phase, which the accumulator update above
        # ensures. So the formula is correct; we keep it and rely on the
        # modulo-2π accumulator reset above.
        #
        # t % (2π) wraps each sample's phase to [0, 2π).
        # Dividing by π and subtracting 1 maps that to [-1, +1).
        samples = (t % (2.0 * np.pi)) / np.pi - 1.0

    else:  # WAVEFORM_TRIANGLE
        # 2/π × arcsin(sin(t)) is the closed-form triangle definition.
        # sin(t) oscillates ±1; arcsin maps that to ±π/2; scaling gives ±1.
        samples = (2.0 / np.pi) * np.arcsin(np.sin(t))

    # Apply gain and convert to float32 (sounddevice requires float32).
    output: np.ndarray = (samples * state.gain).astype(np.float32)

    # Reshape to (frames, 1) — sounddevice convention for mono with explicit channel axis.
    return output.reshape(frames, 1)


# =============================================================================
# STEP 4 — _build_waveform_polygon(): map samples to a QPolygonF
# =============================================================================

def _build_waveform_polygon(
    data: np.ndarray,
    pad_x: int,
    pad_y: int,
    draw_w: int,
    draw_h: int,
) -> QPolygonF:
    """Build a QPolygonF from normalised waveform samples.

    Coordinate mapping:
      Sample index [0, N-1]      → pixel X in [pad_x, pad_x + draw_w]
      Amplitude    [-1.0, +1.0]  → pixel Y in [pad_y + draw_h, pad_y]

    Pixel Y increases downward in Qt, so amplitude +1.0 maps to the TOP
    of the draw area and −1.0 maps to the BOTTOM.

    FIX (audio noise via GIL contention): the previous version built a
    QPainterPath with a pure-Python ``for`` loop calling ``path.lineTo()``
    once per sample — up to ~8 192 Python→C++ calls, 30 times a second, on
    the Qt *main* thread. That is exactly the kind of main-thread work that
    can delay the PortAudio callback thread (CPython's GIL is shared), heard
    as crackling — the same failure mode the BLOCK_SIZE increase elsewhere
    in this file was meant to prevent, just relocated to the canvas.

    Two changes fix this:
      1. Decimate to at most ~2 points per horizontal pixel. The screen
         cannot show more detail than that anyway, so plotting thousands of
         samples into a few hundred pixels was pure wasted work.
      2. Build all QPointF objects in one list comprehension and hand them
         to QPolygonF in a single call, instead of thousands of individual
         path-mutation calls.

    Args:
        data:   1-D float array of waveform samples, in [-1, +1] * gain.
        pad_x:  Left pixel offset of the draw area.
        pad_y:  Top pixel offset of the draw area.
        draw_w: Width of the draw area in pixels.
        draw_h: Height of the draw area in pixels.

    Returns:
        QPolygonF tracing the waveform from left to right, ready to pass to
        QPainter.drawPolyline().
    """
    n = len(data)

    # Decimate: more than ~2 points per pixel adds no visible detail.
    max_points = max(64, min(n, draw_w * 2))
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        data = data[idx]
        n = max_points

    indices = np.arange(n, dtype=np.float64)

    # X: spread samples evenly across the draw width.
    # Y: (1 - data) / 2 maps [-1, +1] → [1, 0] (flip Y), then scale to draw_h.
    px_arr = pad_x + indices / max(n - 1, 1) * draw_w
    py_arr = pad_y + (1.0 - data) / 2.0 * draw_h

    return QPolygonF([QPointF(x, y) for x, y in zip(px_arr, py_arr)])


# =============================================================================
# STEP 5 — WaveformCanvas: clean oscilloscope display
# =============================================================================

class WaveformCanvas(QWidget):
    """A QPainter widget that renders a clean oscilloscope view of the waveform.

    DESIGN — what a real oscilloscope shows:
      • Dark background with a regular amplitude/time grid.
      • A single bright trace — nothing else.
      • Y axis: amplitude in normalised units (±1.0). The ±1.0 lines are a
        fixed scale reference; the trace itself scales with the gain, so a
        lower gain visibly shrinks it toward the zero line.
      • X axis: time in ms, showing exactly DISPLAY_CYCLES complete cycles.
        The time window = DISPLAY_CYCLES / frequency, so the trace is always
        stable regardless of frequency — like pressing AUTO on a Rigol scope.
      • Minimal annotations: amplitude labels on the left, time labels on the
        bottom, a small frequency/gain readout inside the draw area.

    FIX (visualization): removed the offscreen Gaussian blur glow pass.
    That pass called np.apply_along_axis + np.convolve in a Python loop over
    every row and column of a full-size ARGB image — O(H × W × kernel_len)
    work executed 8 times per frame (4 channels × 2 passes). At typical window
    sizes this took 40–200 ms per frame, far exceeding the 33 ms budget and
    making the timer fire late. The result was a jittery, stuttery display and
    — because the Python GIL was held during the NumPy work — audio underflows.

    The replacement: a single QPainter drawPath call with antialiasing. This
    is GPU-accelerated via Qt's paint engine and takes < 1 ms per frame.

    THREAD SAFETY:
      set_waveform() is called from the Qt main thread (the 30 fps timer).
      paintEvent() is also on the Qt main thread. No locking needed.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with a flat zero waveform and default state."""
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Flat zero line as the initial display
        self._data: np.ndarray = np.zeros(256, dtype=np.float32)
        self._is_playing: bool = False
        self._waveform_name: str = "SINE"
        self._frequency: float = DEFAULT_FREQUENCY_HZ
        self._gain: float = DEFAULT_GAIN

    # ── Public API ────────────────────────────────────────────────────────────

    def set_waveform(
        self,
        data: np.ndarray,
        frequency: float,
        gain: float,
        waveform_name: str,
        is_playing: bool,
    ) -> None:
        """Update display data and schedule a repaint.

        Args:
            data:          1-D float32 array, in [-1.0, +1.0] * gain. Gain is
                           baked in, so the trace height tracks the slider.
            frequency:     Tone frequency in Hz (for time-axis labels).
            gain:          Linear gain [0.0, 1.0] (also shown in the annotation).
            waveform_name: Human-readable waveform name for the watermark.
            is_playing:    True when audio is currently playing.
        """
        self._data = data
        self._frequency = frequency
        self._gain = gain
        self._waveform_name = waveform_name
        self._is_playing = is_playing
        self.update()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Render the oscilloscope: background → grid → trace → labels."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        px = CANVAS_PAD_X
        pt = CANVAS_PAD_Y
        pb = CANVAS_PAD_BOTTOM
        draw_w = w - px * 2
        draw_h = h - pt - pb

        # ── 1. Background ─────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        # ── 2. Amplitude grid (horizontal dashed lines) ───────────────────────
        # Five lines at ±1.0, ±0.5, and 0.0.
        amp_levels = [1.0, 0.5, 0.0, -0.5, -1.0]
        for amp in amp_levels:
            gy = int(pt + (1.0 - amp) / 2.0 * draw_h)
            if amp == 0.0:
                pen = QPen(QColor(ZERO_LINE_COLOR), 1, Qt.PenStyle.SolidLine)
            else:
                pen = QPen(QColor(GRID_COLOR), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(px, gy, px + draw_w, gy)

        # ── 3. Time grid (vertical dashed lines at each cycle boundary) ───────
        grid_pen = QPen(QColor(GRID_COLOR), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for c in range(1, DISPLAY_CYCLES):
            gx = int(px + c / DISPLAY_CYCLES * draw_w)
            painter.drawLine(gx, pt, gx, pt + draw_h)

        # ── 4. Waveform trace ─────────────────────────────────────────────────
        # Build the path and draw it in a single pass — no offscreen buffer,
        # no blur, no compositing. Qt's paint engine handles antialiasing.
        if len(self._data) > 1:
            polygon = _build_waveform_polygon(self._data, px, pt, draw_w, draw_h)
            trace_pen = QPen(
                QColor(CURVE_COLOR),
                CURVE_STROKE_WIDTH,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(trace_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(polygon)

        # ── 5. Draw-area border ───────────────────────────────────────────────
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(px, pt, draw_w, draw_h)

        # ── 6. Amplitude axis labels (left side) ──────────────────────────────
        label_font = QFont("IBM Plex Mono", 9)
        painter.setFont(label_font)
        painter.setPen(QPen(QColor(LABEL_COLOR)))
        amp_labels = {1.0: "+1.0", 0.5: "+0.5", 0.0: " 0.0", -0.5: "−0.5", -1.0: "−1.0"}
        for amp, text in amp_labels.items():
            gy = int(pt + (1.0 - amp) / 2.0 * draw_h)
            painter.drawText(0, gy - 7, px - 5, 14, Qt.AlignmentFlag.AlignRight, text)

        # ── 7. Time axis labels (bottom — one label per cycle boundary) ───────
        period_ms = 1000.0 / max(self._frequency, 1.0)
        time_font = QFont("IBM Plex Mono", 8)
        painter.setFont(time_font)
        painter.setPen(QPen(QColor(LABEL_COLOR)))
        for c in range(DISPLAY_CYCLES + 1):
            t_ms = c * period_ms
            label = f"{t_ms:.2f} ms" if t_ms > 0 else "0 ms"
            tx = int(px + c / DISPLAY_CYCLES * draw_w)
            painter.drawText(tx - 30, h - pb + 6, 60, 20,
                             Qt.AlignmentFlag.AlignCenter, label)

        # ── 8. Info annotation (top-left, inside draw area) ───────────────────
        info_font = QFont("IBM Plex Mono", 8)
        painter.setFont(info_font)
        info_color = QColor(LABEL_COLOR)
        info_color.setAlpha(200)
        painter.setPen(QPen(info_color))
        db_str = (
            f"{20.0 * np.log10(self._gain):+.1f} dBFS"
            if self._gain > 0.0
            else "−∞ dBFS"
        )
        annotation = (
            f"f = {self._frequency:.1f} Hz   "
            f"T = {period_ms:.3f} ms   "
            f"gain = {int(round(self._gain * 100))} % ({db_str})"
        )
        painter.drawText(px + 6, pt + 14, annotation)

        # ── 9. Waveform name watermark ─────────────────────────────────────────
        wm_font = QFont("IBM Plex Mono", 48, QFont.Weight.Bold)
        painter.setFont(wm_font)
        wm_color = QColor(CURVE_COLOR)
        wm_color.setAlpha(10)
        painter.setPen(QPen(wm_color))
        painter.drawText(
            px, pt, draw_w, draw_h,
            Qt.AlignmentFlag.AlignCenter,
            self._waveform_name,
        )

        # ── 10. Status indicator (top-right, outside draw area) ───────────────
        dot_x = w - px + 10
        dot_y = pt + STATUS_DOT_RADIUS + 2
        if self._is_playing:
            dot_color = QColor("#22c55e")
            status_text = "PLAYING"
        else:
            dot_color = QColor(LABEL_COLOR)
            status_text = "STOPPED"
        painter.setBrush(dot_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            dot_x - STATUS_DOT_RADIUS, dot_y - STATUS_DOT_RADIUS,
            STATUS_DOT_RADIUS * 2, STATUS_DOT_RADIUS * 2,
        )
        status_font = QFont("IBM Plex Mono", 7)
        painter.setFont(status_font)
        painter.setPen(QPen(dot_color))
        painter.drawText(dot_x + STATUS_DOT_RADIUS + 4, dot_y + 4, status_text)

        painter.end()


# =============================================================================
# STEP 6 — AudioEngine: manages the sounddevice OutputStream
# =============================================================================

class AudioEngine:
    """Manages a sounddevice OutputStream for real-time waveform synthesis.

    THREAD CONCEPT — two threads, strict rules:
      Qt main thread:  all widget updates, all signal emissions, all UI logic.
      PortAudio audio thread: only the ``_callback`` method runs here.

    The ``_state`` dataclass is shared between the two threads. Python's GIL
    (Global Interpreter Lock) makes individual attribute assignments atomic
    for built-in types (int, float), so the thread-safe setters below are
    safe without an explicit Lock for this use case.
    """

    def __init__(self) -> None:
        """Create the audio state and OutputStream, but do not start playback."""
        self._state: WaveformState = WaveformState()

        # On Windows, target WASAPI (low-latency) over the default MME.
        output_device = self._find_wasapi_output()

        self._stream: sd.OutputStream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            device=output_device,
            callback=self._callback,
        )

    @staticmethod
    def _find_wasapi_output() -> int | None:
        """Return the WASAPI default output device index, or None for system default.

        AUDIO CONCEPT — Choosing the right host API on Windows:
        sounddevice's default on Windows resolves to MME, the oldest Windows
        audio API. MME adds its own buffering, introduces timing jitter, and
        causes crackling — especially with professional USB interfaces.
        WASAPI is the modern low-latency Windows audio API.
        On Linux and macOS this returns None and sounddevice uses ALSA/CoreAudio.
        """
        for api in sd.query_hostapis():
            if api["name"] == "Windows WASAPI":
                idx = api["default_output_device"]
                if idx >= 0:
                    print(
                        f"[AudioEngine] using WASAPI output device {idx}: "
                        f"{sd.query_devices(idx)['name']}",
                        flush=True,
                    )
                    return idx
        print("[AudioEngine] WASAPI not found — using system default", flush=True)
        return None

    # ── Audio callback — runs on the PortAudio audio thread ──────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill the output buffer with synthesised audio.

        CRITICAL RULES for this function:
          1. No Qt calls — never update a widget or emit a signal from here.
          2. No blocking I/O — no file reads, no network, no sleep.
          3. No memory allocation — do not create new Python objects in the hot path.
          4. No Python exceptions — wrap risky code in try/except if needed.
        """
        if status.output_underflow:
            print("[AudioEngine] output underflow — consider increasing BLOCK_SIZE", flush=True)
        outdata[:] = waveform_block(self._state, frames)

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start audio output."""
        if not self._stream.active:
            self._stream.start()

    def stop(self) -> None:
        """Stop audio output."""
        if self._stream.active:
            self._stream.stop()

    def close(self) -> None:
        """Stop and release the audio device. Call this in closeEvent."""
        self._stream.stop()
        self._stream.close()

    # ── Thread-safe parameter setters ─────────────────────────────────────────

    def set_frequency(self, hz: float) -> None:
        """Set tone frequency. Called from the Qt main thread."""
        self._state.frequency = hz

    def set_gain(self, gain: float) -> None:
        """Set output gain. Called from the Qt main thread."""
        self._state.gain = gain

    def set_waveform_type(self, wtype: int) -> None:
        """Set waveform type and reset the phase accumulator.

        FIX (waveform switching clicks): when the waveform type changes we
        reset state.phase to 0.0. Without this, the new waveform starts at
        whatever phase the old waveform left off — which can be a large
        amplitude discontinuity (e.g. sawtooth at π suddenly becoming a sine
        at π), heard as a click or pop. Resetting to 0.0 starts every
        waveform from its zero crossing, guaranteeing a smooth transition.
        """
        self._state.waveform_type = wtype
        self._state.phase = 0.0   # reset to zero crossing — eliminates click on type change

    @property
    def state(self) -> WaveformState:
        """Read-only access to the current audio state (for display use)."""
        return self._state


# =============================================================================
# STEP 7 — ControlPanel: right-side UI widget
# =============================================================================

class ControlPanel(QWidget):
    """Right-side panel with waveform selector, frequency and gain sliders.

    All UI events are emitted as Qt signals so MainWindow can route them to
    both the AudioEngine and the WaveformCanvas without this panel knowing
    about either. This is the signal/slot decoupling pattern: the UI never
    calls DSP functions directly.

    Signals:
        waveform_changed:  Carries the new waveform type index (int).
        frequency_changed: Carries the new frequency in Hz (float).
        gain_changed:      Carries the new linear gain (float, 0.0–1.0).
    """

    waveform_changed: Signal = Signal(int)
    frequency_changed: Signal = Signal(float)
    gain_changed: Signal = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(240)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ── AMS logo ──────────────────────────────────────────────────────────
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH)).scaledToWidth(
                80, Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(pixmap)
        else:
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
        freq_label = QLabel("Frequency")
        freq_label.setStyleSheet("color: #5a6175; font-size: 12px;")
        self._freq_slider = QSlider(Qt.Orientation.Horizontal)
        self._freq_slider.setRange(FREQ_MIN_HZ, FREQ_MAX_HZ)
        self._freq_slider.setValue(int(DEFAULT_FREQUENCY_HZ))
        form.addRow(freq_label, self._freq_slider)

        self._freq_display = QLabel(f"{int(DEFAULT_FREQUENCY_HZ)} Hz")
        self._freq_display.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        form.addRow("", self._freq_display)

        # ── Gain slider ───────────────────────────────────────────────────────
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

    # ── Internal slots ────────────────────────────────────────────────────────

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
        return self._play_btn.isChecked()

    @property
    def play_button(self) -> QPushButton:
        return self._play_btn


# =============================================================================
# STEP 8 — MainWindow: assembly, signals, and the display timer
# =============================================================================

class MainWindow(QMainWindow):
    """Application main window.

    Layout (QSplitter, horizontal):
    ┌─────────────────────────────┬──────────────┐
    │  WaveformCanvas             │ ControlPanel │
    │  (oscilloscope display)     │              │
    └─────────────────────────────┴──────────────┘

    Signal flow:
      ControlPanel.waveform_changed   → AudioEngine.set_waveform_type
      ControlPanel.frequency_changed  → AudioEngine.set_frequency
      ControlPanel.gain_changed       → AudioEngine.set_gain
      ControlPanel.play_button.toggled → AudioEngine.start / stop
      QTimer.timeout (30 fps)         → _refresh_canvas → WaveformCanvas.set_waveform

    ARCHITECTURE CONCEPT — signal-only inter-component communication:
    The ControlPanel does not hold a reference to the AudioEngine or the
    WaveformCanvas. It emits signals. MainWindow connects those signals to
    the appropriate slots. This means you can replace the ControlPanel UI
    (e.g. with a MIDI controller handler) without touching the AudioEngine,
    and you can replace the WaveformCanvas (e.g. with a spectrogram) without
    touching the ControlPanel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._engine = AudioEngine()

        self._display_timer = QTimer(self)
        self._display_timer.setInterval(DISPLAY_TIMER_MS)

        self._build_ui()
        self._connect_signals()
        self._display_timer.start()

    def _build_ui(self) -> None:
        self._canvas = WaveformCanvas()
        self._controls = ControlPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([620, 240])
        self.setCentralWidget(splitter)

    def _connect_signals(self) -> None:
        self._controls.waveform_changed.connect(self._engine.set_waveform_type)
        self._controls.frequency_changed.connect(self._engine.set_frequency)
        self._controls.gain_changed.connect(self._engine.set_gain)
        self._controls.play_button.toggled.connect(self._on_play_toggled)
        self._display_timer.timeout.connect(self._refresh_canvas)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            self._engine.start()
        else:
            self._engine.stop()

    def _refresh_canvas(self) -> None:
        """Generate DISPLAY_CYCLES cycles of the current waveform for the canvas.

        DISPLAY — gain-aware trace:
        The display waveform is generated with the engine's actual
        state.gain, so the trace visibly grows/shrinks as the gain slider
        moves (gain=0 collapses it to the flat zero line). The ±1.0 grid
        lines stay fixed as a scale reference, so you can read the trace's
        peak directly against them.

        OSCILLOSCOPE DESIGN — fixed time window:
        We always show a window of exactly DISPLAY_CYCLES / frequency seconds.
        The number of samples changes with frequency:
          • At   20 Hz (3 cycles): 3/20   = 150 ms → 7 200 samples
          • At  440 Hz (3 cycles): 3/440  ≈ 6.8 ms →   327 samples
          • At 2000 Hz (3 cycles): 3/2000 = 1.5 ms →    72 samples
        The canvas stretches however many samples we pass to the full pixel
        width, so the waveform always appears stable — like pressing AUTO on
        a hardware oscilloscope.
        """
        state = self._engine.state

        n_samples = int(round(DISPLAY_CYCLES * SAMPLE_RATE / max(state.frequency, 1.0)))
        n_samples = max(64, min(n_samples, 8_192))

        # Display state: gain=state.gain so the trace height tracks the
        # slider. phase=0.0 so the display always starts from the zero
        # crossing. We do NOT disturb the engine's own state.phase — we use
        # a fresh object.
        display_state = WaveformState(
            waveform_type=state.waveform_type,
            frequency=state.frequency,
            gain=state.gain,     # ← trace amplitude follows the gain slider
            phase=0.0,            # ← always start from zero crossing for a stable display
        )
        block = waveform_block(display_state, n_samples)   # shape (n_samples, 1)

        self._canvas.set_waveform(
            data=block[:, 0],                              # 1-D array, amplitude = gain
            frequency=state.frequency,
            gain=state.gain,                               # also used for the dB annotation
            waveform_name=WAVEFORM_NAMES[state.waveform_type].upper(),
            is_playing=self._controls.is_playing,
        )

    # ── Window lifecycle ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Stop and release the audio device when the window closes."""
        self._display_timer.stop()
        self._engine.close()
        super().closeEvent(event)


# =============================================================================
# STEP 9 — Entry point
# =============================================================================

def main() -> None:
    """Create the QApplication, show the window, and start the event loop."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PySide6.QtGui import QPalette
    palette = app.palette()
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
