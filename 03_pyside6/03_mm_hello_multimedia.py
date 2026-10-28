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
    QImage,
    QLinearGradient,
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
ZERO_LINE_COLOR: str = "#2e3547"    # horizontal zero line — slightly brighter than grid

# Canvas display parameters
DISPLAY_CYCLES: int = 3             # number of complete waveform cycles shown
CANVAS_PAD_X: int = 52             # horizontal padding (pixels) — room for Y-axis labels
CANVAS_PAD_Y: int = 28             # vertical padding (pixels) — room for X-axis labels
CANVAS_PAD_BOTTOM: int = 36        # extra bottom padding for time-axis labels

# Glow effect parameters — used in the two-pass offscreen Gaussian blur
GLOW_SIGMA: float = 4.0            # Gaussian blur σ in pixels — controls bloom radius
GLOW_STROKE_WIDTH: float = 8.0     # stroke width of the glow layer before blurring
GLOW_ALPHA: float = 0.55           # opacity of the composited glow layer
CURVE_STROKE_WIDTH: float = 1.8    # width of the sharp foreground waveform line

# Gradient fill parameters — semi-transparent fill between curve and zero line
FILL_ALPHA_TOP: int = 60           # ARGB alpha at the zero crossing (0–255)
FILL_ALPHA_BOTTOM: int = 0         # ARGB alpha at the amplitude extremes

# Status indicator
STATUS_DOT_RADIUS: int = 5         # radius of the PLAYING/STOPPED dot in px

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
# STEP 4 — Gaussian blur helpers (pure NumPy, no Qt dependency)
# =============================================================================

def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Return a normalised 1-D Gaussian kernel for the given sigma.

    DSP CONCEPT — Gaussian convolution as a blur:
    A Gaussian kernel is the optimal low-pass filter in both spatial and
    frequency domains (it minimises the uncertainty principle product). Applied
    as a 2-D image blur it is separable: a 2-D Gaussian G(x,y) = G(x) * G(y),
    so we can convolve rows then columns — O(n*k) instead of O(n*k²).

    This is mathematically identical to the Gaussian smoothing used in audio
    DSP (e.g. smoothing a spectral envelope) — the same kernel, different domain.

    Args:
        sigma: Standard deviation of the Gaussian in pixels.

    Returns:
        1-D float64 array, length = 6*sigma+1 (captures ±3σ), normalised to
        sum to 1.0 so convolution preserves overall brightness.
    """
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur_rgba(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur to an ARGB uint8 image array.

    Works on each channel independently using two 1-D convolution passes
    (horizontal then vertical). The convolution uses 'reflect' boundary mode
    to avoid dark edges at the image border.

    IMAGE LAYOUT CONCEPT — QImage::Format_ARGB32:
    Qt stores pixels as 32-bit little-endian ARGB: bytes are [B, G, R, A] in
    memory. We reshape the flat buffer into (height, width, 4) and process
    each channel (axis 2) independently.

    Args:
        image: NumPy array of shape (H, W, 4), dtype uint8, channel order BGRA.
        sigma: Gaussian standard deviation in pixels.

    Returns:
        Blurred array of the same shape and dtype.
    """
    kernel = _gaussian_kernel_1d(sigma)
    blurred = image.astype(np.float32)

    # Separable 2-D convolution: horizontal pass then vertical pass.
    # np.apply_along_axis applies the 1-D convolve along the chosen axis
    # for every position along the other axes — equivalent to a nested loop
    # but implemented in C inside NumPy.
    for ch in range(4):
        # Horizontal pass — convolve each row
        blurred[:, :, ch] = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=blurred[:, :, ch]
        )
        # Vertical pass — convolve each column
        blurred[:, :, ch] = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=blurred[:, :, ch]
        )

    return np.clip(blurred, 0, 255).astype(np.uint8)


def _build_waveform_path(
    data: np.ndarray,
    pad_x: int,
    pad_y: int,
    draw_w: int,
    draw_h: int,
) -> QPainterPath:
    """Build a QPainterPath from normalised waveform samples.

    Coordinate mapping:
      Sample index [0, N-1]      → pixel X in [pad_x, pad_x + draw_w]
      Amplitude    [-1.0, +1.0]  → pixel Y in [pad_y + draw_h, pad_y]
    Pixel Y increases downward, so +1.0 amplitude maps to the TOP of the draw area.

    Uses vectorised NumPy to compute all pixel coordinates at once, then
    iterates only to build the QPainterPath — this is the minimum Python-loop
    overhead possible without dropping into C++.

    Args:
        data:   1-D float32 array of waveform samples in [-1.0, 1.0].
        pad_x:  Left pixel offset of the draw area.
        pad_y:  Top pixel offset of the draw area.
        draw_w: Width of the draw area in pixels.
        draw_h: Height of the draw area in pixels.

    Returns:
        QPainterPath tracing the waveform from left to right.
    """
    n = len(data)
    indices = np.arange(n, dtype=np.float32)

    # Vectorised coordinate computation — no Python loop over samples
    px_arr = (pad_x + indices / max(n - 1, 1) * draw_w).astype(np.float32)
    py_arr = (pad_y + (1.0 - data) / 2.0 * draw_h).astype(np.float32)

    path = QPainterPath()
    path.moveTo(float(px_arr[0]), float(py_arr[0]))
    for i in range(1, n):
        path.lineTo(float(px_arr[i]), float(py_arr[i]))
    return path


# =============================================================================
# STEP 4b — WaveformCanvas: QPainter visualisation widget
# =============================================================================

class WaveformCanvas(QWidget):
    """A QPainter widget that draws multiple cycles of the current waveform.

    Rendering pipeline (executed every paintEvent at 30 fps):

      1. OFFSCREEN GLOW LAYER
         Draw the waveform stroke (width = GLOW_STROKE_WIDTH) with the accent
         colour onto a QImage (Format_ARGB32). Convert to a NumPy array via
         the QImage memory buffer (zero-copy), apply a separable Gaussian blur
         (sigma = GLOW_SIGMA) using pure NumPy, convert back to QPixmap, and
         composite onto the canvas with GLOW_ALPHA opacity.

      2. GRADIENT FILL
         Build a closed QPainterPath (waveform curve + return along zero line)
         and fill it with a QLinearGradient from the accent colour (at the zero
         line, semi-transparent) to fully transparent (at the amplitude extremes).

      3. SHARP FOREGROUND LINE
         Draw the waveform path again at CURVE_STROKE_WIDTH with full opacity —
         the crisp top layer that sits above the glow bloom.

      4. GRID, AXES, LABELS, STATUS, WATERMARK
         All drawn with QPainter directly on the widget surface.

    THREAD SAFETY:
      ``set_waveform()`` is called from the Qt main thread (the 30 fps timer).
      ``paintEvent()`` is also on the Qt main thread. No locking needed.

    DESIGN CONCEPT — model/view separation:
      The canvas has zero knowledge of the AudioEngine. It receives a NumPy
      array and renders it. Replacing the canvas (e.g. with a spectrogram)
      requires no changes to the audio layer.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with a flat zero waveform and default state."""
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # DISPLAY_CYCLES copies of one cycle, stitched together
        self._data: np.ndarray = np.zeros(1_024, dtype=np.float32)

        # Playback state — set by MainWindow to drive the status indicator
        self._is_playing: bool = False

        # Waveform name shown as watermark — set by MainWindow
        self._waveform_name: str = "SINE"

        # Frequency in Hz — used to compute the time-axis labels
        self._frequency: float = DEFAULT_FREQUENCY_HZ

        # Linear gain [0.0, 1.0] — used to annotate the display; the waveform
        # samples already carry this scaling so no extra multiply is needed here
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
            data:          1-D float32 array of waveform samples already scaled
                           by gain, i.e. values in [-gain, +gain].
            frequency:     Current tone frequency in Hz (for time-axis labels).
            gain:          Linear gain in [0.0, 1.0]. Stored so paintEvent can
                           draw the zero-line and amplitude labels at fixed ±1.0
                           while the wave itself only reaches ±gain — exactly
                           what a hardware oscilloscope shows.
            waveform_name: Human-readable waveform name for the watermark.
            is_playing:    True when audio is currently playing.
        """
        self._data = data
        self._frequency = frequency
        self._gain = gain
        self._waveform_name = waveform_name
        self._is_playing = is_playing
        self.update()   # schedule repaint via Qt's event loop

    # ── Rendering ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Full rendering pipeline: glow → fill → curve → decorations."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        px = CANVAS_PAD_X
        pt = CANVAS_PAD_Y
        pb = CANVAS_PAD_BOTTOM
        draw_w = w - px * 2
        draw_h = h - pt - pb
        cy = pt + draw_h // 2   # pixel row of the zero-amplitude line

        # ── 1. Background ─────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        # ── 2. Watermark — waveform name in large faint text ──────────────────
        wm_font = QFont("IBM Plex Mono", 52, QFont.Weight.Bold)
        painter.setFont(wm_font)
        wm_color = QColor(CURVE_COLOR)
        wm_color.setAlpha(14)
        painter.setPen(QPen(wm_color))
        painter.drawText(
            px, pt, draw_w, draw_h,
            Qt.AlignmentFlag.AlignCenter,
            self._waveform_name,
        )

        # ── 3. Amplitude grid (horizontal) ────────────────────────────────────
        # Five lines: +1.0, +0.5, 0.0, −0.5, −1.0
        amp_levels = [1.0, 0.5, 0.0, -0.5, -1.0]
        for amp in amp_levels:
            gy = int(pt + (1.0 - amp) / 2.0 * draw_h)
            if amp == 0.0:
                pen = QPen(QColor(ZERO_LINE_COLOR), 1)
            else:
                pen = QPen(QColor(GRID_COLOR), 1)
            pen.setStyle(Qt.PenStyle.DashLine if amp != 0.0 else Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(px, gy, w - px, gy)

        # ── 4. Time grid (vertical) — one line per cycle boundary ─────────────
        grid_pen = QPen(QColor(GRID_COLOR), 1)
        grid_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for c in range(1, DISPLAY_CYCLES):
            gx = int(px + c / DISPLAY_CYCLES * draw_w)
            painter.drawLine(gx, pt, gx, pt + draw_h)

        # ── 5. Glow layer — offscreen QImage → NumPy Gaussian blur → composite ──
        #
        # RENDERING CONCEPT — offscreen compositing:
        # We render to a QImage (off-screen pixel buffer) rather than directly
        # to the widget. This gives us access to the raw pixel data as a NumPy
        # array (via QImage.bits()), where we apply a real Gaussian blur.
        # The blurred result is composited back onto the widget with reduced
        # opacity — producing the glow bloom effect.
        #
        # QImage.bits() returns a memoryview of the pixel buffer with zero copy.
        # np.frombuffer() wraps it without copying. The array shape is
        # (height * width * 4,) because QImage stores pixels packed — we
        # reshape to (H, W, 4) for per-channel processing.
        glow_image = QImage(w, h, QImage.Format.Format_ARGB32)
        glow_image.fill(Qt.GlobalColor.transparent)
        gp = QPainter(glow_image)
        gp.setRenderHint(QPainter.RenderHint.Antialiasing)
        glow_color = QColor(CURVE_COLOR)
        glow_color.setAlpha(200)
        gp.setPen(QPen(glow_color, GLOW_STROKE_WIDTH, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        gp.setBrush(Qt.BrushStyle.NoBrush)
        glow_path = _build_waveform_path(self._data, px, pt, draw_w, draw_h)
        gp.drawPath(glow_path)
        gp.end()

        # Convert QImage pixel buffer to NumPy array — zero-copy via memoryview
        ptr = glow_image.bits()                          # memoryview of raw ARGB bytes
        arr = np.frombuffer(ptr, dtype=np.uint8).copy() # copy needed: buffer is read-only
        arr = arr.reshape((h, w, 4))                     # (H, W, 4) — channels: B,G,R,A

        # Apply separable Gaussian blur — the mathematically correct glow
        blurred = _gaussian_blur_rgba(arr, GLOW_SIGMA)

        # Write blurred pixels back into a new QImage and composite onto canvas
        blurred_image = QImage(
            blurred.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32
        )
        painter.setOpacity(GLOW_ALPHA)
        painter.drawImage(0, 0, blurred_image)
        painter.setOpacity(1.0)

        # ── 6. Gradient fill — area between curve and zero line ───────────────
        #
        # Build a closed path: waveform curve forward, then return along zero.
        # Fill with a vertical linear gradient: accent colour (semi-transparent)
        # at the zero line, fully transparent at top and bottom edges.
        # This is the standard DAW waveform fill style.
        fill_path = QPainterPath(glow_path)   # copy the already-built path
        fill_path.lineTo(w - px, cy)          # close along the zero line
        fill_path.lineTo(px, cy)
        fill_path.closeSubpath()

        gradient = QLinearGradient(0, pt, 0, pt + draw_h)
        top_color = QColor(CURVE_COLOR)
        top_color.setAlpha(0)
        mid_color = QColor(CURVE_COLOR)
        mid_color.setAlpha(FILL_ALPHA_TOP)
        gradient.setColorAt(0.0, top_color)
        gradient.setColorAt(0.5, mid_color)
        gradient.setColorAt(1.0, top_color)

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # ── 7. Sharp foreground waveform line ─────────────────────────────────
        painter.setPen(QPen(
            QColor(CURVE_COLOR),
            CURVE_STROKE_WIDTH,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        ))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(glow_path)

        # ── 8. Amplitude axis labels (left side) ──────────────────────────────
        label_font = QFont("IBM Plex Mono", 9)
        painter.setFont(label_font)
        painter.setPen(QPen(QColor(LABEL_COLOR)))
        amp_labels = {1.0: "+1.0", 0.5: "+0.5", 0.0: " 0.0", -0.5: "−0.5", -1.0: "−1.0"}
        for amp, text in amp_labels.items():
            gy = int(pt + (1.0 - amp) / 2.0 * draw_h)
            painter.drawText(0, gy - 6, px - 4, 14, Qt.AlignmentFlag.AlignRight, text)

        # ── 9. Time axis labels (bottom) ──────────────────────────────────────
        # Label each cycle boundary with its time in milliseconds.
        # Period T = 1 / frequency. Labels: 0 ms, T ms, 2T ms, … , N*T ms.
        period_ms = 1000.0 / max(self._frequency, 1.0)
        time_font = QFont("IBM Plex Mono", 8)
        painter.setFont(time_font)
        painter.setPen(QPen(QColor(LABEL_COLOR)))
        for c in range(DISPLAY_CYCLES + 1):
            t_ms = c * period_ms
            label = f"{t_ms:.2f} ms" if t_ms > 0 else "0 ms"
            tx = int(px + c / DISPLAY_CYCLES * draw_w)
            painter.drawText(tx - 28, h - pb + 6, 56, 20,
                             Qt.AlignmentFlag.AlignCenter, label)

        # ── 10. Border around the draw area ───────────────────────────────────
        border_pen = QPen(QColor(AXIS_COLOR), 1)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(px, pt, draw_w, draw_h)

        # ── 11. Status indicator (top-right corner) ───────────────────────────
        dot_x = w - px + 8
        dot_y = pt + STATUS_DOT_RADIUS + 2
        if self._is_playing:
            dot_color = QColor("#22c55e")   # green — playing
            status_text = "PLAYING"
        else:
            dot_color = QColor("#5a6175")   # muted — stopped
            status_text = "STOPPED"
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            dot_x - STATUS_DOT_RADIUS, dot_y - STATUS_DOT_RADIUS,
            STATUS_DOT_RADIUS * 2, STATUS_DOT_RADIUS * 2,
        )
        status_font = QFont("IBM Plex Mono", 7)
        painter.setFont(status_font)
        painter.setPen(QPen(dot_color))
        painter.drawText(dot_x + STATUS_DOT_RADIUS + 4, dot_y + 4, status_text)

        # ── 12. Frequency + gain annotation (top-left, inside draw area) ────────
        info_font = QFont("IBM Plex Mono", 8)
        painter.setFont(info_font)
        info_color = QColor(LABEL_COLOR)
        info_color.setAlpha(180)
        painter.setPen(QPen(info_color))
        # dBFS: 20 × log10(gain). Guard against log(0) with a floor of -∞ → "−∞ dB".
        if self._gain > 0.0:
            db_str = f"{20.0 * np.log10(self._gain):+.1f} dBFS"
        else:
            db_str = "−∞ dBFS"
        annotation = (
            f"f = {self._frequency:.1f} Hz   T = {period_ms:.3f} ms   "
            f"gain = {int(round(self._gain * 100))} % ({db_str})"
        )
        painter.drawText(px + 8, pt + 16, annotation)

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

        # AUDIO CONCEPT — Choosing the right host API on Windows:
        # sounddevice's default on Windows resolves to MME, the oldest and
        # least reliable Windows audio API. MME adds its own buffering on top
        # of the block size, introduces timing jitter, and causes crackling and
        # noise interference — especially with professional USB interfaces like
        # the Focusrite Scarlett/Clarett series.
        #
        # We instead target WASAPI (Windows Audio Session API), which is the
        # modern low-latency Windows audio API. We query it by name at runtime
        # so the device index is always resolved correctly, regardless of how
        # many audio devices are connected or in what order.
        #
        # WASAPI EQUIVALENT in Web Audio:
        #   AudioContext({ latencyHint: 'interactive' }) — both select the
        #   lowest-latency path to the hardware.
        output_device = self._find_wasapi_output()

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
            device=output_device,
            callback=self._callback,
        )

    @staticmethod
    def _find_wasapi_output() -> int | None:
        """Return the default output device index for the WASAPI host API.

        Queries host APIs by name at runtime so the index is always correct,
        even if devices are added, removed, or reordered between runs.

        Returns:
            The WASAPI default output device index, or None to fall back to
            sounddevice's own default (better than MME, still not ideal).
        """
        for api in sd.query_hostapis():
            if api["name"] == "Windows WASAPI":
                idx = api["default_output_device"]
                if idx >= 0:
                    print(f"[AudioEngine] using WASAPI output device {idx}: "
                          f"{sd.query_devices(idx)['name']}", flush=True)
                    return idx
        print("[AudioEngine] WASAPI not found — falling back to system default", flush=True)
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
        """Generate DISPLAY_CYCLES cycles of the current waveform and send to the canvas.

        This runs on the Qt main thread at 30 fps. It reads the current audio
        state (frequency, waveform type, gain) and synthesises exactly
        DISPLAY_CYCLES complete cycles for display.

        OSCILLOSCOPE DESIGN — fixed time window:
        We always show a window of exactly DISPLAY_CYCLES / frequency seconds.
        This means the number of samples rendered scales with frequency:
          • At  20 Hz  (3 cycles): 3 / 20   =  150 ms → 7 200 samples
          • At 440 Hz  (3 cycles): 3 / 440  ≈  6.8 ms → 327 samples
          • At 2000 Hz (3 cycles): 3 / 2000 =  1.5 ms → 72 samples
        The canvas maps however many samples we pass to the full pixel width,
        so the user directly sees the relationship T = 1/f: at high frequency
        the cycle is physically narrower; at low frequency it is wider. This is
        exactly what a Rigol or Tektronix oscilloscope shows.

        The minimum of 64 samples ensures the path has enough points to look
        smooth even at very high frequencies where a single cycle spans only
        a few pixels.

        GAIN — we pass state.gain into display_state (not 1.0). The waveform
        samples in the output array are therefore in [-gain, +gain]. The canvas
        keeps its amplitude grid fixed at ±1.0, so the student sees the wave
        shrink toward the zero line as gain decreases — identical to turning
        the VOLTS/DIV knob on a hardware oscilloscope.

        This is independent of the audio callback — the canvas always shows
        clean cycles starting from phase 0, regardless of the audio block
        boundaries or the engine's current phase accumulator.
        """
        state = self._engine.state

        # Fixed time window: always show exactly DISPLAY_CYCLES complete cycles.
        # n_samples = round(DISPLAY_CYCLES × SAMPLE_RATE / frequency).
        # Clamped to [64, 8192] to avoid degenerate arrays at the slider extremes.
        n_samples = int(round(DISPLAY_CYCLES * SAMPLE_RATE / max(state.frequency, 1.0)))
        n_samples = max(64, min(n_samples, 8_192))

        # Temporary state for display only — do not disturb the engine's phase.
        # gain is taken directly from state so the canvas amplitude reflects the
        # real output level, not a fictional unit-gain shape.
        display_state = WaveformState(
            waveform_type=state.waveform_type,
            frequency=state.frequency,
            gain=state.gain,   # ← real gain: the wave scales on screen
            phase=0.0,         # always start from phase zero for a clean display
        )
        block = waveform_block(display_state, n_samples)   # shape (n_samples, 1)
        self._canvas.set_waveform(
            data=block[:, 0],                              # 1-D array in [-gain, +gain]
            frequency=state.frequency,
            gain=state.gain,                               # forwarded for annotation
            waveform_name=WAVEFORM_NAMES[state.waveform_type].upper(),
            is_playing=self._controls.is_playing,
        )

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
