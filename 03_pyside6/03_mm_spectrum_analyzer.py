# =============================================================================
# 03_mm_spectrum_analyzer.py — Real-Time Spectrum Analyzer
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A professional real-time audio spectrum analyzer with a live configurable
#   IIR filter pipeline. Microphone input is captured via a sounddevice
#   InputStream callback.
#
#   Each audio block is windowed (Hann), transformed with numpy.fft.rfft,
#   converted to dBFS, and passed to the Qt main thread via a queue.Queue.
#   A QTimer fires at 30 fps, drains the queue, and updates the QPainter
#   spectrum display.
#
#   A QComboBox selects the filter type (lowpass, highpass, bandpass, bandstop).
#   A QSlider controls the cutoff frequency. The filter uses scipy.signal.butter
#   with output='sos' and scipy.signal.sosfilt with carried state (zi) — the
#   professional pattern for stateful real-time IIR filtering. The filtered
#   signal is sent to the audio output so you can hear the filter working.
#   Both the raw and filtered spectra are overlaid on the same canvas.
#
#   A mini filter-response panel (QPainter) shows the Bode magnitude plot of
#   the current filter, computed with scipy.signal.sosfreqz, updated on every
#   filter parameter change.
#
#   WEBGL EQUIVALENT:
#     sounddevice InputStream callback  ↔  MediaStreamSource + AudioWorklet
#     numpy.fft.rfft                    ↔  AnalyserNode.getFloatFrequencyData()
#     scipy.signal.butter + sosfilt     ↔  BiquadFilterNode (single 2nd-order
#                                           section; sosfilt chains multiple)
#     SpectrumCanvas QPainter           ↔  Canvas 2D drawing on AnalyserNode data
#
# How to run:
#   uv sync                                         # first time only
#   uv run python 03_mm_spectrum_analyzer.py
#
# Ubuntu system dependencies (install once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0 libportaudio2
#
# Connections to other seminars:
#   Demo 1 (hello_multimedia)  — introduces the sounddevice callback pattern.
#   Demo 2 (waveform_player)   — introduces the queue.Queue audio/UI bridge and
#                                 the threading.Lock pattern used here for filter
#                                 state swaps.
#   Demo 5 (multimedia_workbench) — integrates SpectrumCanvas and AudioProcessor
#                                    directly as reusable components.
# =============================================================================

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfreqz
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# STEP 1 — Constants
# =============================================================================

WINDOW_TITLE: str = "Spectrum Analyzer — Multimediaprogrammierung Seminar 03"
WINDOW_MIN_WIDTH: int = 1_000
WINDOW_MIN_HEIGHT: int = 580

# Audio I/O
SAMPLE_RATE: int = 48_000       # Hz — standard professional sample rate
BLOCK_SIZE: int = 1_024         # frames per callback block
                                # larger than Demo 1/2: gives more FFT resolution
                                # at the cost of slightly higher latency

# FFT parameters
NFFT: int = 2_048               # FFT size — must be >= BLOCK_SIZE for zero-padding
                                # Frequency resolution = SAMPLE_RATE / NFFT
                                # = 48000 / 2048 ≈ 23.4 Hz per bin
DBFS_FLOOR: float = -90.0       # dBFS — bins below this are clamped (log(0) guard)

# Display frequency range (Hz) — matches the standard audio spectrum display
FREQ_MIN_HZ: float = 20.0       # lower limit of human hearing
FREQ_MAX_HZ: float = 20_000.0   # upper limit of human hearing

# Standard audio frequency grid lines (Hz) — logarithmically spaced
FREQ_GRID_HZ: tuple[float, ...] = (
    20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000
)

# dBFS grid lines
DBFS_GRID: tuple[float, ...] = (-90.0, -60.0, -30.0, -12.0, -6.0, 0.0)

# Display refresh
DISPLAY_FPS: int = 30
DISPLAY_TIMER_MS: int = 1_000 // DISPLAY_FPS    # 33 ms

# Filter parameter ranges
CUTOFF_MIN_HZ: int = 20
CUTOFF_MAX_HZ: int = 20_000
DEFAULT_CUTOFF_HZ: int = 1_000
DEFAULT_FILTER_ORDER: int = 4
FILTER_TYPES: tuple[str, ...] = ("lowpass", "highpass", "bandpass", "bandstop")
FILTER_TYPE_LABELS: tuple[str, ...] = ("Low-pass", "High-pass", "Band-pass", "Band-stop")

# Canvas colours — dark theme, consistent with the course HTML design system
CANVAS_BG: str = "#0d0f14"
GRID_COLOR: str = "#1a1f2b"
AXIS_COLOR: str = "#252b38"
LABEL_COLOR: str = "#5a6175"
RAW_COLOR_RGBA: tuple[int, int, int, int] = (79, 156, 249, 160)    # --accent blue
FILTERED_COLOR_RGBA: tuple[int, int, int, int] = (167, 139, 250, 160)  # --accent2 purple
CURSOR_COLOR: str = "#fb923c"   # --orange
RESPONSE_COLOR: str = "#34d399"  # --green — filter response plot

# AMS logo — one directory above this script (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"


# =============================================================================
# STEP 2 — FilterState: IIR filter design and stateful execution
# =============================================================================

@dataclass
class FilterState:
    """Holds the current IIR filter design and its real-time execution state.

    DSP CONCEPT — SOS filter state (zi):
    An IIR filter has memory: its output at time n depends not just on the
    current input but on past inputs and past outputs. In a real-time system
    the audio arrives in blocks. Between blocks the filter's internal state
    (the delay line values) must be preserved — otherwise each block is
    filtered from silence, producing a transient (pop or click) at every
    block boundary, 47 times per second at 48 000 Hz / 1 024 frames.

    The ``zi`` array (initial conditions for sosfilt) is the filter's memory.
    It has shape ``(n_sections, 2)`` where n_sections = order / 2 for SOS form.
    sosfilt returns the updated zi at the end of each block:

        filtered, zi = sosfilt(sos, block, zi=zi)

    The returned zi is fed back in as the initial condition for the next block.
    This is mathematically equivalent to running the filter on the entire
    continuous signal — just split into pieces for real-time processing.

    WHY sosfilt_zi:
    sosfilt_zi(sos) returns the initial conditions that produce a step response
    (i.e. if the input were a constant 1.0 for all past time). Multiplying by 0
    gives the correct initial conditions for a signal that was 0.0 for all past
    time — which is what we want at the start of a session (silence before
    the microphone was opened).

    WHY output='sos' (second-order sections):
    A 4th-order Butterworth filter in direct-form (b, a) coefficients develops
    numerical instability at low cutoff frequencies (below ~100 Hz at 48 kHz).
    SOS form splits the filter into 2nd-order sections chained in series.
    Each section has well-conditioned coefficients. Always use output='sos'.

    Attributes:
        filter_type: One of 'lowpass', 'highpass', 'bandpass', 'bandstop'.
        cutoff_hz: Cutoff frequency in Hz. For bandpass/bandstop this is the
                   centre frequency; bandwidth is fixed at 1 octave.
        order: Filter order (2, 4, 6, or 8). Higher = steeper roll-off.
        sos: SOS coefficient array, shape (n_sections, 6). None until rebuilt.
        zi: Filter state array, shape (n_sections, 2). None until rebuilt.
    """

    filter_type: str = "lowpass"
    cutoff_hz: float = float(DEFAULT_CUTOFF_HZ)
    order: int = DEFAULT_FILTER_ORDER
    sos: np.ndarray | None = field(default=None, repr=False)
    zi: np.ndarray | None = field(default=None, repr=False)

    def rebuild(self, samplerate: int) -> None:
        """Design the filter and initialise its state.

        For bandpass and bandstop the cutoff defines the centre frequency.
        The band edges are set one octave apart (centre / √2, centre × √2),
        which gives a one-octave bandwidth — a reasonable default for a
        demonstration filter.

        Args:
            samplerate: Audio sample rate in Hz. Used to normalise Wn.
        """
        nyq: float = samplerate / 2.0

        if self.filter_type in ("bandpass", "bandstop"):
            # One-octave band: [cutoff / sqrt(2), cutoff * sqrt(2)]
            low = max(self.cutoff_hz / (2 ** 0.5), 1.0)
            high = min(self.cutoff_hz * (2 ** 0.5), nyq - 1.0)
            wn: float | tuple[float, float] = (low / nyq, high / nyq)
        else:
            wn = min(self.cutoff_hz, nyq - 1.0) / nyq

        self.sos = butter(
            N=self.order,
            Wn=wn,
            btype=self.filter_type,
            output="sos",
        )

        # Initialise zi for a signal that was silence before t=0.
        # sosfilt_zi gives the step-response initial conditions;
        # multiplying by 0 gives the zero-signal initial conditions.
        self.zi = sosfilt_zi(self.sos) * 0.0

    @property
    def is_ready(self) -> bool:
        """True if the filter has been built and is ready to apply."""
        return self.sos is not None and self.zi is not None


# =============================================================================
# STEP 3 — AudioProcessor: duplex stream with real-time filter
# =============================================================================

class AudioProcessor:
    """Manages a sounddevice duplex stream (microphone in, monitor out).

    The callback runs entirely on the PortAudio audio thread. It:
      1. Copies the input block (never holds a reference to indata).
      2. Applies the SOS filter with carried state zi.
      3. Writes filtered audio to outdata for monitoring.
      4. Pushes (raw_block, filtered_block) to a queue.Queue for display.

    Filter state is swapped atomically using a threading.Lock. The Lock
    protects the entire filter_state object — sos and zi together — because
    rebuilding a filter replaces both. An inconsistent state (old sos, new zi)
    would produce garbage output.

    THREAD CONCEPT — queue.Queue as the audio/UI bridge:
    The sounddevice callback runs on a high-priority OS audio thread. Qt
    widgets live on the main thread. These two threads must never share
    mutable state directly — the GIL does not protect non-atomic operations,
    and Qt's event loop is not re-entrant.

    queue.Queue is the standard Python solution: it is thread-safe by design
    (uses an internal threading.Lock), and its put_nowait / get_nowait methods
    are safe to call from any thread. The callback pushes; the QTimer drains.

    The queue has maxsize=4. If the UI falls behind (e.g. the window is
    minimised), put_nowait raises queue.Full and the callback discards the
    block rather than blocking. Audio playback continues uninterrupted.
    """

    def __init__(self) -> None:
        """Initialise the processor. Call start() to open the stream."""
        self._filter_state: FilterState = FilterState()
        self._filter_state.rebuild(SAMPLE_RATE)
        self._lock: threading.Lock = threading.Lock()

        # Queue for (raw_block, filtered_block) tuples — float32, shape (N,)
        self._audio_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=4)

        self._overflow_count: int = 0   # counts input overflows for diagnostics
        self._stream: sd.Stream | None = None

    # ── Audio callback — runs on the PortAudio audio thread ──────────────────

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Process one block of audio from microphone to speaker.

        Args:
            indata:    Input buffer, shape (frames, channels), float32. READ-ONLY.
            outdata:   Output buffer to fill, shape (frames, channels), float32.
            frames:    Number of frames in this block.
            time_info: PortAudio timing struct (not used here).
            status:    Stream health flags — check for overflows.
        """
        # CRITICAL: always copy indata before using it.
        # indata is a view into PortAudio's internal ring buffer. After this
        # callback returns, PortAudio overwrites it with the next block.
        # Storing a reference (not a copy) to indata leads to silent data
        # corruption — one of the most confusing bugs in audio programming.
        raw: np.ndarray = indata[:, 0].copy()   # mono: take channel 0

        if status.input_overflow:
            self._overflow_count += 1
            if self._overflow_count % 10 == 1:   # rate-limit log spam
                print(
                    f"[AudioProcessor] input overflow ×{self._overflow_count} "
                    f"— consider increasing BLOCK_SIZE",
                    flush=True,
                )

        # Apply the filter with carried state (zi).
        # This entire block is protected by the Lock so a concurrent set_filter()
        # call cannot swap sos/zi in the middle of the filter application.
        with self._lock:
            fs = self._filter_state
            if fs.is_ready:
                filtered, fs.zi = sosfilt(fs.sos, raw, zi=fs.zi)
                filtered = filtered.astype(np.float32)
            else:
                filtered = raw.copy()

        # Write filtered audio to the output for monitoring.
        # Reshape to (frames, channels) — outdata is always 2-D.
        if outdata.shape[1] == 1:
            outdata[:, 0] = filtered
        else:
            # Duplicate mono filtered signal to all output channels
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = filtered

        # Push (raw, filtered) to the display queue — non-blocking.
        # If the queue is full the UI is behind; discard the block.
        # Audio output continues regardless — the UI and audio are decoupled.
        try:
            self._audio_queue.put_nowait((raw, filtered))
        except queue.Full:
            pass

    # ── Filter management ──────────────────────────────────────────────────────

    def set_filter(self, new_state: FilterState) -> None:
        """Swap the active filter state atomically.

        Rebuilds the new state's SOS and zi before acquiring the lock, so the
        lock hold time is minimal — just the pointer swap, not the filter design.
        Resetting zi on filter change avoids a transient pop: the new filter
        starts from silence rather than from the previous filter's internal state.

        Args:
            new_state: The new FilterState. Must have rebuild() called first.
        """
        new_state.rebuild(SAMPLE_RATE)
        with self._lock:
            self._filter_state = new_state

    def get_filter_state(self) -> FilterState:
        """Return a copy of the current filter state (for UI display).

        Returns:
            A copy of the active FilterState. Safe to read on the main thread.
        """
        with self._lock:
            import copy
            return copy.copy(self._filter_state)

    # ── Stream lifecycle ───────────────────────────────────────────────────────

    def start(self, input_device: int | None = None) -> None:
        """Open the duplex stream and start the callback.

        Args:
            input_device: sounddevice device index for the input. None = default.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

        try:
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=(input_device, None),   # (input, output) — None = default output
                channels=(1, 2),               # 1 input channel (mono mic), 2 output channels
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Cannot open audio device: {exc}") from exc

    def stop(self) -> None:
        """Stop the stream (keeps the device open for restart)."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def close(self) -> None:
        """Stop and release the audio device. Call from closeEvent."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def audio_queue(self) -> queue.Queue[tuple[np.ndarray, np.ndarray]]:
        """Queue polled by MainWindow's QTimer for (raw, filtered) blocks."""
        return self._audio_queue


# =============================================================================
# STEP 4 — fft_magnitude_db(): pure NumPy FFT with Hann windowing
# =============================================================================

# Pre-compute the Hann window once at module load — reused every callback block.
# Creating a new window array every block would allocate memory in the hot path.
_HANN_WINDOW: np.ndarray = np.hanning(NFFT).astype(np.float32)


def fft_magnitude_db(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Hann window, compute the real FFT, and convert to dBFS.

    DSP CONCEPT — Windowing and spectral leakage:
    The FFT assumes the input signal repeats periodically with period NFFT.
    If the signal does not complete an exact whole number of cycles within
    the NFFT window (which is almost always the case for arbitrary audio),
    the edges of the window contain a discontinuity — the end of the block
    does not connect smoothly to the beginning. The FFT interprets this
    discontinuity as high-frequency energy, smearing spectral energy from
    strong components into neighbouring frequency bins. This smearing is
    called spectral leakage.

    The Hann window (raised cosine) tapers both ends of the block smoothly
    to zero, eliminating the edge discontinuity. The trade-off:
      - Less leakage (sharper spectral peaks for pure tones).
      - Slightly lower frequency resolution (each bin is a weighted average
        of a small neighbourhood rather than a single frequency).
    For audio spectrum displays the Hann window is the standard choice.

    DSP CONCEPT — rfft vs fft:
    Audio signals are real-valued (no imaginary component). np.fft.rfft
    exploits this: it computes only the positive-frequency half of the
    spectrum (bins 0 to NFFT//2 inclusive), discarding the redundant
    negative-frequency mirror. This halves the computation and output size
    compared to np.fft.fft. Always use rfft for real-valued audio.

    Frequency resolution = SAMPLE_RATE / NFFT = 48000 / 2048 ≈ 23.4 Hz/bin.
    Number of output bins  = NFFT // 2 + 1 = 1025.

    Args:
        block: 1-D float32 audio array of any length ≤ NFFT.

    Returns:
        freqs: Frequency axis in Hz, shape (NFFT//2 + 1,).
        mag_db: Magnitude in dBFS, shape (NFFT//2 + 1,), floor at DBFS_FLOOR.
    """
    # Zero-pad or truncate to NFFT samples
    n = len(block)
    if n < NFFT:
        padded = np.zeros(NFFT, dtype=np.float32)
        padded[:n] = block
    else:
        padded = block[:NFFT].copy()

    # Apply the Hann window
    windowed = padded * _HANN_WINDOW

    # Real FFT — output shape: (NFFT//2 + 1,)
    spectrum = np.fft.rfft(windowed)

    # Convert to dBFS.
    # 1. np.abs() gives the magnitude of each complex bin.
    # 2. Divide by (NFFT / 2) to normalise for the FFT size and one-sided output.
    #    This makes a full-scale sine wave appear at 0 dBFS regardless of NFFT.
    # 3. 20 * log10() converts linear amplitude to decibels.
    # 4. + 1e-9 (epsilon) prevents log10(0) = -inf for silent bins.
    magnitude = np.abs(spectrum) / (NFFT / 2.0)
    mag_db = 20.0 * np.log10(magnitude + 1e-9)
    mag_db = np.clip(mag_db, DBFS_FLOOR, 0.0)   # clamp to [floor, 0 dBFS]

    # Frequency axis: bin k corresponds to frequency k * (SAMPLE_RATE / NFFT)
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / SAMPLE_RATE).astype(np.float32)

    return freqs, mag_db.astype(np.float32)


# =============================================================================
# STEP 5 — SpectrumCanvas: QPainter spectrum display
# =============================================================================

class SpectrumCanvas(QWidget):
    """QPainter widget displaying the real-time audio spectrum.

    Overlays two spectra:
      - Raw microphone input: blue (--accent)
      - Filter output:        purple (--accent2)

    The frequency axis is logarithmic — one octave spans the same pixel
    width regardless of where it falls on the axis. This matches human
    auditory perception and is the standard for professional spectrum
    analyzers and equalizers.

    A frequency cursor follows the mouse and shows Hz and dBFS at that point.
    An optional peak-hold mode retains the maximum dBFS at each frequency bin.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with empty spectra."""
        super().__init__(parent)
        self.setMinimumSize(500, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)   # receive mouseMoveEvent without button press

        n_bins: int = NFFT // 2 + 1
        self._freqs: np.ndarray = np.zeros(n_bins, dtype=np.float32)
        self._raw_db: np.ndarray = np.full(n_bins, DBFS_FLOOR, dtype=np.float32)
        self._filtered_db: np.ndarray = np.full(n_bins, DBFS_FLOOR, dtype=np.float32)

        # Peak-hold arrays — hold the maximum seen at each bin
        self._peak_raw: np.ndarray = np.full(n_bins, DBFS_FLOOR, dtype=np.float32)
        self._peak_filtered: np.ndarray = np.full(n_bins, DBFS_FLOOR, dtype=np.float32)
        self._peak_hold: bool = False

        # Cursor state
        self._cursor_x: int = -1    # -1 means no cursor (mouse outside widget)

    # ── Public interface ───────────────────────────────────────────────────────

    def set_spectrum(
        self,
        raw_db: np.ndarray,
        filtered_db: np.ndarray,
        freqs: np.ndarray,
    ) -> None:
        """Update the displayed spectra and repaint.

        Args:
            raw_db:      Raw spectrum magnitude in dBFS, shape (n_bins,).
            filtered_db: Filtered spectrum magnitude in dBFS, shape (n_bins,).
            freqs:       Frequency axis in Hz, shape (n_bins,).
        """
        self._freqs = freqs
        self._raw_db = raw_db
        self._filtered_db = filtered_db

        if self._peak_hold:
            np.maximum(self._peak_raw, raw_db, out=self._peak_raw)
            np.maximum(self._peak_filtered, filtered_db, out=self._peak_filtered)

        self.update()

    def set_peak_hold(self, enabled: bool) -> None:
        """Enable or disable peak hold.

        Args:
            enabled: If True, peak values accumulate. If False, reset peaks.
        """
        self._peak_hold = enabled
        if not enabled:
            n = len(self._raw_db)
            self._peak_raw = np.full(n, DBFS_FLOOR, dtype=np.float32)
            self._peak_filtered = np.full(n, DBFS_FLOOR, dtype=np.float32)

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _freq_to_px(self, freq_hz: float) -> float:
        """Map a frequency in Hz to a pixel X position on the log axis.

        DSP CONCEPT — Log frequency axis:
        Human hearing perceives pitch logarithmically: each doubling of
        frequency (one octave) sounds like the same interval regardless of
        where it falls on the scale. 100→200 Hz sounds like the same step
        as 1000→2000 Hz or 10000→20000 Hz.

        On a linear frequency axis, the bass (20–200 Hz) would occupy only
        1 % of the width, making it impossible to distinguish. On a log axis
        the bass, midrange, and treble each occupy roughly equal screen width
        — matching how musicians and engineers think about frequency content.

        The formula:
            px = width × log10(f / FREQ_MIN_HZ) / log10(FREQ_MAX_HZ / FREQ_MIN_HZ)

        Maps [FREQ_MIN_HZ, FREQ_MAX_HZ] → [0, width] on a log scale.
        log10(20/20) = 0 → px = 0.
        log10(20000/20) = log10(1000) = 3 → px = width.
        Each decade (×10) spans width / 3 pixels (since log10(1000)/3 = 1).

        Args:
            freq_hz: Frequency in Hz, clamped to [FREQ_MIN_HZ, FREQ_MAX_HZ].

        Returns:
            Pixel X position (float).
        """
        import math
        freq_hz = max(FREQ_MIN_HZ, min(freq_hz, FREQ_MAX_HZ))
        log_range: float = math.log10(FREQ_MAX_HZ / FREQ_MIN_HZ)
        return self.width() * math.log10(freq_hz / FREQ_MIN_HZ) / log_range

    def _dbfs_to_py(self, db: float) -> float:
        """Map a dBFS value to a pixel Y position.

        Maps [DBFS_FLOOR, 0] → [height, 0].
        0 dBFS is at the top; DBFS_FLOOR is at the bottom.

        Args:
            db: Magnitude in dBFS.

        Returns:
            Pixel Y position (float).
        """
        db = max(DBFS_FLOOR, min(db, 0.0))
        fraction: float = (db - DBFS_FLOOR) / (0.0 - DBFS_FLOOR)   # 0 at floor, 1 at 0 dBFS
        return self.height() * (1.0 - fraction)

    def _px_to_freq(self, px: int) -> float:
        """Map a pixel X position to a frequency in Hz (inverse of _freq_to_px).

        Args:
            px: Pixel X position.

        Returns:
            Frequency in Hz.
        """
        import math
        frac = max(0.0, min(1.0, px / max(self.width(), 1)))
        log_range: float = math.log10(FREQ_MAX_HZ / FREQ_MIN_HZ)
        return FREQ_MIN_HZ * (10.0 ** (frac * log_range))

    def _px_to_dbfs(self, px: int) -> float:
        """Return the raw spectrum dBFS at the nearest frequency bin to px.

        Args:
            px: Pixel X position.

        Returns:
            dBFS value, or DBFS_FLOOR if no data.
        """
        if len(self._freqs) == 0 or len(self._raw_db) == 0:
            return DBFS_FLOOR
        target_hz = self._px_to_freq(px)
        idx = int(np.argmin(np.abs(self._freqs - target_hz)))
        return float(self._raw_db[idx])

    # ── Mouse tracking ─────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Track cursor X for the frequency readout."""
        self._cursor_x = event.pos().x()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        """Hide the cursor when the mouse leaves the widget."""
        self._cursor_x = -1
        self.update()

    # ── QPainter rendering ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw background, grid, spectra, peak hold, and cursor."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # ── Background ────────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        # ── dBFS grid lines ────────────────────────────────────────────────────
        painter.setFont(QFont("IBM Plex Mono", 9))
        for db in DBFS_GRID:
            py = int(self._dbfs_to_py(db))
            painter.setPen(QPen(QColor(GRID_COLOR), 1))
            painter.drawLine(0, py, w, py)
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            label = f"{int(db):+d} dBFS" if db != 0 else " 0 dBFS"
            painter.drawText(4, py - 2, label)

        # ── Frequency grid lines ───────────────────────────────────────────────
        for freq in FREQ_GRID_HZ:
            px = int(self._freq_to_px(freq))
            painter.setPen(QPen(QColor(GRID_COLOR), 1))
            painter.drawLine(px, 0, px, h)
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            label = f"{int(freq / 1000)}k" if freq >= 1_000 else str(int(freq))
            painter.drawText(px + 2, h - 4, label)

        # ── Filtered spectrum (purple, behind raw) ─────────────────────────
        if len(self._filtered_db) > 0 and len(self._freqs) > 0:
            self._draw_spectrum(painter, self._freqs, self._filtered_db, FILTERED_COLOR_RGBA)

        # ── Raw spectrum (blue, on top) ────────────────────────────────────
        if len(self._raw_db) > 0 and len(self._freqs) > 0:
            self._draw_spectrum(painter, self._freqs, self._raw_db, RAW_COLOR_RGBA)

        # ── Peak hold lines ───────────────────────────────────────────────────
        if self._peak_hold and len(self._freqs) > 0:
            self._draw_peak_line(painter, self._freqs, self._peak_raw, RAW_COLOR_RGBA)
            self._draw_peak_line(painter, self._freqs, self._peak_filtered, FILTERED_COLOR_RGBA)

        # ── Frequency cursor ───────────────────────────────────────────────────
        if self._cursor_x >= 0:
            cursor_hz = self._px_to_freq(self._cursor_x)
            cursor_db = self._px_to_dbfs(self._cursor_x)
            painter.setPen(QPen(QColor(CURSOR_COLOR), 1))
            painter.drawLine(self._cursor_x, 0, self._cursor_x, h)
            tooltip = (
                f"{cursor_hz:.0f} Hz  {cursor_db:.1f} dBFS"
                if cursor_hz < 1_000
                else f"{cursor_hz / 1_000:.2f} kHz  {cursor_db:.1f} dBFS"
            )
            painter.setPen(QPen(QColor(CURSOR_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 10))
            tx = min(self._cursor_x + 6, w - 160)
            painter.drawText(tx, 18, tooltip)

        painter.end()

    def _draw_spectrum(
        self,
        painter: QPainter,
        freqs: np.ndarray,
        mag_db: np.ndarray,
        rgba: tuple[int, int, int, int],
    ) -> None:
        """Draw a filled spectrum path for the given data arrays.

        The spectrum is drawn as a filled path: the top edge follows the
        magnitude curve and the bottom edge is the floor line (DBFS_FLOOR).
        This gives a solid filled shape, consistent with professional
        spectrum analyzer displays.

        Args:
            painter: Active QPainter.
            freqs:   Frequency array in Hz, shape (n_bins,).
            mag_db:  Magnitude in dBFS, shape (n_bins,).
            rgba:    Fill colour as (R, G, B, A).
        """
        w = self.width()
        py_floor = int(self._dbfs_to_py(DBFS_FLOOR))

        # Build the filled path.
        # Only include bins within [FREQ_MIN_HZ, FREQ_MAX_HZ].
        path = QPainterPath()
        started = False

        for i in range(len(freqs)):
            f = float(freqs[i])
            if f < FREQ_MIN_HZ or f > FREQ_MAX_HZ:
                continue
            px = self._freq_to_px(f)
            py = self._dbfs_to_py(float(mag_db[i]))
            if not started:
                path.moveTo(px, py_floor)   # start at the floor
                path.lineTo(px, py)
                started = True
            else:
                path.lineTo(px, py)

        if not started:
            return

        # Close the path back to the floor line
        path.lineTo(self._freq_to_px(float(freqs[-1])), py_floor)
        path.closeSubpath()

        r, g, b, a = rgba
        painter.setPen(Qt.PenStyle.NoPen)
        from PySide6.QtGui import QBrush
        painter.setBrush(QBrush(QColor(r, g, b, a)))
        painter.drawPath(path)

    def _draw_peak_line(
        self,
        painter: QPainter,
        freqs: np.ndarray,
        peak_db: np.ndarray,
        rgba: tuple[int, int, int, int],
    ) -> None:
        """Draw the peak-hold line as a thin stroke above the spectrum fill.

        Args:
            painter:  Active QPainter.
            freqs:    Frequency array in Hz.
            peak_db:  Peak dBFS values, shape (n_bins,).
            rgba:     Line colour as (R, G, B, A).
        """
        r, g, b, _ = rgba
        pen = QPen(QColor(r, g, b, 220), 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        started = False
        for i in range(len(freqs)):
            f = float(freqs[i])
            if f < FREQ_MIN_HZ or f > FREQ_MAX_HZ:
                continue
            px = self._freq_to_px(f)
            py = self._dbfs_to_py(float(peak_db[i]))
            if not started:
                path.moveTo(px, py)
                started = True
            else:
                path.lineTo(px, py)

        painter.drawPath(path)


# =============================================================================
# STEP 5b — FilterResponseView: mini Bode magnitude plot
# =============================================================================

class FilterResponseView(QWidget):
    """A compact QPainter plot showing the filter's frequency response.

    Uses scipy.signal.sosfreqz to compute the magnitude response on a
    log-frequency grid and draws it as a line plot on the same dBFS/Hz
    axes as SpectrumCanvas (same coordinate helpers for consistency).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with a flat (all-pass) response."""
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._response_db: np.ndarray = np.zeros(512, dtype=np.float32)
        self._response_hz: np.ndarray = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, 512)

    def update_response(self, filter_state: FilterState) -> None:
        """Recompute and display the filter's magnitude response.

        sosfreqz computes the frequency response of a filter in SOS form
        at the specified frequencies. We evaluate it on a log-spaced grid
        matching the SpectrumCanvas axis so the plot is visually aligned.

        Args:
            filter_state: The current filter state. Must have sos set.
        """
        if filter_state.sos is None:
            return

        # Log-spaced frequency grid from FREQ_MIN_HZ to FREQ_MAX_HZ
        import math
        freqs_hz = np.logspace(
            math.log10(FREQ_MIN_HZ),
            math.log10(FREQ_MAX_HZ),
            512,
        )
        # Normalise to [0, 1] (where 1 = Nyquist)
        worN = freqs_hz / (SAMPLE_RATE / 2.0)
        _, h = sosfreqz(filter_state.sos, worN=worN, fs=2.0)
        mag_db = 20.0 * np.log10(np.abs(h) + 1e-9)
        self._response_db = np.clip(mag_db, DBFS_FLOOR, 6.0).astype(np.float32)
        self._response_hz = freqs_hz.astype(np.float32)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the filter response as a green line plot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor("#0d0f14"))

        # 0 dBFS and −6 dBFS grid lines
        for db in (-90.0, -6.0, 0.0):
            py = int(h * (1.0 - (db - DBFS_FLOOR) / (6.0 - DBFS_FLOOR)))
            painter.setPen(QPen(QColor(GRID_COLOR), 1))
            painter.drawLine(0, py, w, py)

        # Response curve
        import math
        log_range = math.log10(FREQ_MAX_HZ / FREQ_MIN_HZ)

        path = QPainterPath()
        started = False
        for i in range(len(self._response_hz)):
            f = float(self._response_hz[i])
            db = float(self._response_db[i])
            px = w * math.log10(f / FREQ_MIN_HZ) / log_range
            py = h * (1.0 - (db - DBFS_FLOOR) / (6.0 - DBFS_FLOOR))
            py = max(0.0, min(float(h), py))
            if not started:
                path.moveTo(px, py)
                started = True
            else:
                path.lineTo(px, py)

        painter.setPen(QPen(QColor(RESPONSE_COLOR), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(QPen(QColor(LABEL_COLOR)))
        painter.setFont(QFont("IBM Plex Mono", 8))
        painter.drawText(4, h - 4, "Filter response")
        painter.end()


# =============================================================================
# STEP 6 — FilterControlPanel: filter parameters and device selector
# =============================================================================

class FilterControlPanel(QWidget):
    """Right-side panel: filter type, cutoff, order, device, peak hold.

    Signals:
        filter_changed: Emitted when any filter parameter changes.
                        Carries a new FilterState (already rebuilt).
        device_changed: Emitted when a new input device is selected.
                        Carries the sounddevice device index (int).
        peak_hold_changed: Emitted when the peak hold checkbox changes.
                           Carries the new bool value.
    """

    filter_changed: Signal = Signal(object)    # FilterState
    device_changed: Signal = Signal(int)
    peak_hold_changed: Signal = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the control panel UI."""
        super().__init__(parent)
        self.setFixedWidth(260)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all controls."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

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

        layout.addSpacing(4)

        # ── Filter controls group ─────────────────────────────────────────────
        filter_group = QGroupBox("IIR Filter")
        filter_group.setStyleSheet(
            "QGroupBox { color: #a78bfa; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        form = QFormLayout(filter_group)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Filter type selector
        self._type_combo = QComboBox()
        self._type_combo.addItems(list(FILTER_TYPE_LABELS))
        form.addRow("Type", self._type_combo)

        # Cutoff frequency slider (log-mapped display label)
        self._cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self._cutoff_slider.setRange(CUTOFF_MIN_HZ, CUTOFF_MAX_HZ)
        self._cutoff_slider.setValue(DEFAULT_CUTOFF_HZ)
        self._cutoff_label = QLabel(f"{DEFAULT_CUTOFF_HZ} Hz")
        self._cutoff_label.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        form.addRow("Cutoff", self._cutoff_slider)
        form.addRow("", self._cutoff_label)

        # Filter order spinbox — only even orders for SOS Butterworth
        self._order_spin = QSpinBox()
        self._order_spin.setRange(2, 8)
        self._order_spin.setSingleStep(2)
        self._order_spin.setValue(DEFAULT_FILTER_ORDER)
        form.addRow("Order", self._order_spin)

        layout.addWidget(filter_group)

        # ── Filter response view ───────────────────────────────────────────────
        self._response_view = FilterResponseView()
        layout.addWidget(self._response_view)

        # ── Device selector ───────────────────────────────────────────────────
        device_group = QGroupBox("Input Device")
        device_group.setStyleSheet(
            "QGroupBox { color: #4f9cf9; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        dev_layout = QVBoxLayout(device_group)

        self._device_combo = QComboBox()
        self._device_indices: list[int] = []
        self._populate_devices()
        dev_layout.addWidget(self._device_combo)

        layout.addWidget(device_group)

        # ── Peak hold checkbox ────────────────────────────────────────────────
        self._peak_check = QCheckBox("Peak hold")
        self._peak_check.setStyleSheet("color: #d8dce6; font-size: 12px;")
        layout.addWidget(self._peak_check)

        # ── Legend ────────────────────────────────────────────────────────────
        legend_raw = QLabel("■ Raw input")
        legend_raw.setStyleSheet("color: #4f9cf9; font-size: 11px; font-family: 'IBM Plex Mono';")
        legend_filt = QLabel("■ Filtered output")
        legend_filt.setStyleSheet("color: #a78bfa; font-size: 11px; font-family: 'IBM Plex Mono';")
        layout.addWidget(legend_raw)
        layout.addWidget(legend_filt)

        layout.addStretch()

        # ── Internal connections ───────────────────────────────────────────────
        self._type_combo.currentIndexChanged.connect(self._emit_filter)
        self._cutoff_slider.valueChanged.connect(self._on_cutoff_changed)
        self._order_spin.valueChanged.connect(self._emit_filter)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._peak_check.toggled.connect(self.peak_hold_changed)

        # Emit initial filter state
        self._emit_filter()

    def _populate_devices(self) -> None:
        """Populate the device combo box with available input devices."""
        self._device_combo.clear()
        self._device_indices.clear()
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    name = dev["name"]
                    self._device_combo.addItem(f"{i}: {name}")
                    self._device_indices.append(i)
        except Exception:
            self._device_combo.addItem("Default input device")
            self._device_indices.append(-1)

    def _on_cutoff_changed(self, value: int) -> None:
        """Update the cutoff label and emit the filter change."""
        if value >= 1_000:
            self._cutoff_label.setText(f"{value / 1_000:.1f} kHz")
        else:
            self._cutoff_label.setText(f"{value} Hz")
        self._emit_filter()

    def _on_device_changed(self, index: int) -> None:
        """Emit the selected device index."""
        if 0 <= index < len(self._device_indices):
            dev_idx = self._device_indices[index]
            self.device_changed.emit(dev_idx if dev_idx >= 0 else 0)

    def _emit_filter(self) -> None:
        """Build a new FilterState from current controls and emit it."""
        fs = FilterState(
            filter_type=FILTER_TYPES[self._type_combo.currentIndex()],
            cutoff_hz=float(self._cutoff_slider.value()),
            order=self._order_spin.value(),
        )
        fs.rebuild(SAMPLE_RATE)
        self._response_view.update_response(fs)
        self.filter_changed.emit(fs)


# =============================================================================
# STEP 7 — MainWindow: assembly, QTimer polling, device management
# =============================================================================

class MainWindow(QMainWindow):
    """Application main window.

    Layout (QSplitter, horizontal):
    ┌───────────────────────────────────┬───────────────────┐
    │  SpectrumCanvas                   │ FilterControlPanel│
    │  (live QPainter spectrum display) │                   │
    └───────────────────────────────────┴───────────────────┘

    The QTimer at 30 fps drains the AudioProcessor's audio_queue.
    Each (raw, filtered) block pair is transformed via fft_magnitude_db
    and sent to SpectrumCanvas.set_spectrum(). All Qt calls happen here
    on the main thread — the audio callback never touches Qt.
    """

    def __init__(self) -> None:
        """Create processor, panels, timer, and open the audio device."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._processor = AudioProcessor()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(DISPLAY_TIMER_MS)

        self._build_ui()
        self._connect_signals()

        # Start the audio stream with the default input device
        self._start_audio()

        self._poll_timer.start()

    # ── UI assembly ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create and arrange the spectrum canvas and control panel."""
        self._canvas = SpectrumCanvas()
        self._controls = FilterControlPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([740, 260])

        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Listening on default input device…")

    # ── Signal connections ─────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect all signals between panels, processor, and timer."""
        # Filter parameter changes → processor
        self._controls.filter_changed.connect(self._on_filter_changed)

        # Device selection → restart the audio stream
        self._controls.device_changed.connect(self._on_device_changed)

        # Peak hold toggle → canvas
        self._controls.peak_hold_changed.connect(self._canvas.set_peak_hold)

        # 30 fps poll timer → drain audio queue
        self._poll_timer.timeout.connect(self._poll_audio)

    # ── Audio management ───────────────────────────────────────────────────────

    def _start_audio(self, device_index: int | None = None) -> None:
        """Open the audio stream on the specified input device.

        Args:
            device_index: sounddevice device index, or None for default.
        """
        try:
            self._processor.start(device_index)
        except RuntimeError as exc:
            QMessageBox.critical(self, "Audio device error", str(exc))

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_filter_changed(self, filter_state: FilterState) -> None:
        """Pass the new filter state to the audio processor.

        Args:
            filter_state: The new FilterState with rebuilt SOS and zi.
        """
        self._processor.set_filter(filter_state)

    def _on_device_changed(self, device_index: int) -> None:
        """Restart the audio stream with the newly selected input device.

        Args:
            device_index: sounddevice device index.
        """
        self._processor.stop()
        self._start_audio(device_index)
        try:
            dev = sd.query_devices(device_index)
            self.statusBar().showMessage(f"Input: {dev['name']}")
        except Exception:
            self.statusBar().showMessage(f"Input device {device_index}")

    def _poll_audio(self) -> None:
        """Drain the audio queue and update the spectrum canvas.

        Called at 30 fps by the QTimer on the Qt main thread. Takes the
        latest (raw, filtered) block from the queue, runs the FFT, and
        sends the result to the canvas. Older blocks are discarded so the
        display always shows the most recent audio.
        """
        latest: tuple[np.ndarray, np.ndarray] | None = None
        try:
            while True:
                latest = self._processor.audio_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            raw_block, filtered_block = latest
            freqs, raw_db = fft_magnitude_db(raw_block)
            _, filtered_db = fft_magnitude_db(filtered_block)
            self._canvas.set_spectrum(raw_db, filtered_db, freqs)

    # ── Window lifecycle ───────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Stop the timer and release the audio device on close."""
        self._poll_timer.stop()
        self._processor.close()
        super().closeEvent(event)


# =============================================================================
# Entry point
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
# End of 03_mm_spectrum_analyzer.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
