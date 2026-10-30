# =============================================================================
# 03_mm_multimedia_workbench.py — Full Multimedia Workbench
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A complete, professional multimedia analysis and processing workbench
#   combining all components from demos 1–4 into a single multi-panel
#   application.
#
#   Open a video file (audio extracted automatically via ffmpeg). The left
#   side shows the video frame display with a live DSP selector. The bottom
#   left shows a WaveformView of the audio with a live playhead. The centre
#   shows a real-time SpectrumCanvas driven by the playback stream (not the
#   microphone). A TimelineWidget provides full transport controls. A
#   ParameterPanel exposes all DSP parameters for both audio and video.
#
#   All components communicate exclusively through signals — no direct
#   cross-panel function calls. The architecture mirrors a professional
#   broadcast monitoring or media analysis tool. QSettings persists window
#   geometry, panel sizes, last opened file, and device selection between
#   sessions.
#
#   ARCHITECTURE CONCEPT — Signal-only inter-component communication:
#   No panel holds a reference to another panel. Every interaction goes
#   through MainWindow via signals. This is what makes the components
#   reusable and testable in isolation: replace the VideoDisplay with a
#   camera feed, replace the ParameterPanel with a MIDI controller, and
#   neither side needs to change.
#
#   ARCHITECTURE CONCEPT — Model / View separation:
#   MediaSession is the model. It has no Qt dependencies. The view
#   components (VideoDisplay, WaveformView, SpectrumCanvas) have no
#   knowledge of MediaSession. They receive data through signals routed
#   by MainWindow.
#
#   WEBGL EQUIVALENT:
#     sounddevice OutputStream   ↔  AudioContext (master clock for A/V sync)
#     outputBufferDacTime        ↔  AudioContext.currentTime
#     numpy.fft.rfft             ↔  AnalyserNode.getFloatFrequencyData()
#     scipy.signal.butter+sosfilt↔  BiquadFilterNode (SOS chains multiple)
#     cv2.VideoCapture.read()    ↔  HTMLVideoElement.requestVideoFrameCallback()
#     cv2.cvtColor + QImage      ↔  drawImage(video, canvas)
#     cv2.Canny / cv2.GaussianBlur ↔ WebGL fragment shader (per-pixel DSP)
#
# How to run:
#   uv sync                                             # first time only
#   uv run python 03_mm_multimedia_workbench.py
#
# Ubuntu system dependencies (install once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0 \
#                    libportaudio2 libsndfile1 ffmpeg libopencv-dev
#
# Connections to other seminars:
#   Demo 1 (hello_multimedia)     — phase accumulation, audio callback pattern
#   Demo 2 (waveform_player)      — WaveformView, PlaybackEngine, DspChain
#   Demo 3 (spectrum_analyzer)    — SpectrumCanvas, FilterState, AudioProcessor
#   Demo 4 (video_player)         — VideoDisplay, MediaFile, AudioPlaybackEngine,
#                                    frame DSP pipeline, A/V sync loop
# =============================================================================

from __future__ import annotations

import copy
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from math import gcd, log10
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi, sosfreqz
from PySide6.QtCore import Qt, QSettings, QSize, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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

WINDOW_TITLE: str = "Multimedia Workbench — Multimediaprogrammierung Seminar 03"
WINDOW_MIN_WIDTH: int = 1_280
WINDOW_MIN_HEIGHT: int = 720

# QSettings keys — used to persist state between sessions
SETTINGS_ORG: str = "TU Ilmenau AMS"
SETTINGS_APP: str = "MM Seminar 03 Workbench"
SETTINGS_GEOMETRY: str = "mainwindow/geometry"
SETTINGS_SPLITTER_H: str = "splitter/horizontal"
SETTINGS_SPLITTER_LEFT: str = "splitter/left_vertical"
SETTINGS_SPLITTER_CENTRE: str = "splitter/centre_vertical"
SETTINGS_RECENT_FILES: str = "files/recent"
SETTINGS_RECENT_MAX: int = 5   # maximum entries in the Recent Files list

# Audio I/O
DEVICE_SAMPLE_RATE: int = 48_000    # Hz — standard professional rate
AUDIO_BLOCK_SIZE: int = 1_024       # frames per callback; larger = more FFT resolution

# FFT parameters (same as Demo 3)
NFFT: int = 2_048                   # FFT size; frequency resolution = 48000/2048 ≈ 23.4 Hz/bin
DBFS_FLOOR: float = -90.0           # dBFS floor for spectrum display

# Spectrum display frequency range
FREQ_MIN_HZ: float = 20.0
FREQ_MAX_HZ: float = 20_000.0
FREQ_GRID_HZ: tuple[float, ...] = (
    20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000
)
DBFS_GRID: tuple[float, ...] = (-90.0, -60.0, -30.0, -12.0, -6.0, 0.0)

# Display timers
DISPLAY_FPS: int = 30
DISPLAY_TIMER_MS: int = 1_000 // DISPLAY_FPS    # 33 ms

# IIR filter parameter ranges
CUTOFF_MIN_HZ: int = 20
CUTOFF_MAX_HZ: int = 20_000
DEFAULT_CUTOFF_HZ: int = 1_000
DEFAULT_FILTER_ORDER: int = 4
FILTER_TYPES: tuple[str, ...] = ("lowpass", "highpass", "bandpass", "bandstop")
FILTER_TYPE_LABELS: tuple[str, ...] = ("Low-pass", "High-pass", "Band-pass", "Band-stop")

# Video DSP modes (same as Demo 4)
DSP_PASSTHROUGH: str = "Passthrough"
DSP_GRAYSCALE: str = "Grayscale"
DSP_CANNY: str = "Canny Edges"
DSP_BLUR: str = "Gaussian Blur"
DSP_EQUALIZE: str = "Histogram EQ"
DSP_MOTION: str = "Motion Diff"
DSP_MODES: tuple[str, ...] = (
    DSP_PASSTHROUGH, DSP_GRAYSCALE, DSP_CANNY,
    DSP_BLUR, DSP_EQUALIZE, DSP_MOTION,
)

CANNY_T1_DEFAULT: int = 50
CANNY_T2_DEFAULT: int = 150
BLUR_KSIZE_DEFAULT: int = 15

# ffmpeg audio extraction command template (same as Demo 4)
FFMPEG_EXTRACT_CMD: list[str] = [
    "ffmpeg", "-i", "{input}", "-vn",
    "-acodec", "pcm_s16le",
    "-ar", str(DEVICE_SAMPLE_RATE),
    "-ac", "2",
    "-y", "{output}",
]
SUPPORTED_VIDEO_FILTER: str = (
    "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.m4v)"
)

# Canvas / UI colours — dark theme, consistent with the course HTML design system
CANVAS_BG: str = "#0d0f14"
GRID_COLOR: str = "#1a1f2b"
ZERO_LINE_COLOR: str = "#252b38"
LABEL_COLOR: str = "#5a6175"
ACCENT_BLUE_RGBA: tuple[int, int, int, int] = (79, 156, 249, 200)   # --accent
ACCENT_PURPLE_RGBA: tuple[int, int, int, int] = (167, 139, 250, 160) # --accent2
PLAYHEAD_COLOR: str = "#fb923c"     # --orange
RESPONSE_COLOR: str = "#34d399"     # --green (filter response plot)
CURSOR_COLOR: str = "#fb923c"

# AMS logo — one directory above this script (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"

# Pre-computed Hann window for FFT (reused every block; never allocated in callback)
_HANN_WINDOW: np.ndarray = np.hanning(NFFT).astype(np.float32)


# =============================================================================
# STEP 2 — Data models (pure Python, zero Qt dependencies)
# =============================================================================

@dataclass
class VideoMeta:
    """Metadata from a VideoCapture object.

    Attributes:
        fps: Frames per second.
        frame_count: Total frame count (may be approximate for some codecs).
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_seconds: Total duration in seconds (computed from fps and frame_count).
    """

    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_seconds = self.frame_count / max(self.fps, 1e-6)


@dataclass
class FilterState:
    """IIR filter design and real-time SOS execution state.

    DSP CONCEPT — SOS filter state (zi):
    In a real-time system, audio arrives in blocks. Between blocks the filter's
    internal state (the delay line values) must be preserved — otherwise each
    block is filtered from silence, producing a transient at every boundary
    (47 times per second at 48 000 Hz / 1 024 frames).

    ``zi`` carries the filter memory between blocks:
        filtered, zi = sosfilt(sos, block, zi=zi)

    The returned zi is fed into the next block's sosfilt call. This is
    mathematically equivalent to running the filter on the entire continuous
    signal — just split into pieces for real-time processing.

    WHY output='sos': A 4th-order Butterworth in b,a form overflows at low
    cutoff frequencies. SOS form splits into 2nd-order sections, each with
    well-conditioned coefficients. Always use output='sos'.

    Attributes:
        filter_type: One of 'lowpass', 'highpass', 'bandpass', 'bandstop'.
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order (2, 4, 6, or 8).
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

        For bandpass/bandstop, the cutoff defines the centre frequency and
        the band is one octave wide (centre / √2 to centre × √2).

        Args:
            samplerate: Audio sample rate in Hz.
        """
        nyq: float = samplerate / 2.0

        if self.filter_type in ("bandpass", "bandstop"):
            low = max(self.cutoff_hz / (2 ** 0.5), 1.0)
            high = min(self.cutoff_hz * (2 ** 0.5), nyq - 1.0)
            wn: float | tuple[float, float] = (low / nyq, high / nyq)
        else:
            wn = min(self.cutoff_hz, nyq - 1.0) / nyq

        self.sos = butter(N=self.order, Wn=wn, btype=self.filter_type, output="sos")
        # sosfilt_zi gives step-response initial conditions; ×0 gives silence start.
        self.zi = sosfilt_zi(self.sos) * 0.0

    @property
    def is_ready(self) -> bool:
        """True if the filter has been designed and is ready to apply."""
        return self.sos is not None and self.zi is not None


# =============================================================================
# STEP 3 — MediaSession: model layer (no Qt)
# =============================================================================

class MediaSession:
    """Holds all state for one open media file.

    Manages the video capture, audio buffer, and audio playback engine as a
    unit. Exposes open(), close(), play(), pause(), stop(), and seek() so
    MainWindow never touches the underlying objects directly.

    ARCHITECTURE CONCEPT — Model / View separation:
    This class is the model. It has no Qt dependencies. It can be unit-tested
    independently of any Qt widget. MainWindow is the controller that mediates
    between this model and the view components.
    """

    def __init__(self) -> None:
        """Initialise with no file open."""
        self._cap: cv2.VideoCapture | None = None
        self._meta: VideoMeta | None = None
        self._audio_data: np.ndarray | None = None      # float32, (frames, 2)
        self._engine: AudioPlaybackEngine | None = None
        self._filter_state: FilterState = FilterState()
        self._filter_state.rebuild(DEVICE_SAMPLE_RATE)
        self._filter_lock: threading.Lock = threading.Lock()
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        # Sequential-read optimisation: track next expected frame so we only
        # issue CAP_PROP_POS_FRAMES on genuine discontinuities (seeks).
        # Initialised to an impossible value to force a seek on first read.
        self._next_frame_idx: int = -999_999
        # Guards all VideoCapture access from the decode thread and main thread.
        # cv2.VideoCapture is NOT thread-safe; concurrent calls corrupt libavcodec.
        self._cap_lock: threading.Lock = threading.Lock()
        # Stored so reopen_video() can recreate the capture after a seek.
        self._video_path: Path | None = None

    # ── File open ──────────────────────────────────────────────────────────────

    def open(self, video_path: Path) -> VideoMeta:
        """Open a video file: open video, extract audio, create engine.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMeta with fps, frame_count, width, height.

        Raises:
            ValueError: If the video cannot be opened.
            RuntimeError: If ffmpeg fails (audio extraction falls back to silence).
        """
        self.close()   # release any previously open resources

        # ── Video ──────────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(
                f"OpenCV cannot open '{video_path.name}'.\n"
                "Ensure the file exists and the codec is installed.\n"
                "Ubuntu: sudo apt install libopencv-dev"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise ValueError(
                f"'{video_path.name}' reports invalid properties "
                f"(fps={fps}, frames={frame_count}, {width}×{height})."
            )

        self._cap = cap
        self._video_path = video_path
        self._next_frame_idx = -999_999
        self._meta = VideoMeta(fps=fps, frame_count=frame_count, width=width, height=height)

        # ── Audio extraction ───────────────────────────────────────────────────
        audio_data: np.ndarray
        try:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="mm_workbench_")
            tmp_wav = Path(self._tmp_dir.name) / "audio.wav"

            cmd = [
                arg.replace("{input}", str(video_path)).replace("{output}", str(tmp_wav))
                for arg in FFMPEG_EXTRACT_CMD
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (exit {result.returncode}).\n"
                    f"{result.stderr[-600:] if result.stderr else '(no output)'}"
                )
            audio_data, _ = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
            if audio_data.shape[1] < 2:
                audio_data = np.repeat(audio_data, 2, axis=1)  # mono → stereo

        except (RuntimeError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            # Non-fatal: fill with silence and surface the warning to the caller
            n_silent = int(self._meta.duration_seconds * DEVICE_SAMPLE_RATE)
            audio_data = np.zeros((n_silent, 2), dtype=np.float32)
            # Re-raise so MainWindow can show a QMessageBox.warning
            # But only after the model is partially open (video is ready).
            self._audio_data = audio_data
            self._engine = AudioPlaybackEngine()
            self._engine.load(audio_data)
            raise

        self._audio_data = audio_data
        self._engine = AudioPlaybackEngine()
        self._engine.load(audio_data)
        return self._meta

    # ── Frame access ───────────────────────────────────────────────────────────

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """Decode one BGR frame, seeking only when necessary.

        The original implementation called CAP_PROP_POS_FRAMES on every tick.
        For compressed codecs (H.264, H.265) this forces the codec to find the
        nearest keyframe and re-decode every delta frame in the GOP — up to
        50–150 wasted decodes per displayed frame at 60 Hz, causing the stutter.

        Fix: track _next_frame_idx. If the requested frame is sequential (within
        SEEK_TOLERANCE frames), just call cap.read() — the codec is already
        positioned correctly. Only seek on genuine discontinuities.

        SEEK_TOLERANCE of 2 frames (~67 ms at 30 fps) is imperceptible to the
        viewer and avoids costly seeks for minor A/V drift corrections.

        Thread safety: _cap_lock is held for the entire read so the
        VideoDecodeThread and main thread never access cap concurrently.

        Args:
            frame_idx: Zero-based frame index to decode.

        Returns:
            BGR frame as shape (H, W, 3) uint8 NumPy array, or None on error.
        """
        with self._cap_lock:
            if self._cap is None:
                return None

            SEEK_TOLERANCE = 2   # frames; within this window, read sequentially

            if abs(self._next_frame_idx - frame_idx) > SEEK_TOLERANCE:
                # Discontinuity — must seek (scrub, stop→play, large A/V drift)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                self._next_frame_idx = frame_idx

            ret, frame = self._cap.read()
            if not ret:
                return None

            self._next_frame_idx += 1
            return frame

    def reset_sequential_position(self) -> None:
        """Force a seek on the next read_frame() call.

        Call this after any operation that changes the VideoCapture position
        externally (e.g. a seek triggered from the transport bar).
        Sets _next_frame_idx to an impossible value so the tolerance check
        in read_frame() always triggers a real seek.
        """
        self._next_frame_idx = -999_999

    def reopen_video(self) -> bool:
        """Release and recreate the VideoCapture under _cap_lock.

        Called before issuing any seek. Releasing and reopening resets all
        internal libavcodec async thread state, which can become corrupted
        after end-of-file or after concurrent access. The OS page cache keeps
        the file resident so the cost is negligible.

        Returns:
            True on success, False if the file cannot be reopened.
        """
        if self._video_path is None or self._meta is None:
            return False

        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()

            cap = cv2.VideoCapture(str(self._video_path))
            if not cap.isOpened():
                self._cap = None
                return False

            self._cap = cap
            self._next_frame_idx = -999_999
            return True

    # ── Audio filter management ────────────────────────────────────────────────

    def set_filter(self, new_state: FilterState) -> None:
        """Swap the audio filter state atomically.

        Rebuilds the new state's SOS/zi before acquiring the lock so the lock
        hold time is minimal (just the pointer swap). Resetting zi on filter
        change avoids a transient pop.

        Args:
            new_state: A new FilterState (rebuild() will be called here).
        """
        new_state.rebuild(DEVICE_SAMPLE_RATE)
        with self._filter_lock:
            self._filter_state = new_state

    def get_filter_state(self) -> FilterState:
        """Return a shallow copy of the current filter state (for UI use).

        Returns:
            A copy of the active FilterState. Safe to read on the main thread.
        """
        with self._filter_lock:
            return copy.copy(self._filter_state)

    # ── Playback control ───────────────────────────────────────────────────────

    def play(self) -> None:
        """Start or resume audio playback."""
        if self._engine is not None:
            self._engine.play()

    def pause(self) -> None:
        """Pause audio playback."""
        if self._engine is not None:
            self._engine.pause()

    def stop(self) -> None:
        """Stop audio playback and reset to the beginning."""
        if self._engine is not None:
            self._engine.stop()

    def seek(self, video_frame: int) -> None:
        """Seek both audio and video to a target frame.

        Converts the video frame index to an audio frame index and calls
        AudioPlaybackEngine.seek() under its internal lock. The caller should
        pause() before seek() to avoid a brief glitch.

        Args:
            video_frame: Target video frame index (0-based).
        """
        if self._engine is not None and self._meta is not None:
            audio_frame = int(video_frame / self._meta.fps * DEVICE_SAMPLE_RATE)
            self._engine.seek(audio_frame)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release all resources (VideoCapture, audio engine, temp directory)."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
        self._meta = None
        self._audio_data = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def meta(self) -> VideoMeta | None:
        """Video metadata, available after open()."""
        return self._meta

    @property
    def audio_data(self) -> np.ndarray | None:
        """Full audio buffer as float32 (frames, 2), or None if not loaded."""
        return self._audio_data

    @property
    def engine(self) -> "AudioPlaybackEngine | None":
        """The AudioPlaybackEngine driving A/V sync. None if no file is open."""
        return self._engine

    @property
    def is_playing(self) -> bool:
        """True if the audio engine is currently active."""
        return self._engine is not None and self._engine.is_playing


# =============================================================================
# STEP 4 — AudioPlaybackEngine: OutputStream as A/V master clock + spectrum
# =============================================================================

class AudioPlaybackEngine:
    """sounddevice OutputStream that serves as the A/V master clock.

    SYNC CONCEPT — outputBufferDacTime as master clock:
    The audio DAC runs on a crystal oscillator with parts-per-million accuracy.
    PortAudio schedules the callback at precisely the right time and provides
    time_info.outputBufferDacTime: the wall-clock time at which the first
    sample of this block will reach the DAC. This is more accurate than
    time.time() (subject to NTP jumps) or time.perf_counter() (no DAC sync).

    The main thread computes the video frame to display as:
        elapsed = dac_time - anchor_dac_time
        audio_frame = anchor_audio_frame + int(elapsed * DEVICE_SAMPLE_RATE)
        video_frame = int(audio_frame / DEVICE_SAMPLE_RATE * fps)

    This is audio-driven video presentation — the foundation of VLC, ffplay,
    and every professional A/V player.

    SPECTRUM: The callback also applies the live IIR filter (FilterState from
    MediaSession) to the current block and pushes (raw_block, filtered_block)
    to a queue.Queue. The SpectrumCanvas timer drains this queue at 30 fps.
    This replaces the microphone input of Demo 3 with the playback audio,
    so the spectrum shows what you are hearing.
    """

    def __init__(self) -> None:
        """Initialise with no audio loaded. Call load() before play()."""
        self._data: np.ndarray | None = None    # float32, (frames, 2)
        self._frame_pos: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

        # A/V sync queues
        self._dac_queue: queue.Queue[float] = queue.Queue(maxsize=8)
        self._finished_queue: queue.Queue[bool] = queue.Queue(maxsize=2)

        # Spectrum queue: (raw_mono_block, filtered_mono_block) — float32, shape (N,)
        # maxsize=4: if the display falls behind, discard old blocks (display stays live)
        self._spectrum_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=4)

        # Filter state pointer — swapped atomically from MediaSession.set_filter()
        self._filter_state: FilterState = FilterState()
        self._filter_state.rebuild(DEVICE_SAMPLE_RATE)
        self._filter_lock: threading.Lock = threading.Lock()

    # ── Buffer management ──────────────────────────────────────────────────────

    def load(self, audio_data: np.ndarray) -> None:
        """Load audio data into the engine.

        Args:
            audio_data: float32 array, shape (frames, 2), values in [-1.0, 1.0].
        """
        was_active = self._stream is not None and self._stream.active
        if was_active:
            self._stream.stop()
        with self._lock:
            self._data = audio_data
            self._frame_pos = 0

    # ── Audio callback — PortAudio audio thread ────────────────────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill outdata from the audio buffer; push DAC time and spectrum data.

        CRITICAL RULES:
          1. No Qt calls — never update a widget or emit a signal from here.
          2. No memory allocation — use pre-allocated buffers.
          3. No blocking — no file I/O, no sleep, no blocking queue operations.
          4. No Python exceptions — wrap risky code in try/except.

        Args:
            outdata:   Output buffer, shape (frames, 2), float32. Write in-place.
            frames:    Number of frames requested.
            time_info: PortAudio timing struct carrying outputBufferDacTime.
            status:    Stream health flags.
        """
        if status.output_underflow:
            print("[AudioPlaybackEngine] output underflow", flush=True)

        with self._lock:
            if self._data is None:
                outdata[:] = 0.0
                return

            pos = self._frame_pos
            n_total = self._data.shape[0]
            remaining = n_total - pos

            if remaining <= 0:
                outdata[:] = 0.0
                self._frame_pos = n_total
                try:
                    self._finished_queue.put_nowait(True)
                except queue.Full:
                    pass
                return

            if remaining < frames:
                outdata[:remaining] = self._data[pos:pos + remaining]
                outdata[remaining:] = 0.0
                self._frame_pos = n_total
                try:
                    self._finished_queue.put_nowait(True)
                except queue.Full:
                    pass
            else:
                outdata[:] = self._data[pos:pos + frames]
                self._frame_pos += frames

            # Push DAC timestamp to the A/V sync queue.
            # SYNC CONCEPT: outputBufferDacTime is the hardware DAC timestamp for
            # the first sample of this block — the most accurate timing available.
            try:
                self._dac_queue.put_nowait(time_info.outputBufferDacTime)
            except queue.Full:
                try:
                    self._dac_queue.get_nowait()
                    self._dac_queue.put_nowait(time_info.outputBufferDacTime)
                except queue.Empty:
                    pass

            # Compute spectrum data and push to spectrum queue for SpectrumCanvas.
            # CRITICAL: copy indata-equivalent (outdata) before the lock releases.
            # The spectrum shows what the user hears: apply the IIR filter to the
            # mono mix of the current block, then push (raw, filtered).
            raw_mono = outdata[:, 0].copy()   # mono mix from left channel

        # Apply IIR filter outside the main lock to minimise hold time.
        # THREAD CONCEPT: we hold _filter_lock only for the filter application.
        with self._filter_lock:
            fs = self._filter_state
            if fs.is_ready:
                filtered_mono, fs.zi = sosfilt(fs.sos, raw_mono, zi=fs.zi)
                filtered_mono = filtered_mono.astype(np.float32)
            else:
                filtered_mono = raw_mono.copy()

        try:
            self._spectrum_queue.put_nowait((raw_mono, filtered_mono))
        except queue.Full:
            pass   # display is behind — discard block, audio continues

    # ── Filter management ──────────────────────────────────────────────────────

    def set_filter(self, new_state: FilterState) -> None:
        """Swap the active IIR filter state atomically.

        Rebuilds the new state before acquiring the lock (expensive design
        step outside the lock, cheap pointer swap inside). Resetting zi on
        filter change avoids a transient pop.

        Args:
            new_state: The new FilterState. rebuild() is called here.
        """
        new_state.rebuild(DEVICE_SAMPLE_RATE)
        with self._filter_lock:
            self._filter_state = new_state

    # ── Playback control ───────────────────────────────────────────────────────

    def _ensure_stream(self) -> None:
        """Create the OutputStream if it does not yet exist."""
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=DEVICE_SAMPLE_RATE,
                channels=2,
                blocksize=AUDIO_BLOCK_SIZE,
                dtype="float32",
                callback=self._callback,
            )

    def play(self) -> None:
        """Start or resume audio playback."""
        self._ensure_stream()
        if not self._stream.active:
            self._stream.start()

    def pause(self) -> None:
        """Pause playback without resetting the position."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def stop(self) -> None:
        """Stop playback and reset to the beginning."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()
        with self._lock:
            self._frame_pos = 0

    def seek(self, audio_frame: int) -> None:
        """Seek to a specific audio frame.

        SYNC CONCEPT — Atomic seek:
        We hold self._lock for the entire seek operation. The callback cannot
        read _frame_pos between our write and any related state update. Caller
        should pause() before seek() to avoid a brief glitch.

        Args:
            audio_frame: Target audio frame index.
        """
        with self._lock:
            if self._data is not None:
                self._frame_pos = max(0, min(audio_frame, self._data.shape[0]))

    def close(self) -> None:
        """Stop and release the audio device. Call from closeEvent."""
        if self._stream is not None:
            if self._stream.active:
                self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def dac_queue(self) -> queue.Queue[float]:
        """Queue delivering outputBufferDacTime values to the main thread."""
        return self._dac_queue

    @property
    def finished_queue(self) -> queue.Queue[bool]:
        """Queue delivering end-of-file signal to the main thread."""
        return self._finished_queue

    @property
    def spectrum_queue(self) -> queue.Queue[tuple[np.ndarray, np.ndarray]]:
        """Queue delivering (raw_mono, filtered_mono) blocks to SpectrumCanvas."""
        return self._spectrum_queue

    @property
    def frame_pos(self) -> int:
        """Current audio frame position (approximate — not lock-protected)."""
        return self._frame_pos

    @property
    def is_playing(self) -> bool:
        """True if the audio stream is currently active."""
        return self._stream is not None and self._stream.active


# =============================================================================
# STEP 5 — FFT helper (reused from Demo 3, pure NumPy)
# =============================================================================

def fft_magnitude_db(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Hann window, compute the real FFT, and convert to dBFS.

    DSP CONCEPT — Windowing and spectral leakage:
    The FFT assumes the signal repeats periodically. A discontinuity at the
    window boundary (the end of the block does not connect to the start) is
    interpreted as high-frequency energy — spectral leakage. The Hann window
    tapers both ends to zero, eliminating the discontinuity. Trade-off: less
    leakage (sharper peaks), slightly lower frequency resolution.

    DSP CONCEPT — rfft vs fft:
    Audio is real-valued. np.fft.rfft computes only the positive-frequency
    half of the spectrum — twice as efficient as np.fft.fft for real inputs.
    Always use rfft for audio.

    Args:
        block: 1-D float32 audio array, length ≤ NFFT.

    Returns:
        freqs: Frequency axis in Hz, shape (NFFT//2 + 1,).
        mag_db: Magnitude in dBFS, shape (NFFT//2 + 1,), floor at DBFS_FLOOR.
    """
    n = len(block)
    if n < NFFT:
        padded = np.zeros(NFFT, dtype=np.float32)
        padded[:n] = block
    else:
        padded = block[:NFFT].copy()

    windowed = padded * _HANN_WINDOW
    spectrum = np.fft.rfft(windowed)

    # Convert to dBFS: normalise for FFT size, apply log, clamp to floor.
    # +1e-9 prevents log10(0) = -inf for silent bins.
    magnitude = np.abs(spectrum) / (NFFT / 2.0)
    mag_db = 20.0 * np.log10(magnitude + 1e-9)
    mag_db = np.clip(mag_db, DBFS_FLOOR, 0.0)

    freqs = np.fft.rfftfreq(NFFT, d=1.0 / DEVICE_SAMPLE_RATE).astype(np.float32)
    return freqs, mag_db.astype(np.float32)


# =============================================================================
# STEP 6 — Frame DSP functions (pure OpenCV/NumPy, reused from Demo 4)
# =============================================================================

def process_passthrough(frame: np.ndarray) -> np.ndarray:
    """Return the frame unmodified."""
    return frame


def process_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale (ITU-R BT.601 luminance) and return as 3-channel BGR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def process_canny(frame: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """Canny edge detection: grayscale → non-max suppression → hysteresis threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.merge([edges, edges, edges])


def process_blur(frame: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur for noise reduction or depth-of-field simulation."""
    ksize = max(1, ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=0)


def process_equalize(frame: np.ndarray) -> np.ndarray:
    """Per-channel histogram equalisation for contrast enhancement."""
    b, g, r = cv2.split(frame)
    return cv2.merge([cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)])


def process_motion(prev: np.ndarray | None, frame: np.ndarray) -> np.ndarray:
    """Frame differencing: |prev - curr| thresholded for motion detection."""
    if prev is None or prev.shape != frame.shape:
        return np.zeros_like(frame)
    diff = cv2.absdiff(prev, frame)
    _, mask = cv2.threshold(
        cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY
    )
    return cv2.merge([mask, mask, mask])


# =============================================================================
# STEP 7 — VideoDisplay: shows one video frame (reused from Demo 4)
# =============================================================================

class VideoDisplay(QWidget):
    """Displays one BGR video frame at a time, scaled to the widget size.

    CONCEPT — BGR → RGB:
    OpenCV uses BGR channel order (historical Windows BITMAPINFOHEADER legacy).
    Qt's QImage.Format_RGB888 expects R, G, B. Skipping the conversion swaps
    red and blue channels silently — a very common and confusing bug.
    Always call cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) before QImage.

    CONCEPT — bytes_per_line in QImage:
    Always pass w * 3 as the fourth argument to QImage for RGB888 frames.
    Without it, Qt uses default 4-byte-aligned row strides, producing
    corrupted frames for widths not divisible by 4/3.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with a dark placeholder frame."""
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None

    def show_frame(self, bgr_frame: np.ndarray) -> None:
        """Convert a BGR frame to QPixmap and schedule a repaint.

        Args:
            bgr_frame: OpenCV frame, shape (H, W, 3), dtype uint8, BGR order.
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        # CRITICAL: provide bytes_per_line = w * 3 (see docstring)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw background and current pixmap, centred."""
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))
        if self._pixmap is None:
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 11))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Open a video file to start playback"
            )
        else:
            px = (w - self._pixmap.width()) // 2
            py = (h - self._pixmap.height()) // 2
            painter.drawPixmap(px, py, self._pixmap)
        painter.end()


# =============================================================================
# STEP 8 — WaveformView: peak-reduced waveform with live playhead (from Demo 2)
# =============================================================================

class WaveformView(QWidget):
    """QPainter waveform canvas with peak-reduction rendering and a playhead.

    DSP CONCEPT — Peak reduction:
    A long audio file at 48 000 Hz contains hundreds of thousands of frames.
    The canvas is ~600 px wide. Drawing every sample causes 100× overdraw and
    misses transients (a brief click between two stride-sampled points). Peak
    reduction: for each pixel column find (min, max) of all covered samples
    and draw a vertical line. O(N), preserves every transient. This is how
    every professional DAW (Pro Tools, Logic, Reaper, Audacity) renders waveforms.

    Signals:
        seek_requested: Emitted on click/drag. Carries the target video frame.
    """

    seek_requested: Signal = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with empty state."""
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._peaks_min: np.ndarray = np.array([])
        self._peaks_max: np.ndarray = np.array([])
        self._n_frames: int = 0
        self._fps: float = 25.0
        self._playhead_frame: int = 0

    def set_audio(self, mono: np.ndarray, n_frames: int, fps: float) -> None:
        """Compute peak-reduced display data from a mono audio array.

        Args:
            mono: 1-D float32 audio array (mono mix of the full audio buffer).
            n_frames: Total video frame count (for playhead mapping).
            fps: Video frame rate (for click-to-seek conversion).
        """
        self._n_frames = n_frames
        self._fps = fps
        w = max(self.width(), 1)
        n = len(mono)
        frames_per_px = n / w

        self._peaks_min = np.empty(w, dtype=np.float32)
        self._peaks_max = np.empty(w, dtype=np.float32)

        for px in range(w):
            start = int(px * frames_per_px)
            end = min(int((px + 1) * frames_per_px), n)
            if start >= end:
                self._peaks_min[px] = 0.0
                self._peaks_max[px] = 0.0
            else:
                chunk = mono[start:end]
                self._peaks_min[px] = float(chunk.min())
                self._peaks_max[px] = float(chunk.max())

        self.update()

    def set_playhead(self, video_frame: int) -> None:
        """Update the playhead position and schedule a repaint.

        Args:
            video_frame: Current video frame index.
        """
        self._playhead_frame = video_frame
        self.update()

    def _amp_to_py(self, amp: float, h: int) -> int:
        """Map amplitude [-1, 1] to pixel Y. Centre at h//2; +1 → top."""
        return int(h / 2.0 * (1.0 - amp))

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Seek to clicked frame on mouse press."""
        if self._n_frames > 0:
            frac = max(0.0, min(1.0, event.pos().x() / max(self.width(), 1)))
            self.seek_requested.emit(int(frac * self._n_frames))

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Seek while dragging (scrubbing)."""
        if self._n_frames > 0 and event.buttons() & Qt.MouseButton.LeftButton:
            frac = max(0.0, min(1.0, event.pos().x() / max(self.width(), 1)))
            self.seek_requested.emit(int(frac * self._n_frames))

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw background, zero line, waveform, and playhead."""
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        if len(self._peaks_min) == 0:
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 9))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Audio waveform")
            painter.end()
            return

        cy = h // 2
        painter.setPen(QPen(QColor(ZERO_LINE_COLOR), 1))
        painter.drawLine(0, cy, w, cy)

        r, g, b, a = ACCENT_BLUE_RGBA
        pen = QPen(QColor(r, g, b, a), 1)
        painter.setPen(pen)
        n_px = min(len(self._peaks_min), w)
        for px in range(n_px):
            py_min = self._amp_to_py(float(self._peaks_min[px]), h)
            py_max = self._amp_to_py(float(self._peaks_max[px]), h)
            painter.drawLine(px, py_min, px, py_max)

        # Playhead
        if self._n_frames > 0:
            ph_px = int(self._playhead_frame / self._n_frames * w)
            painter.setPen(QPen(QColor(PLAYHEAD_COLOR), 2))
            painter.drawLine(ph_px, 0, ph_px, h)

        painter.end()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        """Recompute peak-reduced data at the new width after resize."""
        self.update()


# =============================================================================
# STEP 9 — SpectrumCanvas: real-time spectrum display (adapted from Demo 3)
# =============================================================================

class SpectrumCanvas(QWidget):
    """QPainter spectrum display showing the real-time audio spectrum.

    Driven by the playback stream (not the microphone): the AudioPlaybackEngine
    pushes (raw_mono, filtered_mono) blocks; the QTimer in MainWindow drains
    them and calls set_spectrum() here.

    Overlays two spectra:
      - Raw audio:     blue (--accent)
      - Filtered audio: purple (--accent2)

    The frequency axis is logarithmic — one octave spans equal pixel width,
    matching human auditory perception. This is the standard for professional
    spectrum analyzers and equalizers.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with empty spectra."""
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        n_bins: int = NFFT // 2 + 1
        self._freqs: np.ndarray = np.zeros(n_bins, dtype=np.float32)
        self._raw_db: np.ndarray = np.full(n_bins, DBFS_FLOOR, dtype=np.float32)
        self._filtered_db: np.ndarray = np.full(n_bins, DBFS_FLOOR, dtype=np.float32)
        self._cursor_x: int = -1

    def set_spectrum(
        self, raw_db: np.ndarray, filtered_db: np.ndarray, freqs: np.ndarray
    ) -> None:
        """Update the displayed spectra and schedule a repaint.

        Args:
            raw_db:      Raw spectrum in dBFS, shape (n_bins,).
            filtered_db: Filtered spectrum in dBFS, shape (n_bins,).
            freqs:       Frequency axis in Hz, shape (n_bins,).
        """
        self._freqs = freqs
        self._raw_db = raw_db
        self._filtered_db = filtered_db
        self.update()

    def _freq_to_px(self, freq_hz: float) -> float:
        """Map a frequency in Hz to pixel X on the log axis.

        DSP CONCEPT — Log frequency axis:
        Human hearing is logarithmic: 100→200 Hz sounds like the same step as
        1000→2000 Hz. A linear axis compresses the bass into 1 % of the width.
        Log axis: px = width × log10(f / 20) / log10(20000 / 20).
        """
        freq_hz = max(FREQ_MIN_HZ, min(freq_hz, FREQ_MAX_HZ))
        log_range = log10(FREQ_MAX_HZ / FREQ_MIN_HZ)
        return self.width() * log10(freq_hz / FREQ_MIN_HZ) / log_range

    def _dbfs_to_py(self, db: float) -> float:
        """Map dBFS to pixel Y. 0 dBFS at top; DBFS_FLOOR at bottom."""
        db = max(DBFS_FLOOR, min(db, 0.0))
        fraction = (db - DBFS_FLOOR) / (0.0 - DBFS_FLOOR)
        return self.height() * (1.0 - fraction)

    def _px_to_freq(self, px: int) -> float:
        """Map pixel X to frequency in Hz (inverse of _freq_to_px)."""
        frac = max(0.0, min(1.0, px / max(self.width(), 1)))
        return FREQ_MIN_HZ * (10.0 ** (frac * log10(FREQ_MAX_HZ / FREQ_MIN_HZ)))

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        self._cursor_x = event.pos().x()
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self._cursor_x = -1
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw background, grids, spectra, and frequency cursor."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        # dBFS grid lines
        painter.setFont(QFont("IBM Plex Mono", 8))
        for db in DBFS_GRID:
            py = int(self._dbfs_to_py(db))
            painter.setPen(QPen(QColor(GRID_COLOR), 1))
            painter.drawLine(0, py, w, py)
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            painter.drawText(2, py - 1, f"{int(db):+d}")

        # Frequency grid lines
        for freq in FREQ_GRID_HZ:
            px = int(self._freq_to_px(freq))
            painter.setPen(QPen(QColor(GRID_COLOR), 1))
            painter.drawLine(px, 0, px, h)
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            label = f"{int(freq // 1000)}k" if freq >= 1_000 else str(int(freq))
            painter.drawText(px + 2, h - 3, label)

        # Filtered spectrum (purple, behind raw)
        if len(self._filtered_db) > 0:
            self._draw_spectrum_fill(painter, self._freqs, self._filtered_db, ACCENT_PURPLE_RGBA)

        # Raw spectrum (blue, on top)
        if len(self._raw_db) > 0:
            self._draw_spectrum_fill(painter, self._freqs, self._raw_db, ACCENT_BLUE_RGBA)

        # Frequency cursor
        if self._cursor_x >= 0:
            cursor_hz = self._px_to_freq(self._cursor_x)
            if len(self._freqs) > 0:
                idx = int(np.argmin(np.abs(self._freqs - cursor_hz)))
                cursor_db = float(self._raw_db[idx]) if idx < len(self._raw_db) else DBFS_FLOOR
            else:
                cursor_db = DBFS_FLOOR
            painter.setPen(QPen(QColor(CURSOR_COLOR), 1))
            painter.drawLine(self._cursor_x, 0, self._cursor_x, h)
            label = (
                f"{cursor_hz:.0f} Hz  {cursor_db:.1f} dBFS"
                if cursor_hz < 1_000
                else f"{cursor_hz / 1_000:.2f} kHz  {cursor_db:.1f} dBFS"
            )
            painter.setPen(QPen(QColor(CURSOR_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 10))
            tx = min(self._cursor_x + 6, w - 180)
            painter.drawText(tx, 16, label)

        painter.end()

    def _draw_spectrum_fill(
        self,
        painter: QPainter,
        freqs: np.ndarray,
        mag_db: np.ndarray,
        rgba: tuple[int, int, int, int],
    ) -> None:
        """Draw a filled spectrum path between the magnitude curve and the floor."""
        py_floor = int(self._dbfs_to_py(DBFS_FLOOR))
        path = QPainterPath()
        started = False
        for i in range(len(freqs)):
            f = float(freqs[i])
            if f < FREQ_MIN_HZ or f > FREQ_MAX_HZ:
                continue
            px = self._freq_to_px(f)
            py = self._dbfs_to_py(float(mag_db[i]))
            if not started:
                path.moveTo(px, py_floor)
                path.lineTo(px, py)
                started = True
            else:
                path.lineTo(px, py)
        if not started:
            return
        path.lineTo(self._freq_to_px(float(freqs[-1])), py_floor)
        path.closeSubpath()
        r, g, b, a = rgba
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(r, g, b, a)))
        painter.drawPath(path)


# =============================================================================
# STEP 10 — TimelineWidget: transport controls, scrubber, frame counter
# =============================================================================

class TimelineWidget(QWidget):
    """Transport controls with Play/Pause, Stop, scrubber, and time/frame display.

    Signals:
        play_requested: Play/Pause toggle clicked.
        stop_requested: Stop button clicked.
        seek_requested: Scrubber moved or waveform clicked. Carries video frame index.
    """

    play_requested: Signal = Signal()
    stop_requested: Signal = Signal()
    seek_requested: Signal = Signal(int)   # video frame index

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the transport bar."""
        super().__init__(parent)
        self.setFixedHeight(68)
        self._frame_count: int = 0
        self._fps: float = 25.0
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all transport controls."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(4)

        # ── Scrubber ──────────────────────────────────────────────────────────
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 10_000)
        self._scrubber.setValue(0)
        self._scrubber.setEnabled(False)
        self._scrubber.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #252b38; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0;"
            "  background: #4f9cf9; border-radius: 6px; }"
            "QSlider::sub-page:horizontal { background: #4f9cf9; border-radius: 2px; }"
        )
        layout.addWidget(self._scrubber)

        # ── Button row ────────────────────────────────────────────────────────
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        btn_play_style = (
            "QPushButton { background: rgba(79,156,249,0.12); color: #4f9cf9;"
            "  border: 1px solid rgba(79,156,249,0.4); border-radius: 6px;"
            "  padding: 5px 12px; font-size: 12px; font-family: 'IBM Plex Mono'; min-width: 80px; }"
            "QPushButton:checked { background: rgba(251,146,60,0.14); color: #fb923c;"
            "  border-color: rgba(251,146,60,0.5); }"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )
        btn_stop_style = (
            "QPushButton { background: rgba(79,156,249,0.06); color: #5a6175;"
            "  border: 1px solid #252b38; border-radius: 6px;"
            "  padding: 5px 12px; font-size: 12px; font-family: 'IBM Plex Mono'; min-width: 70px; }"
            "QPushButton:hover { color: #d8dce6; border-color: #4f9cf9; }"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setEnabled(False)
        self._play_btn.setStyleSheet(btn_play_style)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(btn_stop_style)

        self._time_label = QLabel("00:00.000 / 00:00.000")
        self._time_label.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 12px;"
        )

        self._frame_label = QLabel("frame — / —")
        self._frame_label.setStyleSheet(
            "color: #5a6175; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )

        row_layout.addWidget(self._play_btn)
        row_layout.addWidget(self._stop_btn)
        row_layout.addSpacing(8)
        row_layout.addWidget(self._time_label)
        row_layout.addStretch()
        row_layout.addWidget(self._frame_label)
        layout.addWidget(row)

        # ── Internal connections ───────────────────────────────────────────────
        self._play_btn.toggled.connect(self._on_play_toggled)
        self._stop_btn.clicked.connect(self.stop_requested)
        self._scrubber.sliderMoved.connect(self._on_scrubber_moved)

    def _on_play_toggled(self, checked: bool) -> None:
        self._play_btn.setText("⏸  Pause" if checked else "▶  Play")
        self.play_requested.emit()

    def _on_scrubber_moved(self, value: int) -> None:
        if self._frame_count > 0:
            frame = int(value / 10_000 * self._frame_count)
            self.seek_requested.emit(frame)

    def set_file_loaded(self, frame_count: int, fps: float) -> None:
        """Enable controls after a file is loaded.

        Args:
            frame_count: Total video frame count.
            fps: Video frame rate.
        """
        self._frame_count = frame_count
        self._fps = max(fps, 1e-6)
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._scrubber.setEnabled(True)
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")

    def update_position(self, video_frame: int) -> None:
        """Update time display, frame counter, and scrubber position.

        Args:
            video_frame: Current video frame index.
        """
        if self._frame_count == 0:
            return

        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = sec - m * 60
            return f"{m:02d}:{s:06.3f}"

        cur_s = video_frame / self._fps
        tot_s = self._frame_count / self._fps
        self._time_label.setText(f"{fmt(cur_s)} / {fmt(tot_s)}")
        self._frame_label.setText(f"frame {video_frame:,} / {self._frame_count:,}")

        self._scrubber.blockSignals(True)
        self._scrubber.setValue(int(video_frame / self._frame_count * 10_000))
        self._scrubber.blockSignals(False)

    def set_stopped(self) -> None:
        """Reset the play button to the unplayed state."""
        self._play_btn.blockSignals(True)
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")
        self._play_btn.blockSignals(False)

    def reset(self) -> None:
        """Reset all controls to initial (no file) state."""
        self._frame_count = 0
        self._fps = 25.0
        self._time_label.setText("00:00.000 / 00:00.000")
        self._frame_label.setText("frame — / —")
        self._play_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._scrubber.setEnabled(False)
        self._scrubber.setValue(0)
        self.set_stopped()

    @property
    def is_playing(self) -> bool:
        """True when the Play button is in the playing state."""
        return self._play_btn.isChecked()


# =============================================================================
# STEP 11 — ParameterPanel: all DSP controls for audio and video
# =============================================================================

class ParameterPanel(QWidget):
    """Right-side panel exposing all DSP parameters for audio and video.

    ARCHITECTURE CONCEPT — ParameterPanel emits signals, never touches models:
    This panel emits audio_filter_changed(FilterState) and video_dsp_changed(str).
    It never calls MediaSession or AudioPlaybackEngine directly. MainWindow
    connects the signals to the appropriate model methods. This keeps the panel
    reusable: swap MainWindow for a different controller and the panel works
    unchanged.

    Signals:
        audio_filter_changed: New FilterState (already rebuilt) when any audio
                              filter parameter changes.
        video_dsp_changed:    New DSP mode string when the video mode changes.
    """

    audio_filter_changed: Signal = Signal(object)   # FilterState
    video_dsp_changed: Signal = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the parameter panel."""
        super().__init__(parent)
        self.setFixedWidth(275)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all controls: logo, author, audio filter group, video DSP group."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── AMS logo ──────────────────────────────────────────────────────────
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pix = QPixmap(str(LOGO_PATH)).scaledToWidth(
                80, Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(pix)
        else:
            logo_label.setText("AMS Group – TU Ilmenau")
            logo_label.setStyleSheet("color: #4f9cf9; font-size: 10px;")
        layout.addWidget(logo_label)

        # ── Author credit ─────────────────────────────────────────────────────
        author_label = QLabel("Renato Fera P.")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_label.setStyleSheet("color: #5a6175; font-size: 10px; letter-spacing: 0.5px;")
        layout.addWidget(author_label)

        layout.addSpacing(4)

        # ── Audio DSP group ───────────────────────────────────────────────────
        audio_group = QGroupBox("Audio IIR Filter")
        audio_group.setStyleSheet(
            "QGroupBox { color: #4f9cf9; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        audio_form = QFormLayout(audio_group)
        audio_form.setSpacing(8)
        audio_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._filter_type_combo = QComboBox()
        self._filter_type_combo.addItems(list(FILTER_TYPE_LABELS))
        audio_form.addRow("Type", self._filter_type_combo)

        self._cutoff_slider = QSlider(Qt.Orientation.Horizontal)
        self._cutoff_slider.setRange(CUTOFF_MIN_HZ, CUTOFF_MAX_HZ)
        self._cutoff_slider.setValue(DEFAULT_CUTOFF_HZ)
        audio_form.addRow("Cutoff", self._cutoff_slider)

        self._cutoff_label = QLabel(f"{DEFAULT_CUTOFF_HZ} Hz")
        self._cutoff_label.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        audio_form.addRow("", self._cutoff_label)

        self._order_spin = QSpinBox()
        self._order_spin.setRange(2, 8)
        self._order_spin.setSingleStep(2)
        self._order_spin.setValue(DEFAULT_FILTER_ORDER)
        audio_form.addRow("Order", self._order_spin)

        layout.addWidget(audio_group)

        # ── Filter response mini-plot ──────────────────────────────────────────
        self._response_view = _FilterResponseView()
        layout.addWidget(self._response_view)

        # Legend
        legend_raw = QLabel("■ Raw audio    ■ Filtered audio")
        legend_raw.setStyleSheet(
            "color: #5a6175; font-size: 10px; font-family: 'IBM Plex Mono';"
        )
        layout.addWidget(legend_raw)

        # ── Video DSP group ───────────────────────────────────────────────────
        video_group = QGroupBox("Video Frame DSP")
        video_group.setStyleSheet(
            "QGroupBox { color: #a78bfa; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        video_form = QFormLayout(video_group)
        video_form.setSpacing(8)
        video_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._dsp_mode_combo = QComboBox()
        for mode in DSP_MODES:
            self._dsp_mode_combo.addItem(mode)
        video_form.addRow("Mode", self._dsp_mode_combo)

        # Canny controls — visible only in Canny Edges mode
        self._canny_t1_label = QLabel("T1")
        self._canny_t1 = QSlider(Qt.Orientation.Horizontal)
        self._canny_t1.setRange(0, 500)
        self._canny_t1.setValue(CANNY_T1_DEFAULT)
        self._canny_t1_val = QLabel(str(CANNY_T1_DEFAULT))
        self._canny_t1_val.setFixedWidth(30)
        self._canny_t1_val.setStyleSheet("color: #4f9cf9; font-size: 11px;")

        self._canny_t2_label = QLabel("T2")
        self._canny_t2 = QSlider(Qt.Orientation.Horizontal)
        self._canny_t2.setRange(0, 500)
        self._canny_t2.setValue(CANNY_T2_DEFAULT)
        self._canny_t2_val = QLabel(str(CANNY_T2_DEFAULT))
        self._canny_t2_val.setFixedWidth(30)
        self._canny_t2_val.setStyleSheet("color: #4f9cf9; font-size: 11px;")

        t1_row = QWidget()
        t1_layout = QHBoxLayout(t1_row)
        t1_layout.setContentsMargins(0, 0, 0, 0)
        t1_layout.addWidget(self._canny_t1)
        t1_layout.addWidget(self._canny_t1_val)

        t2_row = QWidget()
        t2_layout = QHBoxLayout(t2_row)
        t2_layout.setContentsMargins(0, 0, 0, 0)
        t2_layout.addWidget(self._canny_t2)
        t2_layout.addWidget(self._canny_t2_val)

        video_form.addRow(self._canny_t1_label, t1_row)
        video_form.addRow(self._canny_t2_label, t2_row)

        self._canny_widgets: list[QWidget] = [
            self._canny_t1_label, t1_row, self._canny_t2_label, t2_row
        ]
        self._set_canny_visible(False)

        layout.addWidget(video_group)
        layout.addStretch()

        # ── Internal connections ───────────────────────────────────────────────
        # Audio filter: any change → rebuild FilterState → emit audio_filter_changed
        self._filter_type_combo.currentIndexChanged.connect(self._emit_audio_filter)
        self._cutoff_slider.valueChanged.connect(self._on_cutoff_changed)
        self._order_spin.valueChanged.connect(self._emit_audio_filter)

        # Video DSP mode → emit video_dsp_changed
        self._dsp_mode_combo.currentTextChanged.connect(self._on_dsp_mode_changed)

        # Canny sliders → update value labels
        self._canny_t1.valueChanged.connect(lambda v: self._canny_t1_val.setText(str(v)))
        self._canny_t2.valueChanged.connect(lambda v: self._canny_t2_val.setText(str(v)))

        # Emit initial filter state so the engine starts with the default
        self._emit_audio_filter()

    def _set_canny_visible(self, visible: bool) -> None:
        for w in self._canny_widgets:
            w.setVisible(visible)

    def _on_cutoff_changed(self, value: int) -> None:
        """Update the cutoff label and emit the filter change."""
        self._cutoff_label.setText(
            f"{value / 1_000:.1f} kHz" if value >= 1_000 else f"{value} Hz"
        )
        self._emit_audio_filter()

    def _on_dsp_mode_changed(self, mode: str) -> None:
        """Toggle Canny controls visibility and emit video_dsp_changed."""
        self._set_canny_visible(mode == DSP_CANNY)
        self.video_dsp_changed.emit(mode)

    def _emit_audio_filter(self) -> None:
        """Build a FilterState from current controls and emit audio_filter_changed."""
        fs = FilterState(
            filter_type=FILTER_TYPES[self._filter_type_combo.currentIndex()],
            cutoff_hz=float(self._cutoff_slider.value()),
            order=self._order_spin.value(),
        )
        fs.rebuild(DEVICE_SAMPLE_RATE)
        self._response_view.update_response(fs)
        self.audio_filter_changed.emit(fs)

    @property
    def current_video_dsp(self) -> str:
        """Currently selected video DSP mode string."""
        return self._dsp_mode_combo.currentText()

    @property
    def canny_t1(self) -> int:
        """Current Canny lower threshold."""
        return self._canny_t1.value()

    @property
    def canny_t2(self) -> int:
        """Current Canny upper threshold."""
        return self._canny_t2.value()


class _FilterResponseView(QWidget):
    """Compact Bode magnitude plot using scipy.signal.sosfreqz (from Demo 3)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._resp_db: np.ndarray = np.zeros(256, dtype=np.float32)
        self._resp_hz: np.ndarray = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, 256)

    def update_response(self, fs: FilterState) -> None:
        """Recompute the filter's magnitude response on a log-spaced grid."""
        if fs.sos is None:
            return
        freqs_hz = np.logspace(log10(FREQ_MIN_HZ), log10(FREQ_MAX_HZ), 256)
        worN = freqs_hz / (DEVICE_SAMPLE_RATE / 2.0)
        _, h = sosfreqz(fs.sos, worN=worN, fs=2.0)
        mag_db = 20.0 * np.log10(np.abs(h) + 1e-9)
        self._resp_db = np.clip(mag_db, DBFS_FLOOR, 6.0).astype(np.float32)
        self._resp_hz = freqs_hz.astype(np.float32)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#0d0f14"))

        log_range = log10(FREQ_MAX_HZ / FREQ_MIN_HZ)
        db_range = 6.0 - DBFS_FLOOR

        path = QPainterPath()
        started = False
        for i in range(len(self._resp_hz)):
            f = float(self._resp_hz[i])
            db = float(self._resp_db[i])
            px = w * log10(f / FREQ_MIN_HZ) / log_range
            py = max(0.0, min(float(h), h * (1.0 - (db - DBFS_FLOOR) / db_range)))
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
        painter.drawText(3, h - 3, "Filter response")
        painter.end()


# =============================================================================
# STEP 11b — VideoDecodeThread: background frame decoder
# =============================================================================

class VideoDecodeThread(threading.Thread):
    """Dedicated background thread that pre-decodes video frames.

    ARCHITECTURE — Why a separate decode thread:
    cv2.VideoCapture.read() blocks the calling thread for as long as it
    takes the codec to decode one frame. For H.264/H.265 at high resolution
    this can be 5–30 ms. When decode runs on the Qt main thread (inside the
    QTimer callback) it blocks the event loop: the UI freezes, the timer
    fires again before the previous frame is shown, and the result is the
    frame-skipping that was observed.

    Moving decode to a background thread decouples codec latency from the
    display cadence:
      - The decode thread runs continuously, always 1–2 frames ahead of
        where the audio clock currently is.
      - The main thread merely picks up the already-decoded frame from
        _frame_queue and displays it — a near-zero-cost operation.
      - If the main thread is briefly late (e.g. during a DSP pass),
        the pre-decoded frame waits in the queue; no frame is dropped.

    FRAME QUEUE:
    _frame_queue holds (frame_idx, bgr_ndarray) tuples.
    maxsize=4: enough to absorb a single heavy DSP frame on the display
    side without falling behind.

    SYNC PROTOCOL:
    The decode thread reads the audio clock's target frame index from
    _target_frame (written by the main thread each tick) and decodes
    forward from there, staying at most PREFETCH=4 frames ahead.

    STOP / FLUSH:
    _stop_event: set by stop() to terminate the thread loop.
    _flush_event: set by flush() to discard stale frames after a seek.
    """

    def __init__(self, session: "MediaSession") -> None:
        super().__init__(name="VideoDecodeThread", daemon=True)
        self._session = session
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._wake_event = threading.Event()

        # (frame_idx, bgr_frame) pairs ready for display
        self._frame_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=4)

        self._target_frame: int = 0
        self._playing: bool = False

    # ── Control ───────────────────────────────────────────────────────────────

    def set_playing(self, playing: bool, target_frame: int = 0) -> None:
        """Signal the thread to start or pause decoding."""
        with self._lock:
            self._playing = playing
            if playing:
                self._target_frame = target_frame
        self._wake_event.set()

    def set_target(self, target_frame: int) -> None:
        """Update the target frame index (called by main thread each tick).

        No locking needed for a single integer write on CPython (GIL).
        """
        self._target_frame = target_frame

    def flush(self, new_target: int) -> None:
        """Discard all buffered frames and restart from new_target.

        Call this after a seek so stale pre-decoded frames are not shown.
        """
        with self._lock:
            self._target_frame = new_target
        self._flush_event.set()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._wake_event.set()

    def get_frame(self) -> tuple[int, np.ndarray] | None:
        """Return the next pre-decoded frame, or None if the queue is empty."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal the thread to exit and wait for it to finish."""
        self._stop_event.set()
        self._wake_event.set()
        self.join(timeout=2.0)

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Decode loop — runs until stop() is called."""
        decode_pos: int = 0

        while not self._stop_event.is_set():

            # Idle when paused
            with self._lock:
                playing = self._playing
            if not playing:
                self._wake_event.clear()
                self._wake_event.wait(timeout=0.1)
                continue

            # Handle flush (seek)
            if self._flush_event.is_set():
                self._flush_event.clear()
                with self._lock:
                    decode_pos = self._target_frame
                self._session.reset_sequential_position()

            # Stay at most PREFETCH frames ahead of the audio clock
            PREFETCH = 4
            target = self._target_frame

            if decode_pos > target + PREFETCH:
                self._stop_event.wait(timeout=0.005)
                continue

            meta = self._session.meta
            if meta is None:
                self._stop_event.wait(timeout=0.05)
                continue

            if decode_pos >= meta.frame_count:
                # End of file — idle until a seek or stop
                self._stop_event.wait(timeout=0.05)
                continue

            bgr_frame = self._session.read_frame(decode_pos)
            if bgr_frame is not None:
                try:
                    self._frame_queue.put((decode_pos, bgr_frame), timeout=0.05)
                    decode_pos += 1
                except queue.Full:
                    pass   # timeout — loop and retry
            else:
                decode_pos += 1   # skip corrupted/missing frame


# =============================================================================
# STEP 12 — MainWindow: assembly, signal connections, A/V loop, QSettings
# =============================================================================

class MainWindow(QMainWindow):
    """Application main window for the Full Multimedia Workbench.

    Layout (nested QSplitters):

    ┌─────────────────────────────────────────────────────────────────┬────────────────┐
    │  left (QSplitter, Vertical)                                     │  ParameterPanel│
    │  ┌────────────────────────────────────────────────────────┐     │                │
    │  │  VideoDisplay (video frame + DSP overlay)              │     │                │
    │  ├────────────────────────────────────────────────────────┤     │                │
    │  │  WaveformView (audio waveform + live playhead)         │     │                │
    │  └────────────────────────────────────────────────────────┘     │                │
    ├─────────────────────────────────────────────────────────────────┤                │
    │  centre (QSplitter, Vertical)                                   │                │
    │  ┌────────────────────────────────────────────────────────┐     │                │
    │  │  SpectrumCanvas (real-time audio spectrum)             │     │                │
    │  ├────────────────────────────────────────────────────────┤     │                │
    │  │  TimelineWidget (transport controls + scrubber)        │     │                │
    │  └────────────────────────────────────────────────────────┘     │                │
    └─────────────────────────────────────────────────────────────────┴────────────────┘
    Status bar: time | frame | A/V sync offset | buffer status | fps

    Signal flow (documented in _connect_signals()):

    TimelineWidget.seek_requested(int)
        → MainWindow._on_seek(frame)
            → MediaSession.seek(frame)          [audio seek, atomic]
            → VideoDisplay.show_frame(...)      [immediate frame display]
            → WaveformView.set_playhead(frame)  [waveform cursor]

    AudioPlaybackEngine.dac_queue (via QTimer at 30 fps)
        → MainWindow._poll_av()
            → compute target_video_frame from DAC time
            → VideoDisplay.show_frame(processed_frame)
            → SpectrumCanvas.set_spectrum(raw_db, filtered_db, freqs)
            → WaveformView.set_playhead(target_frame)
            → TimelineWidget.update_position(target_frame)
            → statusBar update with sync offset

    ParameterPanel.audio_filter_changed(FilterState)
        → MediaSession.set_filter(fs)           [filter hot-swap, thread-safe]
        → AudioPlaybackEngine.set_filter(fs)    [SOS/zi rebuilt under lock]

    ParameterPanel.video_dsp_changed(str)
        → MainWindow._current_video_dsp stored  [read in _poll_av per frame]
    """

    def __init__(self) -> None:
        """Initialise the workbench: create model, build UI, restore settings."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Model
        self._session: MediaSession = MediaSession()

        # Background video decode thread — pre-decodes frames ahead of the audio clock
        self._decode_thread: VideoDecodeThread = VideoDecodeThread(self._session)
        self._decode_thread.start()

        # A/V sync state
        self._dac_time_anchor: float | None = None      # DAC time at last play/seek
        self._audio_frame_anchor: int = 0               # audio frame at last play/seek
        self._prev_bgr_frame: np.ndarray | None = None  # for motion diff

        # Current video DSP mode (read in _poll_av per frame)
        self._current_video_dsp: str = DSP_PASSTHROUGH

        # QSettings — persists window geometry, splitter sizes, recent files
        self._settings: QSettings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        # Shared QTimer: drives A/V sync loop + spectrum updates at 30 fps
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(DISPLAY_TIMER_MS)

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._restore_settings()
        self._poll_timer.start()

    # ── UI assembly ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Assemble all panels into nested QSplitters."""
        # ── View components ────────────────────────────────────────────────────
        self._video_display = VideoDisplay()
        self._waveform_view = WaveformView()
        self._spectrum_canvas = SpectrumCanvas()
        self._timeline = TimelineWidget()
        self._param_panel = ParameterPanel()

        # ── Left vertical splitter: video (top) + waveform (bottom) ───────────
        self._splitter_left = QSplitter(Qt.Orientation.Vertical)
        self._splitter_left.addWidget(self._video_display)
        self._splitter_left.addWidget(self._waveform_view)
        self._splitter_left.setStretchFactor(0, 3)   # video gets 3× more space
        self._splitter_left.setStretchFactor(1, 1)
        self._splitter_left.setSizes([480, 120])

        # ── Centre vertical splitter: spectrum (top) + timeline (bottom) ──────
        self._splitter_centre = QSplitter(Qt.Orientation.Vertical)
        self._splitter_centre.addWidget(self._spectrum_canvas)
        self._splitter_centre.addWidget(self._timeline)
        self._splitter_centre.setStretchFactor(0, 1)
        self._splitter_centre.setStretchFactor(1, 0)
        self._splitter_centre.setSizes([300, 68])

        # ── Horizontal root splitter: left | centre | right ────────────────────
        self._splitter_h = QSplitter(Qt.Orientation.Horizontal)
        self._splitter_h.addWidget(self._splitter_left)
        self._splitter_h.addWidget(self._splitter_centre)
        self._splitter_h.addWidget(self._param_panel)
        self._splitter_h.setStretchFactor(0, 2)
        self._splitter_h.setStretchFactor(1, 1)
        self._splitter_h.setStretchFactor(2, 0)
        self._splitter_h.setSizes([620, 380, 275])

        self.setCentralWidget(self._splitter_h)

        # ── Status bar ─────────────────────────────────────────────────────────
        self.statusBar().setStyleSheet(
            "QStatusBar { background: #161a22; color: #5a6175;"
            " font-family: 'IBM Plex Mono'; font-size: 11px;"
            " border-top: 1px solid #252b38; }"
            "QStatusBar::item { border: none; }"
        )
        self.statusBar().showMessage("Ready — open a video file to begin (File > Open)")

    # ── Menu bar ───────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        """Create the full menu bar: File, View, Help."""
        menubar = self.menuBar()

        # ── File menu ──────────────────────────────────────────────────────────
        file_menu = menubar.addMenu("File")

        open_action = file_menu.addAction("Open Video File…")
        open_action.setShortcut("Ctrl+O")
        # SIGNAL: open_action.triggered → self._open_file
        # Purpose: opens QFileDialog, loads video+audio, updates all panels
        open_action.triggered.connect(self._open_file)

        # Recent Files submenu — populated from QSettings
        self._recent_menu = file_menu.addMenu("Recent Files")
        self._update_recent_menu()

        file_menu.addSeparator()

        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        # ── View menu ──────────────────────────────────────────────────────────
        view_menu = menubar.addMenu("View")

        reset_panels_action = view_menu.addAction("Reset Panel Sizes")
        # SIGNAL: reset_panels_action.triggered → self._reset_panels
        # Purpose: restore default QSplitter sizes when panels have been dragged
        reset_panels_action.triggered.connect(self._reset_panels)

        # ── Help menu ──────────────────────────────────────────────────────────
        help_menu = menubar.addMenu("Help")

        about_action = help_menu.addAction("About")
        # SIGNAL: about_action.triggered → self._show_about
        # Purpose: display course and author information
        about_action.triggered.connect(self._show_about)

    # ── Signal connections ─────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect every signal between panels, model, and timer.

        Each connect() call is documented: source → destination → purpose.
        ARCHITECTURE CONCEPT: no panel holds a reference to another panel.
        All inter-component communication goes through MainWindow via signals.
        """
        # ── Transport → model ──────────────────────────────────────────────────
        # TimelineWidget.play_requested → MainWindow._on_play_requested
        # Purpose: toggle audio playback on/off; anchor DAC time for A/V sync
        self._timeline.play_requested.connect(self._on_play_requested)

        # TimelineWidget.stop_requested → MainWindow._on_stop
        # Purpose: stop audio, reset all positions to frame 0
        self._timeline.stop_requested.connect(self._on_stop)

        # TimelineWidget.seek_requested(frame) → MainWindow._on_seek(frame)
        # Purpose: atomic seek of audio+video; immediate frame display
        self._timeline.seek_requested.connect(self._on_seek)

        # ── Waveform scrubbing → seek ──────────────────────────────────────────
        # WaveformView.seek_requested(frame) → MainWindow._on_seek(frame)
        # Purpose: clicking/dragging the waveform canvas seeks both audio and video
        self._waveform_view.seek_requested.connect(self._on_seek)

        # ── Parameter panel → model ────────────────────────────────────────────
        # ParameterPanel.audio_filter_changed(FilterState) → MainWindow._on_audio_filter
        # Purpose: hot-swap the IIR filter in the audio engine (thread-safe)
        self._param_panel.audio_filter_changed.connect(self._on_audio_filter_changed)

        # ParameterPanel.video_dsp_changed(str) → MainWindow._on_video_dsp_changed
        # Purpose: update the current video DSP mode; applied per-frame in _poll_av
        self._param_panel.video_dsp_changed.connect(self._on_video_dsp_changed)

        # ── Poll timer → A/V sync loop ─────────────────────────────────────────
        # QTimer.timeout → MainWindow._poll_av
        # Purpose: drain DAC queue at 30 fps; decode frame; update all view panels
        self._poll_timer.timeout.connect(self._poll_av)

    # ── File open ──────────────────────────────────────────────────────────────

    def _open_file(self, path_str: str | None = None) -> None:
        """Open a video file: stop current session, open model, update all views.

        Args:
            path_str: Path string to open directly (e.g. from Recent Files).
                      If None, opens a QFileDialog.
        """
        if not path_str:
            path_str, _ = QFileDialog.getOpenFileName(
                self, "Open Video File", "", SUPPORTED_VIDEO_FILTER
            )
        if not path_str:
            return   # user cancelled

        video_path = Path(path_str)

        # Stop current playback and reset all state before loading a new file
        self._decode_thread.set_playing(False)
        self._decode_thread.flush(0)
        self._session.stop()
        self._dac_time_anchor = None
        self._audio_frame_anchor = 0
        self._prev_bgr_frame = None
        self._timeline.reset()
        self.statusBar().showMessage(f"Opening {video_path.name}…")

        # ── Open model ─────────────────────────────────────────────────────────
        meta: VideoMeta | None = None
        try:
            meta = self._session.open(video_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot open video", str(exc))
            self.statusBar().showMessage("Error — could not open video file")
            return
        except (RuntimeError, Exception) as exc:
            # Audio extraction failed but model is partially open (meta available)
            meta = self._session.meta
            if meta is None:
                QMessageBox.critical(self, "Cannot open file", str(exc))
                self.statusBar().showMessage("Error — could not open file")
                return
            QMessageBox.warning(
                self, "Audio extraction failed",
                f"{exc}\n\nPlayback will continue without audio (silent)."
            )

        # ── Update all view components ─────────────────────────────────────────
        self.setWindowTitle(f"{video_path.name} — {WINDOW_TITLE}")
        self._timeline.set_file_loaded(meta.frame_count, meta.fps)

        # Populate WaveformView with the full audio buffer (mono mix)
        audio = self._session.audio_data
        if audio is not None:
            mono = audio.mean(axis=1).astype(np.float32)   # stereo → mono
            self._waveform_view.set_audio(mono, meta.frame_count, meta.fps)

        # Display the first video frame immediately
        first_frame = self._session.read_frame(0)
        if first_frame is not None:
            self._video_display.show_frame(self._apply_video_dsp(first_frame))

        self._timeline.update_position(0)

        # Send the current filter state to the engine (in case user changed it
        # while no file was open; also synchronises the new engine's filter)
        self._on_audio_filter_changed(self._session.get_filter_state())

        self.statusBar().showMessage(
            f"Loaded: {video_path.name}  —  "
            f"{meta.width}×{meta.height} @ {meta.fps:.3f} fps  —  "
            f"Audio: {DEVICE_SAMPLE_RATE:,} Hz stereo"
        )

        # Add to recent files list
        self._add_recent_file(str(video_path))

    # ── Playback control slots ─────────────────────────────────────────────────

    def _on_play_requested(self) -> None:
        """Toggle playback. Anchors the DAC time on transition to playing."""
        if self._timeline.is_playing:
            # Transitioning to playing — anchor will be set on first DAC tick
            self._dac_time_anchor = None
            self._audio_frame_anchor = (
                self._session.engine.frame_pos if self._session.engine else 0
            )
            current_frame = int(
                self._audio_frame_anchor / DEVICE_SAMPLE_RATE * (
                    self._session.meta.fps if self._session.meta else 30.0
                )
            )
            self._decode_thread.set_playing(True, target_frame=current_frame)
            self._session.play()
        else:
            self._decode_thread.set_playing(False)
            self._session.pause()

    def _on_stop(self) -> None:
        """Stop playback, reset positions, display frame 0."""
        self._decode_thread.set_playing(False)
        self._decode_thread.flush(0)
        self._session.stop()
        self._dac_time_anchor = None
        self._audio_frame_anchor = 0
        self._prev_bgr_frame = None
        self._timeline.set_stopped()
        self._timeline.update_position(0)
        self._waveform_view.set_playhead(0)

        frame = self._session.read_frame(0)
        if frame is not None:
            self._video_display.show_frame(self._apply_video_dsp(frame))

    def _on_seek(self, video_frame: int) -> None:
        """Seek audio and video atomically to a target video frame.

        SYNC CONCEPT — Atomic seek:
        Steps 1–7 must complete as an atomic unit from the UI perspective:
          1. Pause audio (callback no longer advances _frame_pos)
          2. Seek audio (under internal lock)
          3. Seek video (VideoCapture.set, sequential after lock release)
          4. Decode and display the target frame immediately
          5. Reset DAC time anchor so the sync loop re-anchors on resume
          6. Update timeline and waveform playhead
          7. Resume audio if it was playing

        Args:
            video_frame: Target video frame index (0-based).
        """
        meta = self._session.meta
        if meta is None:
            return

        was_playing = self._session.is_playing

        # Step 1: Pause audio and decode thread so neither advances during seek
        if was_playing:
            self._session.pause()
        self._decode_thread.set_playing(False)

        # Step 2: Seek audio (lock acquired inside MediaSession.seek)
        self._session.seek(video_frame)
        target_audio_frame = int(video_frame / meta.fps * DEVICE_SAMPLE_RATE)

        # Step 2b: Reopen VideoCapture to reset libavcodec state, then flush
        # the decode thread's queue so no stale pre-decoded frames are shown.
        self._session.reopen_video()
        self._decode_thread.flush(video_frame)

        # Step 3 & 4: Decode and display the target frame immediately
        bgr = self._session.read_frame(video_frame)
        if bgr is not None:
            self._prev_bgr_frame = None   # motion diff resets on seek
            processed = self._apply_video_dsp(bgr)
            self._video_display.show_frame(processed)
            self._prev_bgr_frame = bgr.copy()

        # Step 5: Reset DAC anchor — the sync loop will re-anchor on next tick
        self._dac_time_anchor = None
        self._audio_frame_anchor = target_audio_frame

        # Step 6: Update UI
        self._timeline.update_position(video_frame)
        self._waveform_view.set_playhead(video_frame)
        self._update_status_bar(video_frame)

        # Step 7: Resume if was playing
        if was_playing:
            self._decode_thread.set_playing(True, target_frame=video_frame)
            self._session.play()

    # ── DSP slots ─────────────────────────────────────────────────────────────

    def _on_audio_filter_changed(self, fs: FilterState) -> None:
        """Propagate new FilterState to the AudioPlaybackEngine.

        SIGNAL: ParameterPanel.audio_filter_changed → here
        SOURCE: ParameterPanel (already called rebuild() on the FilterState)
        DEST:   AudioPlaybackEngine.set_filter (thread-safe, minimal lock time)

        Args:
            fs: The new FilterState (sos and zi already built).
        """
        if self._session.engine is not None:
            self._session.engine.set_filter(fs)
        # Also update the session's own filter state for future engine swaps
        self._session.set_filter(fs)

    def _on_video_dsp_changed(self, mode: str) -> None:
        """Store the new video DSP mode for use in the per-frame dispatch.

        SIGNAL: ParameterPanel.video_dsp_changed → here
        SOURCE: ParameterPanel (QComboBox.currentTextChanged)
        DEST:   self._current_video_dsp — read in _poll_av per decoded frame

        Args:
            mode: One of DSP_PASSTHROUGH, DSP_GRAYSCALE, DSP_CANNY, etc.
        """
        self._current_video_dsp = mode

    def _apply_video_dsp(self, bgr: np.ndarray) -> np.ndarray:
        """Apply the currently selected video DSP operation to a BGR frame.

        Args:
            bgr: Input BGR frame (H, W, 3) uint8 from OpenCV.

        Returns:
            Processed BGR frame, same shape and dtype.
        """
        mode = self._current_video_dsp

        if mode == DSP_PASSTHROUGH:
            return process_passthrough(bgr)
        elif mode == DSP_GRAYSCALE:
            return process_grayscale(bgr)
        elif mode == DSP_CANNY:
            return process_canny(bgr, self._param_panel.canny_t1, self._param_panel.canny_t2)
        elif mode == DSP_BLUR:
            return process_blur(bgr, BLUR_KSIZE_DEFAULT)
        elif mode == DSP_EQUALIZE:
            return process_equalize(bgr)
        elif mode == DSP_MOTION:
            result = process_motion(self._prev_bgr_frame, bgr)
            return result
        return bgr

    # ── A/V sync loop ──────────────────────────────────────────────────────────

    def _poll_av(self) -> None:
        """30 fps A/V sync loop: drain DAC queue, decode frame, update all views.

        Called on the Qt main thread by QTimer at DISPLAY_TIMER_MS intervals.
        ALL Qt calls happen here — never in the audio callback.

        Steps each tick:
          1. Drain dac_queue — take the latest DAC timestamp (discard older ones).
          2. Drain spectrum_queue — take the latest (raw, filtered) block.
          3. If playing, compute target video frame from audio clock.
          4. Decode the frame with MediaSession.read_frame().
          5. Apply video DSP with _apply_video_dsp().
          6. Display via VideoDisplay.show_frame().
          7. Update WaveformView playhead and TimelineWidget position.
          8. Update SpectrumCanvas with the FFT of the latest audio block.
          9. Update status bar with current time and sync offset.
         10. Check finished_queue for end-of-file.
        """
        engine = self._session.engine
        if engine is None:
            return

        # ── Step 1: Drain DAC queue ────────────────────────────────────────────
        latest_dac: float | None = None
        try:
            while True:
                latest_dac = engine.dac_queue.get_nowait()
        except queue.Empty:
            pass

        # ── Step 2: Drain spectrum queue ───────────────────────────────────────
        latest_spectrum: tuple[np.ndarray, np.ndarray] | None = None
        try:
            while True:
                latest_spectrum = engine.spectrum_queue.get_nowait()
        except queue.Empty:
            pass

        meta = self._session.meta
        if meta is None:
            return

        target_video_frame: int | None = None

        # ── Steps 3–9: Video frame decode and display ──────────────────────────
        if latest_dac is not None and engine.is_playing:
            # Step 3: Compute target video frame from audio clock.
            # Anchor the DAC time on the first tick after play/seek.
            if self._dac_time_anchor is None:
                self._dac_time_anchor = latest_dac

            elapsed_seconds = latest_dac - self._dac_time_anchor
            current_audio_frame = self._audio_frame_anchor + int(
                elapsed_seconds * DEVICE_SAMPLE_RATE
            )
            target_video_frame = int(current_audio_frame / DEVICE_SAMPLE_RATE * meta.fps)
            target_video_frame = max(0, min(target_video_frame, meta.frame_count - 1))

            # Step 3b: Tell the decode thread where the audio clock is
            self._decode_thread.set_target(target_video_frame)

            # Step 4: Pull pre-decoded frames from the queue, discarding any
            # older than the audio clock target (we have already passed them).
            best_frame: np.ndarray | None = None
            best_idx: int = -1

            while True:
                item = self._decode_thread.get_frame()
                if item is None:
                    break   # queue empty — use whatever we have
                frame_idx, bgr_frame = item
                if frame_idx <= target_video_frame:
                    best_frame = bgr_frame
                    best_idx = frame_idx
                else:
                    # Frame is ahead of the clock — put it back and stop
                    try:
                        self._decode_thread._frame_queue.put_nowait((frame_idx, bgr_frame))
                    except queue.Full:
                        pass
                    break

            # Step 5 & 6: Apply DSP and display
            if best_frame is not None:
                processed = self._apply_video_dsp(best_frame)
                self._video_display.show_frame(processed)
                self._prev_bgr_frame = best_frame.copy()
                target_video_frame = best_idx

            # Step 7: Update waveform playhead and timeline
            self._waveform_view.set_playhead(target_video_frame)
            self._timeline.update_position(target_video_frame)

            # Step 9: Update status bar
            self._update_status_bar(target_video_frame, elapsed_seconds)

        # ── Step 8: Update spectrum ────────────────────────────────────────────
        if latest_spectrum is not None:
            raw_block, filtered_block = latest_spectrum
            freqs, raw_db = fft_magnitude_db(raw_block)
            _, filtered_db = fft_magnitude_db(filtered_block)
            self._spectrum_canvas.set_spectrum(raw_db, filtered_db, freqs)

        # ── Step 10: End-of-file check ─────────────────────────────────────────
        try:
            engine.finished_queue.get_nowait()
            self._timeline.set_stopped()
            self._dac_time_anchor = None
            self._audio_frame_anchor = 0
            self.statusBar().showMessage("Playback finished — press Play to replay")
        except queue.Empty:
            pass

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _update_status_bar(
        self,
        video_frame: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Build and display the status bar string.

        Format:
            HH:MM:SS.mmm | frame N / M | Sync: +X ms | fps | resolution | audio Hz | DSP: mode

        Args:
            video_frame: Current video frame index, or None to omit timing info.
            elapsed_seconds: Elapsed audio time since last anchor (for sync offset).
        """
        meta = self._session.meta
        if meta is None:
            return

        parts: list[str] = []

        if video_frame is not None:
            cur_s = video_frame / meta.fps
            m = int(cur_s // 60)
            s = cur_s - m * 60
            parts.append(f"{m:02d}:{s:06.3f}")
            parts.append(f"frame {video_frame:,} / {meta.frame_count:,}")

        if elapsed_seconds is not None:
            # Sync offset: difference between audio clock and expected video time
            expected_audio_s = video_frame / meta.fps if video_frame is not None else 0.0
            sync_ms = abs(elapsed_seconds - expected_audio_s) * 1_000
            sync_str = f"Sync: {sync_ms:.0f} ms" if sync_ms < 1_000 else "Sync: seeking"
            parts.append(sync_str)

        parts.append(f"{meta.fps:.3f} fps")
        parts.append(f"{meta.width}×{meta.height}")
        parts.append(f"audio {DEVICE_SAMPLE_RATE:,} Hz")
        parts.append(f"DSP: {self._current_video_dsp}")

        self.statusBar().showMessage("   │   ".join(parts))

    # ── QSettings persistence ──────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        """Restore window geometry and splitter sizes from QSettings.

        CONCEPT — QSettings persistence:
        QSettings stores key/value pairs in a platform-appropriate location
        (on Ubuntu: ~/.config/TU Ilmenau AMS/MM Seminar 03 Workbench.conf).
        Storing and restoring geometry and splitter state is what distinguishes
        a professional application from a script: the user's layout survives
        between sessions.
        """
        geom = self._settings.value(SETTINGS_GEOMETRY)
        if geom is not None:
            self.restoreGeometry(geom)

        splitter_h = self._settings.value(SETTINGS_SPLITTER_H)
        if splitter_h is not None:
            self._splitter_h.restoreState(splitter_h)

        splitter_left = self._settings.value(SETTINGS_SPLITTER_LEFT)
        if splitter_left is not None:
            self._splitter_left.restoreState(splitter_left)

        splitter_centre = self._settings.value(SETTINGS_SPLITTER_CENTRE)
        if splitter_centre is not None:
            self._splitter_centre.restoreState(splitter_centre)

    def _save_settings(self) -> None:
        """Save window geometry and splitter sizes to QSettings."""
        self._settings.setValue(SETTINGS_GEOMETRY, self.saveGeometry())
        self._settings.setValue(SETTINGS_SPLITTER_H, self._splitter_h.saveState())
        self._settings.setValue(SETTINGS_SPLITTER_LEFT, self._splitter_left.saveState())
        self._settings.setValue(SETTINGS_SPLITTER_CENTRE, self._splitter_centre.saveState())

    def _add_recent_file(self, path_str: str) -> None:
        """Add a path to the recent files list in QSettings and update the menu.

        CONCEPT — Recent files list:
        QSettings stores a list as a semicolon-joined string. We keep the most
        recent SETTINGS_RECENT_MAX paths, removing duplicates and keeping order.

        Args:
            path_str: Absolute path string to add.
        """
        recent_raw = self._settings.value(SETTINGS_RECENT_FILES, "")
        recent: list[str] = [p for p in recent_raw.split(";") if p] if recent_raw else []

        # Remove existing entry (if any) then prepend — most recent first
        recent = [p for p in recent if p != path_str]
        recent.insert(0, path_str)
        recent = recent[:SETTINGS_RECENT_MAX]

        self._settings.setValue(SETTINGS_RECENT_FILES, ";".join(recent))
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        """Rebuild the Recent Files submenu from QSettings.

        CONCEPT — Dynamic QAction menu:
        Each recent file gets its own QAction. The triggered signal passes the
        path via a lambda with a default argument (captures path by value, not
        by reference — avoids the common Python closure-in-loop bug).
        """
        self._recent_menu.clear()
        recent_raw = self._settings.value(SETTINGS_RECENT_FILES, "")
        recent: list[str] = [p for p in recent_raw.split(";") if p] if recent_raw else []

        if not recent:
            no_files_action = self._recent_menu.addAction("(no recent files)")
            no_files_action.setEnabled(False)
            return

        for path_str in recent:
            name = Path(path_str).name
            action = self._recent_menu.addAction(name)
            # Default argument path=path_str captures path by value — not by reference
            action.triggered.connect(lambda checked=False, p=path_str: self._open_file(p))

    def _reset_panels(self) -> None:
        """Reset QSplitter sizes to their defaults."""
        self._splitter_h.setSizes([620, 380, 275])
        self._splitter_left.setSizes([480, 120])
        self._splitter_centre.setSizes([300, 68])

    def _show_about(self) -> None:
        """Display the About dialog with course and author information."""
        QMessageBox.information(
            self,
            "About — Multimedia Workbench",
            "Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia\n\n"
            "Demo 5: Full Multimedia Workbench\n\n"
            "Institut für Medientechnik\n"
            "Technische Universität Ilmenau\n\n"
            "Author: Renato Fera P.\n"
            "linkedin.com/in/renato-profeta\n\n"
            "Combines all components from Demos 1–4:\n"
            "  • AudioPlaybackEngine (A/V master clock)\n"
            "  • Real-time IIR filter + SpectrumCanvas (Demo 3)\n"
            "  • WaveformView + peak reduction (Demo 2)\n"
            "  • VideoDisplay + frame DSP pipeline (Demo 4)\n"
            "  • QSettings persistence (geometry, panels, recent files)",
        )

    # ── Window lifecycle ───────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Stop timer, release all resources, save settings on window close.

        RESOURCE MANAGEMENT:
        Each resource is released in a separate try/except so that one failure
        does not prevent the others from running. An unreleased sounddevice
        stream holds the audio device (ALSA lock on Ubuntu). An unreleased
        VideoCapture holds a file descriptor. An uncleaned TemporaryDirectory
        leaves the extracted WAV on disk.
        """
        self._poll_timer.stop()

        try:
            self._decode_thread.stop()
        except Exception as exc:
            print(f"[MainWindow] Decode thread stop error: {exc}", flush=True)

        try:
            self._session.close()
        except Exception as exc:
            print(f"[MainWindow] Session close error: {exc}", flush=True)

        try:
            self._save_settings()
        except Exception as exc:
            print(f"[MainWindow] Settings save error: {exc}", flush=True)

        super().closeEvent(event)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Create the QApplication, apply the dark palette, and start the event loop."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark Fusion palette — consistent with all course demos and the HTML design system
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0d0f14"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#d8dce6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#161a22"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1f2b"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#161a22"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#d8dce6"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4f9cf9"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d0f14"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#161a22"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#d8dce6"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# =============================================================================
# End of 03_mm_multimedia_workbench.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
