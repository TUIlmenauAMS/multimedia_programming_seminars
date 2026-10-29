# =============================================================================
# 03_mm_video_player.py — Video Player with Audio Sync and Frame DSP
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A professional video player with audio-driven synchronisation and a live
#   frame-level DSP pipeline.
#
#   A video file is opened via File > Open. The audio track is extracted at
#   load time using subprocess + ffmpeg to a temporary WAV file, then loaded
#   with soundfile into a NumPy buffer. Playback uses a sounddevice
#   OutputStream callback as the master clock: PortAudio provides the DAC
#   timestamp (outputBufferDacTime) which is the most accurate timing signal
#   available on the system. A QTimer in the main thread at 60 Hz reads this
#   timestamp from a queue.Queue, computes the target frame index as
#   int(dac_time * fps), seeks VideoCapture with CAP_PROP_POS_FRAMES, decodes
#   the frame, applies the selected DSP operation, converts BGR → RGB →
#   QImage → QPixmap, and displays it on a QLabel.
#
#   Frame DSP pipeline (selectable via QComboBox):
#     Passthrough       — no processing
#     Grayscale         — colour to luminance
#     Canny edges       — hysteresis edge detector (thresholds adjustable)
#     Gaussian blur     — noise reduction / defocus
#     Histogram EQ      — contrast enhancement for poorly exposed footage
#     Motion diff       — frame differencing for motion detection
#
#   The timeline scrubber seeks both audio and video atomically: audio is
#   paused, both stream positions are updated under the audio lock, the target
#   frame is decoded and displayed immediately, then audio resumes.
#
#   WEBGL EQUIVALENT:
#     cv2.VideoCapture.read()         ↔  HTMLVideoElement.requestVideoFrameCallback()
#     cv2.cvtColor + QImage display   ↔  drawImage(video, canvas)
#     sounddevice OutputStream        ↔  AudioContext (master clock for A/V sync)
#     time_info.outputBufferDacTime   ↔  AudioContext.currentTime
#     cv2.Canny / cv2.GaussianBlur    ↔  WebGL fragment shader (per-pixel DSP)
#     subprocess ffmpeg extract       ↔  MediaSource API (demuxing)
#
# How to run:
#   uv sync                                       # first time only
#   uv run python 03_mm_video_player.py
#
# Ubuntu system dependencies (install once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0 \
#                    libportaudio2 libsndfile1 ffmpeg libopencv-dev
#
# Connections to other seminars:
#   Demo 2 (waveform_player) — introduces PlaybackEngine and the
#     queue.Queue/QTimer pattern that AudioPlaybackEngine builds on here.
#   Demo 3 (spectrum_analyzer) — introduces sosfilt with zi state;
#     the audio callback structure here is identical in pattern.
#   Demo 5 (multimedia_workbench) — imports VideoDisplay and
#     AudioPlaybackEngine from this file as reusable components.
# =============================================================================

from __future__ import annotations

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
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# =============================================================================
# STEP 1 — Constants
# =============================================================================

WINDOW_TITLE: str = "Video Player — Multimediaprogrammierung Seminar 03"
WINDOW_MIN_WIDTH: int = 1060
WINDOW_MIN_HEIGHT: int = 620

# Audio I/O
DEVICE_SAMPLE_RATE: int = 48_000    # Hz — target playback rate for extracted audio
AUDIO_BLOCK_SIZE: int = 512         # frames per PortAudio callback block

# Display timer — 60 Hz for smooth video; audio callbacks will pace actual
# frame timing, so 60 Hz polling is sufficient to keep the display current.
DISPLAY_FPS: int = 60
DISPLAY_TIMER_MS: int = 1_000 // DISPLAY_FPS    # 16 ms

# ffmpeg audio extraction command template.
# -i           : input file
# -vn          : no video output (audio only)
# -acodec pcm_s16le : uncompressed 16-bit PCM — universally supported by soundfile
# -ar 48000    : resample to DEVICE_SAMPLE_RATE so no Python-side resampling needed
# -ac 2        : force stereo output (mono sources are duplicated)
# -y           : overwrite output without prompting
FFMPEG_EXTRACT_CMD: list[str] = [
    "ffmpeg", "-i", "{input}", "-vn",
    "-acodec", "pcm_s16le",
    "-ar", str(DEVICE_SAMPLE_RATE),
    "-ac", "2",
    "-y", "{output}",
]

# Supported video file extensions shown in QFileDialog
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    "*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm", "*.flv", "*.m4v",
)
SUPPORTED_VIDEO_FILTER: str = (
    "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.m4v)"
)

# DSP mode identifiers — keys in DSP_MODES dict, used by QComboBox
DSP_PASSTHROUGH: str = "Passthrough"
DSP_GRAYSCALE: str = "Grayscale"
DSP_CANNY: str = "Canny Edges"
DSP_BLUR: str = "Gaussian Blur"
DSP_EQUALIZE: str = "Histogram EQ"
DSP_MOTION: str = "Motion Diff"

DSP_MODES: tuple[str, ...] = (
    DSP_PASSTHROUGH,
    DSP_GRAYSCALE,
    DSP_CANNY,
    DSP_BLUR,
    DSP_EQUALIZE,
    DSP_MOTION,
)

# Canny edge detector default thresholds.
# Threshold1 = lower hysteresis boundary; threshold2 = upper boundary.
# Edges with gradient > threshold2 are definitely edges.
# Edges with gradient between threshold1 and threshold2 are edges only if
# connected to a definite edge. Edges below threshold1 are discarded.
CANNY_THRESHOLD1_DEFAULT: int = 50
CANNY_THRESHOLD2_DEFAULT: int = 150

# Gaussian blur default kernel size (must be odd and positive)
BLUR_KSIZE_DEFAULT: int = 15

# Canvas and UI colours — same dark theme as all course demos
CANVAS_BG: str = "#0d0f14"
LABEL_COLOR: str = "#5a6175"
ACCENT_COLOR: str = "#4f9cf9"
ACCENT2_COLOR: str = "#a78bfa"
ORANGE_COLOR: str = "#fb923c"
GREEN_COLOR: str = "#34d399"

# AMS logo — one directory above this script (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"


# =============================================================================
# STEP 2 — MediaFile: video metadata, audio extraction, audio loading
# =============================================================================

@dataclass
class VideoMeta:
    """Metadata read from a VideoCapture object at open time.

    Attributes:
        fps: Frames per second of the video stream.
        frame_count: Total frame count (may be approximate for some codecs).
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_seconds: Total duration in seconds.
    """

    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_seconds = self.frame_count / max(self.fps, 1e-6)


class MediaFile:
    """Opens a video file and prepares it for synchronised A/V playback.

    Responsibilities:
      - Open the video stream with cv2.VideoCapture and validate it.
      - Extract the audio track to a temporary WAV file using ffmpeg.
      - Load the extracted WAV with soundfile.

    Why ffmpeg for audio extraction:
      OpenCV's VideoCapture decodes video frames only — it does not decode
      audio tracks. Every production-grade video pipeline (VLC, ffplay,
      DaVinci Resolve, ffmpeg itself) handles audio and video as separate
      streams. ffmpeg is the universal demuxer: it handles every container
      (MP4, MKV, AVI, MOV, WebM) and every audio codec (AAC, AC-3, MP3,
      Opus, FLAC) that students will encounter in real-world media files.
      subprocess.run is the correct way to call ffmpeg from Python in a
      professional pipeline — do not use os.system().
    """

    def __init__(self) -> None:
        """Initialise with no file open."""
        self._cap: cv2.VideoCapture | None = None
        self._meta: VideoMeta | None = None
        self._audio_data: np.ndarray | None = None   # float32, (frames, 2)
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._tmp_wav: Path | None = None
        # Tracks the next frame index the codec is positioned at, so read_frame()
        # can avoid issuing a seek when the requested frame is sequential.
        # Initialised to an impossible value to force a seek on the first read.
        self._next_frame_idx: int = -999_999

    # ── Video ─────────────────────────────────────────────────────────────────

    def open_video(self, path: Path) -> VideoMeta:
        """Open a video file and return its metadata.

        Args:
            path: Path to the video file.

        Returns:
            VideoMeta with fps, frame_count, width, height.

        Raises:
            ValueError: If the file cannot be opened or has no video stream.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(
                f"OpenCV cannot open '{path.name}'.\n"
                "Check that the file exists and that the codec is installed.\n"
                f"Ubuntu: sudo apt install libopencv-dev"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise ValueError(
                f"'{path.name}' reports invalid video properties "
                f"(fps={fps}, frames={frame_count}, {width}×{height}).\n"
                "The file may be corrupted or the codec unsupported."
            )

        if self._cap is not None:
            self._cap.release()

        self._cap = cap
        self._meta = VideoMeta(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        return self._meta

    # ── Audio extraction ───────────────────────────────────────────────────────

    def extract_audio(self, video_path: Path) -> Path:
        """Extract the audio track from a video file to a temporary WAV.

        Uses ffmpeg via subprocess.run. The extracted WAV is written to a
        TemporaryDirectory managed by this MediaFile instance and deleted in
        close().

        CONCEPT — Why not OpenCV for audio?
        cv2.VideoCapture reads video frames only. Audio packets in the
        container are not accessible through OpenCV's Python bindings.
        ffmpeg demuxes the container, decodes the audio stream, and writes
        PCM samples. This is the standard pattern in every production A/V
        player that uses OpenCV for video.

        CONCEPT — ffmpeg command breakdown:
          -i {input}           Read the input container (MP4, MKV, etc.)
          -vn                  Discard the video stream — audio output only
          -acodec pcm_s16le    Encode output as 16-bit little-endian PCM
                               (uncompressed; universally supported by soundfile)
          -ar 48000            Resample to DEVICE_SAMPLE_RATE in ffmpeg.
                               Doing this here avoids a separate Python resample step.
          -ac 2                Force 2-channel stereo output.
                               Mono sources are automatically duplicated.
          -y                   Overwrite the output WAV without prompting.

        Args:
            video_path: Path to the video file.

        Returns:
            Path to the extracted temporary WAV file.

        Raises:
            RuntimeError: If ffmpeg is not installed or extraction fails.
        """
        # Create a TemporaryDirectory that persists until close() is called.
        # This ensures the WAV file is available for the entire session.
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="mm_video_")
        tmp_wav = Path(self._tmp_dir.name) / "audio.wav"

        cmd = [
            arg
            .replace("{input}", str(video_path))
            .replace("{output}", str(tmp_wav))
            for arg in FFMPEG_EXTRACT_CMD
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,     # capture stdout and stderr
                text=True,               # decode as UTF-8 strings
                check=False,             # handle non-zero exit ourselves
                timeout=120,             # seconds — prevent hang on huge files
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it with:\n"
                "  sudo apt install ffmpeg\n"
                "ffmpeg is required to extract audio from video files."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg timed out extracting audio from '{video_path.name}'.\n"
                "The file may be very large or the system is under heavy load."
            )

        if result.returncode != 0:
            # ffmpeg writes diagnostic output to stderr
            stderr_tail = result.stderr[-800:] if result.stderr else "(no output)"
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}) "
                f"extracting audio from '{video_path.name}'.\n\n"
                f"ffmpeg output:\n{stderr_tail}\n\n"
                "If the video has no audio track, playback will use silence."
            )

        self._tmp_wav = tmp_wav
        return tmp_wav

    # ── Audio loading ──────────────────────────────────────────────────────────

    def load_audio(self, wav_path: Path) -> np.ndarray:
        """Load the extracted WAV into a float32 NumPy array.

        Uses soundfile (libsndfile) which is the professional standard for
        audio file I/O in Python. Returns shape (frames, 2), float32,
        values in [-1.0, 1.0]. Because ffmpeg already resampled to
        DEVICE_SAMPLE_RATE and set -ac 2, no further processing is needed.

        Args:
            wav_path: Path to the WAV file produced by extract_audio().

        Returns:
            Audio data, shape (frames, 2), dtype float32.

        Raises:
            ValueError: If the WAV cannot be read.
        """
        try:
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        except Exception as exc:
            raise ValueError(f"Cannot read extracted audio: {exc}") from exc

        if data.shape[1] < 2:
            # Mono — duplicate to stereo (should not happen with -ac 2, but guard it)
            data = np.repeat(data, 2, axis=1)

        self._audio_data = data
        return data

    # ── Frame access ───────────────────────────────────────────────────────────

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """Decode one frame, seeking only when necessary.

        KEY FIX — Sequential decode vs. random-access seek:
        The original implementation called CAP_PROP_POS_FRAMES on every single
        display tick. For compressed codecs (H.264, H.265, VP9) this is
        catastrophically expensive: the codec must find the nearest keyframe,
        then decode every delta frame in the GOP to reconstruct the target.
        At 60 Hz polling that means potentially 50–150 wasted decodes per
        displayed frame, causing the stutter described in the bug report.

        The fix: track the next expected frame index (_next_frame_idx).
        If the caller requests the next sequential frame, just call cap.read()
        — the codec is already positioned correctly and this is O(1).
        Only issue a seek when there is a genuine discontinuity (after a scrub,
        after stop→play, or if audio sync drifts by more than SEEK_TOLERANCE
        frames). This reduces seeks to near-zero during normal playback.

        SEEK_TOLERANCE: allow a 2-frame window before corrective seek.
        A/V drift within ±2 frames (~67 ms at 30 fps) is imperceptible.
        Seeking for sub-2-frame drift costs more than it gains.

        Args:
            frame_idx: Zero-based frame index to decode.

        Returns:
            BGR frame as shape (H, W, 3) uint8 NumPy array, or None on error.
        """
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

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release all resources.

        Must be called from MainWindow.closeEvent(). Failure to call this
        will leave the audio device locked and the temp directory on disk.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
        self._audio_data = None

    @property
    def meta(self) -> VideoMeta | None:
        """Video metadata, available after open_video()."""
        return self._meta

    @property
    def audio_data(self) -> np.ndarray | None:
        """Audio buffer, available after load_audio(). Shape (frames, 2), float32."""
        return self._audio_data


# =============================================================================
# STEP 3 — AudioPlaybackEngine: sounddevice OutputStream as master clock
# =============================================================================

class AudioPlaybackEngine:
    """sounddevice OutputStream that serves as the A/V master clock.

    SYNC CONCEPT — Why audio is the master clock:
    The audio DAC runs on a crystal oscillator with parts-per-million
    accuracy. PortAudio schedules the audio callback at precisely the right
    time for the DAC to play the samples without gaps or overlaps. The
    callback receives time_info.outputBufferDacTime: the exact wall-clock
    time (in seconds since stream start) at which the first sample of the
    current block will reach the DAC output pins.

    This is the most accurate timestamp available on any platform:
      - More accurate than time.time() (system clock, subject to NTP jumps)
      - More accurate than time.perf_counter() (no DAC synchronisation)
      - More accurate than QTimer (Qt's timer resolution is ~1 ms)

    Using outputBufferDacTime as the reference, the main thread computes
    the video frame to display as:
        target_frame = int(dac_time * fps)

    This is called audio-driven video presentation and is the foundation of
    how VLC, ffplay, and every professional broadcast player works.

    THREAD CONCEPT:
    The callback runs in a high-priority PortAudio audio thread. It must:
      - Never call Qt functions (no signal.emit(), no widget.update())
      - Never allocate memory
      - Never block on I/O, locks, or system calls
      - Complete before the next callback deadline (AUDIO_BLOCK_SIZE / sample_rate seconds)

    All communication from the callback to the main thread goes through
    queue.Queue, which is thread-safe and non-blocking when used with put_nowait().
    """

    def __init__(self) -> None:
        """Initialise with no audio loaded. Call load() before play()."""
        self._data: np.ndarray | None = None    # float32, shape (frames, 2)
        self._frame_pos: int = 0                # current playback position in frames
        self._lock: threading.Lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._start_dac_time: float = 0.0       # DAC time at stream start (seconds)
        self._start_frame: int = 0              # audio frame index at stream start

        # Queue: callback → main thread, carries outputBufferDacTime values.
        # The main thread reads the latest value at 60 Hz to drive video display.
        # maxsize=8: if the UI thread falls behind, older timestamps are discarded.
        self._dac_queue: queue.Queue[float] = queue.Queue(maxsize=8)

        # Queue: callback → main thread, signals end-of-file.
        self._finished_queue: queue.Queue[bool] = queue.Queue(maxsize=2)

    # ── Buffer management ──────────────────────────────────────────────────────

    def load(self, audio_data: np.ndarray) -> None:
        """Load audio data into the engine.

        Args:
            audio_data: Float32 array, shape (frames, 2), values in [-1.0, 1.0].
        """
        was_active = self._stream is not None and self._stream.active
        if was_active:
            self._stream.stop()

        with self._lock:
            self._data = audio_data
            self._frame_pos = 0
            self._start_frame = 0
            self._start_dac_time = 0.0

    # ── Audio callback — PortAudio audio thread ────────────────────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill outdata from the audio buffer and push DAC timestamp to queue.

        SYNC CONCEPT — outputBufferDacTime:
        time_info.outputBufferDacTime is the time in seconds when the FIRST
        sample of THIS block will reach the DAC. It is computed by PortAudio
        from the hardware clock and accounts for driver latency.

        To convert DAC time to a video frame index:
            audio_seconds = dac_time - stream_start_dac_time + seek_offset_seconds
            target_frame  = int(audio_seconds * fps)

        We simplify here by tracking _start_dac_time and _start_frame so that
        the main thread can compute audio_seconds = dac_time - _start_dac_time
        and then add the seek offset.

        In practice, for this demo we push absolute dac_time and let the main
        thread track the offset, because seek() resets both _start_dac_time
        and _start_frame together under the lock.

        Args:
            outdata: Output buffer to fill, shape (frames, 2), float32.
            frames: Number of frames requested.
            time_info: PortAudio timing struct with outputBufferDacTime.
            status: Stream health flags.
        """
        if status.output_underflow:
            # Underflow means the CPU could not fill the buffer in time.
            # This produces an audible glitch. Log it; do not raise.
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
                # Last partial block — copy what remains, zero-pad the rest
                outdata[:remaining] = self._data[pos:pos + remaining]
                outdata[remaining:] = 0.0
                self._frame_pos = n_total
                try:
                    self._finished_queue.put_nowait(True)
                except queue.Full:
                    pass
            else:
                # Normal full block
                outdata[:] = self._data[pos:pos + frames]
                self._frame_pos += frames

            # Push DAC timestamp to the main thread.
            # The main thread uses this to compute the video frame to display.
            # put_nowait is non-blocking — if the queue is full (UI thread is slow),
            # the oldest timestamp is already stale, so dropping it is correct.
            try:
                self._dac_queue.put_nowait(time_info.outputBufferDacTime)
            except queue.Full:
                # Drain one stale entry and push the fresh one
                try:
                    self._dac_queue.get_nowait()
                    self._dac_queue.put_nowait(time_info.outputBufferDacTime)
                except queue.Empty:
                    pass

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
            self._start_frame = 0
            self._start_dac_time = 0.0

    def seek(self, audio_frame: int) -> None:
        """Seek to a specific audio frame position.

        SYNC CONCEPT — Atomic seek:
        Seeking must be atomic with respect to the audio callback. If the
        callback reads _frame_pos between our write and a later related state
        update, the streams can diverge. We hold the lock for the entire
        seek operation.

        This method is called from the main thread; the caller must pause()
        the stream before calling seek() if it wants to avoid a brief
        audible glitch at the seek point. See MainWindow._on_seek().

        Args:
            audio_frame: Target audio frame index. Clamped to valid range.
        """
        with self._lock:
            if self._data is not None:
                n_total = self._data.shape[0]
                self._frame_pos = max(0, min(audio_frame, n_total))
                self._start_frame = self._frame_pos
                self._start_dac_time = 0.0   # reset on seek; main thread re-anchors

    def close(self) -> None:
        """Stop the stream and release the audio device. Call from closeEvent."""
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
    def frame_pos(self) -> int:
        """Current audio frame position (approximate — not lock-protected)."""
        return self._frame_pos

    @property
    def is_playing(self) -> bool:
        """True if the audio stream is currently active."""
        return self._stream is not None and self._stream.active


# =============================================================================
# STEP 4 — Frame DSP functions: pure OpenCV/NumPy, no Qt
# =============================================================================

def process_passthrough(frame: np.ndarray) -> np.ndarray:
    """Return the frame unmodified.

    Args:
        frame: BGR frame, shape (H, W, 3), uint8.

    Returns:
        The same array unchanged.
    """
    return frame


def process_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert the frame to grayscale, returned as 3-channel BGR.

    VIDEO CONCEPT — Grayscale conversion:
    cv2.COLOR_BGR2GRAY computes luminance using the ITU-R BT.601 formula:
        Y = 0.299 R + 0.587 G + 0.114 B
    These coefficients reflect human visual sensitivity: the eye is most
    sensitive to green, less to red, least to blue. A naive average of the
    three channels (0.333 each) would produce incorrect perceived brightness.

    The result is stacked to 3 channels (cv2.merge) so the downstream
    BGR → RGB → QImage pipeline always receives a 3-channel array.

    Use cases in broadcast/production:
      - Reference monitoring (luminance-only for legal broadcast levels)
      - Preparing input for algorithms that require grayscale (e.g. face detection)
      - Archival conversion of colour-incorrect footage

    Args:
        frame: BGR frame, shape (H, W, 3), uint8.

    Returns:
        3-channel BGR frame with equal R=G=B (grayscale), same shape.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # (H, W), uint8
    return cv2.merge([gray, gray, gray])                     # (H, W, 3), uint8


def process_canny(
    frame: np.ndarray,
    threshold1: int,
    threshold2: int,
) -> np.ndarray:
    """Apply Canny edge detection and return edges as a 3-channel BGR frame.

    VIDEO CONCEPT — Canny edge detector:
    Canny (1986) is the standard first-pass edge detector. It has four stages:
      1. Gaussian smoothing (5×5 kernel) to remove noise before differentiation
      2. Sobel gradient magnitude and direction at every pixel
      3. Non-maximum suppression: thin edges to one pixel wide by zeroing
         pixels that are not local maxima along the gradient direction
      4. Hysteresis thresholding:
           - Pixels with gradient > threshold2 are definite edges
           - Pixels with gradient < threshold1 are discarded
           - Pixels between threshold1 and threshold2 are kept only if
             connected (8-connectivity) to a definite edge

    The two thresholds give the user control over sensitivity:
      - Raising threshold2: fewer false edges (stricter upper bound)
      - Lowering threshold1: shorter edges still connect to definite ones

    Typical values: threshold1 = threshold2 / 3 (OpenCV's recommended ratio).

    Use cases in broadcast/production:
      - Pre-processing for rotoscoping / matte extraction
      - Object boundary detection in video analysis pipelines
      - Visual effects: edge-glow, sketch look, stylisation

    Args:
        frame: BGR frame, shape (H, W, 3), uint8.
        threshold1: Lower hysteresis threshold.
        threshold2: Upper hysteresis threshold.

    Returns:
        3-channel BGR frame with white edges on black background.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)          # (H, W), uint8, binary
    return cv2.merge([edges, edges, edges])                  # (H, W, 3)


def process_blur(frame: np.ndarray, ksize: int) -> np.ndarray:
    """Apply Gaussian blur to the frame.

    VIDEO CONCEPT — Gaussian blur:
    GaussianBlur convolves the frame with a 2-D Gaussian kernel of size
    ksize × ksize. Larger kernels produce more blur and are slower. ksize
    must be odd and positive (OpenCV requirement).

    The kernel is separable: a ksize × ksize 2-D Gaussian = 1-D horizontal
    Gaussian followed by 1-D vertical Gaussian. OpenCV exploits this
    automatically, so complexity is O(ksize * W * H) not O(ksize² * W * H).

    sigma=0 tells OpenCV to compute the standard deviation automatically from
    ksize using σ = 0.3 × ((ksize-1) × 0.5 − 1) + 0.8.

    Use cases in broadcast/production:
      - Noise reduction before encode (reduces DCT artefacts in H.264/H.265)
      - Depth-of-field simulation
      - Privacy masking (face blur)
      - Pre-processing before edge detection or optical flow

    Args:
        frame: BGR frame, shape (H, W, 3), uint8.
        ksize: Kernel size (must be odd and > 0). Clamped to nearest odd integer.

    Returns:
        Blurred BGR frame, same shape.
    """
    # Clamp ksize to a valid odd integer >= 1
    ksize = max(1, ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=0)


def process_equalize(frame: np.ndarray) -> np.ndarray:
    """Apply per-channel histogram equalisation for contrast enhancement.

    VIDEO CONCEPT — Histogram equalisation:
    Histogram equalisation redistributes pixel intensities so that every
    intensity level is used roughly equally. For a well-exposed image with
    intensities spread across [0, 255], the effect is small. For a dark or
    washed-out image, it dramatically improves contrast and reveals detail.

    cv2.equalizeHist works on single-channel uint8 images. We apply it
    independently to the B, G, and R channels. This can shift colour balance
    (white balance shifts). A more perceptually correct approach would be
    to convert to HSV, equalise only the V (value) channel, and convert back
    — but per-channel RGB equalisation is more commonly used in practice.

    Use cases in broadcast/production:
      - Correcting under-exposed surveillance or security footage
      - Normalising inconsistent exposure across a multi-camera shoot
      - Pre-processing for computer vision (face recognition, licence plates)

    Args:
        frame: BGR frame, shape (H, W, 3), uint8.

    Returns:
        Equalised BGR frame, same shape.
    """
    b, g, r = cv2.split(frame)
    return cv2.merge([
        cv2.equalizeHist(b),
        cv2.equalizeHist(g),
        cv2.equalizeHist(r),
    ])


def process_motion(prev_frame: np.ndarray | None, frame: np.ndarray) -> np.ndarray:
    """Compute the absolute difference between consecutive frames.

    VIDEO CONCEPT — Frame differencing / motion detection:
    cv2.absdiff(prev, curr) computes |prev[i,j] − curr[i,j]| per pixel.
    Pixels where nothing moved are black (near zero). Pixels where something
    moved are bright (large difference).

    A threshold is applied to suppress noise: small differences from codec
    artefacts, sensor noise, and slight camera vibration are zeroed out.
    Remaining bright pixels mark regions of motion.

    Use cases in broadcast/production:
      - Real-time motion detection for surveillance systems
      - Shot-change detection in video editing pipelines
      - Video compression pre-analysis (where to spend more bits)
      - Background subtraction for keying

    Args:
        prev_frame: The previous BGR frame, or None on the first frame.
        frame: The current BGR frame, shape (H, W, 3), uint8.

    Returns:
        3-channel BGR motion map, same shape. White = motion, black = static.
    """
    if prev_frame is None or prev_frame.shape != frame.shape:
        # First frame or shape mismatch — return black frame
        return np.zeros_like(frame)

    diff = cv2.absdiff(prev_frame, frame)

    # Threshold: suppress codec noise (differences < 20 are likely artefacts)
    _, mask = cv2.threshold(
        cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY),
        thresh=20,
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    return cv2.merge([mask, mask, mask])


# =============================================================================
# STEP 4b — VideoDecodeThread: background frame decoder
# =============================================================================

class VideoDecodeThread(threading.Thread):
    """Dedicated background thread that pre-decodes video frames.

    ARCHITECTURE — Why a separate decode thread:
    cv2.VideoCapture.read() blocks the calling thread for as long as it
    takes the codec to decode one frame. For H.264/H.265 at high resolution
    this can be 5–30 ms. When decode runs on the Qt main thread (inside the
    QTimer callback) it blocks the event loop for that duration: the UI
    freezes, the timer fires again before the previous frame is shown, and
    the result is the stutter/frame-skipping the user reported.

    Moving decode to a background thread decouples the codec latency from
    the display cadence:
      - The decode thread runs continuously, always 1–2 frames ahead of
        where the audio clock currently is.
      - The main thread merely picks up the already-decoded frame from
        _frame_queue and displays it — a near-zero-cost operation.
      - If the main thread is briefly late (e.g. during a DSP pass),
        the pre-decoded frame waits in the queue; no frame is dropped.

    FRAME QUEUE:
    _frame_queue holds (frame_idx, bgr_ndarray) tuples.
    maxsize=4: enough to absorb a single heavy DSP frame on the display
    side without falling behind. Larger buffers waste memory and introduce
    display latency.

    SYNC PROTOCOL:
    The decode thread reads the audio clock's target frame index from
    _target_frame (written by the main thread under _lock) and decodes
    forward from there. This ensures the buffer tracks the audio position
    even after a seek.

    STOP / FLUSH:
    _stop_event: set by stop() to terminate the thread loop.
    _flush_event: set by flush() to discard stale pre-decoded frames after
    a seek and immediately re-anchor to the new position.
    """

    def __init__(self, media: "MediaFile") -> None:  # forward-ref; MediaFile defined above
        super().__init__(name="VideoDecodeThread", daemon=True)
        self._media = media
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._wake_event = threading.Event()   # poke the thread when paused

        # (frame_idx, bgr_frame) pairs ready for display
        self._frame_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=4)

        # Current target: the audio clock tells us where to decode next.
        # Written by set_target() from the main thread; read by run() loop.
        self._target_frame: int = 0
        self._playing: bool = False

    # ── Control ───────────────────────────────────────────────────────────────

    def set_playing(self, playing: bool, target_frame: int = 0) -> None:
        """Signal the thread to start or pause decoding.

        Args:
            playing: True to decode, False to idle.
            target_frame: Frame index to decode from when resuming.
        """
        with self._lock:
            self._playing = playing
            if playing:
                self._target_frame = target_frame
        self._wake_event.set()

    def set_target(self, target_frame: int) -> None:
        """Update the target frame index (called by main thread each tick).

        The decode thread uses this to stay ahead of the audio clock.
        No locking needed for a single integer write on CPython (GIL).
        """
        self._target_frame = target_frame

    def flush(self, new_target: int) -> None:
        """Discard all buffered frames and restart from new_target.

        Call this after a seek so stale pre-decoded frames are not shown.
        The flush is acknowledged by the decode thread resetting its
        internal position and refilling the queue from new_target.

        Args:
            new_target: Frame index to restart decoding from.
        """
        with self._lock:
            self._target_frame = new_target
        self._flush_event.set()
        # Drain the queue so the display thread does not consume stale frames
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._wake_event.set()

    def get_frame(self) -> tuple[int, np.ndarray] | None:
        """Return the next pre-decoded frame, or None if the queue is empty.

        Called from the main thread in the A/V sync loop.
        Non-blocking: returns None immediately if no frame is ready.
        """
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
        """Decode loop — runs until stop() is called.

        Decodes frames sequentially from the current target position,
        placing them in _frame_queue. Blocks when the queue is full
        (the display is keeping up or faster than decode, which is fine).
        Idles when _playing is False.
        """
        decode_pos: int = 0   # frame index the codec is currently positioned at

        while not self._stop_event.is_set():

            # ── Idle when paused ──────────────────────────────────────────────
            with self._lock:
                playing = self._playing
            if not playing:
                self._wake_event.clear()
                self._wake_event.wait(timeout=0.1)
                continue

            # ── Handle flush (seek) ───────────────────────────────────────────
            if self._flush_event.is_set():
                self._flush_event.clear()
                with self._lock:
                    decode_pos = self._target_frame
                self._media.reset_sequential_position()

            # ── Decode one frame ahead of the audio clock ─────────────────────
            # Stay at most PREFETCH frames ahead so we do not waste memory or
            # decode frames the user will never see (e.g. during a seek in
            # progress). PREFETCH=4 matches the queue maxsize.
            PREFETCH = 4
            target = self._target_frame

            if decode_pos > target + PREFETCH:
                # We are far ahead — sleep briefly and let the clock catch up
                self._stop_event.wait(timeout=0.005)
                continue

            meta = self._media.meta
            if meta is None:
                self._stop_event.wait(timeout=0.05)
                continue

            if decode_pos >= meta.frame_count:
                # End of file — idle until a seek or stop
                self._stop_event.wait(timeout=0.05)
                continue

            bgr_frame = self._media.read_frame(decode_pos)
            if bgr_frame is not None:
                try:
                    # Blocking put: if the queue is full, wait until the display
                    # thread consumes a frame. This naturally throttles the decoder
                    # to the display rate without busy-waiting.
                    self._frame_queue.put((decode_pos, bgr_frame), timeout=0.05)
                    decode_pos += 1
                except queue.Full:
                    pass   # timeout — loop and retry
            else:
                # read_frame returned None — skip this index (corrupted frame)
                decode_pos += 1


# =============================================================================
# STEP 5 — VideoDisplay: QWidget that shows a single video frame
# =============================================================================

class VideoDisplay(QWidget):
    """QWidget that displays one video frame at a time.

    Receives BGR frames from OpenCV, converts to RGB, wraps in a QImage,
    and scales to the widget size maintaining aspect ratio.

    CONCEPT — BGR → RGB conversion:
    OpenCV uses BGR channel order everywhere — a legacy from its early days
    interfacing with Windows BITMAPINFOHEADER structures, which stored bytes
    as B, G, R. Qt's QImage.Format_RGB888 expects R, G, B order. Mixing the
    two produces frames with swapped red and blue channels: skin tones look
    cyan, sky looks orange. This is the single most common bug when
    integrating OpenCV with any other library.

    Always convert: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    There are no exceptions.

    CONCEPT — bytes_per_line in QImage:
    QImage(data, w, h, fmt) uses Qt's default row stride: rows aligned to
    4-byte boundaries. For a 3-channel RGB frame where w is not divisible
    by 4/3, Qt assumes a wider row than the actual array, reads into the
    wrong memory, and produces green-shifted or visually corrupted frames.

    Always provide bytes_per_line = w * 3 explicitly:
        QImage(data.data, w, h, w * 3, QImage.Format.Format_RGB888)

    This is non-optional. The bug is subtle and frame-size dependent — it
    will not appear on all videos and is very hard to diagnose.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with a blank dark frame."""
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None

    def show_frame(self, bgr_frame: np.ndarray) -> None:
        """Convert a BGR frame to QPixmap and schedule a repaint.

        Args:
            bgr_frame: OpenCV frame, shape (H, W, 3), dtype uint8, BGR channel order.
        """
        h, w = bgr_frame.shape[:2]

        # CRITICAL: Convert BGR (OpenCV) to RGB (Qt) before building QImage.
        # Skipping this step swaps the red and blue channels silently.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Ensure the array is contiguous in memory (required by QImage).
        # cv2 operations do not always return contiguous arrays.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        # CRITICAL: Provide bytes_per_line = w * 3.
        # Omitting this causes Qt to use default 4-byte-aligned row strides,
        # producing corrupted frames for widths not divisible by 4/3.
        bytes_per_line = w * 3
        qimg = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Scale to the widget's current size, maintaining aspect ratio.
        # Qt.SmoothTransformation uses bilinear interpolation — better quality
        # than Qt.FastTransformation (nearest-neighbour) at the cost of a
        # small amount of CPU per frame.
        self._pixmap = QPixmap.fromImage(qimg).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw the current pixmap centred on the widget."""
        painter = QPainter(self)
        w, h = self.width(), self.height()

        # Dark background
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        if self._pixmap is None:
            # No frame loaded — draw a placeholder
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 11))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open a video file to start playback",
            )
        else:
            # Centre the scaled pixmap on the widget
            px = (w - self._pixmap.width()) // 2
            py = (h - self._pixmap.height()) // 2
            painter.drawPixmap(px, py, self._pixmap)

        painter.end()


# =============================================================================
# STEP 6 — DspControlPanel: DSP mode selector with conditional Canny controls
# =============================================================================

class DspControlPanel(QWidget):
    """Right-side panel: DSP mode selector and mode-specific parameters.

    The Canny threshold controls are shown only when Canny Edges mode is
    selected and hidden otherwise. This is the professional approach: expose
    only parameters relevant to the current mode. Showing irrelevant controls
    confuses users and implies those controls have an effect when they do not.

    Signals:
        mode_changed: Emitted when the DSP mode QComboBox changes.
                      Carries the new mode string.
    """

    mode_changed: Signal = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the DSP control panel."""
        super().__init__(parent)
        self.setFixedWidth(270)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all controls."""
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

        # ── DSP mode group ────────────────────────────────────────────────────
        dsp_group = QGroupBox("Frame DSP Pipeline")
        dsp_group.setStyleSheet(
            "QGroupBox { color: #a78bfa; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        dsp_form = QFormLayout(dsp_group)
        dsp_form.setSpacing(10)
        dsp_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # DSP mode selector
        self._mode_combo = QComboBox()
        for mode in DSP_MODES:
            self._mode_combo.addItem(mode)
        self._mode_combo.setStyleSheet(
            "QComboBox { background: #161a22; color: #d8dce6; border: 1px solid #252b38;"
            " border-radius: 4px; padding: 4px 8px; font-family: 'IBM Plex Mono';"
            " font-size: 12px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #161a22; color: #d8dce6;"
            " selection-background-color: #252b38; }"
        )
        dsp_form.addRow("Mode", self._mode_combo)

        # ── Canny parameters (shown only in Canny Edges mode) ─────────────────
        # These widgets are created here and added to the form. Their visibility
        # is toggled in _on_mode_changed(). The parent QGroupBox resizes
        # automatically when hidden children no longer contribute to layout height.

        self._canny_label_t1 = QLabel("Threshold 1")
        self._canny_label_t1.setStyleSheet("color: #d8dce6; font-size: 12px;")
        self._canny_t1 = QSlider(Qt.Orientation.Horizontal)
        self._canny_t1.setRange(0, 500)
        self._canny_t1.setValue(CANNY_THRESHOLD1_DEFAULT)
        self._canny_t1.setToolTip(
            "Lower hysteresis threshold.\n"
            "Edges between T1 and T2 are kept only if connected to a definite edge."
        )

        self._canny_val_t1 = QLabel(str(CANNY_THRESHOLD1_DEFAULT))
        self._canny_val_t1.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        self._canny_val_t1.setFixedWidth(36)

        t1_row = QWidget()
        t1_layout = QHBoxLayout(t1_row)
        t1_layout.setContentsMargins(0, 0, 0, 0)
        t1_layout.setSpacing(6)
        t1_layout.addWidget(self._canny_t1)
        t1_layout.addWidget(self._canny_val_t1)

        self._canny_label_t2 = QLabel("Threshold 2")
        self._canny_label_t2.setStyleSheet("color: #d8dce6; font-size: 12px;")
        self._canny_t2 = QSlider(Qt.Orientation.Horizontal)
        self._canny_t2.setRange(0, 500)
        self._canny_t2.setValue(CANNY_THRESHOLD2_DEFAULT)
        self._canny_t2.setToolTip(
            "Upper hysteresis threshold.\n"
            "Pixels with gradient > T2 are definite edges. Recommended: T2 = 3 × T1."
        )

        self._canny_val_t2 = QLabel(str(CANNY_THRESHOLD2_DEFAULT))
        self._canny_val_t2.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )
        self._canny_val_t2.setFixedWidth(36)

        t2_row = QWidget()
        t2_layout = QHBoxLayout(t2_row)
        t2_layout.setContentsMargins(0, 0, 0, 0)
        t2_layout.setSpacing(6)
        t2_layout.addWidget(self._canny_t2)
        t2_layout.addWidget(self._canny_val_t2)

        dsp_form.addRow(self._canny_label_t1, t1_row)
        dsp_form.addRow(self._canny_label_t2, t2_row)

        # Collect all Canny-specific widgets for bulk show/hide
        self._canny_widgets: list[QWidget] = [
            self._canny_label_t1, t1_row,
            self._canny_label_t2, t2_row,
        ]

        layout.addWidget(dsp_group)

        # ── Video info group (populated when a file is opened) ─────────────────
        info_group = QGroupBox("Video Info")
        info_group.setStyleSheet(
            "QGroupBox { color: #5a6175; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        info_form = QFormLayout(info_group)
        info_form.setSpacing(6)
        info_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        label_style = "color: #d8dce6; font-family: 'IBM Plex Mono'; font-size: 11px;"

        self._info_file = QLabel("—")
        self._info_file.setStyleSheet(label_style)
        self._info_file.setWordWrap(True)
        info_form.addRow("File", self._info_file)

        self._info_res = QLabel("—")
        self._info_res.setStyleSheet(label_style)
        info_form.addRow("Resolution", self._info_res)

        self._info_fps = QLabel("—")
        self._info_fps.setStyleSheet(label_style)
        info_form.addRow("Frame rate", self._info_fps)

        self._info_dur = QLabel("—")
        self._info_dur.setStyleSheet(label_style)
        info_form.addRow("Duration", self._info_dur)

        self._info_sr = QLabel("—")
        self._info_sr.setStyleSheet(label_style)
        info_form.addRow("Audio rate", self._info_sr)

        layout.addWidget(info_group)
        layout.addStretch()

        # ── Internal connections ──────────────────────────────────────────────
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._canny_t1.valueChanged.connect(
            lambda v: self._canny_val_t1.setText(str(v))
        )
        self._canny_t2.valueChanged.connect(
            lambda v: self._canny_val_t2.setText(str(v))
        )

        # Initialise: hide Canny widgets (mode starts at Passthrough)
        self._set_canny_visible(False)

    def _set_canny_visible(self, visible: bool) -> None:
        """Show or hide all Canny-specific controls."""
        for w in self._canny_widgets:
            w.setVisible(visible)

    def _on_mode_changed(self, mode: str) -> None:
        """Toggle Canny widget visibility and emit mode_changed."""
        self._set_canny_visible(mode == DSP_CANNY)
        self.mode_changed.emit(mode)

    # ── Public interface ───────────────────────────────────────────────────────

    def set_video_info(self, meta: VideoMeta, filename: str) -> None:
        """Populate the video info section.

        Args:
            meta: VideoMeta loaded from the file.
            filename: Display name (stem only, not full path).
        """
        self._info_file.setText(filename)
        self._info_res.setText(f"{meta.width} × {meta.height}")
        self._info_fps.setText(f"{meta.fps:.3f} fps")

        dur = meta.duration_seconds
        mins = int(dur // 60)
        secs = dur - mins * 60
        self._info_dur.setText(f"{mins:02d}:{secs:05.2f}")

        self._info_sr.setText(f"{DEVICE_SAMPLE_RATE:,} Hz")

    @property
    def current_mode(self) -> str:
        """Currently selected DSP mode string."""
        return self._mode_combo.currentText()

    @property
    def canny_threshold1(self) -> int:
        """Current Canny lower threshold."""
        return self._canny_t1.value()

    @property
    def canny_threshold2(self) -> int:
        """Current Canny upper threshold."""
        return self._canny_t2.value()


# =============================================================================
# STEP 7 — TransportBar: play/pause/stop, time display, scrubber
# =============================================================================

class TransportBar(QWidget):
    """Horizontal bar with playback controls, time display, and scrubber.

    Signals:
        play_requested: Play/Pause toggle clicked.
        stop_requested: Stop button clicked.
        seek_requested: Scrubber dragged. Carries target video frame index.
    """

    play_requested: Signal = Signal()
    stop_requested: Signal = Signal()
    seek_requested: Signal = Signal(int)   # carries target VIDEO frame index

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the transport bar."""
        super().__init__(parent)
        self.setFixedHeight(64)
        self._frame_count: int = 0
        self._fps: float = 25.0
        self._build_ui()

    def _build_ui(self) -> None:
        """Create transport controls."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(4)

        # ── Scrubber ──────────────────────────────────────────────────────────
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 10_000)
        self._scrubber.setValue(0)
        self._scrubber.setEnabled(False)
        self._scrubber.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #252b38;"
            " border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px;"
            " margin: -4px 0; background: #4f9cf9; border-radius: 6px; }"
            "QSlider::sub-page:horizontal { background: #4f9cf9; border-radius: 2px; }"
        )
        layout.addWidget(self._scrubber)

        # ── Buttons and time ──────────────────────────────────────────────────
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)

        btn_style_play = (
            "QPushButton {"
            "  background: rgba(79,156,249,0.12); color: #4f9cf9;"
            "  border: 1px solid rgba(79,156,249,0.4); border-radius: 6px;"
            "  padding: 5px 12px; font-size: 12px; font-family: 'IBM Plex Mono';"
            "  min-width: 80px;"
            "}"
            "QPushButton:checked {"
            "  background: rgba(251,146,60,0.14); color: #fb923c;"
            "  border-color: rgba(251,146,60,0.5);"
            "}"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )
        btn_style_stop = (
            "QPushButton {"
            "  background: rgba(79,156,249,0.06); color: #5a6175;"
            "  border: 1px solid #252b38; border-radius: 6px;"
            "  padding: 5px 12px; font-size: 12px; font-family: 'IBM Plex Mono';"
            "  min-width: 70px;"
            "}"
            "QPushButton:hover { color: #d8dce6; border-color: #4f9cf9; }"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setEnabled(False)
        self._play_btn.setStyleSheet(btn_style_play)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(btn_style_stop)

        self._time_label = QLabel("00:00.000 / 00:00.000")
        self._time_label.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 12px;"
        )

        self._frame_label = QLabel("frame — / —")
        self._frame_label.setStyleSheet(
            "color: #5a6175; font-family: 'IBM Plex Mono'; font-size: 11px;"
        )

        btn_layout.addWidget(self._play_btn)
        btn_layout.addWidget(self._stop_btn)
        btn_layout.addSpacing(8)
        btn_layout.addWidget(self._time_label)
        btn_layout.addStretch()
        btn_layout.addWidget(self._frame_label)

        layout.addWidget(btn_row)

        # ── Internal connections ──────────────────────────────────────────────
        self._play_btn.toggled.connect(self._on_play_toggled)
        self._stop_btn.clicked.connect(self.stop_requested)
        self._scrubber.sliderMoved.connect(self._on_scrubber_moved)

    # ── Internal slots ─────────────────────────────────────────────────────────

    def _on_play_toggled(self, checked: bool) -> None:
        self._play_btn.setText("⏸  Pause" if checked else "▶  Play")
        self.play_requested.emit()

    def _on_scrubber_moved(self, value: int) -> None:
        """Convert scrubber position (0–10000) to a video frame index."""
        if self._frame_count > 0:
            frame = int(value / 10_000 * self._frame_count)
            self.seek_requested.emit(frame)

    # ── Public interface ───────────────────────────────────────────────────────

    def set_file_loaded(self, frame_count: int, fps: float) -> None:
        """Enable controls after a file is successfully loaded."""
        self._frame_count = frame_count
        self._fps = max(fps, 1e-6)
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._scrubber.setEnabled(True)
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")

    def update_position(self, video_frame: int) -> None:
        """Update the time display, frame counter, and scrubber.

        Args:
            video_frame: Current video frame index.
        """
        if self._frame_count == 0:
            return

        def fmt_time(seconds: float) -> str:
            m = int(seconds // 60)
            s = seconds - m * 60
            return f"{m:02d}:{s:06.3f}"

        current_s = video_frame / self._fps
        total_s = self._frame_count / self._fps
        self._time_label.setText(f"{fmt_time(current_s)} / {fmt_time(total_s)}")
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
        """Reset display to initial state."""
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
        """True when the Play button is in the pressed state."""
        return self._play_btn.isChecked()


# =============================================================================
# STEP 8 — MainWindow: file open, timer loop, seek, DSP, status bar
# =============================================================================

class MainWindow(QMainWindow):
    """Application main window.

    Layout:
    ┌────────────────────────────────────────────────────┬──────────────────┐
    │  VideoDisplay (frame canvas, fills available space)│  DspControlPanel │
    ├────────────────────────────────────────────────────┤                  │
    │  TransportBar (scrubber + controls, fixed height)  │                  │
    └────────────────────────────────────────────────────┴──────────────────┘
    Status bar: time | frame | fps | resolution | audio rate | DSP mode

    A/V sync loop:
      1. AudioPlaybackEngine._callback() runs at AUDIO_BLOCK_SIZE / 48000 Hz
         intervals on the PortAudio audio thread.
      2. It pushes time_info.outputBufferDacTime to _audio_engine.dac_queue.
      3. A QTimer at 60 Hz fires _poll_av_sync() on the Qt main thread.
      4. _poll_av_sync() drains dac_queue, computes target video frame, decodes
         frame, applies DSP, displays via VideoDisplay.show_frame().

    All Qt operations happen on the main thread.
    All audio operations happen on the PortAudio audio thread.
    No cross-thread Qt calls, no audio calls on the main thread.
    """

    def __init__(self) -> None:
        """Initialise all components."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._media = MediaFile()
        self._audio_engine = AudioPlaybackEngine()

        # Background video decode thread — pre-decodes frames off the main thread
        # so codec latency never blocks the Qt event loop.
        self._decode_thread = VideoDecodeThread(self._media)
        self._decode_thread.start()

        # DAC time anchor: the DAC time at which the current playback segment started.
        # Reset on seek and on play-from-stop.
        self._dac_time_anchor: float | None = None

        # Audio frame position at the start of the current playback segment.
        # Combined with _dac_time_anchor, we can compute:
        #   elapsed = dac_time - _dac_time_anchor
        #   audio_frame = _audio_frame_anchor + elapsed * DEVICE_SAMPLE_RATE
        #   video_frame = int(audio_frame / DEVICE_SAMPLE_RATE * fps)
        self._audio_frame_anchor: int = 0

        # Previous frame for motion diff (held between display updates)
        self._prev_bgr_frame: np.ndarray | None = None

        # QTimer drives the A/V sync loop at 60 Hz from the main thread.
        self._av_timer = QTimer(self)
        self._av_timer.setInterval(DISPLAY_TIMER_MS)

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._av_timer.start()

    # ── UI assembly ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create and arrange all panels."""
        self._video_display = VideoDisplay()
        self._transport = TransportBar()
        self._dsp_panel = DspControlPanel()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._video_display, stretch=1)
        left_layout.addWidget(self._transport)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._dsp_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([780, 270])

        self.setCentralWidget(splitter)

        # Status bar
        self._status_bar = self.statusBar()
        self._status_bar.setStyleSheet(
            "QStatusBar { background: #161a22; color: #5a6175;"
            " font-family: 'IBM Plex Mono'; font-size: 11px; border-top: 1px solid #252b38; }"
            "QStatusBar::item { border: none; }"
        )
        self._status_bar.showMessage("Ready — open a video file to begin")

    def _build_menu(self) -> None:
        """Create the File menu."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = file_menu.addAction("Open Video File…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

    # ── Signal connections ─────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect all signals between UI components, engine, and timer."""
        # Transport ↔ engine
        self._transport.play_requested.connect(self._on_play_requested)
        self._transport.stop_requested.connect(self._on_stop)
        self._transport.seek_requested.connect(self._on_seek)

        # DSP panel → mode tracking (no separate slot needed; _poll_av_sync reads live)
        # The mode_changed signal is connected here for status bar update only.
        self._dsp_panel.mode_changed.connect(self._on_dsp_mode_changed)

        # A/V sync timer → poll loop
        self._av_timer.timeout.connect(self._poll_av_sync)

    # ── File open ──────────────────────────────────────────────────────────────

    def _open_file(self) -> None:
        """Open a video file: stop current, open video, extract audio, update UI."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            SUPPORTED_VIDEO_FILTER,
        )
        if not path_str:
            return   # user cancelled

        video_path = Path(path_str)

        # Stop current playback and release resources before loading a new file
        self._audio_engine.stop()
        self._decode_thread.set_playing(False)
        self._decode_thread.flush(0)
        self._media.close()
        self._dac_time_anchor = None
        self._audio_frame_anchor = 0
        self._prev_bgr_frame = None
        self._transport.reset()

        self._status_bar.showMessage(f"Opening {video_path.name}…")

        # ── Step 1: Open video stream ──────────────────────────────────────────
        try:
            meta = self._media.open_video(video_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot open video", str(exc))
            self._status_bar.showMessage("Error — could not open video file")
            return

        # ── Step 2: Extract audio track ────────────────────────────────────────
        audio_data: np.ndarray | None = None
        try:
            wav_path = self._media.extract_audio(video_path)
            audio_data = self._media.load_audio(wav_path)
        except (RuntimeError, ValueError) as exc:
            # Non-fatal: continue with silent playback and warn the user.
            # Some video files have no audio track; this is a valid case.
            QMessageBox.warning(
                self,
                "Audio extraction failed",
                f"{exc}\n\nPlayback will continue without audio.",
            )
            # Create a silent audio buffer matching the video duration
            n_silent = int(meta.duration_seconds * DEVICE_SAMPLE_RATE)
            audio_data = np.zeros((n_silent, 2), dtype=np.float32)

        # ── Step 3: Load audio into engine ─────────────────────────────────────
        self._audio_engine.load(audio_data)

        # ── Step 4: Update UI ──────────────────────────────────────────────────
        self.setWindowTitle(f"{video_path.name} — {WINDOW_TITLE}")
        self._transport.set_file_loaded(meta.frame_count, meta.fps)
        self._dsp_panel.set_video_info(meta, video_path.name)

        # Display first frame immediately so the user sees the video loaded
        first_frame = self._media.read_frame(0)
        if first_frame is not None:
            processed = self._apply_dsp(first_frame)
            self._video_display.show_frame(processed)

        self._transport.update_position(0)
        self._update_status_bar(video_frame=0)
        self._status_bar.showMessage(
            f"Loaded: {video_path.name}  —  "
            f"{meta.width}×{meta.height} @ {meta.fps:.2f} fps  —  "
            f"Audio: {DEVICE_SAMPLE_RATE:,} Hz stereo"
        )

    # ── Playback control slots ─────────────────────────────────────────────────

    def _on_play_requested(self) -> None:
        """Toggle between play and pause."""
        if self._transport.is_playing:
            # Transitioning to playing — anchor the DAC time for sync.
            # We will set _dac_time_anchor on the first DAC timestamp we receive.
            self._dac_time_anchor = None
            self._audio_frame_anchor = self._audio_engine.frame_pos

            # Tell decode thread where we are and start it pre-buffering
            meta = self._media.meta
            if meta is not None:
                start_frame = int(
                    self._audio_frame_anchor / DEVICE_SAMPLE_RATE * meta.fps
                )
                self._decode_thread.flush(start_frame)
                self._decode_thread.set_playing(True, start_frame)

            self._audio_engine.play()
        else:
            self._audio_engine.pause()
            self._decode_thread.set_playing(False)

    def _on_stop(self) -> None:
        """Stop playback and reset to the beginning."""
        self._audio_engine.stop()
        self._decode_thread.set_playing(False)
        self._decode_thread.flush(0)
        self._dac_time_anchor = None
        self._audio_frame_anchor = 0
        self._prev_bgr_frame = None
        self._transport.set_stopped()
        self._transport.update_position(0)

        # Display frame 0
        if self._media.meta is not None:
            frame = self._media.read_frame(0)
            if frame is not None:
                self._video_display.show_frame(self._apply_dsp(frame))

    def _on_seek(self, video_frame: int) -> None:
        """Seek both audio and video to a target frame.

        SYNC CONCEPT — Atomic seek:
        A non-atomic seek (audio first, video second, or vice versa) creates
        a window where one stream has moved and the other has not. If the A/V
        sync loop fires in this window, it reads an inconsistent state and
        displays the wrong frame. The pattern here avoids this:

          1. Pause audio — the callback no longer advances _frame_pos.
          2. Acquire the audio lock (inside seek()) — now _frame_pos is frozen.
          3. Seek audio: _frame_pos = target_audio_frame.
          4. Seek video: cap.set(CAP_PROP_POS_FRAMES, video_frame).
          5. Decode and display the target frame immediately.
          6. Reset DAC time anchor so the A/V loop re-anchors after resume.
          7. Release lock, resume audio.

        The user sees the seeked frame instantly while the button is held.
        When they release and click Play, audio and video resume from the
        same position.

        Args:
            video_frame: Target video frame index (0-based).
        """
        meta = self._media.meta
        if meta is None:
            return

        was_playing = self._audio_engine.is_playing

        # Step 1: Pause audio so the callback is not running during seek
        if was_playing:
            self._audio_engine.pause()

        # Step 2–3: Seek audio (lock acquired inside seek())
        target_audio_frame = int(video_frame / meta.fps * DEVICE_SAMPLE_RATE)
        self._audio_engine.seek(target_audio_frame)

        # Step 4: Seek video
        # (The lock is released after seek() returns, so we are safe here)
        bgr_frame = self._media.read_frame(video_frame)

        # Step 5: Display the seeked frame immediately
        if bgr_frame is not None:
            self._prev_bgr_frame = None   # motion diff resets on seek
            processed = self._apply_dsp(bgr_frame)
            self._video_display.show_frame(processed)
            self._prev_bgr_frame = bgr_frame.copy()

        # Step 6: Reset DAC time anchor — the sync loop will re-anchor on next tick.
        # Flush the decode thread so stale pre-buffered frames are discarded and
        # it restarts from the new position.
        self._dac_time_anchor = None
        self._audio_frame_anchor = target_audio_frame
        self._decode_thread.flush(video_frame)

        # Update transport display
        self._transport.update_position(video_frame)
        self._update_status_bar(video_frame)

        # Step 7: Resume if was playing
        if was_playing:
            self._decode_thread.set_playing(True, video_frame)
            self._audio_engine.play()

    def _on_dsp_mode_changed(self, mode: str) -> None:
        """Update status bar when DSP mode changes."""
        self._update_status_bar()

    # ── A/V sync loop ──────────────────────────────────────────────────────────

    def _poll_av_sync(self) -> None:
        """60 Hz A/V sync loop: audio-clock-driven frame display.

        Called on the Qt main thread by QTimer at DISPLAY_TIMER_MS intervals.

        FIX vs original implementation:
        The original called MediaFile.read_frame() (which issues a
        CAP_PROP_POS_FRAMES seek and decodes inline) directly here on the
        main thread. That blocked the event loop for 5–30 ms per tick on
        H.264 content, causing the stutter. The fix splits the work:

          VideoDecodeThread (background, always running):
            Decodes frames sequentially from the video file into _frame_queue,
            staying PREFETCH frames ahead of the audio clock at all times.

          _poll_av_sync (main thread, 60 Hz):
            1. Drain the DAC timestamp queue from the audio callback.
            2. Compute target_video_frame from the audio clock.
            3. Inform the decode thread of the new target (set_target()).
            4. Pull the closest pre-decoded frame from _frame_queue.
               This is now O(1) — no I/O, no codec call.
            5. Apply DSP (still on main thread; fast for most modes).
            6. Display via VideoDisplay.show_frame().
            7. Check for end-of-file.

        FRAME SELECTION from queue:
        The queue may contain frames from slightly before or after the
        audio-clock target. We drain it until we find the frame whose
        index is closest to target_video_frame, then display that one.
        Frames strictly older than target are discarded (skip ahead).
        Frames strictly newer are kept for the next tick (hold back).
        This achieves audio-driven presentation without ever seeking.
        """
        # ── Step 1: Drain DAC queue ────────────────────────────────────────────
        latest_dac: float | None = None
        try:
            while True:
                latest_dac = self._audio_engine.dac_queue.get_nowait()
        except queue.Empty:
            pass

        meta = self._media.meta
        if meta is None:
            return   # no file loaded

        # ── Steps 2–6: Compute target, consume pre-decoded frame, display ───────
        if latest_dac is not None and self._audio_engine.is_playing:
            # Anchor the DAC time on the first tick after play/seek.
            if self._dac_time_anchor is None:
                self._dac_time_anchor = latest_dac

            # Compute elapsed audio time since anchor
            elapsed_seconds = latest_dac - self._dac_time_anchor

            # Absolute audio frame position
            current_audio_frame = self._audio_frame_anchor + int(
                elapsed_seconds * DEVICE_SAMPLE_RATE
            )

            # Target video frame from the audio clock
            target_video_frame = int(current_audio_frame / DEVICE_SAMPLE_RATE * meta.fps)
            target_video_frame = max(0, min(target_video_frame, meta.frame_count - 1))

            # Inform decode thread so it pre-buffers the right section
            self._decode_thread.set_target(target_video_frame)

            # Pull pre-decoded frames from queue, discarding any that are older
            # than the audio clock target (we have already passed those frames).
            # Stop as soon as we have the best available frame.
            best_frame: np.ndarray | None = None
            best_idx: int = -1

            while True:
                item = self._decode_thread.get_frame()
                if item is None:
                    break   # queue empty — use whatever we have
                frame_idx, bgr_frame = item
                if frame_idx <= target_video_frame:
                    # This frame is at or before our target; keep it as best so far
                    best_frame = bgr_frame
                    best_idx = frame_idx
                else:
                    # This frame is ahead of our target — put it back and stop
                    # Re-queue by inserting directly; if queue full, discard it
                    # (the decode thread will re-produce it shortly).
                    try:
                        self._decode_thread._frame_queue.put_nowait((frame_idx, bgr_frame))
                    except queue.Full:
                        pass
                    break

            if best_frame is not None:
                processed = self._apply_dsp(best_frame)
                self._video_display.show_frame(processed)
                self._prev_bgr_frame = best_frame.copy()
                self._transport.update_position(best_idx)
                self._update_status_bar(best_idx)
            else:
                # Queue was empty this tick (codec is catching up) — update
                # transport with the audio-clock position so the UI stays live.
                self._transport.update_position(target_video_frame)
                self._update_status_bar(target_video_frame)

        # ── Step 7: Check for end-of-file ──────────────────────────────────────
        try:
            self._audio_engine.finished_queue.get_nowait()
            self._decode_thread.set_playing(False)
            self._transport.set_stopped()
            self._dac_time_anchor = None
            self._audio_frame_anchor = 0
            self._status_bar.showMessage("Playback finished")
        except queue.Empty:
            pass

    # ── DSP dispatch ───────────────────────────────────────────────────────────

    def _apply_dsp(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Apply the currently selected DSP operation to a BGR frame.

        Args:
            bgr_frame: Input frame from OpenCV, shape (H, W, 3), uint8.

        Returns:
            Processed BGR frame, same shape and dtype.
        """
        mode = self._dsp_panel.current_mode

        if mode == DSP_PASSTHROUGH:
            return process_passthrough(bgr_frame)

        elif mode == DSP_GRAYSCALE:
            return process_grayscale(bgr_frame)

        elif mode == DSP_CANNY:
            return process_canny(
                bgr_frame,
                threshold1=self._dsp_panel.canny_threshold1,
                threshold2=self._dsp_panel.canny_threshold2,
            )

        elif mode == DSP_BLUR:
            return process_blur(bgr_frame, ksize=BLUR_KSIZE_DEFAULT)

        elif mode == DSP_EQUALIZE:
            return process_equalize(bgr_frame)

        elif mode == DSP_MOTION:
            result = process_motion(self._prev_bgr_frame, bgr_frame)
            return result

        # Fallback (should never be reached if DSP_MODES is kept in sync)
        return bgr_frame

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _update_status_bar(self, video_frame: int | None = None) -> None:
        """Update the status bar with current playback state.

        Args:
            video_frame: Current video frame index, or None to omit frame info.
        """
        meta = self._media.meta
        if meta is None:
            return

        parts: list[str] = []

        if video_frame is not None:
            current_s = video_frame / meta.fps
            m = int(current_s // 60)
            s = current_s - m * 60
            parts.append(f"{m:02d}:{s:06.3f}")
            parts.append(f"frame {video_frame:,} / {meta.frame_count:,}")

        parts.append(f"{meta.fps:.3f} fps")
        parts.append(f"{meta.width}×{meta.height}")
        parts.append(f"audio {DEVICE_SAMPLE_RATE:,} Hz")
        parts.append(f"DSP: {self._dsp_panel.current_mode}")

        self._status_bar.showMessage("   │   ".join(parts))

    # ── Window lifecycle ───────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Release all resources on window close.

        RESOURCE MANAGEMENT:
        Failing to release resources here leaves the system in a bad state:
          - An unreleased sounddevice stream holds the audio device. On Linux,
            this locks the ALSA/PulseAudio device and prevents other applications
            (including the next run of this demo) from opening it.
          - An unreleased VideoCapture holds the file descriptor. On some systems
            this prevents the file from being deleted.
          - An uncleaned TemporaryDirectory leaves the extracted WAV on disk.
            In a long session with many files opened, this wastes disk space.

        Always release in closeEvent. Use try/except around each release in
        case one failure does not prevent the others from running.
        """
        self._av_timer.stop()

        try:
            self._audio_engine.close()
        except Exception as exc:
            print(f"[MainWindow] Audio engine close error: {exc}", flush=True)

        try:
            self._media.close()
        except Exception as exc:
            print(f"[MainWindow] MediaFile close error: {exc}", flush=True)

        super().closeEvent(event)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Create the QApplication, apply the dark palette, and start the event loop."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark Fusion palette — consistent with all other course demos
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
# End of 03_mm_video_player.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
