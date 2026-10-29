# =============================================================================
# 03_mm_waveform_player.py — Waveform Viewer and DSP Player
# =============================================================================
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# -----------------------------------------------------------------------------
# Description:
#   A professional audio file player with a full-featured waveform display
#   and an offline DSP effects chain.
#
#   Open any audio file via File > Open. Two decoders are chained:
#
#     1. soundfile (libsndfile) — primary decoder.
#        Handles: WAV, FLAC, OGG/Vorbis, AIFF, CAF, W64, and more.
#        libsndfile is the professional standard for uncompressed and losslessly
#        compressed audio. It does NOT support MP3, AAC, M4A, OPUS, or WMA —
#        formats that carry patent/licence history and require dedicated codecs.
#
#     2. FFmpeg — universal fallback decoder.
#        Handles: MP3, AAC/M4A, OPUS, WMA, AC3, FLAC (also), and virtually
#        every container/codec combination in professional use.
#        FFmpeg is invoked as a subprocess that decodes the file and pipes raw
#        32-bit float PCM to stdout. This avoids temp files and gives us the
#        same NumPy array shape we would get from soundfile, so the rest of the
#        pipeline is format-agnostic.
#        If FFmpeg is not installed, the app still runs — only libsndfile
#        formats are available, and a clear error explains what to install.
#
#   The entire file is decoded into a NumPy buffer. The waveform is rendered
#   in QPainter using peak-reduction: for each pixel column the min and max
#   amplitude of all covered samples are drawn as a vertical line. This
#   preserves visual accuracy at any zoom level without overdrawing.
#
#   A DSP effects chain (gain, fade in/out, DC removal, normalisation) can be
#   applied offline. The original (blue) and processed (purple) waveforms are
#   overlaid on the same canvas. A scrubber supports seeking during playback.
#
#   Playback uses a sounddevice OutputStream callback with a threading.Lock-
#   protected frame pointer. This is the reusable audio transport pattern that
#   Demo 5 (the full workbench) integrates directly.
#
#   WEBGL EQUIVALENT:
#     soundfile.read()          ↔  AudioContext.decodeAudioData()
#     FFmpeg subprocess pipe    ↔  MediaDecoder API / WebCodecs API
#     PlaybackEngine._callback  ↔  ScriptProcessorNode / AudioWorkletProcessor
#     DspChain functions        ↔  offline AudioBuffer processing in Web Audio
#     WaveformView QPainter     ↔  AnalyserNode + Canvas 2D drawImage()
#
# How to run:
#   uv sync                                       # first time only
#   uv run python 03_mm_waveform_player.py
#
# Ubuntu system dependencies (install once):
#   sudo apt install libgl1 libegl1 libxcb-cursor0 libxcb-xinerama0 \
#                    libportaudio2 libsndfile1 ffmpeg
#
# Windows system dependencies:
#   FFmpeg: https://www.gyan.dev/ffmpeg/builds/  (add ffmpeg/bin to PATH)
#   All Python packages are bundled — no other system deps needed.
#
# Connections to other seminars:
#   Demo 1 (hello_multimedia) — introduces the sounddevice callback pattern
#     that PlaybackEngine uses here. Read Demo 1 first.
#   Demo 5 (multimedia_workbench) — imports WaveformView and PlaybackEngine
#     from this file directly as reusable components.
# =============================================================================

from __future__ import annotations

import queue
import shutil
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QTimer, Signal, QRect
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QLinearGradient,
    QBrush,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, resample_poly, sosfilt

# =============================================================================
# STEP 1 — Constants
# =============================================================================

WINDOW_TITLE: str = "Waveform Player — Multimediaprogrammierung Seminar 03"
WINDOW_MIN_WIDTH: int = 960
WINDOW_MIN_HEIGHT: int = 560

# Audio I/O
DEVICE_SAMPLE_RATE: int = (
    48_000  # Hz — target playback rate; files are resampled to this
)
BLOCK_SIZE: int = 512  # frames per callback block

# ---------------------------------------------------------------------------
# Decoder availability
# ---------------------------------------------------------------------------
# shutil.which() searches PATH for the ffmpeg executable — the same mechanism
# the shell uses. Returns the full path string if found, None if not found.
# Works identically on Ubuntu (looks for "ffmpeg") and Windows (looks for
# "ffmpeg.exe", but shutil.which handles the .exe extension automatically).
#
# We resolve this once at import time so the rest of the code can branch on
# FFMPEG_BIN without repeated PATH lookups. None means FFmpeg is absent and
# only libsndfile formats will be available.
FFMPEG_BIN: str | None = shutil.which("ffmpeg")

# ---------------------------------------------------------------------------
# Supported audio file formats
# ---------------------------------------------------------------------------
# The file-dialog filter presents two tiers:
#
#   "Common Audio Files" — a curated list of extensions every audio professional
#     recognises. Shown first so the dialog defaults to it.
#
#   "All Files (*)" — lets the user select any file. If the extension is
#     unknown, AudioBuffer.load() will still attempt both decoders. This is
#     the correct professional behaviour: FFmpeg handles hundreds of containers
#     and the user may legitimately have an unusual format.
#
# Note: the libsndfile-only filter is intentionally NOT separated from the
# FFmpeg filter. Splitting them would confuse students and is unnecessary —
# the decoders are chained transparently inside AudioBuffer.load().
#
# libsndfile formats (always available):
_SF_EXTS = "*.wav *.flac *.ogg *.aiff *.aif *.caf *.w64"
# FFmpeg-only formats (available when FFMPEG_BIN is set):
_FF_EXTS = "*.mp3 *.m4a *.aac *.opus *.wma *.ac3 *.mp4 *.mkv *.webm"

if FFMPEG_BIN:
    SUPPORTED_FILTER: str = (
        f"Common Audio Files ({_SF_EXTS} {_FF_EXTS});;"
        f"All Files (*)"
    )
else:
    # FFmpeg not found — only advertise libsndfile formats, but still allow
    # "All Files" so advanced users can attempt any format via libsndfile.
    SUPPORTED_FILTER: str = (
        f"Audio Files — libsndfile only ({_SF_EXTS});;"
        f"All Files (*)"
    )

# DC-removal filter cutoff
DC_HIGHPASS_HZ: float = 20.0  # Hz — anything below 20 Hz is inaudible; treat as DC

# Normalisation target
NORMALISE_TARGET_DBFS: float = -0.1  # dBFS — peak level after normalisation

# Display
DISPLAY_FPS: int = 30
DISPLAY_TIMER_MS: int = 1_000 // DISPLAY_FPS  # 33 ms

# Canvas colours — dark theme, consistent with the course HTML design system
CANVAS_BG: str = "#0d0f14"
ZERO_LINE_COLOR: str = "#252b38"
ORIGINAL_COLOR: str = "#4f9cf9"  # --accent blue  — original waveform
PROCESSED_COLOR: str = "#a78bfa"  # --accent2 purple — processed waveform
PLAYHEAD_COLOR: str = "#fb923c"  # --orange — playhead line
LABEL_COLOR: str = "#5a6175"  # --muted

SELECTION_COLOR: str = "#4f9cf9"   # selection highlight (blue, semi-transparent)
LOOP_MARKER_COLOR: str = "#fb923c" # loop boundary markers (orange)

# EQ band centre / corner frequencies (Hz)
EQ_LOW_FREQ:  float = 80.0    # low shelf corner
EQ_MID_FREQ:  float = 1_000.0 # mid peak centre
EQ_HIGH_FREQ: float = 8_000.0 # high shelf corner
EQ_MID_Q:     float = 0.707   # mid band Q (Butterworth, ~1 octave)

# AMS logo — one directory above this script (repo root)
LOGO_PATH: Path = Path(__file__).resolve().parent.parent / "ams_group_tui.png"


# =============================================================================
# STEP 2 — FFmpeg fallback decoder
# =============================================================================
#
# WHY A SUBPROCESS INSTEAD OF A PYTHON LIBRARY?
# -----------------------------------------------
# pydub, librosa, and audioread all wrap FFmpeg internally, but they add their
# own abstractions, dependencies, and conversion steps. Calling FFmpeg directly
# via subprocess gives us:
#   • Zero additional Python dependencies (FFmpeg is already required by
#     other demos in this course).
#   • Full transparency — students can see and run the exact FFmpeg command
#     themselves in a terminal to understand what it does.
#   • Direct access to raw PCM bytes via stdout — no temp files, no extra
#     copy, minimal latency.
#
# THE FFMPEG COMMAND EXPLAINED:
#   ffmpeg
#     -hide_banner -loglevel error   suppress version banner and info logs;
#                                    only errors are printed to stderr
#     -i <path>                      input file (any format FFmpeg understands)
#     -vn                            discard video streams (e.g. in MP4/MKV)
#     -acodec pcm_f32le              encode output as 32-bit float little-endian
#                                    PCM — the native format of sounddevice and
#                                    NumPy on all modern hardware (x86/ARM)
#     -f f32le                       output container: raw byte stream, no header
#     -ac <n>                        force output channel count (preserve original)
#     -ar <rate>                     force output sample rate (we resample to
#                                    DEVICE_SAMPLE_RATE here, saving a second
#                                    resample_poly pass later)
#     pipe:1                         write output to stdout (file descriptor 1)
#                                    instead of a file; we read it in Python
#
# WHY pipe:1 INSTEAD OF A TEMP FILE?
#   A temp file requires: disk write → disk read → delete. For a 50 MB MP3
#   that's 50 MB of decoded PCM (~200 MB at float32) written to disk just to
#   be read back immediately. Piping to stdout keeps everything in memory and
#   is significantly faster, especially on SSDs with write throttling.
#
# WINDOWS NOTE:
#   On Windows, a subprocess that opens a console window will briefly flash
#   a black terminal behind the app. We suppress this with:
#     creationflags=subprocess.CREATE_NO_WINDOW  (Windows-only flag)
#   On Linux/macOS this flag is not available and must be omitted. We branch
#   on sys.platform to handle both.


def _decode_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode any audio file to float32 PCM using FFmpeg as a subprocess.

    This is the fallback decoder used when soundfile (libsndfile) cannot
    read the file — most commonly for MP3, AAC/M4A, OPUS, and WMA files.

    The function shells out to ``ffmpeg``, instructs it to decode the input
    to raw 32-bit float little-endian PCM at ``DEVICE_SAMPLE_RATE``, and
    reads the resulting bytes from stdout. The bytes are then interpreted as
    a NumPy float32 array and reshaped to ``(frames, channels)``.

    AUDIO CONCEPT — Why float32 little-endian PCM?
    ------------------------------------------------
    Float32 is the universal internal format for digital audio processing:
      • Range is not hard-clipped at ±1.0 (unlike int16/int24), so intermediate
        calculations can exceed full scale without wrapping — headroom for DSP.
      • NumPy, sounddevice, and scipy all operate natively in float32 without
        conversion overhead.
      • Little-endian matches x86 and ARM byte order — zero byte-swap cost.

    Args:
        path: Path to any audio file supported by FFmpeg.

    Returns:
        Tuple of (data, samplerate) where data is float32, shape
        (frames, channels), and samplerate is DEVICE_SAMPLE_RATE.

    Raises:
        RuntimeError: If FFmpeg is not installed or returns a non-zero exit code.
        ValueError: If the decoded byte stream cannot be interpreted as audio
                    (e.g. the file is corrupt or has zero channels).
    """
    if FFMPEG_BIN is None:
        raise RuntimeError(
            "FFmpeg is not installed or not found on PATH.\n"
            "Ubuntu:  sudo apt install ffmpeg\n"
            "Windows: https://www.gyan.dev/ffmpeg/builds/  (add ffmpeg/bin to PATH)\n"
            f"File '{path.name}' requires FFmpeg to decode."
        )

    # ── Step 1: probe the file to get channel count ───────────────────────────
    # We ask FFmpeg to tell us the channel count before decoding, so we can
    # correctly reshape the raw byte stream. Without this we would have to
    # assume mono or stereo and risk misinterpreting the data.
    #
    # ffprobe is FFmpeg's companion tool for metadata. We ask it for:
    #   -select_streams a:0    first audio stream only
    #   -show_entries ...      only the fields we need
    #   -of csv=p=0            output as bare CSV (no key names, no headers)
    ffprobe_bin = shutil.which("ffprobe") or FFMPEG_BIN.replace("ffmpeg", "ffprobe")

    probe_cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate",
        "-of", "csv=p=0",
        str(path),
    ]

    _kw: dict = {}
    if sys.platform == "win32":
        # Suppress the console window that Windows would open for the subprocess.
        # CREATE_NO_WINDOW is a Windows-only process creation flag.
        _kw["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        probe_result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=True,
            **_kw,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # ffprobe unavailable or failed — fall back to assuming stereo.
        # This is a safe default: if the file is actually mono, FFmpeg will
        # upmix it to stereo via -ac 2 below.
        n_channels_native = 2
    else:
        # Parse "channels,sample_rate\n" → e.g. "2,44100\n"
        parts = probe_result.stdout.strip().split(",")
        try:
            n_channels_native = int(parts[0])
        except (ValueError, IndexError):
            n_channels_native = 2  # safe fallback

    # Clamp to stereo maximum for the output device.
    # Professional interfaces can have 8+ channels, but sounddevice opens a
    # stereo stream. Mixing down to stereo here keeps the playback engine simple.
    n_channels_out = min(n_channels_native, 2)

    # ── Step 2: decode to raw float32 PCM via stdout pipe ────────────────────
    decode_cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "error",   # suppress all non-error output
        "-i", str(path),        # input: any format FFmpeg understands
        "-vn",                  # discard video streams (MP4, MKV, WebM etc.)
        "-acodec", "pcm_f32le", # output codec: raw 32-bit float, little-endian
        "-f", "f32le",          # output format: headerless raw byte stream
        "-ac", str(n_channels_out),          # output channel count
        "-ar", str(DEVICE_SAMPLE_RATE),      # resample to device rate here
        "pipe:1",               # write decoded bytes to stdout
    ]

    try:
        proc = subprocess.run(
            decode_cmd,
            stdout=subprocess.PIPE,  # capture PCM bytes
            stderr=subprocess.PIPE,  # capture error messages
            check=True,              # raise CalledProcessError on non-zero exit
            **_kw,
        )
    except subprocess.CalledProcessError as exc:
        # FFmpeg writes human-readable errors to stderr.
        stderr_msg = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg could not decode '{path.name}':\n{stderr_msg}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"FFmpeg executable not found at '{FFMPEG_BIN}'."
        ) from exc

    # ── Step 3: interpret the raw bytes as a NumPy float32 array ─────────────
    # proc.stdout is a bytes object. Each 4 bytes is one float32 sample.
    # np.frombuffer creates a NumPy array that shares the buffer — no copy.
    # We then reshape from flat (samples,) to (frames, channels).
    raw_bytes = proc.stdout
    if len(raw_bytes) == 0:
        raise ValueError(
            f"FFmpeg produced no output for '{path.name}'. "
            "The file may be corrupt, empty, or contain no audio stream."
        )

    # Validate byte alignment: float32 = 4 bytes per sample
    if len(raw_bytes) % (4 * n_channels_out) != 0:
        # Truncate to the nearest complete frame. This can happen if FFmpeg
        # was interrupted or the file has a partial final frame.
        n_complete = len(raw_bytes) // (4 * n_channels_out) * (4 * n_channels_out)
        raw_bytes = raw_bytes[:n_complete]

    data = np.frombuffer(raw_bytes, dtype="<f4")  # <f4 = little-endian float32
    # Reshape flat sample stream to (frames, channels).
    # C-order (row-major) means samples are interleaved: L R L R L R …
    # which is exactly how sounddevice and soundfile store multichannel audio.
    data = data.reshape(-1, n_channels_out)

    # FFmpeg already resampled to DEVICE_SAMPLE_RATE via -ar above.
    return data, DEVICE_SAMPLE_RATE


# =============================================================================
# STEP 3 — AudioBuffer: decoded audio held in memory
# =============================================================================


@dataclass
class AudioBuffer:
    """Holds the fully decoded audio of one file as a NumPy array.

    All audio data is stored as float32 in shape ``(frames, channels)`` —
    the universal format across soundfile, sounddevice, numpy, and scipy.
    Values are in ``[-1.0, 1.0]``.

    Attributes:
        data: Audio samples, shape ``(frames, channels)``, float32.
        samplerate: Sample rate in Hz (always DEVICE_SAMPLE_RATE after load).
        path: Original file path.
        duration_seconds: Total duration in seconds.
    """

    data: np.ndarray  # shape (frames, channels), float32
    samplerate: int  # Hz
    path: Path
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_seconds = self.data.shape[0] / self.samplerate

    @classmethod
    def load(cls, path: Path) -> AudioBuffer:
        """Read any audio file and return an AudioBuffer at DEVICE_SAMPLE_RATE.

        DECODER CHAIN:
        --------------
        This method implements a two-stage decoder chain, which is the
        professional pattern used in cross-format audio applications:

          Stage 1 — soundfile (libsndfile):
            Attempted first. Fast, battle-tested, no subprocess overhead.
            Covers: WAV, FLAC, OGG/Vorbis, AIFF, CAF, W64, RF64, and more.
            Fails silently for MP3, AAC, OPUS, WMA — we catch the exception
            and fall through to Stage 2.

          Stage 2 — FFmpeg subprocess (fallback):
            Used when libsndfile raises an exception. Handles virtually every
            audio/video container and codec in professional use. See the
            _decode_via_ffmpeg() function above for the full explanation of
            how the subprocess pipe works.
            If FFmpeg is also absent, a RuntimeError is raised with
            installation instructions for both Ubuntu and Windows.

        AUDIO CONCEPT — Why resampling matters:
        sounddevice opens the audio device at DEVICE_SAMPLE_RATE (48 000 Hz).
        If you play a 44 100 Hz file through a 48 000 Hz stream without
        resampling, every sample is played at the wrong pitch and speed
        (44 100 / 48 000 ≈ −1.5 semitones, 8.8 % too slow). Always resample
        to the device rate before playback.

        When Stage 1 (soundfile) decodes the file, the native sample rate may
        differ from DEVICE_SAMPLE_RATE, so we resample here using resample_poly.
        When Stage 2 (FFmpeg) decodes, we pass -ar DEVICE_SAMPLE_RATE to FFmpeg
        directly, so no Python-side resampling is needed — FFmpeg's built-in
        resampler (libswresample, the same engine used in professional broadcast
        tools) handles it.

        RESAMPLING CONCEPT — resample_poly:
        resample_poly uses a polyphase filter bank — the same algorithm used
        in professional audio editors and broadcast encoders (Pro Tools, Reaper,
        Nuendo). It is more accurate than numpy.interp (which is linear
        interpolation, not suitable for audio) and faster than
        scipy.signal.resample for rational frequency ratios. It requires integer
        up/down factors, so we reduce the ratio to its simplest form using GCD.

        Args:
            path: Path to any audio file. soundfile is tried first; FFmpeg
                  is used as a fallback if soundfile cannot decode the format.

        Returns:
            AudioBuffer at DEVICE_SAMPLE_RATE.

        Raises:
            ValueError: If neither decoder can read the file.
        """
        from math import gcd

        # ── Stage 1: soundfile (libsndfile) ──────────────────────────────────
        sf_error: Exception | None = None
        try:
            data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
            # always_2d=True ensures shape is always (frames, channels),
            # even for mono files (which would otherwise be 1-D).
        except Exception as exc:
            # libsndfile cannot handle this format (most likely MP3, AAC,
            # OPUS, WMA). Record the error and fall through to Stage 2.
            sf_error = exc
            data = None
            file_sr = None

        if data is not None:
            # soundfile succeeded — resample in Python if needed.
            if file_sr != DEVICE_SAMPLE_RATE:
                g = gcd(DEVICE_SAMPLE_RATE, file_sr)
                up, down = DEVICE_SAMPLE_RATE // g, file_sr // g
                # Resample each channel independently.
                # resample_poly operates on 1-D arrays; we loop over channels
                # and reassemble with np.stack.
                resampled_channels = [
                    resample_poly(data[:, ch], up, down).astype(np.float32)
                    for ch in range(data.shape[1])
                ]
                data = np.stack(resampled_channels, axis=1)
            return cls(data=data, samplerate=DEVICE_SAMPLE_RATE, path=path)

        # ── Stage 2: FFmpeg subprocess fallback ───────────────────────────────
        # soundfile failed (sf_error is set). Try FFmpeg.
        # _decode_via_ffmpeg() already resamples to DEVICE_SAMPLE_RATE via
        # the -ar flag, so no additional Python-side resampling is needed.
        try:
            data, file_sr = _decode_via_ffmpeg(path)
        except (RuntimeError, ValueError) as ffmpeg_exc:
            # Both decoders failed. Report both errors so the student
            # understands exactly what was attempted and why each failed.
            raise ValueError(
                f"Cannot decode '{path.name}'.\n\n"
                f"soundfile (libsndfile) error:\n  {sf_error}\n\n"
                f"FFmpeg error:\n  {ffmpeg_exc}"
            ) from ffmpeg_exc

        return cls(data=data, samplerate=DEVICE_SAMPLE_RATE, path=path)

    @property
    def n_frames(self) -> int:
        """Total number of audio frames (samples per channel)."""
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of audio channels (1 = mono, 2 = stereo)."""
        return self.data.shape[1]

    def mono(self) -> np.ndarray:
        """Return a mono mix of the audio as a 1-D float32 array.

        For a stereo file, averages left and right channels. This is used
        by WaveformView which always displays a mono waveform.
        """
        if self.n_channels == 1:
            return self.data[:, 0]
        return self.data.mean(axis=1).astype(np.float32)


# =============================================================================
# STEP 4 — DspChain: offline DSP processing functions
# =============================================================================


class DspChain:
    """A collection of offline DSP functions applied to an AudioBuffer copy.

    All functions are pure NumPy/SciPy — zero Qt dependencies. They take a
    float32 array of shape (frames, channels) and return one of the same shape.
    They never modify the input in-place; they always return a new array so the
    original buffer is preserved for comparison.
    """

    @staticmethod
    def apply_gain(data: np.ndarray, db: float) -> np.ndarray:
        """Apply a gain in dB to the audio data.

        AUDIO CONCEPT — dB gain:
        Decibels (dB) are a logarithmic unit. +6 dB ≈ double the amplitude.
        −6 dB ≈ half. 0 dB = no change. The conversion from dB to linear
        gain is: linear = 10^(dB / 20).
        This formula applies to amplitude (voltage, sample value). For power
        the denominator would be 10, not 20.

        Args:
            data: Audio array, shape (frames, channels), float32.
            db: Gain in dB. Negative values attenuate; positive values amplify.

        Returns:
            Gained audio array, same shape and dtype.
        """
        linear_gain: float = 10.0 ** (db / 20.0)
        return (data * linear_gain).astype(np.float32)

    @staticmethod
    def apply_fade(
        data: np.ndarray,
        fade_in_ms: float,
        fade_out_ms: float,
        samplerate: int,
    ) -> np.ndarray:
        """Apply linear fade-in and fade-out envelopes.

        AUDIO CONCEPT — Fade envelopes:
        A fade-in ramps amplitude from 0 to 1 at the start of the file.
        A fade-out ramps from 1 to 0 at the end. Without fades, abrupt
        starts and ends produce clicks (discontinuities at zero-crossings).
        np.linspace generates the ramp with exactly the right number of points.

        Args:
            data: Audio array, shape (frames, channels), float32.
            fade_in_ms: Duration of the fade-in in milliseconds.
            fade_out_ms: Duration of the fade-out in milliseconds.
            samplerate: Sample rate in Hz (used to convert ms to frames).

        Returns:
            Audio with fades applied, same shape and dtype.
        """
        result = data.copy()
        n_frames = data.shape[0]

        fade_in_frames = min(int(fade_in_ms * samplerate / 1_000), n_frames)
        fade_out_frames = min(int(fade_out_ms * samplerate / 1_000), n_frames)

        if fade_in_frames > 0:
            # Build ramp from 0.0 to 1.0 over fade_in_frames samples
            ramp = np.linspace(0.0, 1.0, fade_in_frames, dtype=np.float32)
            result[:fade_in_frames] *= ramp[:, np.newaxis]  # broadcast over channels

        if fade_out_frames > 0:
            # Build ramp from 1.0 to 0.0 over fade_out_frames samples
            ramp = np.linspace(1.0, 0.0, fade_out_frames, dtype=np.float32)
            result[-fade_out_frames:] *= ramp[:, np.newaxis]

        return result

    @staticmethod
    def remove_dc(data: np.ndarray, samplerate: int) -> np.ndarray:
        """Remove DC offset with a high-pass Butterworth filter at 20 Hz.

        DSP CONCEPT — DC offset:
        A DC offset is a constant non-zero mean value in an audio signal.
        It appears as a vertical shift of the waveform on the canvas —
        the zero line is no longer the centre. DC offset causes:
          1. Clipping headroom loss (the asymmetric waveform clips earlier
             on one side).
          2. Click at the start of playback (the speaker cone jumps from
             rest position to the DC level).
          3. Saturation in downstream processing (compressors, limiters,
             analogue outputs).

        DC offset is common in recordings from cheap USB audio interfaces
        with poor analogue front-ends. It is invisible to the ear but
        visible on a waveform display and measurable with a DC meter.

        A 2nd-order Butterworth high-pass filter at 20 Hz removes the DC
        component while leaving all audible content (>20 Hz) untouched.
        We use output='sos' (second-order sections) for numerical stability.

        RULE: Always use butter(..., output='sos') and sosfilt. Never use
        lfilter with b, a coefficients for filters above order 2 — they
        overflow at low cutoff frequencies.

        Args:
            data: Audio array, shape (frames, channels), float32.
            samplerate: Sample rate in Hz.

        Returns:
            DC-removed audio, same shape and dtype.
        """
        # Design a 2nd-order Butterworth high-pass filter.
        # Wn is normalised: cutoff / (samplerate / 2).
        sos = butter(
            N=2,
            Wn=DC_HIGHPASS_HZ / (samplerate / 2.0),
            btype="highpass",
            output="sos",
        )
        # Apply per channel. sosfilt processes 1-D arrays.
        result = np.empty_like(data)
        for ch in range(data.shape[1]):
            result[:, ch] = sosfilt(sos, data[:, ch]).astype(np.float32)
        return result

    @staticmethod
    def normalise(data: np.ndarray) -> np.ndarray:
        """Peak-normalise the audio to NORMALISE_TARGET_DBFS.

        AUDIO CONCEPT — Peak normalisation:
        Peak normalisation finds the loudest sample in the file and scales
        the entire file so that sample sits at the target level.
        NORMALISE_TARGET_DBFS = −0.1 dBFS leaves a tiny headroom margin
        below 0 dBFS (full scale) to avoid inter-sample peaks clipping
        when converted to integer formats (e.g. 16-bit WAV on export).

        0 dBFS = maximum representable level (1.0 in float32 notation).
        −0.1 dBFS ≈ 0.989 linear.

        Note: peak normalisation does not change the perceived dynamic range
        or loudness character of the audio. For perceived loudness matching,
        use a loudness-normalisation standard such as EBU R128.

        Args:
            data: Audio array, shape (frames, channels), float32.

        Returns:
            Normalised audio, same shape and dtype.
        """
        peak = np.max(np.abs(data))
        if peak < 1e-9:
            return data.copy()  # silence — do not divide by zero
        target_linear: float = 10.0 ** (NORMALISE_TARGET_DBFS / 20.0)
        return (data * (target_linear / peak)).astype(np.float32)

    @staticmethod
    def apply_eq_band(
        data: np.ndarray,
        samplerate: int,
        freq: float,
        gain_db: float,
        btype: str,
        q: float = 0.707,
    ) -> np.ndarray:
        """Apply a single EQ band (low shelf, peaking, or high shelf).

        DSP CONCEPT — Parametric EQ bands:
        A shelving filter boosts/cuts all frequencies above (high shelf) or
        below (low shelf) a corner frequency. A peaking (bell) filter boosts
        or cuts a band of frequencies centred on freq with width set by Q.
        We implement all three using a 2nd-order Butterworth IIR filter
        designed with scipy.signal.butter (output='sos' for numerical
        stability).

        For gain_db == 0 the function returns the input unchanged to avoid
        applying a unity-gain filter (wastes CPU, adds noise at float32
        precision boundaries).

        Args:
            data: Audio array (frames, channels), float32.
            samplerate: Sample rate in Hz.
            freq: Corner / centre frequency in Hz.
            gain_db: Boost (positive) or cut (negative) in dB.
            btype: 'lowpass' for low shelf, 'highpass' for high shelf,
                   'bandpass' for mid peak.
            q: Bandwidth Q for bandpass only (ignored for shelves).

        Returns:
            Filtered audio, same shape and dtype.
        """
        if abs(gain_db) < 0.01:
            return data  # nothing to do

        nyq = samplerate / 2.0
        wn = freq / nyq

        if btype == "bandpass":
            bw = freq / q
            low = max((freq - bw / 2) / nyq, 0.001)
            high = min((freq + bw / 2) / nyq, 0.999)
            sos = butter(N=2, Wn=[low, high], btype="bandpass", output="sos")
        else:
            wn = min(max(wn, 0.001), 0.999)
            sos = butter(N=2, Wn=wn, btype=btype, output="sos")

        linear_gain = 10.0 ** (gain_db / 20.0)
        result = np.empty_like(data)
        for ch in range(data.shape[1]):
            filtered = sosfilt(sos, data[:, ch]).astype(np.float32)
            # Blend original + (filtered − original) * gain: shelf/peak behaviour
            result[:, ch] = (data[:, ch] + (filtered - data[:, ch]) * (linear_gain - 1.0)).astype(np.float32)
        return result

    @classmethod
    def apply_chain(
        cls,
        buffer: AudioBuffer,
        gain_db: float,
        fade_in_ms: float,
        fade_out_ms: float,
        remove_dc: bool,
        do_normalise: bool,
        eq_low_db: float = 0.0,
        eq_mid_db: float = 0.0,
        eq_high_db: float = 0.0,
        sel_start: int | None = None,
        sel_end: int | None = None,
    ) -> AudioBuffer:
        """Apply the full DSP chain to a copy of the buffer.

        When sel_start / sel_end are provided the chain operates only on that
        region; the rest of the file is left untouched.  Fades are applied to
        the selection boundaries so the loop edges are click-free.  When no
        selection is given the chain runs on the entire file.

        Processing order:
          1. EQ  (low shelf → mid peak → high shelf)
          2. Gain
          3. DC removal
          4. Fade in / fade out  (at region boundaries)
          5. Normalise

        Args:
            buffer: Source AudioBuffer — never modified.
            gain_db: Gain in dB.
            fade_in_ms / fade_out_ms: Fade durations in ms.
            remove_dc: Apply 20 Hz high-pass DC removal.
            do_normalise: Peak-normalise to −0.1 dBFS.
            eq_low_db: Low-shelf gain (±12 dB, corner EQ_LOW_FREQ Hz).
            eq_mid_db: Mid-peak gain (±12 dB, centre EQ_MID_FREQ Hz).
            eq_high_db: High-shelf gain (±12 dB, corner EQ_HIGH_FREQ Hz).
            sel_start: First frame of the selection (None → whole file).
            sel_end: First frame after the selection (None → whole file).

        Returns:
            A new AudioBuffer with the processed audio.
        """
        full_data = buffer.data.copy()

        # Determine the region to process
        total = buffer.n_frames
        if sel_start is not None and sel_end is not None:
            r0 = max(0, min(sel_start, total))
            r1 = max(0, min(sel_end, total))
            if r0 >= r1:
                r0, r1 = 0, total  # degenerate selection → whole file
        else:
            r0, r1 = 0, total

        region = full_data[r0:r1].copy()

        # 1 — EQ
        region = cls.apply_eq_band(region, buffer.samplerate, EQ_LOW_FREQ,  eq_low_db,  "lowpass")
        region = cls.apply_eq_band(region, buffer.samplerate, EQ_MID_FREQ,  eq_mid_db,  "bandpass", EQ_MID_Q)
        region = cls.apply_eq_band(region, buffer.samplerate, EQ_HIGH_FREQ, eq_high_db, "highpass")

        # 2 — Gain
        region = cls.apply_gain(region, gain_db)

        # 3 — DC removal
        if remove_dc:
            region = cls.remove_dc(region, buffer.samplerate)

        # 4 — Fades (always applied at region boundaries for click-free looping)
        region = cls.apply_fade(region, fade_in_ms, fade_out_ms, buffer.samplerate)

        # 5 — Normalise
        if do_normalise:
            region = cls.normalise(region)

        full_data[r0:r1] = region
        return AudioBuffer(data=full_data, samplerate=buffer.samplerate, path=buffer.path)


# =============================================================================
# STEP 5 — WaveformView: QPainter waveform canvas with playhead
# =============================================================================


class WaveformView(QWidget):
    """Professional QPainter waveform canvas.

    Features:
      • Gradient-filled waveform shape (filled path from centre outward),
        anti-aliased, with a bright stroke edge — matches the visual quality
        of Logic Pro / Reaper / Audacity.
      • Zoom: mouse-wheel zooms in/out (1× – 64×); horizontal QScrollBar pans.
      • Selection: left-click-drag draws a selection region (blue highlight).
        Shift-click extends the selection. Right-click clears the selection.
      • Time ruler at the bottom with second / sub-second tick marks.
      • Subtle amplitude grid lines at ±0.5 and ±1.0.
      • Playhead (orange vertical line).

    Signals:
        seek_requested:      Emitted on left click (no drag). Carries frame.
        selection_changed:   Emitted when the selection region changes.
                             Carries (sel_start_frame, sel_end_frame) or
                             (-1, -1) when cleared.
    """

    seek_requested:    Signal = Signal(int)
    selection_changed: Signal = Signal(int, int)

    # Height of the time ruler strip at the bottom of the canvas (px)
    _RULER_H: int = 22

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        # Audio data
        self._orig_audio:  np.ndarray | None = None   # mono float32, full file
        self._proc_audio:  np.ndarray | None = None
        self._n_frames:    int = 0
        self._samplerate:  int = DEVICE_SAMPLE_RATE
        self._has_processed: bool = False

        # Zoom / pan  (view_start / view_end are frame indices)
        self._zoom:       float = 1.0
        self._view_start: int   = 0   # first visible frame
        self._view_end:   int   = 0   # last visible frame (exclusive)

        # Cached peak arrays for the current view window (size = canvas width)
        self._cache_orig_min: np.ndarray = np.array([])
        self._cache_orig_max: np.ndarray = np.array([])
        self._cache_proc_min: np.ndarray = np.array([])
        self._cache_proc_max: np.ndarray = np.array([])
        self._cache_view: tuple[int, int, int] = (-1, -1, -1)  # (start,end,width)

        # Playhead
        self._playhead_frame: int = 0

        # Selection
        self._sel_start:  int = -1
        self._sel_end:    int = -1
        self._drag_anchor: int | None = None   # frame where drag started

    # ── Public interface ───────────────────────────────────────────────────────

    def set_buffer(
        self,
        original: "AudioBuffer | None",
        processed: "AudioBuffer | None" = None,
    ) -> None:
        """Load new buffers, reset view, invalidate cache."""
        if original is None:
            self._orig_audio = None
            self._proc_audio = None
            self._n_frames   = 0
            self._zoom       = 1.0
            self._view_start = 0
            self._view_end   = 0
            self._has_processed = False
            self._invalidate_cache()
            self.update()
            return

        self._n_frames   = original.n_frames
        self._samplerate = original.samplerate
        self._orig_audio = original.mono()
        self._zoom       = 1.0
        self._view_start = 0
        self._view_end   = self._n_frames

        if processed is not None:
            self._proc_audio    = processed.mono()
            self._has_processed = True
        else:
            self._proc_audio    = None
            self._has_processed = False

        self._invalidate_cache()
        self.update()

    def set_playhead(self, frame: int) -> None:
        self._playhead_frame = frame
        self.update()

    def clear_selection(self) -> None:
        self._sel_start = -1
        self._sel_end   = -1
        self.selection_changed.emit(-1, -1)
        self.update()

    @property
    def selection(self) -> tuple[int, int]:
        """Return (sel_start, sel_end) frames, or (-1, -1) if none."""
        if self._sel_start < 0 or self._sel_end < 0:
            return -1, -1
        a, b = sorted([self._sel_start, self._sel_end])
        return a, b

    # ── Zoom helpers ───────────────────────────────────────────────────────────

    def _apply_zoom(self, new_zoom: float, anchor_px: int) -> None:
        """Set zoom level keeping the pixel column anchor_px on the same frame."""
        new_zoom = max(1.0, min(64.0, new_zoom))
        if self._n_frames == 0:
            return
        # Frame under the cursor before zoom
        anchor_frame = self._px_to_frame(anchor_px)
        self._zoom = new_zoom
        visible = int(self._n_frames / self._zoom)
        # Recentre so anchor_frame stays at anchor_px
        w = max(self.width(), 1)
        frac = anchor_px / w
        self._view_start = int(anchor_frame - frac * visible)
        self._view_start = max(0, min(self._view_start, self._n_frames - visible))
        self._view_end   = self._view_start + visible
        self._invalidate_cache()
        self.update()

    def set_scroll(self, value: int) -> None:
        """Called by the external QScrollBar (value = view_start frame)."""
        if self._n_frames == 0:
            return
        visible = int(self._n_frames / self._zoom)
        self._view_start = max(0, min(value, self._n_frames - visible))
        self._view_end   = self._view_start + visible
        self._invalidate_cache()
        self.update()

    def scroll_range(self) -> tuple[int, int, int]:
        """Return (min, max, page_step) for the scroll bar."""
        if self._n_frames == 0:
            return 0, 0, 0
        visible = int(self._n_frames / self._zoom)
        return 0, max(0, self._n_frames - visible), visible

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _invalidate_cache(self) -> None:
        self._cache_view = (-1, -1, -1)

    def _ensure_cache(self) -> None:
        """Recompute peak arrays for the current view window if stale."""
        w = max(self.width(), 1)
        key = (self._view_start, self._view_end, w)
        if key == self._cache_view:
            return
        self._cache_view = key

        if self._orig_audio is None or self._view_end <= self._view_start:
            self._cache_orig_min = np.zeros(w, dtype=np.float32)
            self._cache_orig_max = np.zeros(w, dtype=np.float32)
            self._cache_proc_min = np.zeros(w, dtype=np.float32)
            self._cache_proc_max = np.zeros(w, dtype=np.float32)
            return

        self._cache_orig_min, self._cache_orig_max = self._peak_reduce(
            self._orig_audio, self._view_start, self._view_end, w
        )
        if self._has_processed and self._proc_audio is not None:
            self._cache_proc_min, self._cache_proc_max = self._peak_reduce(
                self._proc_audio, self._view_start, self._view_end, w
            )
        else:
            self._cache_proc_min = np.zeros(w, dtype=np.float32)
            self._cache_proc_max = np.zeros(w, dtype=np.float32)

    @staticmethod
    def _peak_reduce(
        mono: np.ndarray, start: int, end: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Peak-reduce mono[start:end] into `width` columns (vectorised)."""
        region = mono[start:end]
        n = len(region)
        if n == 0:
            return np.zeros(width, np.float32), np.zeros(width, np.float32)

        # Pad so n is divisible by width — avoids a Python loop
        remainder = (-n) % width
        if remainder:
            region = np.concatenate([region, np.full(remainder, np.nan, dtype=np.float32)])
        region = region.reshape(width, -1)

        with np.errstate(all="ignore"):
            peaks_max = np.nanmax(region, axis=1).astype(np.float32)
            peaks_min = np.nanmin(region, axis=1).astype(np.float32)

        np.nan_to_num(peaks_max, nan=0.0, copy=False)
        np.nan_to_num(peaks_min, nan=0.0, copy=False)
        return peaks_min, peaks_max

    def _frame_to_px(self, frame: int) -> int:
        """Map a frame index to pixel X within the current view."""
        if self._view_end <= self._view_start:
            return 0
        w = max(self.width(), 1)
        ratio = (frame - self._view_start) / (self._view_end - self._view_start)
        return int(ratio * w)

    def _px_to_frame(self, px: int) -> int:
        """Map pixel X to a frame index within the current view."""
        if self._n_frames == 0 or self._view_end <= self._view_start:
            return 0
        w = max(self.width(), 1)
        frac = max(0.0, min(1.0, px / w))
        return int(self._view_start + frac * (self._view_end - self._view_start))

    def _wave_h(self) -> int:
        """Height of the waveform area (canvas minus ruler strip)."""
        return max(self.height() - self._RULER_H, 1)

    def _amp_to_py(self, amp: float, h: int) -> int:
        """Amplitude → pixel Y in the waveform area."""
        return int(h / 2.0 * (1.0 - float(amp)))

    # ── Mouse interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self._n_frames == 0:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.clear_selection()
            return
        frame = self._px_to_frame(event.pos().x())
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and self._sel_start >= 0:
            # Extend existing selection
            self._sel_end = frame
        else:
            # Start new selection / seek
            self._drag_anchor = frame
            self._sel_start   = frame
            self._sel_end     = frame
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._n_frames == 0:
            return
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_anchor is not None:
            frame = self._px_to_frame(event.pos().x())
            self._sel_start = self._drag_anchor
            self._sel_end   = frame
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._n_frames == 0:
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_anchor is not None:
            frame = self._px_to_frame(event.pos().x())
            if abs(frame - self._drag_anchor) < 64:
                # Tiny drag treated as a click → seek, clear selection
                self.seek_requested.emit(frame)
                self._sel_start = -1
                self._sel_end   = -1
                self.selection_changed.emit(-1, -1)
            else:
                a, b = sorted([self._drag_anchor, frame])
                self._sel_start = a
                self._sel_end   = b
                self.selection_changed.emit(a, b)
            self._drag_anchor = None
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._n_frames == 0:
            return
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self._apply_zoom(self._zoom * factor, event.position().x())
        self.zoom_changed.emit(self._zoom, self._view_start)

    zoom_changed: Signal = Signal(float, int)   # (zoom, view_start) → scrollbar

    # ── QPainter rendering ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        self._ensure_cache()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w    = self.width()
        wh   = self._wave_h()   # waveform area height
        h    = self.height()

        # ── Background ────────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        if self._orig_audio is None:
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 11))
            painter.drawText(
                QRect(0, 0, w, wh),
                Qt.AlignmentFlag.AlignCenter,
                "Open an audio file to display the waveform",
            )
            self._draw_ruler(painter, w, wh, h)
            painter.end()
            return

        # ── Amplitude grid lines ──────────────────────────────────────────────
        grid_pen = QPen(QColor(ZERO_LINE_COLOR), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for amp in (1.0, 0.5, 0.0, -0.5, -1.0):
            py = self._amp_to_py(amp, wh)
            painter.drawLine(0, py, w, py)

        # ── Selection highlight ───────────────────────────────────────────────
        s0, s1 = self.selection
        if s0 >= 0 and s1 > s0:
            px0 = self._frame_to_px(s0)
            px1 = self._frame_to_px(s1)
            sel_color = QColor(79, 156, 249, 40)
            painter.fillRect(px0, 0, px1 - px0, wh, sel_color)
            # Boundary markers
            marker_pen = QPen(QColor(79, 156, 249, 160), 1, Qt.PenStyle.DashLine)
            painter.setPen(marker_pen)
            painter.drawLine(px0, 0, px0, wh)
            painter.drawLine(px1, 0, px1, wh)

        # ── Processed waveform (purple, filled, behind) ───────────────────────
        if self._has_processed and len(self._cache_proc_max) == w:
            self._draw_waveform(
                painter, w, wh,
                self._cache_proc_min, self._cache_proc_max,
                QColor(167, 139, 250, 60),   # fill
                QColor(167, 139, 250, 160),  # stroke
            )

        # ── Original waveform (blue, filled, on top) ──────────────────────────
        if len(self._cache_orig_max) == w:
            self._draw_waveform(
                painter, w, wh,
                self._cache_orig_min, self._cache_orig_max,
                QColor(79, 156, 249, 70),    # fill
                QColor(79, 156, 249, 220),   # stroke
            )

        # ── Playhead ──────────────────────────────────────────────────────────
        ph_px = self._frame_to_px(self._playhead_frame)
        ph_pen = QPen(QColor(PLAYHEAD_COLOR), 2)
        painter.setPen(ph_pen)
        painter.drawLine(ph_px, 0, ph_px, wh)

        # ── Ruler ─────────────────────────────────────────────────────────────
        self._draw_ruler(painter, w, wh, h)

        painter.end()

    def _draw_waveform(
        self,
        painter: QPainter,
        w: int,
        wh: int,
        peaks_min: np.ndarray,
        peaks_max: np.ndarray,
        fill_color: QColor,
        stroke_color: QColor,
    ) -> None:
        """Draw a gradient-filled waveform shape for one buffer."""
        cy = wh / 2.0

        # Build a closed QPainterPath: top edge (max) left→right,
        # bottom edge (min) right→left, closed at both ends.
        path = QPainterPath()
        path.moveTo(0, cy)
        for px in range(w):
            py_max = self._amp_to_py(float(peaks_max[px]), wh)
            path.lineTo(px, py_max)
        path.lineTo(w - 1, cy)
        for px in range(w - 1, -1, -1):
            py_min = self._amp_to_py(float(peaks_min[px]), wh)
            path.lineTo(px, py_min)
        path.closeSubpath()

        # Vertical gradient: brightest at centre (zero line), transparent at edges
        grad = QLinearGradient(0, 0, 0, wh)
        top_color    = QColor(fill_color)
        top_color.setAlpha(20)
        centre_color = QColor(fill_color)
        centre_color.setAlpha(fill_color.alpha())
        grad.setColorAt(0.0,  top_color)
        grad.setColorAt(0.5,  centre_color)
        grad.setColorAt(1.0,  top_color)

        painter.fillPath(path, QBrush(grad))

        # Stroke — draw only the top-edge and bottom-edge paths separately
        # so the closed ends don't get a stroke
        stroke_pen = QPen(stroke_color, 1)
        painter.setPen(stroke_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _draw_ruler(self, painter: QPainter, w: int, wh: int, h: int) -> None:
        """Draw the time ruler strip at the bottom of the canvas."""
        # Background strip
        ruler_bg = QColor("#0a0c10")
        painter.fillRect(0, wh, w, h - wh, ruler_bg)
        border_pen = QPen(QColor(ZERO_LINE_COLOR), 1)
        painter.setPen(border_pen)
        painter.drawLine(0, wh, w, wh)

        if self._n_frames == 0 or self._samplerate == 0:
            return

        view_dur = (self._view_end - self._view_start) / self._samplerate

        # Choose tick interval: aim for ~80 px between ticks
        raw_interval = view_dur * 80 / max(w, 1)
        nice = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
        interval = nice[-1]
        for n in nice:
            if n >= raw_interval:
                interval = n
                break

        painter.setFont(QFont("IBM Plex Mono", 8))
        tick_pen  = QPen(QColor("#3a4155"), 1)
        label_pen = QPen(QColor(LABEL_COLOR))

        # First tick at or after view_start
        view_start_s = self._view_start / self._samplerate
        first_tick = int(view_start_s / interval) * interval

        t = first_tick
        while True:
            frame = int(t * self._samplerate)
            px = self._frame_to_px(frame)
            if px > w:
                break
            # Tick mark
            painter.setPen(tick_pen)
            painter.drawLine(px, wh + 1, px, wh + 5)
            # Label
            if t >= 60:
                m = int(t // 60)
                s = t - m * 60
                label = f"{m}:{s:05.2f}"
            else:
                label = f"{t:.2f}s" if t < 10 else f"{t:.1f}s"
            painter.setPen(label_pen)
            painter.drawText(px + 3, h - 4, label)
            t += interval

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        self._invalidate_cache()
        self.update()


# =============================================================================
# STEP 6 — PlaybackEngine: sounddevice OutputStream with Lock-protected seek
# =============================================================================


class PlaybackEngine:
    """Manages a sounddevice OutputStream for playing back an AudioBuffer.

    THREAD CONCEPT — Lock-protected frame pointer:
    Demo 1's AudioEngine set simple float/int attributes from the main thread
    and read them in the callback. That is safe for scalars because Python's
    GIL makes individual attribute assignment atomic.

    Here, the callback reads _frame_pos AND increments it:
        pos = self._frame_pos
        self._frame_pos += frames   # read-modify-write — NOT atomic under GIL

    A read-modify-write is not atomic: the main thread could call seek() and
    overwrite _frame_pos between the read and the write, corrupting the
    playback position. We use a threading.Lock to make the entire operation
    atomic:

        with self._lock:
            pos = self._frame_pos
            self._frame_pos += frames

    This guarantees that seek() either runs completely before the callback's
    block, or completely after — never in the middle.

    Data is passed from the callback to the main thread via a queue.Queue.
    queue.Queue is thread-safe by design — it uses its own internal lock.
    The callback pushes frame positions; a QTimer in MainWindow drains the
    queue and updates the UI.
    """

    def __init__(self) -> None:
        """Initialise with no buffer loaded. Call load() before play()."""
        self._buffer: AudioBuffer | None = None
        self._frame_pos: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

        # Loop region (-1 = disabled)
        self._loop_start: int = -1
        self._loop_end:   int = -1
        self._loop_active: bool = False

        # Queue for passing frame positions from the audio thread to the UI thread.
        # The callback pushes; MainWindow's QTimer drains.
        # maxsize=4 so the queue does not grow unboundedly if the UI is slow.
        self._frame_queue: queue.Queue[int] = queue.Queue(maxsize=4)

        # Flag set by the callback when playback reaches end-of-file.
        # MainWindow polls this via the same QTimer.
        self._finished_queue: queue.Queue[bool] = queue.Queue(maxsize=2)

    # ── Buffer management ──────────────────────────────────────────────────────

    def load(self, buffer: AudioBuffer) -> None:
        """Load a new AudioBuffer, stopping playback if currently active.

        Args:
            buffer: The decoded audio to play.
        """
        was_active = self._stream is not None and self._stream.active
        if was_active:
            self._stream.stop()

        with self._lock:
            self._buffer = buffer
            self._frame_pos = 0

        if was_active:
            self._stream.start()

    # ── Audio callback — runs on the PortAudio audio thread ───────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Read frames from the buffer and write them to the output.

        Handles end-of-file by zero-padding the final block and posting to
        _finished_queue. Never calls Qt functions — uses queue.Queue only.

        Args:
            outdata: Buffer to fill, shape (frames, channels), float32.
            frames: Number of frames requested.
            time_info: PortAudio timing (not used here).
            status: Stream health flags.
        """
        if status.output_underflow:
            print("[PlaybackEngine] output underflow", flush=True)

        with self._lock:
            if self._buffer is None:
                outdata[:] = 0.0
                return

            pos = self._frame_pos
            buf = self._buffer.data
            n_frames = buf.shape[0]
            n_channels = buf.shape[1]

            remaining = n_frames - pos

            if remaining <= 0:
                # Past end of file — output silence and signal completion
                outdata[:] = 0.0
                self._frame_pos = n_frames
            elif remaining < frames:
                # Final partial block — copy what remains, zero-pad the rest
                outdata[:remaining, :n_channels] = buf[pos : pos + remaining]
                if n_channels < outdata.shape[1]:
                    outdata[:remaining, n_channels:] = 0.0
                outdata[remaining:] = 0.0
                self._frame_pos = n_frames
                # Signal end-of-file to the main thread (non-blocking put)
                try:
                    self._finished_queue.put_nowait(True)
                except queue.Full:
                    pass
            else:
                # Normal block
                chunk = buf[pos : pos + frames]
                if chunk.shape[1] == outdata.shape[1]:
                    outdata[:] = chunk
                elif chunk.shape[1] == 1:
                    # Mono file → duplicate to all output channels
                    outdata[:] = np.repeat(chunk, outdata.shape[1], axis=1)
                else:
                    # More source channels than output — take first N
                    outdata[:] = chunk[:, : outdata.shape[1]]
                self._frame_pos += frames

            # Push frame position to the UI queue (non-blocking)
            try:
                self._frame_queue.put_nowait(self._frame_pos)
            except queue.Full:
                pass  # UI is not keeping up — drop the update, do not block

    # ── Playback control ───────────────────────────────────────────────────────

    def _ensure_stream(self) -> None:
        """Create the OutputStream if it does not yet exist."""
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=DEVICE_SAMPLE_RATE,
                channels=2,  # always output stereo (mono files are duplicated)
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._callback,
            )

    def play(self) -> None:
        """Start or resume playback."""
        self._ensure_stream()
        if not self._stream.active:
            self._stream.start()

    def pause(self) -> None:
        """Pause playback. Call play() to resume."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def stop(self) -> None:
        """Stop playback and reset the playhead to the beginning."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()
        with self._lock:
            self._frame_pos = 0

    def seek(self, frame: int) -> None:
        """Seek to a specific frame position.

        Args:
            frame: Target frame index. Clamped to [0, n_frames].
        """
        with self._lock:
            if self._buffer is not None:
                self._frame_pos = max(0, min(frame, self._buffer.n_frames))

    def close(self) -> None:
        """Stop and release the audio device. Call from closeEvent."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def frame_queue(self) -> queue.Queue[int]:
        """Queue polled by MainWindow's QTimer for current frame position."""
        return self._frame_queue

    @property
    def finished_queue(self) -> queue.Queue[bool]:
        """Queue polled by MainWindow's QTimer for end-of-file signal."""
        return self._finished_queue

    @property
    def is_playing(self) -> bool:
        """True if the stream is active."""
        return self._stream is not None and self._stream.active


# =============================================================================
# STEP 6 — DspPanel: DSP controls QGroupBox
# =============================================================================


class DspPanel(QWidget):
    """Right-side panel: DSP controls and Apply button.

    Emits:
        dsp_requested: When Apply DSP is clicked. Carries (gain_db, fade_in_ms,
                       fade_out_ms, remove_dc, normalise) as a tuple.
    """

    dsp_requested: Signal = Signal(float, float, float, bool, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the DSP control panel."""
        super().__init__(parent)
        self.setFixedWidth(260)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all DSP controls."""
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

        # ── DSP controls group ────────────────────────────────────────────────
        group = QGroupBox("DSP Effects Chain")
        group.setStyleSheet(
            "QGroupBox { color: #a78bfa; font-size: 12px; border: 1px solid #252b38;"
            " border-radius: 4px; margin-top: 8px; padding-top: 10px; }"
        )
        form = QFormLayout(group)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Gain spinbox: −12 to +12 dB in 0.5 dB steps
        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(-12.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setValue(0.0)
        self._gain_spin.setSuffix(" dB")
        form.addRow("Gain", self._gain_spin)

        # Fade in
        self._fade_in_spin = QDoubleSpinBox()
        self._fade_in_spin.setRange(0.0, 5_000.0)
        self._fade_in_spin.setSingleStep(10.0)
        self._fade_in_spin.setValue(0.0)
        self._fade_in_spin.setSuffix(" ms")
        form.addRow("Fade in", self._fade_in_spin)

        # Fade out
        self._fade_out_spin = QDoubleSpinBox()
        self._fade_out_spin.setRange(0.0, 5_000.0)
        self._fade_out_spin.setSingleStep(10.0)
        self._fade_out_spin.setValue(0.0)
        self._fade_out_spin.setSuffix(" ms")
        form.addRow("Fade out", self._fade_out_spin)

        # DC removal checkbox
        self._dc_check = QCheckBox("Remove DC offset")
        self._dc_check.setStyleSheet("color: #d8dce6; font-size: 12px;")
        form.addRow("", self._dc_check)

        # Normalise checkbox
        self._norm_check = QCheckBox("Normalise to −0.1 dBFS")
        self._norm_check.setStyleSheet("color: #d8dce6; font-size: 12px;")
        form.addRow("", self._norm_check)

        layout.addWidget(group)

        # ── Apply button ──────────────────────────────────────────────────────
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(167,139,250,0.10);"
            "  color: #a78bfa;"
            "  border: 1px solid rgba(167,139,250,0.4);"
            "  border-radius: 6px;"
            "  padding: 8px 16px;"
            "  font-size: 13px;"
            "  font-family: 'IBM Plex Mono';"
            "}"
            "QPushButton:hover { background: rgba(167,139,250,0.2); }"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )
        layout.addWidget(self._apply_btn)

        # ── Legend ────────────────────────────────────────────────────────────
        legend = QLabel("■ Original   ■ Processed")
        legend.setStyleSheet(
            "color: #5a6175; font-size: 11px; font-family: 'IBM Plex Mono';"
        )
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(legend)

        layout.addStretch()

        # ── Internal connection ───────────────────────────────────────────────
        self._apply_btn.clicked.connect(self._on_apply)

    def _on_apply(self) -> None:
        """Emit dsp_requested with current control values."""
        self.dsp_requested.emit(
            self._gain_spin.value(),
            self._fade_in_spin.value(),
            self._fade_out_spin.value(),
            self._dc_check.isChecked(),
            self._norm_check.isChecked(),
        )

    def set_file_loaded(self, loaded: bool) -> None:
        """Enable or disable the Apply button depending on whether a file is open.

        Args:
            loaded: True when an audio file has been successfully loaded.
        """
        self._apply_btn.setEnabled(loaded)


# =============================================================================
# STEP 7 — TransportBar: play/pause/stop controls and time display
# =============================================================================


class TransportBar(QWidget):
    """Horizontal bar with playback controls, time display, and scrubber.

    Signals:
        play_requested: Play/Pause button clicked.
        stop_requested: Stop button clicked.
        seek_requested: Scrubber moved. Carries target frame index.
    """

    play_requested: Signal = Signal()
    stop_requested: Signal = Signal()
    seek_requested: Signal = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the transport bar UI."""
        super().__init__(parent)
        self.setFixedHeight(60)
        self._n_frames: int = 0
        self._samplerate: int = DEVICE_SAMPLE_RATE
        self._build_ui()

    def _build_ui(self) -> None:
        """Create all transport controls."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(10)

        # Play/Pause toggle button
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setEnabled(False)
        self._play_btn.setFixedWidth(90)
        self._play_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(79,156,249,0.12); color: #4f9cf9;"
            "  border: 1px solid rgba(79,156,249,0.4); border-radius: 6px;"
            "  padding: 6px 12px; font-size: 12px; font-family: 'IBM Plex Mono';"
            "}"
            "QPushButton:checked {"
            "  background: rgba(251,146,60,0.14); color: #fb923c;"
            "  border-color: rgba(251,146,60,0.5);"
            "}"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )

        # Stop button
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(79,156,249,0.06); color: #5a6175;"
            "  border: 1px solid #252b38; border-radius: 6px;"
            "  padding: 6px 12px; font-size: 12px; font-family: 'IBM Plex Mono';"
            "}"
            "QPushButton:hover { color: #d8dce6; border-color: #4f9cf9; }"
            "QPushButton:disabled { color: #3a3f50; border-color: #252b38; }"
        )

        # Time display
        self._time_label = QLabel("00:00.000 / 00:00.000")
        self._time_label.setStyleSheet(
            "color: #4f9cf9; font-family: 'IBM Plex Mono'; font-size: 12px;"
        )
        self._time_label.setFixedWidth(210)

        # Scrubber (timeline slider)
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, 1000)
        self._scrubber.setValue(0)
        self._scrubber.setEnabled(False)

        layout.addWidget(self._play_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._time_label)
        layout.addWidget(self._scrubber, stretch=1)

        # Internal connections
        self._play_btn.toggled.connect(self._on_play_toggled)
        self._stop_btn.clicked.connect(self.stop_requested)
        self._scrubber.sliderMoved.connect(self._on_scrubber_moved)

    # ── Internal slots ─────────────────────────────────────────────────────────

    def _on_play_toggled(self, checked: bool) -> None:
        self._play_btn.setText("⏸  Pause" if checked else "▶  Play")
        self.play_requested.emit()

    def _on_scrubber_moved(self, value: int) -> None:
        """Convert scrubber position (0–1000) to a frame index and emit."""
        if self._n_frames > 0:
            frame = int(value / 1_000 * self._n_frames)
            self.seek_requested.emit(frame)

    # ── Public interface ───────────────────────────────────────────────────────

    def set_file_loaded(self, n_frames: int, samplerate: int) -> None:
        """Enable controls when a file is loaded.

        Args:
            n_frames: Total frames in the loaded file.
            samplerate: Sample rate of the loaded file (Hz).
        """
        self._n_frames = n_frames
        self._samplerate = samplerate
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._scrubber.setEnabled(True)
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")

    def update_position(self, frame: int) -> None:
        """Update the time display and scrubber position.

        Args:
            frame: Current playback frame position.
        """
        if self._n_frames == 0:
            return

        # Format current and total time as MM:SS.mmm
        def fmt(seconds: float) -> str:
            m = int(seconds // 60)
            s = seconds - m * 60
            return f"{m:02d}:{s:06.3f}"

        current_s = frame / self._samplerate
        total_s = self._n_frames / self._samplerate
        self._time_label.setText(f"{fmt(current_s)} / {fmt(total_s)}")

        # Update scrubber (block signals to avoid triggering seek_requested)
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(int(frame / self._n_frames * 1_000))
        self._scrubber.blockSignals(False)

    def set_stopped(self) -> None:
        """Reset the play button to the stopped state."""
        self._play_btn.blockSignals(True)
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")
        self._play_btn.blockSignals(False)

    @property
    def is_playing(self) -> bool:
        """True when the Play button is in the pressed (playing) state."""
        return self._play_btn.isChecked()


# =============================================================================
# STEP 8 — MainWindow: assembly, File menu, signals, QTimer polling
# =============================================================================


class MainWindow(QMainWindow):
    """Application main window.

    Layout:
    ┌─────────────────────────────────────────────┬──────────────┐
    │  WaveformView (waveform canvas)             │  DspPanel    │
    ├─────────────────────────────────────────────┤              │
    │  TransportBar (play/stop/scrubber/time)     │              │
    └─────────────────────────────────────────────┴──────────────┘

    The QTimer at 30 fps drains the PlaybackEngine's frame_queue and
    finished_queue, updating the UI on the main thread. The audio callback
    never touches Qt — all communication goes through queue.Queue.
    """

    def __init__(self) -> None:
        """Initialise engine, panels, timer, and file menu."""
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._engine = PlaybackEngine()
        self._original_buffer: AudioBuffer | None = None
        self._processed_buffer: AudioBuffer | None = None

        # QTimer drives all UI updates from the audio thread's queues
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(DISPLAY_TIMER_MS)

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._poll_timer.start()

    # ── UI assembly ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create and arrange all panels."""
        self._waveform = WaveformView()
        self._transport = TransportBar()
        self._dsp = DspPanel()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._waveform, stretch=1)
        left_layout.addWidget(self._transport)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._dsp)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 260])

        self.setCentralWidget(splitter)

    def _build_menu(self) -> None:
        """Create the File menu."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = file_menu.addAction("Open Audio File…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect all signals between panels, engine, and timer."""
        # Transport → engine
        self._transport.play_requested.connect(self._on_play_requested)
        self._transport.stop_requested.connect(self._on_stop)
        self._transport.seek_requested.connect(self._on_seek)

        # Waveform canvas scrubbing → engine seek
        self._waveform.seek_requested.connect(self._on_seek)

        # DSP panel → apply chain
        self._dsp.dsp_requested.connect(self._on_apply_dsp)

        # 30 fps poll timer → drain audio queues
        self._poll_timer.timeout.connect(self._poll_engine)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_file(self) -> None:
        """Open a file dialog and load the selected audio file."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            SUPPORTED_FILTER,
        )
        if not path_str:
            return  # user cancelled

        try:
            buf = AudioBuffer.load(Path(path_str))
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return

        self._original_buffer = buf
        self._processed_buffer = None

        # Update the engine
        self._engine.stop()
        self._engine.load(buf)

        # Update the UI
        file_name = buf.path.name
        self.setWindowTitle(f"{file_name} — {WINDOW_TITLE}")
        self._waveform.set_buffer(buf, None)
        self._transport.set_file_loaded(buf.n_frames, buf.samplerate)
        self._transport.update_position(0)
        self._dsp.set_file_loaded(True)

    def _on_play_requested(self) -> None:
        """Toggle playback state."""
        if self._transport.is_playing:
            self._engine.play()
        else:
            self._engine.pause()

    def _on_stop(self) -> None:
        """Stop playback and reset to the beginning."""
        self._engine.stop()
        self._transport.set_stopped()
        self._transport.update_position(0)
        self._waveform.set_playhead(0)

    def _on_seek(self, frame: int) -> None:
        """Seek the engine and update the waveform playhead.

        Args:
            frame: Target frame index.
        """
        self._engine.seek(frame)
        self._waveform.set_playhead(frame)
        if self._original_buffer is not None:
            self._transport.update_position(frame)

    def _on_apply_dsp(
        self,
        gain_db: float,
        fade_in_ms: float,
        fade_out_ms: float,
        remove_dc: bool,
        do_normalise: bool,
    ) -> None:
        """Run the DSP chain on a background thread and swap the buffer.

        The heavy NumPy/SciPy computation runs on a worker thread so the
        audio callback (which runs on the PortAudio thread) is never starved
        by main-thread work.  The buffer swap itself is done back on the main
        thread via a QTimer single-shot, which is the only Qt-safe way to
        touch the engine from a non-Qt thread.

        Args:
            gain_db: Gain in dB.
            fade_in_ms: Fade-in duration in ms.
            fade_out_ms: Fade-out duration in ms.
            remove_dc: Apply DC removal filter.
            do_normalise: Apply peak normalisation.
        """
        if self._original_buffer is None:
            return

        # Disable button while processing to prevent double-clicks.
        self._dsp.set_file_loaded(False)

        # Snapshot the source buffer so the worker holds its own reference.
        src = self._original_buffer

        def _worker() -> None:
            processed = DspChain.apply_chain(
                src,
                gain_db,
                fade_in_ms,
                fade_out_ms,
                remove_dc,
                do_normalise,
            )
            # Hand result back to the main thread via a zero-delay timer.
            QTimer.singleShot(0, lambda: self._finish_dsp(processed))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_dsp(self, processed: "AudioBuffer") -> None:
        """Swap in the processed buffer (called on the main thread).

        Args:
            processed: The freshly computed AudioBuffer from the DSP worker.
        """
        self._processed_buffer = processed

        # Save position before load() resets it, then restore after.
        with self._engine._lock:
            saved_pos = self._engine._frame_pos
        was_playing = self._engine.is_playing

        self._engine.load(processed)

        with self._engine._lock:
            self._engine._frame_pos = min(saved_pos, processed.n_frames)

        if was_playing:
            self._engine.play()

        self._waveform.set_buffer(self._original_buffer, processed)
        self._dsp.set_file_loaded(True)

    def _poll_engine(self) -> None:
        """Drain the engine's queues and update the UI.

        Called at 30 fps by the QTimer. Reads the latest frame position from
        frame_queue and updates the transport and waveform playhead. Also
        checks finished_queue for end-of-file.
        """
        # Drain frame_queue — take the latest value, discard older ones
        latest_frame: int | None = None
        try:
            while True:
                latest_frame = self._engine.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_frame is not None:
            self._waveform.set_playhead(latest_frame)
            self._transport.update_position(latest_frame)

        # Check for end-of-file
        try:
            self._engine.finished_queue.get_nowait()
            # Playback reached end-of-file — reset the transport button
            self._transport.set_stopped()
        except queue.Empty:
            pass

    # ── Window lifecycle ───────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Stop the timer and release the audio device on close."""
        self._poll_timer.stop()
        self._engine.close()
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
# End of 03_mm_waveform_player.py
# Author   : Renato Fera P.
# LinkedIn : https://www.linkedin.com/in/renato-profeta/
# Course   : Multimediaprogrammierung — Chapter 03: PySide6 for Multimedia
# Institute: Institut für Medientechnik, Technische Universität Ilmenau
# =============================================================================
