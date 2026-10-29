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
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
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

    @classmethod
    def apply_chain(
        cls,
        buffer: AudioBuffer,
        gain_db: float,
        fade_in_ms: float,
        fade_out_ms: float,
        remove_dc: bool,
        do_normalise: bool,
    ) -> AudioBuffer:
        """Apply the full DSP chain to a copy of the buffer.

        The processing order matters:
          1. Gain — scale the amplitude first
          2. DC removal — filter on the gained signal
          3. Fade — apply envelope after filtering (avoids filter transients at edges)
          4. Normalise — always last, scales the final result

        Args:
            buffer: The source AudioBuffer. Never modified.
            gain_db: Gain in dB (−12 to +12).
            fade_in_ms: Fade-in duration in ms.
            fade_out_ms: Fade-out duration in ms.
            remove_dc: If True, apply the 20 Hz high-pass DC removal filter.
            do_normalise: If True, peak-normalise the result to −0.1 dBFS.

        Returns:
            A new AudioBuffer with the processed audio.
        """
        data = buffer.data.copy()

        data = cls.apply_gain(data, gain_db)

        if remove_dc:
            data = cls.remove_dc(data, buffer.samplerate)

        data = cls.apply_fade(data, fade_in_ms, fade_out_ms, buffer.samplerate)

        if do_normalise:
            data = cls.normalise(data)

        return AudioBuffer(data=data, samplerate=buffer.samplerate, path=buffer.path)


# =============================================================================
# STEP 5 — WaveformView: QPainter waveform canvas with playhead
# =============================================================================


class WaveformView(QWidget):
    """QPainter waveform canvas with peak-reduction rendering and a playhead.

    Displays two waveforms overlaid:
      - Original buffer: blue (--accent)
      - Processed buffer: purple (--accent2)

    The waveform is rendered with peak-reduction: for each pixel column,
    the minimum and maximum amplitudes of all audio samples it covers are
    found and drawn as a vertical line. This is the correct approach for
    all professional audio editors (Pro Tools, Logic, Reaper, Audacity).

    Clicking or dragging on the canvas emits ``seek_requested(int)`` with
    the target frame index, enabling scrubbing.

    Signals:
        seek_requested: Emitted on click or drag. Carries the target frame.
    """

    seek_requested: Signal = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise with empty buffers."""
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Peak-reduced display arrays — recomputed in set_buffer()
        self._orig_min: np.ndarray = np.array([])
        self._orig_max: np.ndarray = np.array([])
        self._proc_min: np.ndarray = np.array([])
        self._proc_max: np.ndarray = np.array([])

        self._n_frames: int = 0  # total audio frames (for playhead mapping)
        self._playhead_frame: int = 0  # current playhead position in frames
        self._has_processed: bool = False

    # ── Public interface ───────────────────────────────────────────────────────

    def set_buffer(
        self,
        original: AudioBuffer | None,
        processed: AudioBuffer | None = None,
    ) -> None:
        """Compute peak-reduced display data and repaint.

        DSP CONCEPT — Peak reduction:
        A 10-second stereo file at 48 000 Hz contains 480 000 frames.
        The canvas is perhaps 800 px wide. Drawing every sample would require
        drawing 600 points per pixel column — most of which are invisible
        because they overwrite each other. Worse, striding (taking every Nth
        sample) misses transients: a brief loud click between two quiet samples
        would be invisible if the stride skips it.

        Peak-reduction solves both problems: for each pixel column, find the
        minimum and maximum sample values across all frames it covers, then
        draw a vertical line between those two extremes. This is O(N) in the
        number of samples (one pass) and preserves every transient — no sample
        is ever skipped.

        This is how every professional DAW renders waveforms.

        Args:
            original: The source AudioBuffer (always provided when a file is open).
            processed: The DSP-processed AudioBuffer, or None if no DSP has been applied.
        """
        if original is None:
            self._orig_min = np.array([])
            self._orig_max = np.array([])
            self._proc_min = np.array([])
            self._proc_max = np.array([])
            self._n_frames = 0
            self._has_processed = False
            self.update()
            return

        self._n_frames = original.n_frames
        self._orig_min, self._orig_max = self._peak_reduce(original.mono())

        if processed is not None:
            self._proc_min, self._proc_max = self._peak_reduce(processed.mono())
            self._has_processed = True
        else:
            self._proc_min = np.array([])
            self._proc_max = np.array([])
            self._has_processed = False

        self.update()

    def set_playhead(self, frame: int) -> None:
        """Update the playhead position and repaint.

        Args:
            frame: Current playback position in audio frames.
        """
        self._playhead_frame = frame
        self.update()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _peak_reduce(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-pixel-column min and max from a mono audio array.

        Args:
            mono: 1-D float32 array of audio samples.

        Returns:
            Tuple of (peaks_min, peaks_max), each of length self.width().
        """
        n = len(mono)
        width = max(self.width(), 1)
        # Number of audio frames each pixel column covers
        frames_per_px = n / width

        peaks_min = np.empty(width, dtype=np.float32)
        peaks_max = np.empty(width, dtype=np.float32)

        for px in range(width):
            start = int(px * frames_per_px)
            end = int((px + 1) * frames_per_px)
            end = min(end, n)
            if start >= end:
                peaks_min[px] = 0.0
                peaks_max[px] = 0.0
            else:
                chunk = mono[start:end]
                peaks_min[px] = float(chunk.min())
                peaks_max[px] = float(chunk.max())

        return peaks_min, peaks_max

    def _frame_to_px(self, frame: int) -> int:
        """Convert a frame index to a pixel X position."""
        if self._n_frames == 0:
            return 0
        return int(frame / self._n_frames * self.width())

    def _px_to_frame(self, px: int) -> int:
        """Convert a pixel X position to a frame index."""
        if self._n_frames == 0:
            return 0
        frac = max(0.0, min(1.0, px / max(self.width(), 1)))
        return int(frac * self._n_frames)

    def _amp_to_py(self, amp: float, h: int) -> int:
        """Map amplitude [-1, 1] to pixel Y. Centre at h//2; +1 → top."""
        return int(h / 2.0 * (1.0 - amp))

    # ── Mouse interaction ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """Seek to the clicked position."""
        if self._n_frames > 0:
            self.seek_requested.emit(self._px_to_frame(event.pos().x()))

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """Seek while dragging (scrubbing)."""
        if self._n_frames > 0 and event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._px_to_frame(event.pos().x()))

    # ── QPainter rendering ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Draw background, zero line, waveforms, and the playhead."""
        painter = QPainter(self)
        w, h = self.width(), self.height()

        # ── Background ────────────────────────────────────────────────────────
        painter.fillRect(0, 0, w, h, QColor(CANVAS_BG))

        if len(self._orig_min) == 0:
            # No file loaded — draw a placeholder message
            painter.setPen(QPen(QColor(LABEL_COLOR)))
            painter.setFont(QFont("IBM Plex Mono", 11))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open an audio file to display the waveform",
            )
            painter.end()
            return

        # ── Zero (centre) line ────────────────────────────────────────────────
        cy = h // 2
        painter.setPen(QPen(QColor(ZERO_LINE_COLOR), 1))
        painter.drawLine(0, cy, w, cy)

        # ── Processed waveform (purple, behind original) ───────────────────
        if self._has_processed and len(self._proc_min) == w:
            proc_pen = QPen(QColor(PROCESSED_COLOR), 1)
            proc_pen.setColor(QColor(167, 139, 250, 140))  # accent2 with alpha
            painter.setPen(proc_pen)
            for px in range(w):
                py_min = self._amp_to_py(float(self._proc_min[px]), h)
                py_max = self._amp_to_py(float(self._proc_max[px]), h)
                painter.drawLine(px, py_min, px, py_max)

        # ── Original waveform (blue, on top) ──────────────────────────────
        if len(self._orig_min) == w:
            orig_pen = QPen(QColor(ORIGINAL_COLOR), 1)
            orig_pen.setColor(QColor(79, 156, 249, 200))  # accent with alpha
            painter.setPen(orig_pen)
            for px in range(w):
                py_min = self._amp_to_py(float(self._orig_min[px]), h)
                py_max = self._amp_to_py(float(self._orig_max[px]), h)
                painter.drawLine(px, py_min, px, py_max)

        # ── Playhead ──────────────────────────────────────────────────────────
        ph_px = self._frame_to_px(self._playhead_frame)
        painter.setPen(QPen(QColor(PLAYHEAD_COLOR), 2))
        painter.drawLine(ph_px, 0, ph_px, h)

        painter.end()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        """Recompute peak data at the new width after a resize."""
        # The peak arrays are width-dependent. After a resize, the previous
        # arrays have the wrong length. Trigger a recompute by re-calling
        # set_buffer — but we don't store the original AudioBuffer here.
        # Instead, we just call update() and let the paint use whatever we have.
        # A full recompute would require storing a reference to the AudioBuffer,
        # which Demo 5 does. For Demo 2 the simple approach is sufficient.
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
        self._apply_btn = QPushButton("Apply DSP")
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
        """Run the DSP chain and update the processed waveform overlay.

        Args:
            gain_db: Gain in dB.
            fade_in_ms: Fade-in duration in ms.
            fade_out_ms: Fade-out duration in ms.
            remove_dc: Apply DC removal filter.
            do_normalise: Apply peak normalisation.
        """
        if self._original_buffer is None:
            return

        self._processed_buffer = DspChain.apply_chain(
            self._original_buffer,
            gain_db,
            fade_in_ms,
            fade_out_ms,
            remove_dc,
            do_normalise,
        )
        self._waveform.set_buffer(self._original_buffer, self._processed_buffer)

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
