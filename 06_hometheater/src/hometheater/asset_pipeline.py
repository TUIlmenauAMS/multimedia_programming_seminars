# =============================================================================
# asset_pipeline.py — Home Theater Demo, Chapter 05 Companion
# =============================================================================
#
# PURPOSE
# -------
# QSpatialSound (Qt Spatial Audio) cannot decode a multi-channel FLAC or split
# channels at runtime. Its `source` property accepts only a QUrl pointing to an
# already-authored, single-channel (or stereo) WAV/MP3 file. Therefore this
# module runs *before* the Qt application starts:
#
#   1. Reads the 6-channel 5.1 FLAC master (clip_audio_51.flac)
#   2. Splits it into six discrete mono WAV files
#   3. Derives stereo and mono downmix WAV files for the lower tiers
#   4. Caches all results under cache/<film-key>/ so re-processing only
#      happens when the source file actually changes
#
# This file has zero PySide6 dependency — it can be run and tested standalone:
#   uv run python -m hometheater.asset_pipeline
#
# COORDINATE / CHANNEL ORDER
# --------------------------
# The FLAC specification (https://xiph.org/flac/format.html §6.1) mandates
# a fixed channel assignment for 6-channel audio:
#
#   Index 0 — front-left   (FL)
#   Index 1 — front-right  (FR)
#   Index 2 — front-center (FC / C)
#   Index 3 — LFE (subwoofer)
#   Index 4 — back-left    (Ls / surround-left)
#   Index 5 — back-right   (Rs / surround-right)
#
# This is a FORMAT-LEVEL GUARANTEE, not a heuristic. Do not detect or guess
# the order at runtime — a silent swap of e.g. LFE↔Center is a common and
# hard-to-hear bug. The indices are named constants below; use them everywhere.
#
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# STEP 1: FLAC channel index constants
# These are the ONLY place the magic indices 0-5 appear. Everywhere else in
# this file uses these names, so a future format change requires editing one
# block, not hunting through the code.
# ---------------------------------------------------------------------------

CH_FL  = 0  # front-left
CH_FR  = 1  # front-right
CH_FC  = 2  # front-center
CH_LFE = 3  # low-frequency effects (subwoofer)
CH_LS  = 4  # surround-left  (back-left in FLAC spec)
CH_RS  = 5  # surround-right (back-right in FLAC spec)

EXPECTED_CHANNELS = 6  # a valid 5.1 FLAC must have exactly this many


# ---------------------------------------------------------------------------
# STEP 2: Output file name registry
# Single source of truth for the filenames audio_engine.py will reference.
# Stored as a plain dict so manifest.json can expose them to Three.js too.
# ---------------------------------------------------------------------------

OUTPUT_FILES = {
    # 5.1 tier — genuinely discrete channels from the FLAC master
    "fl":             "left.wav",
    "fr":             "right.wav",
    "fc":             "center.wav",
    "lfe":            "lfe.wav",
    "ls":             "surround_left.wav",
    "rs":             "surround_right.wav",
    # Stereo tier — ITU-R BS.775-1 passive downmix (derived)
    "stereo_l":       "downmix_stereo_left.wav",
    "stereo_r":       "downmix_stereo_right.wav",
    # Mono tier — fold of the stereo downmix (derived, simplified)
    "mono":           "downmix_mono.wav",
}

# Target peak level after normalization. 0.98 leaves a small headroom margin
# to avoid digital clipping on floating-point rounding at the DAC boundary.
PEAK_TARGET = 0.98

# WAV output subtype written by _write_mono_wav().
# Changing this constant automatically busts the cache (it is embedded in the
# cache directory name via _make_cache_key) so old float32 files are never
# reused after a format change.
WAV_SUBTYPE = "PCM_16"


# ---------------------------------------------------------------------------
# STEP 3: Cache key — keyed on filename + mtime hash so re-processing happens
# automatically if the user replaces the FLAC with a different version.
# ---------------------------------------------------------------------------

class CacheKey(NamedTuple):
    film_slug: str   # e.g. "clip_audio_51"
    mtime_hex: str   # 8-char hex of the file's mtime, stable and fast


def _make_cache_key(flac_path: Path) -> CacheKey:
    """
    Derive a stable, human-readable cache directory name from the source file.
    Using mtime (not a full MD5 of the content) keeps startup fast — a 500 MB
    FLAC would take seconds to hash in full. mtime changes whenever the file is
    replaced, which is the only case we need to detect.

    WAV_SUBTYPE is mixed into the hash so that changing the output format
    (e.g. FLOAT → PCM_16) produces a different cache directory name and forces
    a clean rebuild rather than reusing incompatible files from the old cache.
    """
    slug = flac_path.stem  # filename without extension
    mtime_int = int(flac_path.stat().st_mtime * 1000)  # milliseconds, integer
    # XOR with a hash of WAV_SUBTYPE so a subtype change shifts the key.
    subtype_hash = int(hashlib.md5(WAV_SUBTYPE.encode()).hexdigest()[:8], 16)
    mtime_hex = format((mtime_int ^ subtype_hash) & 0xFFFFFFFF, "08x")
    return CacheKey(film_slug=slug, mtime_hex=mtime_hex)


def _cache_dir(cache_root: Path, key: CacheKey) -> Path:
    """Return the directory path for this film+mtime combination."""
    return cache_root / f"{key.film_slug}_{key.mtime_hex}"


def _cache_is_complete(cache_dir: Path) -> bool:
    """
    A cache is complete only if ALL expected output files exist.
    A partial cache (e.g. from a previous interrupted run) is treated as
    missing — better to re-process cleanly than to play silence from a
    half-written WAV.
    """
    return all((cache_dir / fname).exists() for fname in OUTPUT_FILES.values())


# ---------------------------------------------------------------------------
# STEP 4: Core audio helpers
# ---------------------------------------------------------------------------

def _peak_normalize(audio: np.ndarray) -> np.ndarray:
    """
    CONCEPT: Gain staging after a mix.

    When multiple channels are summed (downmix), their peaks add. A naive
    fold-down of a 5.1 master can easily produce sample values > 1.0, which
    clip at the DAC output or in Qt's mixer. This function scans the entire
    array for the absolute peak and scales the whole signal so that peak
    lands at PEAK_TARGET (0.98).

    This is intentionally a separate, named function — gain staging is a
    distinct, teachable concept, not a detail to bury inside a larger routine.

    Args:
        audio: float32 or float64 numpy array, any shape.

    Returns:
        Scaled copy of the input array. Does NOT modify in place.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        # Guard against silence / near-silence — avoid division by zero.
        return audio.copy()
    return (audio * (PEAK_TARGET / peak)).astype(audio.dtype)


def _write_mono_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    """
    Write a single-channel 16-bit integer PCM WAV.

    IMPORTANT: We write PCM_16 (signed 16-bit integer), NOT float32.
    Qt's QSpatialSound backend on Windows uses WASAPI, which requires integer
    PCM. Float32 WAV (subtype "FLOAT", format code 3) causes WASAPI to silently
    refuse to open the source — play() appears to succeed but produces no audio.
    PCM_16 is universally supported across all Qt audio backends (Windows
    WASAPI, macOS CoreAudio, Linux ALSA/PulseAudio) and is the safe default.

    The quantisation step from float32 → int16 is inaudible at normal listening
    levels: 16-bit gives 96 dB of dynamic range, well above the ~60–70 dB
    range of a typical film mix. The _peak_normalize() step applied to downmix
    channels before writing ensures samples stay within [−1.0, 1.0] so no
    clipping occurs during the int16 conversion.
    """
    # Ensure exactly 1-D (mono) before writing.
    if samples.ndim != 1:
        raise ValueError(
            f"_write_mono_wav: expected 1-D array, got shape {samples.shape}"
        )
    sf.write(str(path), samples.astype(np.float32), samplerate, subtype=WAV_SUBTYPE)


# ---------------------------------------------------------------------------
# STEP 5: Downmix formulas — documented with citation
# ---------------------------------------------------------------------------

def _stereo_downmix(
    fl: np.ndarray,
    fr: np.ndarray,
    fc: np.ndarray,
    ls: np.ndarray,
    rs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CONCEPT: ITU-R BS.775-1 passive downmix matrix.

    This is the same technique a real AV receiver or Dolby Pro Logic encoder
    applies when a 5.1 source must be reproduced on a two-channel system.
    The matrix coefficients are taken verbatim from ITU-R BS.775-1 §4.3:

        Lo = FL + 0.707 * FC + 0.707 * Ls
        Ro = FR + 0.707 * FC + 0.707 * Rs

    The factor 0.707 ≈ 1/√2 ≈ −3 dB. It is not arbitrary — it preserves
    perceived loudness when a mono-compatible signal (C, Ls, Rs) is summed
    into two channels. At full gain (1.0) the center channel would punch
    through the mix by +6 dB relative to L/R; −3 dB is the standard
    broadcast compromise.

    LFE is intentionally excluded from the fold-down. Sub-bass content
    at normal monitoring levels would dominate a 2-ch fold; real broadcast
    downmixers also omit LFE (or attenuate by −10 dB or more) for the same
    reason. We omit it entirely here — the student hears the LFE disappear
    when switching to Stereo tier, which is itself a teaching moment.

    Reference: ITU-R BS.775-1 (1994), "Multichannel stereophonic sound system
    with and without accompanying picture", Section 4.3 — Downmix equations.

    Args:
        fl, fr, fc, ls, rs: 1-D float arrays, equal length, same samplerate.
            LFE is not a parameter — it is deliberately excluded.

    Returns:
        (lo, ro): stereo downmix pair, NOT yet normalized (call
            _peak_normalize on each before writing).
    """
    C = 0.707  # ITU-R BS.775-1 downmix coefficient (≈ 1/√2)
    lo = fl + C * fc + C * ls
    ro = fr + C * fc + C * rs
    return lo, ro


def _mono_downmix(lo: np.ndarray, ro: np.ndarray) -> np.ndarray:
    """
    CONCEPT: Mono fold of a stereo pair.

    IMPORTANT — documented simplification:
    This is NOT the exact broadcast-spec mono matrix (which would apply
    separate coefficients to FL, FR, FC, Ls, Rs individually). Instead it
    folds the already-computed stereo downmix pair with equal weighting:

        M = 0.7071 * (Lo + Ro)

    The 0.7071 (≈ 1/√2) factor compensates for the +6 dB sum of two equal
    signals in an in-phase mono fold — without it, the mono output would be
    6 dB louder than the stereo pair, which is perceptually confusing when
    switching tiers.

    This simplification is acceptable for the demo's teaching purpose
    (demonstrating the transition from full spatial imaging to a point
    source), but should not be presented to students as the industry-standard
    mono downmix matrix without this caveat.

    Args:
        lo, ro: stereo downmix pair from _stereo_downmix(), equal length.

    Returns:
        1-D float array: mono signal, NOT yet normalized.
    """
    return 0.7071 * (lo + ro)


# ---------------------------------------------------------------------------
# STEP 6: Main pipeline entry point
# ---------------------------------------------------------------------------

class PipelineResult(NamedTuple):
    """Returned by run_pipeline() on success."""
    cache_dir: Path          # directory containing all output WAVs
    file_map: dict[str, Path]  # key → absolute Path, e.g. "fc" → .../center.wav
    was_cached: bool         # True if the cache already existed, False if rebuilt


def run_pipeline(
    flac_path: Path,
    cache_root: Path,
    force: bool = False,
) -> PipelineResult:
    """
    Top-level asset pipeline. Call this once at application startup before
    constructing QSpatialSound instances.

    Args:
        flac_path:   Path to the 6-channel 5.1 FLAC master (clip_audio_51.flac).
        cache_root:  Path to the cache/ directory (created if absent).
        force:       If True, re-process even if a complete cache exists.
                     Useful during development; not exposed in the production UI.

    Returns:
        PipelineResult with the cache directory and a map from logical key
        (e.g. "fc", "stereo_l") to the absolute Path of the WAV file.

    Raises:
        FileNotFoundError:  if flac_path does not exist.
        ValueError:         if the FLAC does not have exactly 6 channels.
        RuntimeError:       if any output file cannot be written.
    """

    # -- Validate input existence -------------------------------------------
    if not flac_path.exists():
        raise FileNotFoundError(
            f"5.1 audio master not found: {flac_path}\n"
            f"Expected location: assets/media/{flac_path.name}\n"
            f"Download from: https://media.xiph.org/tearsofsteel/tearsofsteel-surround.flac\n"
            f"Rename to: clip_audio_51.flac"
        )

    # -- Determine cache location -------------------------------------------
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _make_cache_key(flac_path)
    out_dir = _cache_dir(cache_root, key)

    # -- Cache hit: return immediately without re-reading the FLAC ----------
    if not force and out_dir.exists() and _cache_is_complete(out_dir):
        file_map = {k: out_dir / fname for k, fname in OUTPUT_FILES.items()}
        return PipelineResult(cache_dir=out_dir, file_map=file_map, was_cached=True)

    # -- Cache miss or forced rebuild: process the FLAC ---------------------
    print(f"[asset_pipeline] Processing {flac_path.name} → {out_dir.name}/")
    t0 = time.perf_counter()

    out_dir.mkdir(parents=True, exist_ok=True)

    # STEP 6a: Read the full 6-channel FLAC into memory.
    # soundfile returns (samples: ndarray shape [N, channels], samplerate: int).
    # We read as float32 to avoid an extra conversion step later.
    try:
        raw, sr = sf.read(str(flac_path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read FLAC file: {flac_path}\n"
            f"soundfile error: {exc}"
        ) from exc

    # STEP 6b: Validate channel count.
    # A 5.1 source MUST have exactly 6 channels. Anything else (stereo master,
    # incorrectly labelled file) means the channel-index constants above are
    # wrong and we would silently assign the wrong audio to the wrong speaker.
    actual_ch = raw.shape[1]
    if actual_ch != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected a {EXPECTED_CHANNELS}-channel (5.1) FLAC, "
            f"but {flac_path.name} has {actual_ch} channel(s).\n"
            f"If you have a stereo or mono master, the asset pipeline cannot "
            f"derive discrete 5.1 channels from it — a true discrete 5.1 "
            f"source file is required.\n"
            f"Recommended source: https://media.xiph.org/tearsofsteel/"
            f"tearsofsteel-surround.flac"
        )

    print(f"[asset_pipeline]   {raw.shape[0]/sr:.1f}s, {sr} Hz, "
          f"{raw.shape[0]} frames, {actual_ch} ch")

    # STEP 6c: Extract the six discrete channels by fixed FLAC index.
    # These are genuine discrete channels from the master — not synthesized.
    fl  = raw[:, CH_FL]
    fr  = raw[:, CH_FR]
    fc  = raw[:, CH_FC]
    lfe = raw[:, CH_LFE]
    ls  = raw[:, CH_LS]
    rs  = raw[:, CH_RS]

    # STEP 6d: Write the six discrete WAVs (5.1 tier sources).
    # No normalization here — these are the master channels, untouched.
    # Peak normalization is only applied to derived downmixes (see §4.3).
    discrete_map = {
        "fl": fl, "fr": fr, "fc": fc,
        "lfe": lfe, "ls": ls, "rs": rs,
    }
    for key_name, samples in discrete_map.items():
        out_path = out_dir / OUTPUT_FILES[key_name]
        _write_mono_wav(out_path, samples, sr)
        print(f"[asset_pipeline]   wrote {out_path.name}")

    # STEP 6e: Compute ITU-R BS.775-1 stereo downmix.
    # See _stereo_downmix() docstring for the exact matrix and citation.
    lo_raw, ro_raw = _stereo_downmix(fl, fr, fc, ls, rs)

    # Normalize each channel of the stereo pair independently — this preserves
    # the left/right balance of the mix while preventing clipping.
    lo = _peak_normalize(lo_raw)
    ro = _peak_normalize(ro_raw)

    _write_mono_wav(out_dir / OUTPUT_FILES["stereo_l"], lo, sr)
    _write_mono_wav(out_dir / OUTPUT_FILES["stereo_r"], ro, sr)
    print(f"[asset_pipeline]   wrote {OUTPUT_FILES['stereo_l']}, "
          f"{OUTPUT_FILES['stereo_r']}")

    # STEP 6f: Fold stereo downmix to mono.
    # See _mono_downmix() docstring for the simplification caveat.
    mono_raw = _mono_downmix(lo, ro)
    mono = _peak_normalize(mono_raw)

    _write_mono_wav(out_dir / OUTPUT_FILES["mono"], mono, sr)
    print(f"[asset_pipeline]   wrote {OUTPUT_FILES['mono']}")

    # STEP 6g: Write a small metadata sidecar for diagnostics and the README.
    meta = {
        "source_file":    flac_path.name,
        "source_mtime":   flac_path.stat().st_mtime,
        "samplerate_hz":  sr,
        "duration_s":     raw.shape[0] / sr,
        "frames":         raw.shape[0],
        "channels_in":    actual_ch,
        "outputs":        OUTPUT_FILES,
    }
    meta_path = out_dir / "pipeline_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    elapsed = time.perf_counter() - t0
    print(f"[asset_pipeline] Done in {elapsed:.1f}s — "
          f"cache: {out_dir.relative_to(cache_root.parent)}")

    file_map = {k: out_dir / fname for k, fname in OUTPUT_FILES.items()}
    return PipelineResult(cache_dir=out_dir, file_map=file_map, was_cached=False)


# ---------------------------------------------------------------------------
# STEP 7: Standalone test / manual invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run the pipeline standalone, without starting the Qt application.
    Useful for pre-processing assets before the first full launch, or for
    verifying that the FLAC is a valid discrete 5.1 source.

    Usage:
        uv run python -m hometheater.asset_pipeline
    Assumes the project is run from the hometheater-demo/ root directory.
    """
    import sys

    project_root = Path(__file__).resolve().parents[2]  # …/hometheater-demo/
    flac = project_root / "assets" / "media" / "clip_audio_51.flac"
    cache = project_root / "cache"

    print(f"Project root : {project_root}")
    print(f"FLAC source  : {flac}")
    print(f"Cache dir    : {cache}")
    print()

    try:
        result = run_pipeline(flac, cache)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"\n[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    if result.was_cached:
        print("Cache hit — all output files already exist:")
    else:
        print("Processing complete. Output files:")

    for key_name, path in result.file_map.items():
        size_kb = path.stat().st_size // 1024
        print(f"  {key_name:12s}  {path.name}  ({size_kb} KB)")
