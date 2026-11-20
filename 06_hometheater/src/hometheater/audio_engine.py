# =============================================================================
# audio_engine.py — Home Theater Demo, Chapter 05 Companion
# =============================================================================
#
# PURPOSE
# -------
# Owns the entire Qt Spatial Audio graph for the application:
#
#   QAudioEngine          — the binaural renderer (always Headphone mode)
#     └─ QAudioListener   — the virtual listener; pose driven per-frame by
#                           the Three.js camera via bridge.py
#     └─ QAudioRoom       — room geometry + per-wall absorption materials
#     └─ QSpatialSound×9  — one permanent source per unique audio file,
#                           placed at its ITU-R BS.775 speaker position
#
# This module has no knowledge of the UI or the web frontend. It exposes a
# clean Python API that bridge.py calls:
#
#   engine.set_tier(Tier.FIVE_ONE)
#   engine.set_room(RoomVariant.TREATED)
#   engine.play() / pause() / restart()
#   engine.update_listener_pose(x, y, z, qx, qy, qz, qw)
#
# NATIVE ENGINE EQUIVALENT (for students coming from the web demos)
# ----------------------------------------------------------------
#   QAudioEngine(OutputMode.Headphone) ↔ Resonance Audio's binaural decode step
#   QAudioRoom                         ↔ resonanceAudio.setRoomProperties()
#   QSpatialSound.position             ↔ resonanceSource.setFromMatrix()
#   QAudioListener.position/rotation   ↔ resonanceAudio.setListenerFromMatrix()
#
# COORDINATE SYSTEM
# -----------------
# Meters, Y-up, right-handed. This matches Three.js exactly (confirmed in
# scene.js with the same comment block). Positive X = right, positive Y = up,
# positive Z = toward the listener (away from the TV). 0° azimuth = −Z
# direction = facing the TV. All speaker positions below are in this space.
#
#   QAudioEngine.distanceScale = DistanceScaleMeter ensures Qt interprets
#   position values as meters, not the engine's default centimetres.
#
# AUDIO/VIDEO SYNC DESIGN
# -----------------------
# Two constraints drive the transport design:
#
#   1. QSpatialSound.setSource() resets the playback cursor to 0:00.
#      The original 6-source design called setSource() on every tier switch,
#      causing audio to restart from 0:00 while video resumed from the pause
#      position. Fixed by loading all 9 sources once at initialise() time and
#      never calling setSource() again.
#
#   2. QSpatialSound.play() / pause() must never be called after the initial
#      src.play() on each source. These per-source transport calls do not
#      guarantee resume-from-position behaviour across all Qt backends.
#
# Fix — 9 permanent sources, engine-level transport:
#
#   Sources are started ONCE via src.play() on the first call to play() and
#   run continuously thereafter. Per-source pause/stop are never called again
#   (except by restart(), which intentionally resets everything to 0:00).
#
#   Pause/resume use QAudioEngine.pause() / QAudioEngine.resume(), which freeze
#   and unfreeze the entire renderer atomically. All 9 source cursors stop and
#   start together, in lockstep with the video. No drift is possible.
#
#   "Tier switch" is 9 setVolume() calls; it works in both paused and playing
#   states. setVolume() while the engine is paused just sets a property — no
#   audio leaks through, and the correct volumes are already in place when
#   resume() is called.
#
#   Concretely:
#     - First play()      → src.play() on all 9; engine.resume(); apply tier
#                           volumes. _started=True.
#     - Subsequent play() → engine.resume(); _apply_tier(self._tier).
#                           No src.play() ever again.
#     - pause()           → engine.pause(). No setVolume() calls.
#     - set_tier()        → update self._tier; _apply_tier(). Works in both
#                           paused and playing states — setVolume() is safe
#                           while the engine is paused.
#     - restart()         → src.stop() on all 9; _started=False; call play().
#
# Three speaker positions (Center, FL, FR) host two sources each. Only one of
# each pair is ever at volume 1.0 at a time, so there is no bleed or
# comb-filtering artefact between the shared-position pairs.
#
# =============================================================================

from __future__ import annotations

import enum
import math
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtSpatialAudio import (
    QAudioEngine,
    QAudioListener,
    QAudioRoom,
    QSpatialSound,
)
from PySide6.QtGui import QQuaternion, QVector3D


# ---------------------------------------------------------------------------
# STEP 1: Public enumerations — Tier and RoomVariant
# These are the only values the bridge and UI ever pass in. String names are
# used over the wire (JSON-safe) and mapped to these enums defensively.
# ---------------------------------------------------------------------------

class Tier(enum.Enum):
    """
    The three historical surround formats this demo teaches.

    MONO     — single point source at the center speaker position
    STEREO   — left + right only, ITU-R BS.775 passive downmix
    FIVE_ONE — all six discrete channels, genuine spatial audio
    """
    MONO     = "mono"
    STEREO   = "stereo"
    FIVE_ONE = "5.1"


class RoomVariant(enum.Enum):
    """
    Two acoustic environments demonstrating the effect of room treatment.

    TREATED — heavy absorption: bass traps, acoustic panels, curtains
    WET     — bare reflective surfaces: brick, plaster, concrete
    """
    TREATED = "treated"
    WET     = "wet"


# ---------------------------------------------------------------------------
# STEP 2: ITU-R BS.775 virtual speaker layout
# ---------------------------------------------------------------------------
#
# CONCEPT: The ITU-R BS.775 standard defines the loudspeaker angles for 5.1
# reproduction. All five positional speakers sit on a circle of equal radius
# around the listener ("sweet spot"). Equal radius is mandatory for correct
# inter-aural time difference (ITD) imaging — if L were closer than Ls, the
# brain would use the arrival time difference as a spurious distance cue that
# the room geometry doesn't support.
#
# Angles measured from 0° = directly ahead (−Z axis, toward the TV):
#   C    0°    front-center
#   L  −30°    front-left
#   R  +30°    front-right
#   Ls −110°   surround-left  (behind and to the left)
#   Rs +110°   surround-right (behind and to the right)
#
# LFE is non-directional: the human auditory system cannot localise
# sub-bass below ~80 Hz. Its QSpatialSound source is placed near the TV for
# visual plausibility but its acoustic role is purely omnidirectional. This is
# NOT a bug in the code — it reflects the standard's intent.

# Sweet-spot coordinates (meters). This is the geometric center of the
# five-speaker circle, at seated head height. scene.js uses the identical
# constant (SWEET_SPOT) as the camera start position — single source of truth.
SWEET_SPOT = QVector3D(0.0, 1.2, 0.8)  # must match SWEET_SPOT_ORIGIN in scene.js

# Radius of the ITU calibration circle in meters.
# All five positional speakers are placed at this distance from SWEET_SPOT.
ITU_RADIUS = 2.0

# Speaker height — ear level at the sweet spot (same as SWEET_SPOT.y).
# In a real home theater the tweeters are aimed at this height.
SPEAKER_HEIGHT = 1.0


def _itu_position(azimuth_deg: float) -> QVector3D:
    """
    Convert an ITU-R BS.775 azimuth angle to a world-space position.

    Azimuth convention: 0° = straight ahead (−Z), positive = clockwise
    when viewed from above (i.e. +30° is to the right, matching the
    standard's notation where R is at +30°).

    Math: a speaker at azimuth θ, radius r, from sweet spot S is at:
        x = S.x + r * sin(θ)
        z = S.z − r * cos(θ)   ← −cos because 0° maps to −Z, not +Z

    This is the same trigonometry scene.js uses to place the visual
    speaker meshes — the two coordinate systems are identical.
    """
    az = math.radians(azimuth_deg)
    x = SWEET_SPOT.x() + ITU_RADIUS * math.sin(az)
    z = SWEET_SPOT.z() - ITU_RADIUS * math.cos(az)
    return QVector3D(x, SPEAKER_HEIGHT, z)


# LFE placed near the TV on the floor for visual sense only.
_LFE_POSITION = QVector3D(0.5, 0.0, -2.8)


# ---------------------------------------------------------------------------
# STEP 3: Room material tables
# See Section 5.4 of the build plan. QAudioRoom.Material names mirror the
# Resonance Audio material vocabulary already taught in the web demos
# (Section 2 of the theory HTML). Using the same names makes the NATIVE ENGINE
# EQUIVALENT mapping explicit and legible.
#
# IMPORTANT: Qt has no literal "Carpet" preset. The closest absorptive preset
# is Transparent, but that is effectively no surface. We use
# FiberGlassInsulation for the floor in TREATED mode as the most absorptive
# available option, and document this gap explicitly rather than silently
# picking the nearest match.
# ---------------------------------------------------------------------------

# Type alias for the per-wall material mapping expected by _apply_room_materials.
# Key = wall name string, value = QAudioRoom.Material enum member.
_WallMaterials = dict  # {str: QAudioRoom.Material}

_ROOM_MATERIALS: dict[RoomVariant, _WallMaterials] = {
    RoomVariant.TREATED: {
        # Heavy absorption — a professionally treated home theater.
        # See Section 2 of the theory HTML for Sabine formula context.
        "left":    QAudioRoom.Material.CurtainHeavy,
        "right":   QAudioRoom.Material.CurtainHeavy,
        "front":   QAudioRoom.Material.FiberGlassInsulation,
        "back":    QAudioRoom.Material.FiberGlassInsulation,
        "ceiling": QAudioRoom.Material.AcousticCeilingTiles,
        # NOTE: Qt provides no Carpet preset. FiberGlassInsulation is the
        # most absorptive floor-plausible preset in the enum. This is a
        # documented gap — the absorption coefficient is higher than carpet
        # but the visual (scene.js) will still render a carpet texture.
        "floor":   QAudioRoom.Material.FiberGlassInsulation,
    },
    RoomVariant.WET: {
        # Bare reflective surfaces — an untreated room with hard walls.
        "left":    QAudioRoom.Material.BrickBare,
        "right":   QAudioRoom.Material.BrickBare,
        "front":   QAudioRoom.Material.PlasterSmooth,
        "back":    QAudioRoom.Material.PlasterSmooth,
        "ceiling": QAudioRoom.Material.PolishedConcreteOrTile,
        "floor":   QAudioRoom.Material.Marble,
    },
}

# Fine-tuning on top of the material presets.
# The presets do most of the work; these nudge the character, not the magnitude.
_ROOM_REVERB: dict[RoomVariant, dict] = {
    RoomVariant.TREATED: {
        "reflection_gain":   0.3,   # low reflections — panels kill early echoes
        "reverb_gain":       0.4,   # short, controlled tail
        "reverb_time":       0.3,   # ~300ms RT60: typical treated home theater
        "reverb_brightness": 0.4,   # dark tail — high freqs absorbed first
    },
    RoomVariant.WET: {
        "reflection_gain":   0.9,   # strong early reflections off hard walls
        "reverb_gain":       0.9,   # long, washy reverb
        "reverb_time":       1.2,   # ~1.2s RT60: lively untreated room
        "reverb_brightness": 0.8,   # bright tail — hard surfaces reflect highs
    },
}

# Room dimensions (meters): width × height × depth.
# MUST match the Three.js BoxGeometry dimensions in scene.js.
# Both files import this value from manifest.json — single source of truth.
ROOM_DIMENSIONS = QVector3D(6.0, 3.0, 7.0)


# ---------------------------------------------------------------------------
# STEP 4: Permanent source definitions — one entry per unique audio file
# ---------------------------------------------------------------------------
#
# The original design had 6 QSpatialSound objects that swapped files on tier
# changes via setSource(). setSource() resets the playback cursor to 0:00,
# causing audio/video desync after a tier switch.
#
# The new design has 9 permanent sources — one per unique audio file across all
# three tiers. setSource() is called exactly once per source at initialise()
# time and never again. Tier switching is handled entirely by setVolume().
#
# Source table layout:
#   index | file key  | ITU position | active in tier(s)
#   ------+-----------+--------------+-------------------
#     0   | mono      | Center  0°   | MONO
#     1   | stereo_l  | FL    −30°   | STEREO
#     2   | stereo_r  | FR    +30°   | STEREO
#     3   | fc        | Center  0°   | FIVE_ONE
#     4   | fl        | FL    −30°   | FIVE_ONE
#     5   | fr        | FR    +30°   | FIVE_ONE
#     6   | ls        | Ls  −110°    | FIVE_ONE
#     7   | rs        | Rs  +110°    | FIVE_ONE
#     8   | lfe       | LFE (floor)  | FIVE_ONE
#
# Center, FL, and FR each host two sources at the same spatial position.
# Only one is ever unmuted per tier, so there is no bleed between them.

# Each entry: (channel_key, azimuth_deg_or_None_for_lfe, display_name)
# azimuth=None signals "use _LFE_POSITION instead of _itu_position()".
_SOURCE_DEFS: list[tuple[str, float | None, str]] = [
    ("mono",     0.0,   "Center/mono"),    # 0 — MONO tier center
    ("stereo_l", -30.0, "FL/stereo-L"),    # 1 — STEREO tier front-left
    ("stereo_r",  30.0, "FR/stereo-R"),    # 2 — STEREO tier front-right
    ("fc",        0.0,  "Center/fc"),      # 3 — 5.1 tier center
    ("fl",       -30.0, "FL/fl"),          # 4 — 5.1 tier front-left
    ("fr",        30.0, "FR/fr"),          # 5 — 5.1 tier front-right
    ("ls",      -110.0, "Ls/ls"),          # 6 — 5.1 tier surround-left
    ("rs",       110.0, "Rs/rs"),          # 7 — 5.1 tier surround-right
    ("lfe",      None,  "LFE/lfe"),        # 8 — 5.1 tier LFE (floor pos)
]

# Tier → volume per source index (matches _SOURCE_DEFS order above).
# 1.0 = active and audible; 0.0 = running silently in lockstep.
# All 9 sources play continuously regardless of tier — only their volume
# changes. This guarantees every source's cursor stays in sync with the video.
_TIER_VOLUMES: dict[Tier, list[float]] = {
    #                  0     1     2     3     4     5     6     7     8
    #               mono  st_l  st_r    fc    fl    fr    ls    rs   lfe
    Tier.MONO:    [1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    Tier.STEREO:  [0.0,  1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    Tier.FIVE_ONE:[0.0,  0.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
}


# ---------------------------------------------------------------------------
# STEP 5: SpatialAudioEngine — the main class
# ---------------------------------------------------------------------------

class SpatialAudioEngine(QObject):
    """
    Encapsulates the complete Qt Spatial Audio graph for the home theater demo.

    Lifecycle:
        engine = SpatialAudioEngine(parent=main_window)
        engine.initialise(file_map)   # call after asset_pipeline completes
        # bridge wires up signals/slots
        engine.play()
        engine.set_tier(Tier.FIVE_ONE)
        engine.set_room(RoomVariant.WET)

    Signals emitted (consumed by bridge.py → forwarded to JS):
        ready()                        — engine is initialised, sources loaded
        error(message: str)            — user-readable error string
        state_changed(tier, room, playing: bool)  — after any state transition

    Audio/video sync guarantee:
        setSource() is called exactly once per QSpatialSound source, inside
        _init_sources(), and never again. src.play() is called exactly once per
        source on the first user-initiated play(), and never again. All
        subsequent pause/resume cycles go through QAudioEngine.pause() and
        QAudioEngine.resume(), which freeze and unfreeze the entire renderer
        atomically. Tier switching and mute/solo only call setVolume(), which
        is safe in both paused and playing states and never resets any cursor.
    """

    # -- Signals ------------------------------------------------------------
    # NATIVE ENGINE EQUIVALENT: these map to the QWebChannel signals described
    # in Section 6 of the build plan (engine_ready, asset_error, state_changed).

    ready          = Signal()
    error          = Signal(str)
    state_changed  = Signal(str, str, bool)   # tier_name, room_name, is_playing

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        # -- QAudioEngine -------------------------------------------------
        # NATIVE ENGINE EQUIVALENT:
        #   OutputMode.Headphone ↔ Resonance Audio's binaural decode step
        #
        # OutputMode.Headphone activates Qt's HRTF-based binaural renderer.
        # This is NEVER exposed as a user toggle — the entire premise of this
        # demo is headphone virtualisation. If the user has speakers connected,
        # they should put on headphones. The UI states this explicitly.
        self._engine = QAudioEngine(self)
        self._engine.setOutputMode(QAudioEngine.OutputMode.Headphone)

        # distanceScale tells Qt what one "unit" in position coordinates means.
        # QAudioEngine.DistanceScaleMeter is the correct constant to pass here:
        # it evaluates to 100.0 and instructs the engine to treat position values
        # as metres rather than its internal default of centimetres.
        # Using 1.0 directly would tell Qt that every position unit is 1 cm,
        # placing the virtual speakers 100× closer than their metre-unit
        # coordinates intend — the distance-attenuation model then reduces them
        # to near-silence. Use the named constant so the intent is explicit.
        self._engine.setDistanceScale(QAudioEngine.DistanceScaleMeter)

        # -- QAudioListener -----------------------------------------------
        # NATIVE ENGINE EQUIVALENT:
        #   QAudioListener ↔ resonanceAudio.setListenerFromMatrix()
        #
        # The listener is the virtual "head" in the scene. Its position and
        # orientation are updated every frame from the Three.js camera transform
        # via bridge.py → update_listener_pose(). It is never set by UI code
        # directly.
        self._listener = QAudioListener(self._engine)
        self._listener.setPosition(SWEET_SPOT)
        # Listener faces −Z (toward the TV) at startup, matching the camera.
        # The identity quaternion (no rotation) is (w=1, x=0, y=0, z=0).
        self._listener.setRotation(QQuaternion(1, 0, 0, 0))

        # -- QAudioRoom ---------------------------------------------------
        # NATIVE ENGINE EQUIVALENT:
        #   QAudioRoom ↔ resonanceAudio.setRoomProperties()
        #
        # The room defines early reflections and reverb tail. Its dimensions
        # MUST match the Three.js BoxGeometry in scene.js — both read from
        # manifest.json (single source of truth).
        self._room = QAudioRoom(self._engine)
        self._room.setDimensions(ROOM_DIMENSIONS)
        self._room.setPosition(QVector3D(0.0, ROOM_DIMENSIONS.y() / 2, 0.0))

        # -- QSpatialSound×9 ----------------------------------------------
        # Nine permanent source objects — one per unique audio file across
        # all three tiers. Constructed ONCE here and never destroyed or
        # recreated. setSource() is called once per source at initialise()
        # time and NEVER again after that.
        self._sources: list[QSpatialSound] = []
        self._source_names: list[str] = []  # parallel list for diagnostics

        # -- State --------------------------------------------------------
        # Default tier is MONO — the app starts silent (paused) with the mono
        # downmix routed through the center speaker only. The student hears
        # a single point source from straight ahead before exploring stereo and
        # 5.1. This also matches hud.js's initial _state.tier = "mono" so the
        # HUD and engine are always in sync from the first state_changed signal.
        self._tier       = Tier.MONO
        self._room_var   = RoomVariant.TREATED
        self._is_playing = False
        self._file_map: dict[str, Path] = {}  # set by initialise()
        self._initialised = False

        # _started tracks whether src.play() has ever been called on the sources.
        # src.play() is called exactly once per source on the first play() call.
        # After that, all transport goes through QAudioEngine.pause()/resume().
        # _started=False → first play() call invokes src.play() on all 9 sources
        # _started=True  → subsequent play() calls only call engine.resume() and
        #                   _apply_tier(); src.play() is never called again
        self._started = False

    # -----------------------------------------------------------------------
    # STEP 6: Initialisation — called once after asset_pipeline completes
    # -----------------------------------------------------------------------

    def initialise(self, file_map: dict[str, Path]) -> None:
        """
        Build the nine QSpatialSound sources and apply the default state.

        Must be called after asset_pipeline.run_pipeline() returns successfully.
        Emits ready() on success, error(message) on any failure.

        Args:
            file_map: dict mapping channel key → Path of the WAV file.
                      Produced by asset_pipeline.PipelineResult.file_map.
                      Must contain keys: mono, stereo_l, stereo_r,
                      fc, fl, fr, ls, rs, lfe.
        """
        self._file_map = file_map

        print("[audio_engine] initialise() file_map:", flush=True)
        for key, path in file_map.items():
            exists = path.exists()
            print(f"  [{key}] {'OK' if exists else 'MISSING'} {path}", flush=True)

        # Validate that every WAV the engine will ever need actually exists
        # before constructing any QSpatialSound object.
        all_keys_needed = {key for key, _, _ in _SOURCE_DEFS}
        missing_keys = all_keys_needed - set(file_map.keys())
        if missing_keys:
            self.error.emit(
                f"file_map is missing required channel keys: {sorted(missing_keys)}. "
                f"Present keys: {sorted(file_map.keys())}"
            )
            return

        missing_files = [
            str(file_map[k]) for k in all_keys_needed
            if not file_map[k].exists()
        ]
        if missing_files:
            print(f"[audio_engine] Missing WAV files: {missing_files}")
            msg = (
                "Asset pipeline output is incomplete. Missing WAV files:\n"
                + "\n".join(f"  {m}" for m in missing_files)
                + "\nRe-run asset_pipeline.py or delete the cache/ directory "
                "to trigger a rebuild."
            )
            self.error.emit(msg)
            return

        try:
            self._init_sources()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(
                f"Failed to initialise audio engine:\n{exc}\n\n"
                "If 'from PySide6.QtSpatialAudio import QAudioEngine' raised "
                "an ImportError, the platform's Qt Multimedia backend plugin "
                "is likely missing — this is an environment issue, not a code "
                "bug. See README.md for the fix."
            )
            return

        # engine.start() MUST come before _apply_room() and _apply_tier().
        # QSpatialSound.setSource() is silently ignored if the engine has not
        # been started yet — the source URL is dropped and play() will produce
        # no audio. Room materials and tier source URLs are only applied once
        # the engine is running and ready to accept them.
        print("[audio_engine] calling engine.start()...", flush=True)
        self._engine.start()
        print(f"[audio_engine] engine.start() returned, output mode={self._engine.outputMode()}", flush=True)

        # Immediately pause the engine so sources don't audibly play during
        # the loading overlay. Sources will be silent until the user presses Play.
        self._engine.pause()
        print("[audio_engine] engine paused after start — waiting for user play()", flush=True)

        try:
            self._apply_room(self._room_var)
            self._apply_tier(self._tier)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Failed to apply initial room/tier configuration:\n{exc}")
            return

        self._initialised = True
        print("[audio_engine] emitting ready()", flush=True)
        self.ready.emit()

    def _init_sources(self) -> None:
        """
        Construct the nine permanent QSpatialSound instances and load their files.

        Every source receives its WAV file here via setSource(), and setSource()
        is NEVER called again anywhere else in this class. This is the core of
        the audio/video sync design: QSpatialSound.setSource() resets the
        playback cursor to 0:00. By loading all files once at startup and
        switching tiers purely through setVolume(), no cursor is ever reset
        after playback begins.

        Sources that share a speaker position (mono/fc at Center, stereo_l/fl
        at FL, stereo_r/fr at FR) are placed at exactly the same QVector3D.
        Only one of each pair is ever unmuted at a time, so the student never
        hears both simultaneously.

        CONCEPT: Spatial audio sources are positioned in 3-D space. The engine
        computes the angle and distance from the listener to each source every
        audio frame and applies the appropriate HRTF filter. For the binaural
        virtualiser to work correctly, the sources must sit at the ITU-R BS.775
        angles around the sweet spot — not at the visual mesh positions (which
        are scaled GLB models placed for aesthetics).
        """
        print("[audio_engine] _init_sources(): constructing 9 QSpatialSound objects...", flush=True)

        for channel_key, azimuth, display_name in _SOURCE_DEFS:
            src = QSpatialSound(self._engine)

            # Position: LFE gets its special floor position; all others use
            # the ITU-R BS.775 circle at the sweet spot.
            if azimuth is None:
                pos = _LFE_POSITION
            else:
                pos = _itu_position(azimuth)
            src.setPosition(pos)

            src.setSize(0.0)
            src.setDistanceCutoff(10.0)
            src.setLoops(QSpatialSound.Loops.Infinite)
            src.setAutoPlay(False)

            # Load the WAV file. This is the ONLY call to setSource() for this
            # source — it will not be called again on tier switches.
            wav_path = self._file_map[channel_key]
            url = QUrl.fromLocalFile(str(wav_path))
            src.setSource(url)

            # Start fully muted. _apply_tier() sets the correct tier volumes
            # immediately after _init_sources() returns.
            src.setVolume(0.0)

            self._sources.append(src)
            self._source_names.append(display_name)

            print(
                f"  [source] {display_name}: "
                f"file={wav_path.name} "
                f"pos=({pos.x():.2f}, {pos.y():.2f}, {pos.z():.2f}) "
                f"exists={wav_path.exists()}",
                flush=True,
            )

        print(f"[audio_engine] _init_sources(): {len(self._sources)} sources created", flush=True)

    # -----------------------------------------------------------------------
    # STEP 7: Room switching
    # -----------------------------------------------------------------------

    def _apply_room(self, variant: RoomVariant) -> None:
        """
        Apply wall materials and reverb parameters for the given room variant.

        CONCEPT: QAudioRoom.Material maps directly to absorption coefficients
        in the same vocabulary as Resonance Audio's room-materials.js (already
        documented in Section 2 of the theory HTML). The visual (Three.js) wall
        materials are updated in the same bridge call that triggers this method —
        the plan's requirement that visual and acoustic properties change together
        is enforced at the call-site level (bridge.py set_room slot).

        NATIVE ENGINE EQUIVALENT:
            QAudioRoom.setWallMaterial() ↔ resonanceAudio.setRoomProperties()
        """
        mats = _ROOM_MATERIALS[variant]
        self._room.setWallMaterial(QAudioRoom.Wall.LeftWall,   mats["left"])
        self._room.setWallMaterial(QAudioRoom.Wall.RightWall,  mats["right"])
        self._room.setWallMaterial(QAudioRoom.Wall.FrontWall,  mats["front"])
        self._room.setWallMaterial(QAudioRoom.Wall.BackWall,   mats["back"])
        self._room.setWallMaterial(QAudioRoom.Wall.Floor,      mats["floor"])
        self._room.setWallMaterial(QAudioRoom.Wall.Ceiling,    mats["ceiling"])

        rev = _ROOM_REVERB[variant]
        self._room.setReflectionGain(rev["reflection_gain"])
        self._room.setReverbGain(rev["reverb_gain"])
        self._room.setReverbTime(rev["reverb_time"])
        self._room.setReverbBrightness(rev["reverb_brightness"])

    # -----------------------------------------------------------------------
    # STEP 8: Tier switching — volume only, never setSource()
    # -----------------------------------------------------------------------

    def _apply_tier(self, tier: Tier) -> None:
        """
        Reconfigure source volumes for the given tier.

        This method ONLY calls setVolume(). It never calls setSource() or
        src.play(). All nine sources were loaded at initialise() time and
        started once on the first play(). Switching tier makes the right sources
        audible (vol=1.0) and silences the others (vol=0.0). Because no source
        is ever re-loaded, no playback cursor is reset, and audio stays in sync
        with the video after every tier change.

        setVolume() is safe to call while the engine is paused — it sets a
        property on the source object and takes effect when the engine resumes.
        No audio leaks through a paused engine regardless of volume values.

        The volume list in _TIER_VOLUMES maps to source indices in _SOURCE_DEFS:
            [0]=mono  [1]=stereo_l  [2]=stereo_r
            [3]=fc    [4]=fl        [5]=fr
            [6]=ls    [7]=rs        [8]=lfe
        """
        volumes = _TIER_VOLUMES[tier]

        print(
            f"[audio_engine] _apply_tier({tier.value}): "
            f"setting volumes on {len(volumes)} sources",
            flush=True,
        )

        for i, volume in enumerate(volumes):
            src = self._sources[i]
            name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
            src.setVolume(volume)
            status = "ACTIVE" if volume > 0.0 else "muted"
            print(f"  [{i}] {name}: vol={volume:.1f} ({status})", flush=True)

    # -----------------------------------------------------------------------
    # STEP 9: Public transport controls
    # -----------------------------------------------------------------------

    @Slot()
    def play(self) -> None:
        """
        Start or resume playback.

        FIRST CALL (_started is False):
            Calls src.play() on all nine sources to begin continuous playback,
            then calls engine.resume() to unfreeze the renderer, then applies
            the active tier's volumes. After this, _started is True and
            src.play() is NEVER called again on any source.

        SUBSEQUENT CALLS (_started is True):
            Calls engine.resume() to unfreeze the renderer, then applies the
            active tier's volumes. Does NOT call src.play() — sources are
            already running inside the engine.

        All 9 sources are started on the first call regardless of their volume.
        Sources with vol=0.0 run silently — their cursors advance in lockstep
        with the active ones and with the video. This is what makes instantaneous
        tier switching possible: a source that was muted while two others played
        is at exactly the right position when it becomes the active source.
        """
        print(
            f"[audio_engine] play() called, _started={self._started}, "
            f"_is_playing={self._is_playing}",
            flush=True,
        )

        if not self._initialised:
            print("[audio_engine] play() ERROR: not initialised", flush=True)
            self.error.emit("play() called before initialise(). "
                            "Wait for engine_ready signal.")
            return

        if not self._started:
            # First play: call src.play() on every source to start continuous
            # playback inside the engine. All sources begin at 0:00 and advance
            # together once the engine resumes. Volumes are still 0.0 at this
            # point (set during _init_sources), so nothing is audible yet.
            print("[audio_engine] play(): first start — calling src.play() on all sources", flush=True)
            for i, src in enumerate(self._sources):
                name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
                src.play()
                print(f"  [play/start] {name}: src.play() called", flush=True)
            self._started = True
        else:
            print("[audio_engine] play(): resuming via engine.resume()", flush=True)

        # Unfreeze the engine. All 9 source cursors resume from exactly where
        # they stopped — no drift, no reset.
        self._engine.resume()

        # Apply the active tier's volumes. This is what makes the correct
        # channels audible. Called on both first-start and resume paths.
        self._is_playing = True
        self._apply_tier(self._tier)

        self._emit_state()
        print("[audio_engine] play() done, _is_playing=True", flush=True)

    @Slot()
    def pause(self) -> None:
        """
        Pause playback by freezing the entire audio engine.

        QAudioEngine.pause() suspends the renderer atomically — all 9 source
        cursors freeze at exactly the same moment. QAudioEngine.resume() in
        play() unfreezes them all together. No cursor drift is possible between
        pause and resume.

        While the engine is paused, setVolume() calls from mute/solo operations
        are safe and effective — they update source properties that will be
        applied when the engine resumes. No audio leaks through.
        """
        print("[audio_engine] pause() called", flush=True)

        if not self._started:
            print("[audio_engine] pause(): sources not yet started — no-op", flush=True)
            self._is_playing = False
            self._emit_state()
            return

        self._engine.pause()

        self._is_playing = False
        self._emit_state()
        print("[audio_engine] pause() done, _is_playing=False", flush=True)

    @Slot()
    def pause_all(self) -> None:
        """
        Pause all audio output cleanly.

        Public alias for pause(), called by MainWindow.closeEvent() to stop
        audible output before the Qt audio graph is torn down. Delegates
        entirely to pause() so all state bookkeeping stays in one place.
        """
        self.pause()

    @Slot()
    def restart(self) -> None:
        """
        Stop all sources (resets all cursors to 0:00) and restart from scratch.

        This deliberately resets every source to the beginning — it is intended
        for the explicit "restart clip" button, not for tier switching. After
        this call, all nine cursors are synchronised at 0:00, so the audio/video
        sync guarantee holds as long as the video is also reset to 0:00 by the
        caller (bridge.py is responsible for that).

        Implementation: src.stop() resets the cursor. _started is set to False
        so the next play() call takes the first-start path and calls src.play()
        on all sources again. This is the only code path that intentionally
        resets _started.
        """
        print("[audio_engine] restart() called — stopping all sources", flush=True)

        # Pause the engine first so stopping individual sources doesn't produce
        # audio glitches from the sources that haven't been stopped yet.
        self._engine.pause()

        for i, src in enumerate(self._sources):
            name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
            src.stop()
            print(f"  [restart] {name}: src.stop() called", flush=True)

        self._started = False
        self._is_playing = False

        # play() will take the first-start path, call src.play() on all 9,
        # call engine.resume(), and apply the active tier's volumes.
        self.play()

    # -----------------------------------------------------------------------
    # STEP 10: State-switching slots (called by bridge.py)
    # -----------------------------------------------------------------------

    @Slot(str)
    def set_tier(self, tier_name: str) -> None:
        """Switch to a new surround tier.

        Tier switching is a pure volume operation: _apply_tier() only calls
        setVolume() on the 9 sources. No source is re-loaded, no cursor is
        reset, and no engine pause/resume cycle is needed.

        setVolume() is safe in both paused and playing states. While paused,
        the new volumes are stored on the source objects and applied when the
        engine resumes. While playing, they take effect within the next audio
        buffer (~10ms) with no glitch and no cursor reset.
        """
        print(f"[audio_engine] set_tier('{tier_name}') called, _is_playing={self._is_playing}", flush=True)

        try:
            new_tier = Tier(tier_name)
        except ValueError:
            print(f"[audio_engine] set_tier() ERROR: unknown tier '{tier_name}'", flush=True)
            self.error.emit(
                f"Unknown tier '{tier_name}'. "
                f"Valid values: {[t.value for t in Tier]}"
            )
            return

        if new_tier == self._tier:
            print(f"[audio_engine] set_tier() no-op: already on {tier_name}", flush=True)
            return

        self._tier = new_tier
        self._apply_tier(self._tier)

        self._emit_state()
        print(f"[audio_engine] set_tier() done: now on {new_tier.value}", flush=True)

    @Slot(str)
    def set_room(self, room_name: str) -> None:
        """Switch room acoustic variant.

        Room material changes take effect immediately on the reverb tail and
        early reflections. No pause/resume is needed; the transition is
        inaudible as a click or glitch on all tested Qt backends.
        """
        print(f"[audio_engine] set_room('{room_name}') called, _is_playing={self._is_playing}", flush=True)

        try:
            new_room = RoomVariant(room_name)
        except ValueError:
            print(f"[audio_engine] set_room() ERROR: unknown room '{room_name}'", flush=True)
            self.error.emit(
                f"Unknown room variant '{room_name}'. "
                f"Valid values: {[r.value for r in RoomVariant]}"
            )
            return

        if new_room == self._room_var:
            print(f"[audio_engine] set_room() no-op: already on {room_name}", flush=True)
            return

        self._apply_room(new_room)
        self._room_var = new_room

        self._emit_state()
        print(f"[audio_engine] set_room() done: now on {new_room.value}", flush=True)

    @Slot(str, bool)
    def set_speaker_mute(self, channel_key: str, muted: bool) -> None:
        """
        Mute or unmute a single speaker channel by channel key.

        Called by bridge.py whenever the HUD's mute or solo state changes.
        The muted flag already encodes the full effective state computed by
        hud.js _applySoloState() — this method just applies it.

        setVolume() is safe while the engine is paused. The user can configure
        mute and solo freely during pause; all changes take effect correctly
        when playback resumes.

        Channel key → source index mapping (matches _SOURCE_DEFS):
            "mono"     → 0    "stereo_l" → 1    "stereo_r" → 2
            "fc"       → 3    "fl"       → 4    "fr"       → 5
            "ls"       → 6    "rs"       → 7    "lfe"      → 8
        """
        # Build the key→index map lazily on first use.
        if not hasattr(self, "_channel_key_to_index"):
            self._channel_key_to_index: dict[str, int] = {
                key: i for i, (key, _, _) in enumerate(_SOURCE_DEFS)
            }

        idx = self._channel_key_to_index.get(channel_key)
        if idx is None:
            print(f"[audio_engine] set_speaker_mute(): unknown channel key '{channel_key}'", flush=True)
            return

        # The effective volume respects both the tier (is this source active in
        # the current tier?) and the mute flag. An inactive-tier source stays
        # at 0.0 regardless of the mute flag.
        tier_volume = _TIER_VOLUMES[self._tier][idx]
        effective_volume = 0.0 if muted else tier_volume

        src = self._sources[idx]
        src.setVolume(effective_volume)

        name = self._source_names[idx]
        print(
            f"[audio_engine] set_speaker_mute('{channel_key}', {muted}): "
            f"{name} vol={effective_volume:.1f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # STEP 11: Per-frame listener pose update (called by bridge.py at ~60 Hz)
    # -----------------------------------------------------------------------

    _pose_call_count: int = 0  # class-level counter for throttled logging

    @Slot(float, float, float, float, float, float, float)
    def update_listener_pose(
        self,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ) -> None:
        """
        Update the listener's world-space position and orientation.

        Called every animation frame from the Three.js camera transform via
        bridge.py. The camera's world position and quaternion are passed
        directly — no smoothing or interpolation is applied here.

        CONCEPT: The binaural renderer recalculates the angle and distance
        from the listener to each source every audio buffer (~10ms). Updating
        the listener pose here feeds that calculation continuously, so moving
        off-axis is immediately audible as a change in the stereo image — not
        just a volume change, but a shift in the HRTF filter applied to each
        source. This is what makes the sweet-spot concept tangible.

        NATIVE ENGINE EQUIVALENT:
            QAudioListener.setPosition()  ↔ resonanceAudio.setListenerFromMatrix()
            QAudioListener.setRotation()  ↔ (same, orientation component)

        No smoothing/interpolation on the audio side. If motion looks jittery,
        fix it in the JS camera update (scene.js), not here — adding a filter
        here would desync visual facing from acoustic facing.
        """
        SpatialAudioEngine._pose_call_count += 1
        n = SpatialAudioEngine._pose_call_count
        if n == 1 or n % 300 == 0:
            print(f"[audio_engine] update_listener_pose #{n}: "
                  f"pos=({x:.3f}, {y:.3f}, {z:.3f}) "
                  f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})", flush=True)
        self._listener.setPosition(QVector3D(x, y, z))
        self._listener.setRotation(QQuaternion(qw, qx, qy, qz))

    # -----------------------------------------------------------------------
    # STEP 12: Internal helpers
    # -----------------------------------------------------------------------

    def _emit_state(self) -> None:
        """Emit state_changed with the current confirmed engine state."""
        print(f"[audio_engine] state_changed emitted: tier={self._tier.value} "
              f"room={self._room_var.value} is_playing={self._is_playing}", flush=True)
        self.state_changed.emit(
            self._tier.value,
            self._room_var.value,
            self._is_playing,
        )

    # -- Read-only state properties for bridge.py ---------------------------

    @property
    def tier(self) -> Tier:
        return self._tier

    @property
    def room_variant(self) -> RoomVariant:
        return self._room_var

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    @property
    def is_started(self) -> bool:
        """True once src.play() has been called on all sources."""
        return self._started
