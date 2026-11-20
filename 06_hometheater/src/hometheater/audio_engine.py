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
#     └─ QSpatialSound×6  — one virtual speaker per ITU-R BS.775 position
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
# NOT a bug in the code — comment it explicitly so students don't mistake the
# placement for a directional design.

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


# Speaker descriptor: azimuth (degrees), display name, channel key
# (matches asset_pipeline.OUTPUT_FILES keys used to build WAV paths)
_SPEAKER_DEFS = [
    # (azimuth_deg,  name,              channel_key,  is_lfe)
    (   0.0,        "Center",           "fc",          False),
    ( -30.0,        "Front-Left",       "fl",          False),
    (  30.0,        "Front-Right",      "fr",          False),
    (-110.0,        "Surround-Left",    "ls",          False),
    ( 110.0,        "Surround-Right",   "rs",          False),
]
# LFE is defined separately because its position is non-standard.
# Placed near the TV on the floor for visual sense only.
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
# STEP 4: Tier → source/volume configuration table
# ---------------------------------------------------------------------------
#
# CONCEPT: Tier switching never destroys or recreates QSpatialSound objects.
# Qt audio graph objects are expensive to construct and their construction
# can produce audible glitches. Instead, all six sources exist permanently;
# tier switching only changes:
#   (a) which WAV file each source points to (source URL)
#   (b) each source's volume (1.0 = active, 0.0 = silent / muted)
#
# The table below maps Tier → list of (channel_key, volume) pairs.
# A channel_key of None means "silence this source" (volume 0, no URL change).

_TIER_CONFIG: dict[Tier, list[tuple[str | None, float]]] = {
    # MONO: a single point-source at the center position.
    # The mono downmix is routed through the center speaker for pedagogical
    # clarity — the student hears everything collapse to one point.
    Tier.MONO: [
        ("mono",  1.0),   # Center   ← mono downmix
        (None,    0.0),   # FL       ← silent
        (None,    0.0),   # FR       ← silent
        (None,    0.0),   # Ls       ← silent
        (None,    0.0),   # Rs       ← silent
        (None,    0.0),   # LFE      ← silent
    ],

    # STEREO: left + right downmix only. Center, surrounds, LFE muted.
    # The student hears the stereo image narrow to the front arc.
    Tier.STEREO: [
        (None,          0.0),  # Center   ← silent
        ("stereo_l",    1.0),  # FL       ← stereo left downmix
        ("stereo_r",    1.0),  # FR       ← stereo right downmix
        (None,          0.0),  # Ls       ← silent
        (None,          0.0),  # Rs       ← silent
        (None,          0.0),  # LFE      ← silent
    ],

    # FIVE_ONE: all six discrete channels from the FLAC master.
    # This is the only tier where the student hears genuinely discrete
    # surround content — not a synthesized approximation.
    Tier.FIVE_ONE: [
        ("fc",   1.0),  # Center
        ("fl",   1.0),  # FL
        ("fr",   1.0),  # FR
        ("ls",   1.0),  # Ls
        ("rs",   1.0),  # Rs
        ("lfe",  1.0),  # LFE
    ],
}

# Speaker order must match _TIER_CONFIG list positions above.
# Index 0 = Center, 1 = FL, 2 = FR, 3 = Ls, 4 = Rs, 5 = LFE
_SPEAKER_ORDER = ["fc", "fl", "fr", "ls", "rs", "lfe"]


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
        # QQuaternion() default constructor produces (0,0,0,0) — a zero quaternion,
        # which is mathematically invalid and breaks the HRTF renderer entirely.
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

        # -- QSpatialSound×6 ----------------------------------------------
        # Six permanent source objects — one per ITU virtual speaker position.
        # They are constructed ONCE here and never destroyed or recreated.
        # Tier switching only changes their source URL and volume.
        self._sources: list[QSpatialSound] = []
        self._source_names: list[str] = []  # parallel list for diagnostics

        # -- State --------------------------------------------------------
        # Default tier is MONO — the app starts silent (paused) with the mono
        # downmix routed through the center speaker only.  The student hears
        # a single point source from straight ahead before exploring stereo and
        # 5.1.  This also matches hud.js's initial _state.tier = "mono" so the
        # HUD and engine are always in sync from the first state_changed signal.
        self._tier       = Tier.MONO
        self._room_var   = RoomVariant.TREATED
        self._is_playing = False
        self._file_map: dict[str, Path] = {}  # set by initialise()
        self._initialised = False

    # -----------------------------------------------------------------------
    # STEP 6: Initialisation — called once after asset_pipeline completes
    # -----------------------------------------------------------------------

    def initialise(self, file_map: dict[str, Path]) -> None:
        """
        Build the six QSpatialSound sources and apply the default state.

        Must be called after asset_pipeline.run_pipeline() returns successfully.
        Emits ready() on success, error(message) on any failure.

        Args:
            file_map: dict mapping channel key → Path of the WAV file.
                      Produced by asset_pipeline.PipelineResult.file_map.
        """
        self._file_map = file_map

        print("[audio_engine] initialise() file_map:", flush=True)
        for key, path in file_map.items():
            exists = path.exists()
            print(f"  [{key}] {'OK' if exists else 'MISSING'} {path}", flush=True)

        # Validate that every WAV the engine will ever need actually exists
        # before constructing any QSpatialSound object.
        missing = [
            str(p) for p in file_map.values() if not p.exists()
        ]
        if missing:
            print(f"[audio_engine] Missing WAV files: {missing}")
            msg = (
                "Asset pipeline output is incomplete. Missing WAV files:\n"
                + "\n".join(f"  {m}" for m in missing)
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
        Construct the six permanent QSpatialSound instances at their ITU positions.

        CONCEPT: Spatial audio sources are positioned in 3-D space. The engine
        computes the angle and distance from the listener to each source every
        audio frame and applies the appropriate HRTF filter. For the binaural
        virtualiser to work correctly, the sources must sit at the ITU-R BS.775
        angles around the sweet spot — not at the visual mesh positions (which
        are scaled GLB models placed for aesthetics).
        """
        print("[audio_engine] _init_sources(): constructing QSpatialSound objects...", flush=True)
        # Build the five positional speakers from _SPEAKER_DEFS.
        for azimuth, name, key, _ in _SPEAKER_DEFS:
            src = QSpatialSound(self._engine)
            pos = _itu_position(azimuth)
            src.setPosition(pos)
            src.setSize(0.0)
            src.setDistanceCutoff(10.0)
            src.setLoops(QSpatialSound.Loops.Infinite)
            src.setAutoPlay(False)
            self._sources.append(src)
            self._source_names.append(name)
            print(f"  [source] {name} (key={key}) at pos=({pos.x():.2f}, {pos.y():.2f}, {pos.z():.2f})", flush=True)

        lfe_src = QSpatialSound(self._engine)
        lfe_src.setPosition(_LFE_POSITION)
        lfe_src.setSize(0.0)
        lfe_src.setDistanceCutoff(10.0)
        lfe_src.setLoops(QSpatialSound.Loops.Infinite)
        lfe_src.setAutoPlay(False)
        self._sources.append(lfe_src)
        self._source_names.append("LFE")
        print(f"  [source] LFE (key=lfe) at pos=({_LFE_POSITION.x():.2f}, {_LFE_POSITION.y():.2f}, {_LFE_POSITION.z():.2f})", flush=True)
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
    # STEP 8: Tier switching — source-swap, not object recreation
    # -----------------------------------------------------------------------

    def _apply_tier(self, tier: Tier) -> None:
        """
        Reconfigure source URLs and volumes for the given tier.

        This operates on the existing six QSpatialSound objects — it does NOT
        destroy or recreate them. Only source URL and volume change.

        The source order in self._sources is fixed:
            [0]=Center  [1]=FL  [2]=FR  [3]=Ls  [4]=Rs  [5]=LFE
        matching _TIER_CONFIG list positions.
        """
        config = _TIER_CONFIG[tier]
        print(f"[audio_engine] _apply_tier({tier.value}): configuring {len(config)} sources", flush=True)

        for i, (channel_key, volume) in enumerate(config):
            src = self._sources[i]
            name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
            src.setVolume(volume)

            if channel_key is None or volume == 0.0:
                print(f"  [{i}] {name}: MUTED (key={channel_key}, vol={volume})", flush=True)
                continue

            wav_path = self._file_map.get(channel_key)
            if wav_path is None:
                raise RuntimeError(
                    f"No WAV path found for channel key '{channel_key}' "
                    f"(tier={tier.value}, speaker index {i}). "
                    f"Available keys: {list(self._file_map.keys())}"
                )
            url = QUrl.fromLocalFile(str(wav_path))
            src.setSource(url)
            print(f"  [{i}] {name}: vol={volume} source={wav_path.name} exists={wav_path.exists()}", flush=True)

    # -----------------------------------------------------------------------
    # STEP 9: Public transport controls
    # -----------------------------------------------------------------------

    @Slot()
    def play(self) -> None:
        """
        Start playback of all currently-active sources.

        CONCEPT: Starting multiple QSpatialSound instances in a tight Python
        loop is not perfectly synchronised at the sample level — Qt processes
        each call in sequence. In practice the offset is sub-millisecond and
        imperceptible. If perfect sample-accurate sync were required, a
        QAudioEngine-level transport would be needed; that API is not currently
        exposed in PySide6's QtSpatialAudio bindings (see build plan §5.3).
        """
        print("[audio_engine] play() called", flush=True)
        if not self._initialised:
            print("[audio_engine] play() ERROR: not initialised", flush=True)
            self.error.emit("play() called before initialise(). "
                            "Wait for engine_ready signal.")
            return

        for i, src in enumerate(self._sources):
            name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
            vol = src.volume()
            url = src.source().toString()
            print(f"  [play] {name}: vol={vol:.1f} url={url or '(none)'}", flush=True)
            if vol > 0.0:
                src.play()
                print(f"  [play] {name}: src.play() called", flush=True)
            else:
                print(f"  [play] {name}: skipped (muted)", flush=True)

        self._is_playing = True
        self._emit_state()
        print(f"[audio_engine] play() done, _is_playing=True", flush=True)

    @Slot()
    def pause(self) -> None:
        """Pause all active (non-muted) sources. Position is preserved for resume.

        Sources with volume=0.0 were never started by play() — calling pause()
        on them is a no-op in Qt but produces misleading log noise and is
        semantically wrong.  We skip them here for the same reason play() skips
        them: the tier config (not pause) is the authority on which sources are
        active.
        """
        print("[audio_engine] pause() called", flush=True)
        for i, src in enumerate(self._sources):
            name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
            if src.volume() == 0.0:
                print(f"  [pause] {name}: skipped (muted)", flush=True)
                continue
            src.pause()
            print(f"  [pause] {name}: src.pause() called", flush=True)
        self._is_playing = False
        self._emit_state()
        print("[audio_engine] pause() done, _is_playing=False", flush=True)

    @Slot()
    def pause_all(self) -> None:
        """Pause all active sources.

        Public alias for pause(), called by MainWindow.closeEvent() to stop
        playback cleanly before the Qt audio graph is torn down.  Stopping
        sources explicitly avoids the brief audio glitch that some backends
        produce when a playing QSpatialSound is destroyed while its
        QAudioEngine is still alive.

        Delegates entirely to pause() so all state bookkeeping stays in one
        place.
        """
        self.pause()

    @Slot()
    def restart(self) -> None:
        """
        Stop all sources (resets to 0:00) and restart.
        """
        print("[audio_engine] restart() called — stopping all sources", flush=True)
        for i, src in enumerate(self._sources):
            name = self._source_names[i] if i < len(self._source_names) else f"src[{i}]"
            src.stop()
            print(f"  [restart] {name}: src.stop() called", flush=True)
        self._is_playing = False
        self.play()

    # -----------------------------------------------------------------------
    # STEP 10: State-switching slots (called by bridge.py)
    # -----------------------------------------------------------------------

    @Slot(str)
    def set_tier(self, tier_name: str) -> None:
        """Switch to a new surround tier.

        If playback is active, it is paused, the tier is reconfigured, and
        playback resumes automatically.  This is preferable to blocking the
        switch with an error because the demo's whole point is letting the
        listener hear tier differences mid-clip without having to find and
        press pause first.
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

        # Pause before reconfiguring so QSpatialSound.setSource() is never
        # called on a source that is mid-playback (some backends glitch on that).
        was_playing = self._is_playing
        if was_playing:
            print("[audio_engine] set_tier() pausing for reconfigure", flush=True)
            self.pause()

        self._apply_tier(new_tier)
        self._tier = new_tier

        if was_playing:
            print("[audio_engine] set_tier() resuming after reconfigure", flush=True)
            self.play()

        self._emit_state()
        print(f"[audio_engine] set_tier() done: now on {new_tier.value}", flush=True)

    @Slot(str)
    def set_room(self, room_name: str) -> None:
        """Switch room acoustic variant.

        If playback is active, it is paused, the room is reconfigured, and
        playback resumes automatically so the listener can hear the acoustic
        difference without manually stopping and restarting the clip.
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

        was_playing = self._is_playing
        if was_playing:
            print("[audio_engine] set_room() pausing for reconfigure", flush=True)
            self.pause()

        self._apply_room(new_room)
        self._room_var = new_room

        if was_playing:
            print("[audio_engine] set_room() resuming after reconfigure", flush=True)
            self.play()

        self._emit_state()
        print(f"[audio_engine] set_room() done: now on {new_room.value}", flush=True)

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
