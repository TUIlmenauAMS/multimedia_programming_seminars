# =============================================================================
# bridge.py — Home Theater Demo, Chapter 05 Companion
# =============================================================================
#
# PURPOSE
# -------
# A single QObject subclass registered with QWebChannel. It is the ONLY
# communication layer between the Three.js frontend and the Python backend.
# Nothing in the web/ directory talks to audio_engine.py directly — every
# message crosses this bridge.
#
# ARCHITECTURE
# ------------
#
#   Three.js (JS)                    bridge.py (Python)          audio_engine.py
#   ─────────────────────────────    ──────────────────────────  ───────────────
#   new QWebChannel(transport, cb)
#   bridge.update_listener_pose() ──→ @Slot update_listener_pose() → engine
#   bridge.set_tier()             ──→ @Slot set_tier()             → engine
#   bridge.set_room()             ──→ @Slot set_room()             → engine
#   bridge.play()                 ──→ @Slot play()                 → engine
#   bridge.pause()                ──→ @Slot pause()                → engine
#   bridge.restart()              ──→ @Slot restart()              → engine
#
#   bridge.engine_ready   ←── Signal ←── engine.ready
#   bridge.asset_error    ←── Signal ←── engine.error
#   bridge.state_changed  ←── Signal ←── engine.state_changed
#   bridge.video_command  ──────────────────────────────────────→ JS handles
#
# JS → Python: QWebChannel slots (decorated @Slot)
# Python → JS: Qt Signals, exposed automatically by QWebChannel
#
# THROTTLING
# ----------
# update_listener_pose is called every animation frame (~60 Hz). The audio
# engine's setPosition/setRotation calls are cheap, but crossing the
# JS→Python bridge has non-trivial overhead (serialisation + IPC). We
# therefore throttle: only forward the pose to the engine if either the
# position has moved by more than POSE_POSITION_THRESHOLD metres or the
# orientation has changed by more than POSE_ANGLE_THRESHOLD (quaternion
# dot-product check). At the listening sweet spot with minimal movement,
# this reduces bridge calls by ~80% while keeping acoustic accuracy well
# within perceptual thresholds (~1cm / ~1° are well below audible ITD error).
#
# COORDINATE SYSTEM
# -----------------
# Same as audio_engine.py and scene.js: meters, Y-up, right-handed.
# Positive Z = toward listener (away from TV). All position values arriving
# from JS are already in this space (Three.js world coordinates).
#
# =============================================================================

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from .audio_engine import SpatialAudioEngine


# ---------------------------------------------------------------------------
# STEP 1: Throttle thresholds
# These are named constants, not magic numbers — tune them if profiling shows
# the bridge is a CPU bottleneck or if pose updates feel laggy.
# ---------------------------------------------------------------------------

# Minimum position change (meters) to forward a pose update.
# 0.005m = 5mm — well below the ITD-detectable threshold (~1cm).
POSE_POSITION_THRESHOLD: float = 0.005

# Minimum quaternion angular change to forward an orientation update.
# Measured as 1 − |q1·q2|: 0 = identical, 1 = 180° apart.
# 0.0001 corresponds to roughly 0.8° of rotation — imperceptible delay.
POSE_ANGLE_THRESHOLD: float = 0.0001


# ---------------------------------------------------------------------------
# STEP 2: AudioBridge — the QWebChannel contract object
# ---------------------------------------------------------------------------

class AudioBridge(QObject):
    """
    Single QObject registered with QWebChannel.

    Every slot here is callable from JavaScript as:
        bridge.slotName(args)

    Every signal here is receivable in JavaScript as:
        bridge.signalName.connect(callback)

    main.py registers this object once:
        channel = QWebChannel(page)
        channel.registerObject("bridge", audio_bridge_instance)

    The JS side connects with:
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.bridge = channel.objects.bridge;
        });
    """

    # -- Signals (Python → JS) -----------------------------------------------
    #
    # CONCEPT: QWebChannel serialises Qt signals to JSON messages delivered
    # asynchronously to the JS callback. The JS side registers .connect()
    # handlers on these signal names, exactly as it would for a DOM EventEmitter.

    # Fired once when the audio engine is initialised and all assets are
    # confirmed loadable. Gates the loading overlay fade-out in hud.js.
    engine_ready = Signal()

    # Fired on any missing or malformed asset. JS displays the message in the
    # loading overlay using the same error style as the web demos.
    asset_error = Signal(str)       # message: human-readable error string

    # Fired after every successful state transition (tier switch, room switch,
    # play/pause). JS uses this to update the HUD — it reflects CONFIRMED engine
    # state, not the optimistic state the user requested.
    state_changed = Signal(str, str, bool)  # tier_name, room_name, is_playing

    # Fired when the engine needs the JS side to control the <video> element.
    # Possible commands: "play", "pause", "restart"
    # (The video element is muted and controlled by Python to stay in sync with
    # QSpatialSound sources — see §7.3 of the build plan for why the video is
    # always muted.)
    video_command = Signal(str)     # command: "play" | "pause" | "restart"

    def __init__(
        self,
        engine: SpatialAudioEngine,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine

        # Throttle state — last forwarded pose values.
        # Stored as flat floats to avoid per-frame object allocation.
        self._last_px: float = 0.0
        self._last_py: float = 0.0
        self._last_pz: float = 0.0
        self._last_qx: float = 0.0
        self._last_qy: float = 0.0
        self._last_qz: float = 0.0
        self._last_qw: float = 1.0  # identity quaternion

        # Wire engine signals → bridge signals (Python → JS relay).
        # The bridge re-emits rather than forwarding directly so that future
        # code can intercept (log, transform, gate) any signal in one place.
        #
        # NOTE: engine.ready is intentionally NOT wired here.  engine.ready
        # fires during engine.initialise(), which is called in main() before
        # the window is shown and long before the QWebChannel handshake
        # completes.  Forwarding it here would emit engine_ready into the void
        # — JS has not yet called bridge.engine_ready.connect() so it would
        # never receive the signal and the Start button would stay disabled.
        #
        # Instead, JS calls bridge.js_ready() from inside the QWebChannel
        # callback (index.html) — the first moment it is guaranteed to be
        # connected.  js_ready() then emits engine_ready.  See js_ready() below.
        self._engine.error.connect(self._on_engine_error)
        self._engine.state_changed.connect(self._on_state_changed)

    # -----------------------------------------------------------------------
    # STEP 3: Engine signal relay (private — not exposed to JS)
    # -----------------------------------------------------------------------

    @Slot()
    def js_ready(self) -> None:
        print(f"[bridge] js_ready() called, engine.is_initialised={self._engine.is_initialised}", flush=True)
        """Called by JS from inside the QWebChannel callback in index.html.

        This is the ONLY correct place to emit engine_ready, because it is
        the first moment we know that:
          1. The QWebChannel handshake has completed.
          2. JS has called bridge.engine_ready.connect(handler).
          3. The emitted signal will actually be received.

        Emitting engine_ready at engine initialisation time (before the page
        loads) means JS never sees it — the signal fires before the connect()
        call, which is a one-shot subscription, so the event is permanently
        lost and the Start button stays disabled.

        If the engine failed to initialise (is_initialised is False), we emit
        asset_error instead so the HUD shows a readable failure message rather
        than leaving the loading overlay up forever.
        """
        if self._engine.is_initialised:
            self.engine_ready.emit()
        else:
            self.asset_error.emit(
                "Audio engine failed to initialise before the page loaded.\n\n"
                "Check the Python terminal for the original error from "
                "audio_engine.py.  The most common cause is a missing or "
                "malformed WAV file in the cache/ directory — delete cache/ "
                "and re-run to trigger a rebuild."
            )

    def _on_engine_error(self, message: str) -> None:
        """Relay engine.error → bridge.asset_error (JS-visible signal)."""
        self.asset_error.emit(message)

    def _on_state_changed(
        self, tier_name: str, room_name: str, is_playing: bool
    ) -> None:
        """
        Relay engine.state_changed → bridge.state_changed (JS-visible).

        Also drives the video_command signal so the muted <video> element
        in the Three.js scene stays in sync with the audio engine transport.

        CONCEPT: There is no cross-engine sync primitive between QSpatialSound
        and an HTML <video> element. We get as close as possible by emitting
        video_command in the same Python call that the engine confirms its new
        state — the round-trip latency (Python→QWebChannel→JS callback) is
        sub-millisecond on localhost, well within the ~20ms audio buffer size.
        """
        self.state_changed.emit(tier_name, room_name, is_playing)

        # Map confirmed play/pause state to a video element command.
        # "restart" is emitted by the restart() slot below before this
        # relay fires, so we only need play/pause here.
        if is_playing:
            self.video_command.emit("play")
        else:
            self.video_command.emit("pause")

    # -----------------------------------------------------------------------
    # STEP 4: JS → Python slots
    # -----------------------------------------------------------------------

    @Slot(float, float, float, float, float, float, float)
    def update_listener_pose(
        self,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ) -> None:
        """
        Called every animation frame (~60 Hz) from scene.js with the
        Three.js camera's world position and orientation quaternion.

        CONCEPT: Throttling here is a performance optimisation, not an
        acoustic approximation. The human auditory system integrates head
        movement over ~10ms buffers; suppressing updates smaller than 5mm /
        0.8° introduces no perceptible delay or imaging error. Without
        throttling, the QWebChannel IPC overhead at 60 Hz can produce
        measurable CPU load on slower hardware.

        Position threshold:  POSE_POSITION_THRESHOLD (5mm)
        Orientation threshold: POSE_ANGLE_THRESHOLD (≈0.8°)

        No smoothing or interpolation is applied — if motion looks jittery,
        fix it in scene.js (camera damping), not here. Adding a filter here
        would desync the visual facing direction from the acoustic facing
        direction, breaking the binaural imaging.
        """
        # -- Position change check -----------------------------------------
        dp = math.sqrt(
            (x - self._last_px) ** 2 +
            (y - self._last_py) ** 2 +
            (z - self._last_pz) ** 2
        )

        # -- Orientation change check (quaternion dot product) -------------
        # |q1 · q2| = 1 when quaternions are identical, < 1 when different.
        # 1 − |dot| gives a small positive number proportional to the angle.
        dot = abs(
            qx * self._last_qx +
            qy * self._last_qy +
            qz * self._last_qz +
            qw * self._last_qw
        )
        # Clamp to [0, 1] to guard against floating-point values slightly > 1.
        dot = min(dot, 1.0)
        dq = 1.0 - dot

        if dp < POSE_POSITION_THRESHOLD and dq < POSE_ANGLE_THRESHOLD:
            return  # pose hasn't changed enough — skip this frame

        # Update cached pose.
        self._last_px, self._last_py, self._last_pz = x, y, z
        self._last_qx, self._last_qy, self._last_qz, self._last_qw = qx, qy, qz, qw

        # Forward to the audio engine.
        self._engine.update_listener_pose(x, y, z, qx, qy, qz, qw)

    @Slot(str)
    def set_tier(self, tier_name: str) -> None:
        """
        Request a tier switch.

        SAFETY: The engine enforces paused-only switching internally and emits
        engine.error if called while playing. This slot does NOT pre-check —
        that would duplicate logic and create a gap between the UI's assumption
        of what 'paused' means and the engine's actual state. The engine is
        the single source of truth.

        JS disables the tier controls while playing (hud.js), but Python
        defends regardless — the UI state and the engine state could diverge
        on a slow bridge round-trip.

        Args:
            tier_name: "mono" | "stereo" | "5.1"
        """
        self._engine.set_tier(tier_name)

    @Slot(str)
    def set_room(self, room_name: str) -> None:
        """
        Request a room variant switch.

        CONCEPT: Visual and acoustic properties change together — this is the
        chapter's stated principle. The JS side (hud.js) calls this slot AND
        updates the Three.js wall materials in the same event handler, so both
        changes are dispatched in the same animation frame. The bridge does not
        need to coordinate them — the JS caller owns that atomicity.

        Args:
            room_name: "treated" | "wet"
        """
        self._engine.set_room(room_name)

    @Slot()
    def play(self) -> None:
        """
        Start transport. Starts all active QSpatialSound sources.
        video_command("play") is emitted via the state_changed relay.
        """
        self._engine.play()

    @Slot()
    def pause(self) -> None:
        """
        Pause transport. Preserves playback position for resume.
        video_command("pause") is emitted via the state_changed relay.
        Also re-enables tier/room controls in the HUD (JS side handles this
        on receipt of state_changed with is_playing=False).
        """
        self._engine.pause()

    @Slot()
    def restart(self) -> None:
        """
        Stop all sources (resets to 0:00) and restart.

        CONCEPT: This is the post-switch recovery path. After a tier or room
        switch (which requires pausing), the user clicks Play and the bridge
        calls restart() rather than play() so the video and audio both reset
        to 0:00 together. See audio_engine.py restart() for the QSpatialSound
        stop() → play() sequence.

        video_command("restart") is emitted BEFORE the engine restarts so
        that the JS side can reset the <video> element to 0:00 and begin
        seeking before audio playback starts, minimising the A/V offset.
        """
        # Emit video restart command FIRST — give the video element time
        # to seek to 0:00 before audio begins.
        self.video_command.emit("restart")
        self._engine.restart()

    # -----------------------------------------------------------------------
    # STEP 5: State query slots (JS can poll current state on page load)
    # -----------------------------------------------------------------------

    @Slot(str, bool)
    def set_speaker_mute(self, key: str, muted: bool) -> None:
        """Mute or unmute a single virtual speaker by its ITU key (C/L/R/Ls/Rs/LFE).

        The HUD always sends the EFFECTIVE mute state — meaning solo logic is
        already resolved by the JS side before this slot is called. This slot's
        only job is to map the HUD key to the correct engine source for the
        active tier, then apply 0.0 (silenced) or the tier's nominal volume
        (audible).

        TIER MAPPING
        ------------
        Each tier uses a different subset of the 9 engine sources and different
        audio files, so the HUD key → source index mapping is tier-dependent:

          Mono:   C        → source 0 ("mono")
          Stereo: L, R     → sources 1, 2 ("stereo_l", "stereo_r")
          5.1:    C,L,R,Ls,Rs,LFE → sources 3–8 ("fc","fl","fr","ls","rs","lfe")

        A key that is not part of the active tier is a silent no-op — the HUD
        disables those strips, but a stale bridge call could still arrive.

        VOLUME RESTORATION
        ------------------
        When unmuting, we restore _TIER_VOLUMES[tier][idx] rather than 1.0.
        Sources that belong to a different tier have a tier volume of 0.0 and
        must never be accidentally unmuted by a cross-tier call.
        """
        from .audio_engine import _SOURCE_DEFS, _TIER_VOLUMES, Tier

        # HUD key → channel key, keyed by Tier enum.
        # Only speakers active in each tier are listed; anything absent is a no-op.
        _KEY_MAP: dict[Tier, dict[str, str]] = {
            Tier.MONO:    {"C": "mono"},
            Tier.STEREO:  {"L": "stereo_l", "R": "stereo_r"},
            Tier.FIVE_ONE: {
                "C": "fc", "L": "fl", "R": "fr",
                "Ls": "ls", "Rs": "rs", "LFE": "lfe",
            },
        }

        tier = self._engine.tier
        channel_key = _KEY_MAP.get(tier, {}).get(key)
        if channel_key is None:
            # Speaker is not active in the current tier — ignore silently.
            return

        # Resolve source index by matching channel_key in _SOURCE_DEFS.
        # _SOURCE_DEFS is a list of (channel_key, azimuth, display_name) tuples
        # ordered to match self._engine._sources.
        source_keys = [entry[0] for entry in _SOURCE_DEFS]
        try:
            idx = source_keys.index(channel_key)
        except ValueError:
            # Defensive: channel_key not in _SOURCE_DEFS — engine/HUD are out of sync.
            return

        if muted:
            self._engine._sources[idx].setVolume(0.0)
        else:
            # Restore the tier's nominal volume for this source, not a hardcoded 1.0.
            # Sources inactive in this tier have a nominal volume of 0.0 — this
            # guard ensures they can never be accidentally unmuted.
            nominal = _TIER_VOLUMES[tier][idx]
            self._engine._sources[idx].setVolume(nominal)

    @Slot(str)
    def open_url(self, url: str) -> None:
        """Open a URL in the system default browser via QDesktopServices."""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(url))

    @Slot(result=str)
    def get_tier(self) -> str:
        """Return the current tier name. Useful for HUD initialisation."""
        return self._engine.tier.value

    @Slot(result=str)
    def get_room(self) -> str:
        """Return the current room variant name."""
        return self._engine.room_variant.value

    @Slot(result=bool)
    def get_is_playing(self) -> bool:
        """Return current play state."""
        return self._engine.is_playing

    # -----------------------------------------------------------------------
    # STEP 6: (Channel setup is performed by MainWindow._setup_channel in
    # main.py. The AudioBridge constructor wires engine signals; the channel
    # registration and qwebchannel.js injection live in main.py so that the
    # channel lifetime is tied to the window, not to this class.)
    # -----------------------------------------------------------------------
