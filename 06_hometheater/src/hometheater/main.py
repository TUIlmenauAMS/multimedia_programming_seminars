"""
hometheater-demo — main.py
==========================
Native binaural home-theater spatial audio simulator.
Multimediaprogrammierung Chapter 05 companion app.

# TODO: confirm author credit block
Author  : Renato Fera P.
Course  : Multimediaprogrammierung
Institute: TU Ilmenau

ARCHITECTURE OVERVIEW
---------------------
This file owns the outermost application shell:

    QApplication
    └── MainWindow  (QMainWindow)
        ├── QWebEngineView        — hosts the Three.js front-end (web/)
        │   └── QWebEnginePage    — custom page with QWebChannel injected
        ├── QWebChannel           — cross-language bridge; see bridge.py
        ├── HtSchemeHandler       — custom ht:// URL scheme (see §SCHEME below)
        ├── SpatialAudioEngine    — Qt Spatial Audio engine; see audio_engine.py
        └── HtBridge              — QObject slots/signals; see bridge.py

STARTUP SEQUENCE
----------------
1. Parse and validate assets/manifest.json (or show a startup error dialog).
2. Run asset_pipeline.py's prepare() — splits/downmixes the 5.1 FLAC if the
   cache is stale. This is the only blocking work before the window opens.
3. Construct SpatialAudioEngine with the prepared cache paths.
4. Construct HtBridge, wire it to the engine.
5. Register HtSchemeHandler so the web page can reach assets via ht://assets/…
6. Open MainWindow; QWebEngineView loads web/index.html.
7. index.html opens QWebChannel → JS receives the bridge object → HUD wires up.
8. bridge.py fires engine_ready() after confirming all asset URLs are reachable.
   This is the gate that lets the Three.js loading overlay fade to the Start button.

ERROR SURFACE
-------------
Errors that occur before the window is shown are displayed as a QMessageBox.
Errors that occur after the window is shown are routed through
HtBridge.asset_error (Python → JS signal) so they appear in the HUD overlay —
never as a bare Python traceback, never silently swallowed.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# STEP 1: PySide6 imports — group by Qt module for readability.
# QtSpatialAudio requires PySide6 ≥ 6.8; if this import fails the app exits
# with a readable message rather than a raw ImportError traceback.
# ---------------------------------------------------------------------------
try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import (
        QWebEngineProfile,
        QWebEnginePage,
        QWebEngineScript,
        QWebEngineUrlScheme,
        QWebEngineUrlSchemeHandler,
        QWebEngineUrlRequestJob,
    )
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtCore import (
        QUrl,
        QByteArray,
        QBuffer,
        QIODevice,
        QCoreApplication,
        Qt,
    )
    from PySide6.QtGui import QIcon
except ImportError as exc:
    # Keep the error message student-readable: name the likely cause
    # (missing Qt Multimedia backend) rather than dumping the raw exception.
    print(
        "\n[hometheater] Import error — PySide6 or one of its sub-modules is "
        "missing or too old.\n"
        "Make sure you are running inside the project's uv environment:\n\n"
        "    uv sync\n"
        "    uv run python -m hometheater\n\n"
        "If the import still fails after uv sync, the platform's Qt Multimedia "
        "backend plugin may be absent. This is an environment issue, not a code "
        "bug — reinstall PySide6 for your platform.\n\n"
        f"Original error: {exc}\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Local package imports.  asset_pipeline and audio_engine have no PySide6
# dependency other than QUrl, so they can be tested standalone (see §9 of
# the build plan).
# ---------------------------------------------------------------------------
from hometheater.asset_pipeline import run_pipeline
from hometheater.audio_engine import SpatialAudioEngine
from hometheater.bridge import AudioBridge


# ---------------------------------------------------------------------------
# CONCEPT: Paths — single source of truth.
# All path constants live here so no other module needs to know the project
# layout.  Everything is relative to the installed package root so the app
# works regardless of where the user's working directory is.
# ---------------------------------------------------------------------------
_PKG_ROOT   = Path(__file__).parent          # src/hometheater/
_PROJ_ROOT  = _PKG_ROOT.parent.parent        # project root (contains assets/)
ASSETS_DIR  = _PROJ_ROOT / "assets"
MANIFEST    = ASSETS_DIR / "manifest.json"
CACHE_DIR   = _PROJ_ROOT / "cache"
WEB_DIR     = _PKG_ROOT  / "web"


# ===========================================================================
# STEP 2: Custom URL scheme — ht://
# ===========================================================================
# CONCEPT: Custom URL scheme handler (QWebEngineUrlSchemeHandler)
# ----------------------------------------------------------------
# The Three.js front-end runs inside QWebEngineView with a local file origin.
# It needs to fetch the manifest and reference model/media paths at runtime,
# but plain file:// URLs for sibling directories are blocked by the web
# engine's same-origin policy, and http://localhost introduces a dependency on
# a running local server.
#
# The solution: register a custom scheme "ht" (home-theater) so the page can
# fetch assets as:
#
#   ht://assets/manifest.json
#   ht://assets/models/sofa.glb
#   ht://assets/media/sintel.mp4
#   ht://cache/sintel-abc123/left.wav
#
# The handler maps "ht://assets/…" → ASSETS_DIR/… and "ht://cache/…" →
# CACHE_DIR/… and reads the file synchronously into a QBuffer, which satisfies
# QWebEngineUrlRequestJob's requirement that the reply body be provided before
# the handler's scope exits.
#
# IMPORTANT: QWebEngineUrlScheme.registerScheme() must be called BEFORE
# QApplication is constructed — Qt enforces this and will log a warning
# (and silently fail the registration) if you call it afterward.  The
# registration therefore happens at module import time via _register_ht_scheme()
# called at the bottom of this file, before main() runs.

def _register_ht_scheme() -> None:
    """Register the ht:// custom URL scheme with the web engine.

    Must be called before QApplication is constructed.  Idempotent if called
    more than once (Qt ignores re-registration of the same scheme).
    """
    scheme = QWebEngineUrlScheme(b"ht")
    # FetchApiAllowed  : required for fetch() / XMLHttpRequest to work with this
    #                    scheme.  Without it Chromium rejects the request with
    #                    "URL scheme is not supported" before it reaches the handler.
    #                    Added in Qt 6.6 / PySide6 6.6; has no effect on older builds
    #                    but is harmless to include.
    # CorsEnabled      : allows cross-origin requests between ht:// sub-resources
    #                    (e.g. ht://assets/… fetched from ht://app/…).
    # LocalAccessAllowed: permits the handler to read local filesystem paths.
    # SecureScheme     : tells Chromium to treat ht:// as a secure context, which
    #                    is required for certain APIs (Web Audio, etc.).
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.CorsEnabled
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.FetchApiAllowed
    )
    QWebEngineUrlScheme.registerScheme(scheme)


class HtSchemeHandler(QWebEngineUrlSchemeHandler):
    """Serve assets from the local filesystem under the ht:// scheme.

    URL mapping:
        ht://assets/<path>  →  ASSETS_DIR/<path>
        ht://cache/<path>   →  CACHE_DIR/<path>
        ht://app/<path>     →  WEB_DIR/<path>

    The manifest is a special case: ht://assets/manifest.json is served from
    the in-memory normalised dict (set via set_manifest()) rather than the raw
    file on disk, so that JS receives the nested models/media structure that
    scene.js expects (built by _normalise_manifest()) rather than the flat keys
    stored in the JSON file.

    Any other host returns a 404-style error response so the JS side gets a
    clear failure rather than a silent hang.
    """

    # MIME type lookup — extend as new asset types are added.
    _MIME: dict[str, str] = {
        ".html":  "text/html",
        ".css":   "text/css",
        ".js":    "application/javascript",
        ".json":  "application/json",
        ".glb":   "model/gltf-binary",
        ".gltf":  "model/gltf+json",
        ".mp4":   "video/mp4",
        ".webm":  "video/webm",
        ".wav":   "audio/wav",
        ".flac":  "audio/flac",
        ".png":   "image/png",
        ".jpg":   "image/jpeg",
        ".jpeg":  "image/jpeg",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._normalised_manifest_bytes: bytes | None = None

    def set_manifest(self, normalised: dict) -> None:
        """Cache the normalised manifest as UTF-8 JSON bytes for serving to JS.

        Must be called before the page loads so the first fetch of
        ht://assets/manifest.json receives the enriched dict.
        """
        self._normalised_manifest_bytes = json.dumps(normalised, ensure_ascii=False).encode()

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802
        url  = job.requestUrl()
        host = url.host()          # "assets", "cache", or "app"
        path = url.path().lstrip("/")

        # Special case: serve the normalised manifest from memory so JS receives
        # the nested models/media structure rather than the flat on-disk JSON.
        if host == "assets" and path == "manifest.json":
            if self._normalised_manifest_bytes is None:
                print("[ht://] manifest requested before set_manifest() was called.")
                job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
                return
            buf = QBuffer(job)
            buf.setData(QByteArray(self._normalised_manifest_bytes))
            buf.open(QIODevice.OpenModeFlag.ReadOnly)
            job.reply(QByteArray(b"application/json"), buf)
            return

        # Map host to a base directory.
        if host == "assets":
            base = ASSETS_DIR
        elif host == "cache":
            base = CACHE_DIR
        elif host == "app":
            base = WEB_DIR
        else:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return

        file_path = base / path
        if not file_path.exists() or not file_path.is_file():
            print(f"[ht://] Not found: {file_path}")
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return

        # Read the file into a QByteArray and wrap it in a QBuffer so
        # QWebEngineUrlRequestJob can read from it.  The buffer's lifetime must
        # outlast the job; parenting it to the job achieves this.
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            print(f"[ht://] Failed to read {file_path}: {exc}")
            job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
            return

        buf = QBuffer(job)          # job takes ownership → lives long enough
        buf.setData(QByteArray(data))
        buf.open(QIODevice.OpenModeFlag.ReadOnly)

        mime = self._MIME.get(file_path.suffix.lower(), "application/octet-stream")
        job.reply(QByteArray(mime.encode()), buf)


# ===========================================================================
# STEP 3: Manifest loading
# ===========================================================================
# CONCEPT: Manifest-driven asset references
# -----------------------------------------
# All filenames (video, 5.1 FLAC, model GLBs) are user-editable and live in
# assets/manifest.json rather than being hardcoded here.  This keeps the
# Python and JS layers in sync via a single file the user edits once, and lets
# the same codebase work with different Blender Foundation films without any
# code changes.
#
# The manifest is validated here (not in bridge.py) because a missing or
# malformed manifest is a startup-time hard error — the app literally cannot
# run without it.  Validation errors are shown as a pre-window QMessageBox,
# the clearest surface available before the web engine is running.

_REQUIRED_MANIFEST_KEYS = [
    "film_title",
    "film_credit",
    "video_file",
    "audio_51_file",
    "model_sofa",
    "model_tv",
    "model_speaker",
    "room_dimensions",        # {width, height, depth} in metres
]


def _normalise_manifest(manifest: dict) -> dict:
    """Reshape a flat manifest into the nested structure scene.js expects.

    assets/manifest.json uses flat keys for human-friendliness:
        "model_sofa", "model_tv", "model_speaker", "video_file", …

    scene.js reads nested keys:
        manifest.models.sofa, manifest.models.tv, manifest.media.video, …

    This function adds the nested structure as extra keys so that both the
    Python layer (which reads flat keys) and the JS layer (which reads nested
    keys) see a consistent manifest without duplicating data in the JSON file.

    Called once in load_manifest() — the returned dict is what every other
    part of the application, including the ht:// scheme handler serving
    manifest.json to JS, actually uses.
    """
    m = dict(manifest)  # shallow copy — do not mutate the caller's dict

    # Build the nested "models" block scene.js reads via manifest.models?.X
    m["models"] = {
        "sofa":      m.get("model_sofa"),
        "tv":        m.get("model_tv"),
        "speaker":   m.get("model_speaker"),
        # Optional keys — may be absent; scene.js guards with ?. operator
        "center":    m.get("model_center"),
        "subwoofer": m.get("model_subwoofer"),
    }

    # Build the nested "media" block scene.js reads via manifest.media?.video
    m["media"] = {
        "video": m.get("video_file"),
    }

    return m


def load_manifest() -> dict:
    """Load and validate assets/manifest.json.

    Returns the parsed manifest dict on success.
    Raises FileNotFoundError or ValueError with a readable message on failure.
    """
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n\n"
            "Create assets/manifest.json following the README instructions "
            "before running the app.  A template is provided in assets/."
        )

    try:
        with MANIFEST.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"assets/manifest.json is not valid JSON: {exc}\n\n"
            "Fix the syntax error and re-run the app."
        ) from exc

    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(
            f"assets/manifest.json is missing required keys: {missing}\n\n"
            "See the README for the full manifest schema."
        )

    # Validate room_dimensions has the expected sub-keys.
    dims = manifest.get("room_dimensions", {})
    dim_keys = {"width", "height", "depth"}
    missing_dim = dim_keys - set(dims.keys())
    if missing_dim:
        raise ValueError(
            f"manifest.json room_dimensions is missing keys: {missing_dim}\n"
            "Expected: {\"width\": <m>, \"height\": <m>, \"depth\": <m>}"
        )

    # Reshape flat keys into the nested structure scene.js reads.
    # Must happen after validation so error messages still name the flat keys.
    return _normalise_manifest(manifest)


# ===========================================================================
# STEP 4: Web engine page — injects QWebChannel
# ===========================================================================
# CONCEPT: QWebEnginePage subclass for QWebChannel bootstrap
# ----------------------------------------------------------
# QWebChannel requires that the page load qwebchannel.js before any JS calls
# qt.webChannelTransport.  Rather than bundling qwebchannel.js manually,
# PySide6 ships it as a Qt resource accessible via
#   QWebEngineScript.ScriptWorldId.MainWorld
# or via QWebEnginePage.scripts().
#
# The cleanest approach is to call
#   QWebChannel(page, page)
# and then set it on the page with
#   page.setWebChannel(channel)
# which causes Qt to automatically inject the qwebchannel.js helper into the
# page before any user scripts run, making qt.webChannelTransport available
# globally.  index.html's inline script then opens the channel with:
#   new QWebChannel(qt.webChannelTransport, callback)

class HtWebPage(QWebEnginePage):
    """QWebEnginePage subclass that wires QWebChannel and routes console logs.

    Overrides javaScriptConsoleMessage so JS errors are visible in the Python
    terminal during development — QWebEngineView suppresses them by default.
    """

    def javaScriptConsoleMessage(      # noqa: N802
        self,
        level: QWebEnginePage.JavaScriptConsoleMessageLevel,
        message: str,
        line_number: int,
        source_id: str,
    ) -> None:
        level_label = {
            QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel:    "INFO",
            QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel: "WARN",
            QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:   "ERROR",
        }.get(level, "LOG")
        # Print JS console output to the Python terminal so the developer
        # doesn't need to open a separate DevTools window for basic debugging.
        print(f"[JS {level_label}] {source_id}:{line_number}  {message}")


# ===========================================================================
# STEP 5: Main window
# ===========================================================================

class MainWindow(QMainWindow):
    """The application's single top-level window.

    Hosts the QWebEngineView full-screen (no menu bar, no status bar — the HUD
    inside the web view is the entire UI chrome) and owns the bridge and engine
    lifetime.
    """

    def __init__(
        self,
        manifest: dict,
        engine: SpatialAudioEngine,
        bridge: AudioBridge,
    ) -> None:
        super().__init__()
        self._manifest = manifest
        self._engine   = engine
        self._bridge   = bridge

        self._setup_window()
        self._setup_webengine()
        self._setup_channel()
        self._load_page()

    # ------------------------------------------------------------------
    # Window chrome
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowTitle(
            f"Home Theater Spatial Audio — {self._manifest.get('film_title', '')}"
        )
        # Start maximised so the Three.js viewport uses as much screen real
        # estate as possible.  The user can resize freely.
        self.showMaximized()

        # Optional window icon — silently skipped if the file is absent rather
        # than crashing, since the icon is cosmetic.
        icon_path = WEB_DIR / "favicon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    # ------------------------------------------------------------------
    # Web engine setup
    # ------------------------------------------------------------------

    def _setup_webengine(self) -> None:
        """Create the QWebEngineView, configure the profile, install the scheme handler."""

        # True off-the-record profile — no disk cache, no persistent storage.
        #
        # QWebEngineProfile(parent) creates an off-the-record profile. We
        # additionally clear the cache and storage paths and set NoCache so
        # Qt never writes any files between launches. Without this, Qt can
        # serve a stale cached version of hud.js or scene.js from disk even
        # after those files have been updated, because the ht:// scheme is
        # treated like any other cacheable resource.
        profile = QWebEngineProfile(self)
        profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
        profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies
        )
        # Empty string disables disk cache and persistent storage entirely.
        profile.setCachePath("")
        profile.setPersistentStoragePath("")

        # Install the custom ht:// scheme handler on this profile so every
        # page loaded in this view can reach ht://assets/… and ht://cache/….
        self._scheme_handler = HtSchemeHandler(self)
        self._scheme_handler.set_manifest(self._manifest)
        profile.installUrlSchemeHandler(b"ht", self._scheme_handler)

        # Create a custom page that routes JS console messages to Python stdout.
        self._page = HtWebPage(profile, self)

        self._view = QWebEngineView(self)
        self._view.setPage(self._page)
        self.setCentralWidget(self._view)

    # ------------------------------------------------------------------
    # QWebChannel setup
    # ------------------------------------------------------------------

    def _setup_channel(self) -> None:
        """Register the bridge object on a QWebChannel and attach it to the page.

        IMPORTANT: setWebChannel() does NOT auto-inject qwebchannel.js — that
        comment in earlier drafts was wrong.  Qt requires the script to be
        injected explicitly via QWebEngineScript so that qt.webChannelTransport
        is available before index.html's bootstrap runs.

        NOTE on QWebChannel parent: pass `self` (the MainWindow), not
        `self._page`.  Passing the page as parent tangled the teardown order
        and produced "Release of profile requested but WebEnginePage still not
        deleted" on exit.  The channel is associated with the page via
        setWebChannel(), not via the constructor argument.

        NOTE on engine_ready timing: engine_ready must NOT be emitted at engine
        initialisation time (before the page loads).  JS can only receive a
        signal after the QWebChannel handshake completes and the handler has
        been registered with bridge.engine_ready.connect(...).  We therefore
        emit it from AudioBridge.js_ready(), a @Slot that JS calls explicitly
        from inside the QWebChannel callback — the earliest moment we know JS
        is connected and listening.  See bridge.py for the slot definition and
        index.html for the call site.
        """
        channel = QWebChannel(self)                    # parent = MainWindow, not page
        channel.registerObject("htBridge", self._bridge)
        self._page.setWebChannel(channel)
        self._channel = channel

        # Inject Qt's bundled qwebchannel.js at document-creation time so
        # QWebChannel is defined before any user script on the page runs.
        script = QWebEngineScript()
        script.setName("qwebchannel_bootstrap")
        script.setSourceUrl(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        self._page.scripts().insert(script)

        # Wire loadFinished only to catch hard page-load failures (404 on
        # index.html, scheme handler crash, etc.) and surface them through the
        # bridge error signal so they appear in the HUD overlay.
        # engine_ready is NOT emitted here — it is emitted by bridge.js_ready()
        # when JS explicitly calls it from inside the QWebChannel callback.
        self._page.loadFinished.connect(self._on_page_load_finished)

    def _on_page_load_finished(self, ok: bool) -> None:
        """Surface hard page-load failures through the bridge error signal.

        Called once when the page finishes loading.  On success (ok=True) we
        do nothing — engine_ready will arrive via bridge.js_ready() once the
        QWebChannel handshake completes.  On failure we emit asset_error so the
        HUD overlay shows a readable message rather than a silent blank screen.
        """
        # Disconnect immediately: we only care about the first load, and
        # leaving this connected would re-fire on any subsequent navigation.
        self._page.loadFinished.disconnect(self._on_page_load_finished)

        if not ok:
            self._bridge.asset_error.emit(
                "Page failed to load.\n\n"
                "Check that web/index.html exists and that the ht:// scheme "
                "handler is serving files correctly.  Run with the Python "
                "terminal visible to see scheme-handler errors."
            )

    # ------------------------------------------------------------------
    # Page load
    # ------------------------------------------------------------------

    def _load_page(self) -> None:
        """Load web/index.html via ht://app/index.html.

        Loading via the ht:// scheme (rather than a plain file:// URL) keeps
        the page within the ht:// origin, so fetch() calls to ht://assets/…
        are same-origin and are not blocked by the web engine's cross-origin
        policy.  A file:// origin cannot fetch ht:// sub-resources because the
        two schemes have different origins — that is why the manifest fetch
        would fail with "Failed to fetch" if we used QUrl.fromLocalFile here.

        The "app" host is handled by HtSchemeHandler, which maps it to WEB_DIR,
        so ht://app/<path> resolves to src/hometheater/web/<path>.
        """
        index_html = WEB_DIR / "index.html"
        if not index_html.exists():
            # This is a packaging error, not a user error.  Show it prominently.
            raise FileNotFoundError(
                f"web/index.html not found at expected path: {index_html}\n"
                "The web/ directory must be present alongside main.py inside "
                "the hometheater package."
            )
        self._view.setUrl(QUrl("ht://app/index.html"))

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop all audio sources cleanly before the window closes.

        QSpatialSound sources that are still playing when their QAudioEngine is
        destroyed can produce a brief audio glitch on some backends.  Explicitly
        pausing here avoids that.
        """
        self._engine.pause_all()
        super().closeEvent(event)


# ===========================================================================
# STEP 6: Application entry point
# ===========================================================================

def _show_fatal_error(title: str, message: str) -> None:
    """Show a QMessageBox for errors that occur before the main window is open.

    If QApplication has not been created yet (e.g. the manifest parse fails
    before QApplication.__init__), this function creates a minimal QApplication
    just to show the dialog, then exits.
    """
    # QApplication must exist before any widgets can be created.
    app = QApplication.instance() or QApplication(sys.argv)
    box = QMessageBox()
    box.setWindowTitle(f"Home Theater — {title}")
    box.setText(message)
    box.setIcon(QMessageBox.Icon.Critical)
    box.exec()
    sys.exit(1)


def main() -> None:
    """Application entry point.

    Called by the __main__ module (python -m hometheater) and by the
    [project.scripts] entry point defined in pyproject.toml.
    """

    # ------------------------------------------------------------------
    # 6a. Manifest — must exist and be valid before anything else runs.
    # ------------------------------------------------------------------
    try:
        manifest = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        _show_fatal_error("Manifest Error", str(exc))
        return  # unreachable — _show_fatal_error exits, but satisfies type checkers

    # ------------------------------------------------------------------
    # 6b. Asset pipeline — split/downmix the 5.1 FLAC and cache the results.
    #
    # This is the only blocking work before the window opens.  For a typical
    # ~90-second film clip, it takes a few seconds on first run; subsequent
    # runs skip it if the cache is current.  A progress indication in the HUD
    # would require running the pipeline in a QThread — not implemented here
    # (complexity vs. one-time cost tradeoff), but noted as a future improvement.
    # ------------------------------------------------------------------
    audio_51_path = ASSETS_DIR / "media" / manifest["audio_51_file"]
    try:
        cache_paths = run_pipeline(audio_51_path, CACHE_DIR)
    except FileNotFoundError as exc:
        _show_fatal_error(
            "Missing Audio Asset",
            f"The 5.1 audio master was not found:\n{audio_51_path}\n\n"
            f"Place the file at the path above and re-run.\n\nDetail: {exc}",
        )
        return
    except (ValueError, RuntimeError) as exc:
        _show_fatal_error("Asset Pipeline Error", str(exc))
        return

    # ------------------------------------------------------------------
    # 6c. QApplication — must be created before any Qt objects.
    #
    # IMPORTANT: QAudioEngine (inside SpatialAudioEngine) requires a
    # QApplication to exist before it is constructed. Do NOT move the
    # SpatialAudioEngine construction above this line.
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)
    app.setApplicationName("hometheater-demo")
    app.setApplicationVersion("0.1.0")
    # High-DPI scaling is on by default in Qt 6; no explicit flag needed.

    # ------------------------------------------------------------------
    # 6d. SpatialAudioEngine — constructs QAudioEngine, listener, room, sources.
    # SpatialAudioEngine.__init__ takes only an optional parent; the file map
    # from the asset pipeline is passed separately via initialise().
    #
    # parent=app: ensures Qt owns the engine in its object tree for the full
    # application lifetime. With parent=None, Python's GC can collect the
    # engine mid-playback since it has no knowledge of Qt's C++ references.
    # ------------------------------------------------------------------
    try:
        engine = SpatialAudioEngine(parent=app)
    except Exception as exc:
        # SpatialAudioEngine raises on import failure of QtSpatialAudio or on
        # a missing/broken platform multimedia backend.
        _show_fatal_error(
            "Audio Engine Error",
            f"Failed to initialise the spatial audio engine:\n\n{exc}\n\n"
            "If 'from PySide6.QtSpatialAudio import QAudioEngine' is the "
            "source of the error, the platform's Qt Multimedia backend plugin "
            "is likely missing.  Reinstall PySide6 for your platform, or "
            "check that the GStreamer or platform-audio packages are installed.",
        )
        return

    # ------------------------------------------------------------------
    # 6e. Bridge — QObject that wires JS slots/signals to the engine.
    #
    # IMPORTANT: Bridge must be constructed BEFORE engine.initialise() so
    # that engine.error is connected before initialise() runs. If initialise()
    # fails (e.g. QAudioEngine.start() raises), it emits engine.error — but
    # if the bridge isn't connected yet, that signal fires into the void and
    # the error is silently swallowed. We also wire a direct stderr print here
    # so any initialise() error is always visible in the terminal regardless
    # of whether the JS side is connected.
    # ------------------------------------------------------------------
    bridge = AudioBridge(engine=engine)

    # Direct terminal print for any engine error — catches failures that
    # occur before the QWebChannel handshake completes and JS can receive them.
    engine.error.connect(lambda msg: print(f"[audio_engine] ERROR: {msg}", flush=True))

    # Load the per-channel WAV paths produced by the asset pipeline.
    # cache_paths is a PipelineResult NamedTuple; .file_map is the dict
    # mapping channel key → Path that SpatialAudioEngine.initialise() expects.
    print("[audio_engine] calling initialise()...", flush=True)
    engine.initialise(cache_paths.file_map)
    print(f"[audio_engine] initialise() returned, is_initialised={engine.is_initialised}", flush=True)

    # ------------------------------------------------------------------
    # 6f. Main window — assembles view, channel, scheme handler.
    # ------------------------------------------------------------------
    try:
        window = MainWindow(manifest=manifest, engine=engine, bridge=bridge)
    except FileNotFoundError as exc:
        _show_fatal_error("Packaging Error", str(exc))
        return

    # engine_ready() is fired from the bridge once the web channel is open and
    # the JS side has confirmed it can reach all asset URLs.  The timing of
    # that signal is managed entirely by bridge.py; main.py doesn't need to
    # know when it fires.

    window.show()
    sys.exit(app.exec())


# ===========================================================================
# Module-level side effect: register the custom URL scheme BEFORE QApplication.
# This call must stay at module scope (not inside main()) because
# QWebEngineUrlScheme.registerScheme() is documented as requiring execution
# prior to QCoreApplication instantiation.  Placing it here ensures it runs
# during import, which happens before main() is called.
# ===========================================================================
_register_ht_scheme()


if __name__ == "__main__":
    main()
