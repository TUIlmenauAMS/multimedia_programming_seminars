/**
 * hud.js — HUD construction and control wiring for the Home Theater Spatial Audio Simulator
 *
 * Multimediaprogrammierung Chapter 05 companion — native PySide6 desktop app.
 * Author: Renato Fera P. / TU Ilmenau  // TODO: confirm author credit block
 *
 * LOAD ORDER CONTRACT (see index.html):
 *   This file is loaded BEFORE scene.js. It registers window.htSignals synchronously
 *   so that scene.js and index.html's bootstrap script can call into it immediately
 *   after their own initialization. Do not reorder the script tags.
 *
 * ARCHITECTURE:
 *   All HUD DOM is built programmatically here — index.html's #ht-hud div is an
 *   empty mount point. This keeps the HTML shell minimal and the HUD logic co-located
 *   with the event wiring that drives it.
 *
 *   Signal contract (window.htSignals — registered at module parse time):
 *     onEngineReady()                  ← called by index.html when Python fires engine_ready
 *     onAssetError(message)            ← called by scene.js or index.html on any asset failure
 *     onStateChanged(tier, room, play) ← called by index.html when Python fires state_changed
 *     onOffAxisReadout(text)           ← called by scene.js every frame with current offset
 *
 *   These are the only public surface hud.js exposes. scene.js and index.html must
 *   not touch HUD DOM directly.
 */

// ---------------------------------------------------------------------------
// STEP 1: Signal registry — registered synchronously at parse time
// ---------------------------------------------------------------------------

/**
 * window.htSignals is registered here, synchronously, before any other module
 * runs. index.html's bootstrap script reads it after the QWebChannel opens;
 * scene.js reads it in the render loop. Both depend on it existing immediately.
 */
window.htSignals = {
  onEngineReady,
  onAssetError,
  onStateChanged,
  onOffAxisReadout,
};

// ---------------------------------------------------------------------------
// STEP 2: UI state — single source of truth for confirmed engine state
// ---------------------------------------------------------------------------

/**
 * _state — reflects the last confirmed state received from Python via state_changed.
 * The HUD never assumes a switch succeeded; it waits for this to update.
 * Buttons show optimistic feedback (disabled while pending) but labels only update
 * on confirmation. See §5.3 of the build plan on the pause-before-switch constraint.
 */
const _state = {
  tier:       "mono",    // "mono" | "stereo" | "5.1"
  room:       "treated", // "treated" | "wet"
  isPlaying:  false,
  engineReady: false,
  // Mixer panel open/closed — UI-only, not reflected in Python state.
  mixerOpen:  false,
  // Per-speaker mute state for the mixer panel. Stored here so we can restore
  // mute states after a solo is cleared. Keyed by ITU speaker key.
  muteSaved:  { C: false, L: false, R: false, Ls: false, Rs: false, LFE: false },
  soloKey:    null, // which speaker is currently soloed, or null
};

// ---------------------------------------------------------------------------
// STEP 3: DOM references — stored as properties on the _el factory function.
//
// NOTE: _el is declared as a function at the bottom of this file (functions are
// hoisted, so it is available here at parse time). Named element references are
// attached directly to the function object, e.g. _el.startBtn = btn.
// The three dict-style refs (_el.tierBtns, _el.roomBtns, _el.speakerStrips) are
// initialised to empty objects immediately after the function declaration so that
// Object.values() / Object.entries() calls on them never see undefined.
// See the JSDoc on _el at the bottom of this file.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// STEP 4: Build the full HUD DOM — called once at DOMContentLoaded
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", _buildHUD);

function _buildHUD() {
  const root = document.getElementById("ht-hud");
  if (!root) {
    console.error("[hud.js] #ht-hud mount point not found in DOM.");
    return;
  }

  _buildLoadingOverlay(root);
  _buildTopLeft(root);
  _buildTopRight(root);
  _buildBottomBar(root);
  _buildMixerPanel(root);
  _buildFooter(root);
}

// ---------------------------------------------------------------------------
// STEP 5: Loading overlay
// ---------------------------------------------------------------------------

/**
 * _buildLoadingOverlay — the full-screen gate shown until Python fires engine_ready.
 *
 * CONCEPT: even though QWebEngineView is not subject to browser autoplay policy,
 * we keep the explicit click-to-start gesture for two reasons:
 *   1. UX consistency with the rest of the course's web demos, which all require it.
 *   2. Best practice for any audio-initializing application — the user should
 *      consciously start the experience, not have it begin unexpectedly.
 * See §7.5 of the build plan.
 *
 * The Start button is disabled until onEngineReady() is called, ensuring the student
 * never clicks into a half-initialized audio engine.
 */
function _buildLoadingOverlay(root) {
  const overlay = _el("div", "ht-overlay");

  // Headphone reminder — the entire premise of this app depends on headphones.
  const icon = _el("div", "ht-overlay-icon");
  icon.textContent = "🎧";
  overlay.appendChild(icon);

  const title = _el("div", "ht-overlay-title");
  title.textContent = "Home Theater Binaural Simulator";
  overlay.appendChild(title);

  const sub = _el("div", "ht-overlay-sub");
  sub.textContent = "Put on headphones before starting.";
  overlay.appendChild(sub);

  const note = _el("div", "ht-overlay-note");
  note.textContent =
    "All playback is binaurally virtualized via Qt Spatial Audio — " +
    "there is no physical speaker array. 5.1 means five virtual speakers " +
    "placed at ITU-R BS.775 angles and decoded for headphones.";
  overlay.appendChild(note);

  // Loading message — updated while Python initialises.
  const msg = _el("div", "ht-overlay-msg");
  msg.textContent = "Initialising audio engine…";
  overlay.appendChild(msg);
  _el.overlayMsg = msg;

  // Start button — disabled until onEngineReady().
  const btn = _el("button", "ht-btn ht-start-btn");
  btn.textContent = "Start";
  btn.disabled = true;
  btn.setAttribute("aria-label", "Start the experience");
  btn.addEventListener("click", _onStartClick);
  overlay.appendChild(btn);
  _el.startBtn = btn;

  // Error panel — hidden by default; shown by onAssetError().
  const errPanel = _el("div", "ht-error-panel");
  errPanel.setAttribute("role", "alert");
  errPanel.hidden = true;
  const errTitle = _el("div", "ht-error-title");
  errTitle.textContent = "Asset error";
  const errText = _el("div", "ht-error-text");
  errPanel.appendChild(errTitle);
  errPanel.appendChild(errText);
  overlay.appendChild(errPanel);
  _el.errorPanel = errPanel;
  _el.errorText  = errText;

  root.appendChild(overlay);
  _el.overlay = overlay;
}

function _onStartClick() {
  // Guard against double-click or re-entry after the overlay has already been
  // removed — _el.overlay is set to null in the transitionend handler below,
  // so a second call would crash on null.classList.add().
  if (!_el.overlay) return;

  // Fade out the overlay, then remove it from the DOM so it no longer blocks
  // mouse events on the Three.js canvas (which sits behind #ht-hud).
  _el.overlay.classList.add("ht-overlay--fade");
  _el.overlay.addEventListener("transitionend", () => {
    _el.overlay?.remove();
    _el.overlay = null;
  }, { once: true });

  // Do NOT call play() here. Start only dismisses the overlay and enters the
  // scene. The user controls playback explicitly with the Play/Pause button.
  // Audio and video remain silent until the user presses Play.
}

// ---------------------------------------------------------------------------
// STEP 6: Top-left readout — off-axis distance + status line
// ---------------------------------------------------------------------------

/**
 * _buildTopLeft — creates the off-axis position readout and engine status line.
 *
 * CONCEPT: displaying the offset from the sweet spot in real time turns the
 * movement constraint into a teaching moment — the student sees "32 cm left"
 * while hearing the stereo image shift, connecting the physical explanation
 * ("move off-axis and the mix degrades") to their direct experience.
 * See §7.4 of the build plan.
 */
function _buildTopLeft(root) {
  const block = _el("div", "ht-top-left");

  const readout = _el("div", "ht-offaxis-readout");
  readout.textContent = "sweet spot";
  readout.setAttribute("aria-live", "polite");
  block.appendChild(readout);
  _el.offAxisReadout = readout;

  const status = _el("div", "ht-status-line");
  status.textContent = "—";
  block.appendChild(status);
  _el.statusLine = status;

  root.appendChild(block);
}

// ---------------------------------------------------------------------------
// STEP 7: Top-right header — logo, author, LinkedIn, course/institute line
// ---------------------------------------------------------------------------

/**
 * _buildTopRight — course branding block.
 *
 * The LinkedIn badge calls bridge.open_url() which routes through
 * QDesktopServices.openUrl() on the Python side. We never use window.open()
 * here — QWebEngineView's default policy may block popup windows.
 * See §6 of the build plan (open_url slot).
 */
function _buildTopRight(root) {
  const block = _el("div", "ht-top-right");

  // AMS / TU Ilmenau logo text — replace with <img> if a logo asset is added.
  const logo = _el("div", "ht-logo");
  logo.textContent = "TU Ilmenau";
  block.appendChild(logo);

  const author = _el("div", "ht-author");
  author.textContent = "Renato Fera P.";  // TODO: confirm author credit block
  block.appendChild(author);

  const course = _el("div", "ht-course");
  course.textContent = "Multimediaprogrammierung · Chapter 05";
  block.appendChild(course);

  // LinkedIn badge — opens via Python bridge, not window.open().
  const linkedIn = _el("button", "ht-btn ht-linkedin-btn");
  linkedIn.textContent = "LinkedIn";
  linkedIn.setAttribute("aria-label", "Open LinkedIn profile");
  linkedIn.addEventListener("click", () => {
    // bridge.open_url() → Python → QDesktopServices.openUrl(QUrl(...))
    // The URL is a placeholder; replace with the actual profile URL.
    window.htBridge?.open_url("https://www.linkedin.com/in/TODO");  // TODO: confirm LinkedIn URL
  });
  block.appendChild(linkedIn);

  root.appendChild(block);
  _el.headerBlock = block;
}

// ---------------------------------------------------------------------------
// STEP 8: Bottom bar — Play/Pause, Tier buttons, Room buttons
// ---------------------------------------------------------------------------

/**
 * _buildBottomBar — the primary transport and format controls.
 *
 * Layout (bottom center):
 *   [Play/Pause]   [Mono] [Stereo] [5.1]   [Treated] [Wet]
 *
 * Tier and Room buttons are disabled while isPlaying — the QSpatialSound API
 * has no seek-to-timestamp method, so switching mid-playback would cause a restart
 * from 0:00 without synchronised video. The UI enforces pause-before-switch rather
 * than letting the student discover this as a glitch. See §5.3 of the build plan.
 */
function _buildBottomBar(root) {
  const bar = _el("div", "ht-bottom-bar");

  // --- Transport ---
  const transport = _el("div", "ht-transport");

  const playPause = _el("button", "ht-btn ht-playpause-btn");
  playPause.textContent = "▶ Play";
  playPause.setAttribute("aria-label", "Play or pause");
  playPause.addEventListener("click", _onPlayPauseClick);
  transport.appendChild(playPause);
  _el.playPauseBtn = playPause;

  bar.appendChild(transport);

  // --- Tier buttons ---
  const tierGroup = _el("div", "ht-btn-group");
  tierGroup.setAttribute("role", "group");
  tierGroup.setAttribute("aria-label", "Surround tier");

  const tiers = [
    { key: "mono",   label: "Mono"   },
    { key: "stereo", label: "Stereo" },
    { key: "5.1",    label: "5.1"    },
  ];

  for (const { key, label } of tiers) {
    const btn = _el("button", "ht-btn ht-tier-btn");
    btn.textContent = label;
    btn.dataset.tier = key;
    btn.setAttribute("aria-pressed", key === _state.tier ? "true" : "false");
    if (key === _state.tier) btn.classList.add("ht-btn--active");
    btn.addEventListener("click", () => _onTierClick(key));
    tierGroup.appendChild(btn);
    _el.tierBtns[key] = btn;
  }

  bar.appendChild(tierGroup);

  // --- Room buttons ---
  const roomGroup = _el("div", "ht-btn-group");
  roomGroup.setAttribute("role", "group");
  roomGroup.setAttribute("aria-label", "Room acoustic treatment");

  const rooms = [
    { key: "treated", label: "Treated" },
    { key: "wet",     label: "Wet"     },
  ];

  for (const { key, label } of rooms) {
    const btn = _el("button", "ht-btn ht-room-btn");
    btn.textContent = label;
    btn.dataset.room = key;
    btn.setAttribute("aria-pressed", key === _state.room ? "true" : "false");
    if (key === _state.room) btn.classList.add("ht-btn--active");
    btn.addEventListener("click", () => _onRoomClick(key));
    roomGroup.appendChild(btn);
    _el.roomBtns[key] = btn;
  }

  bar.appendChild(roomGroup);

  // --- Mixer toggle ---
  const mixerToggle = _el("button", "ht-btn ht-mixer-toggle-btn");
  mixerToggle.textContent = "Mixer";
  mixerToggle.setAttribute("aria-expanded", "false");
  mixerToggle.setAttribute("aria-controls", "ht-mixer-panel");
  mixerToggle.addEventListener("click", _onMixerToggleClick);
  bar.appendChild(mixerToggle);
  _el.mixerToggleBtn = mixerToggle;

  root.appendChild(bar);
}

// ---------------------------------------------------------------------------
// STEP 9: Mixer panel — per-speaker Mute/Solo strips, tier-aware
// ---------------------------------------------------------------------------

/**
 * _buildMixerPanel — floating panel with one strip per virtual speaker.
 *
 * CONCEPT: the mixer panel makes the channel-separation that defines each tier
 * tangible — the student can solo the center channel and hear that in Mono it
 * carries the full mix, while in 5.1 it carries only dialogue. This directly
 * illustrates the discrete vs. derived content distinction from §4.2.
 *
 * "Tier-aware" means only speakers active in the current tier show their strips
 * as interactive — inactive speakers are visually dimmed and their Mute/Solo
 * buttons are disabled, because muting a source that is already volume=0 in
 * Python would have no audible effect and would confuse the student.
 *
 * Solo behaviour: soloing speaker X mutes all others (saving their prior mute
 * states), and clearing the solo restores those states.
 */
function _buildMixerPanel(root) {
  const panel = _el("div", "ht-mixer-panel");
  panel.id = "ht-mixer-panel";
  panel.hidden = true;
  panel.setAttribute("role", "region");
  panel.setAttribute("aria-label", "Speaker mixer");

  const title = _el("div", "ht-mixer-title");
  title.textContent = "Speaker Mixer";
  panel.appendChild(title);

  // Define all six speaker strips. Display order: C, L, R, Ls, Rs, LFE.
  const speakers = [
    { key: "C",   label: "Center"         },
    { key: "L",   label: "Left"           },
    { key: "R",   label: "Right"          },
    { key: "Ls",  label: "Surr L"         },
    { key: "Rs",  label: "Surr R"         },
    { key: "LFE", label: "LFE"            },
  ];

  for (const { key, label } of speakers) {
    const row = _el("div", "ht-mixer-strip");
    row.dataset.speakerKey = key;

    const lbl = _el("span", "ht-mixer-label");
    lbl.textContent = label;
    row.appendChild(lbl);

    const muteBtn = _el("button", "ht-btn ht-mixer-btn ht-mute-btn");
    muteBtn.textContent = "M";
    muteBtn.setAttribute("aria-label", `Mute ${label}`);
    muteBtn.setAttribute("aria-pressed", "false");
    muteBtn.addEventListener("click", () => _onMuteClick(key));
    row.appendChild(muteBtn);

    const soloBtn = _el("button", "ht-btn ht-mixer-btn ht-solo-btn");
    soloBtn.textContent = "S";
    soloBtn.setAttribute("aria-label", `Solo ${label}`);
    soloBtn.setAttribute("aria-pressed", "false");
    soloBtn.addEventListener("click", () => _onSoloClick(key));
    row.appendChild(soloBtn);

    panel.appendChild(row);
    _el.speakerStrips[key] = { row, muteBtn, soloBtn, label: lbl };
  }

  root.appendChild(panel);
  _el.mixerPanel = panel;

  // Initial render of strip active states for the default tier (mono).
  _updateMixerStrips(_state.tier);
}

// ---------------------------------------------------------------------------
// STEP 10: Footer — film credit, binaural note
// ---------------------------------------------------------------------------

/**
 * _buildFooter — bottom edge credit line, manifest-driven film name.
 *
 * CONCEPT: the film credit is mandatory under CC-BY 3.0 (Blender Foundation).
 * The film name is not hardcoded — it is read from htManifest.credit at runtime
 * so this app works with any Blender Foundation open movie without code changes.
 * See §8 of the build plan on the license/credit requirement.
 */
function _buildFooter(root) {
  const footer = _el("div", "ht-footer");

  // Film credit — populated once htManifest is available (may load slightly
  // after DOMContentLoaded; we re-check in onEngineReady()).
  const credit = _el("span", "ht-credit");
  credit.id = "ht-film-credit";
  credit.textContent = _getFilmCredit();
  footer.appendChild(credit);

  root.appendChild(footer);
}

/**
 * _getFilmCredit — builds the CC-BY credit string from the manifest.
 * Falls back to a placeholder if the manifest is not yet loaded.
 */
function _getFilmCredit() {
  const m = window.htManifest;
  // manifest.json uses "film_credit" as the key (not "credit").
  const c = m?.film_credit ?? m?.credit;
  if (!c) return "Film: [loading…] © Blender Foundation, CC-BY 3.0";
  return `Film: "${c.title}" © ${c.author ?? "Blender Foundation"}, ${c.license ?? "CC-BY 3.0"}`;
}

// ---------------------------------------------------------------------------
// STEP 11: Event handlers — transport, tier, room, mixer
// ---------------------------------------------------------------------------

function _onPlayPauseClick() {
  if (!_state.engineReady) return;

  if (_state.isPlaying) {
    window.htBridge?.pause();
    // Optimistic UI: update label immediately; onStateChanged() confirms.
    _el.playPauseBtn.textContent = "▶ Play";
    // Re-enable tier/room switches now that we're pausing.
    _setFormatControlsDisabled(false);
  } else {
    window.htBridge?.play();
    _el.playPauseBtn.textContent = "⏸ Pause";
    // Disable tier/room switches while playing — QSpatialSound has no seek,
    // so a mid-playback switch would restart from 0:00 without synced video.
    // See §5.3 of the build plan.
    _setFormatControlsDisabled(true);
  }
}

/**
 * _onTierClick — sends a tier switch request to Python.
 *
 * We do not update the active button state here — that happens in onStateChanged()
 * after Python confirms the switch. This prevents a visual lie if Python rejects
 * the request (e.g. if the bridge receives it while playing due to a race).
 */
function _onTierClick(key) {
  if (_state.isPlaying) return;  // defensive — buttons should already be disabled
  if (key === _state.tier) return;

  // Disable all tier buttons while the switch is in-flight.
  for (const btn of Object.values(_el.tierBtns)) btn.disabled = true;

  window.htBridge?.set_tier(key);
}

/**
 * _onRoomClick — sends a room variant switch request to Python.
 * Same optimistic-disable, confirm-on-signal pattern as tier.
 * Simultaneously updates the Three.js wall materials so the visual change
 * is immediate; Python's state_changed will confirm the acoustic change.
 */
function _onRoomClick(key) {
  if (_state.isPlaying) return;
  if (key === _state.room) return;

  for (const btn of Object.values(_el.roomBtns)) btn.disabled = true;

  // Optimistic visual update — scene.js applies the material change immediately.
  // If Python rejects, onStateChanged() will call setRoomVariant() again with
  // the previous state, reverting the visual.
  window.htScene?.setRoomVariant(key);

  window.htBridge?.set_room(key);
}

function _onMixerToggleClick() {
  _state.mixerOpen = !_state.mixerOpen;
  _el.mixerPanel.hidden = !_state.mixerOpen;
  _el.mixerToggleBtn.setAttribute("aria-expanded", String(_state.mixerOpen));
  _el.mixerToggleBtn.classList.toggle("ht-btn--active", _state.mixerOpen);
}

/**
 * _onMuteClick — toggles mute for a single speaker.
 *
 * We track mute state locally (_state.muteSaved) and sync to Python via
 * bridge.set_speaker_mute(key, muted). This is a mixer-panel action, not a
 * tier switch — it doesn't go through the pause-before-switch gate.
 */
function _onMuteClick(key) {
  _state.muteSaved[key] = !_state.muteSaved[key];
  const muted = _state.muteSaved[key];

  const strip = _el.speakerStrips[key];
  strip.muteBtn.setAttribute("aria-pressed", String(muted));
  strip.muteBtn.classList.toggle("ht-mixer-btn--active", muted);
  strip.row.classList.toggle("ht-mixer-strip--muted", muted);

  window.htBridge?.set_speaker_mute(key, muted);
}

/**
 * _onSoloClick — solos a speaker (mutes all others, saves their mute state),
 * or clears the solo (restores saved mute states).
 *
 * CONCEPT: solo is a standard mixing-console feature that lets you audition a
 * single channel in isolation. Here it teaches the student what each discrete
 * channel contributes — soling the Center in 5.1 reveals only the dialogue track.
 */
function _onSoloClick(key) {
  if (_state.soloKey === key) {
    // Clear solo — restore all speakers to their saved mute states.
    _state.soloKey = null;
    for (const k of Object.keys(_state.muteSaved)) {
      const strip = _el.speakerStrips[k];
      strip.soloBtn.setAttribute("aria-pressed", "false");
      strip.soloBtn.classList.remove("ht-mixer-btn--active");
      strip.row.classList.toggle("ht-mixer-strip--muted", _state.muteSaved[k]);
      window.htBridge?.set_speaker_mute(k, _state.muteSaved[k]);
    }
  } else {
    // Set solo — mute all others, preserve their saved states for restoration.
    _state.soloKey = key;
    for (const k of Object.keys(_state.muteSaved)) {
      const isSoloed = k === key;
      const strip = _el.speakerStrips[k];
      strip.soloBtn.setAttribute("aria-pressed", String(isSoloed));
      strip.soloBtn.classList.toggle("ht-mixer-btn--active", isSoloed);
      // Mute state visual: muted if not the soloed speaker.
      strip.row.classList.toggle("ht-mixer-strip--muted", !isSoloed);
      // Send to Python: soloed speaker unmuted, all others muted.
      window.htBridge?.set_speaker_mute(k, !isSoloed);
    }
  }
}

// ---------------------------------------------------------------------------
// STEP 12: Signal handlers — called from outside (index.html bootstrap, scene.js)
// ---------------------------------------------------------------------------

/**
 * onEngineReady — called by index.html's bootstrap script when Python fires
 * the engine_ready signal. Gates the Start button and updates the loading message.
 *
 * CONCEPT: the Start button is intentionally disabled until the audio engine
 * is fully constructed and all asset paths confirmed loadable. If the button
 * were enabled before this, the student might click Start into a half-initialized
 * engine and hear nothing, with no explanation. See §7.5 of the build plan.
 */
function onEngineReady() {
  _state.engineReady = true;

  if (_el.overlayMsg) {
    _el.overlayMsg.textContent = "Engine ready. Headphones on?";
  }
  if (_el.startBtn) {
    _el.startBtn.disabled = false;
    _el.startBtn.focus();
  }

  // Update film credit now that the manifest is confirmed loaded.
  const creditEl = document.getElementById("ht-film-credit");
  if (creditEl) creditEl.textContent = _getFilmCredit();

  _setStatus("Ready");
}

/**
 * onAssetError — called by scene.js or index.html when any asset fails to load.
 * Displays the error in the overlay's error panel, keeping the overlay visible.
 * Never routes to console only — the error must be legible in the UI.
 * See §1 and §6 of the build plan.
 *
 * @param {string} message — human-readable, names the missing file and expected path
 */
function onAssetError(message) {
  // Ensure the overlay is still visible (or re-show it if already dismissed).
  if (_el.overlay) {
    _el.overlay.classList.remove("ht-overlay--fade");
    _el.overlay.hidden = false;
  }

  if (_el.errorPanel) {
    _el.errorText.textContent = message;
    _el.errorPanel.hidden = false;
  }

  if (_el.startBtn) {
    _el.startBtn.disabled = true;
  }

  _setStatus("Error — see overlay");
  console.error("[hud.js] Asset error:", message);
}

/**
 * onStateChanged — called by index.html when Python fires the state_changed signal.
 *
 * This is the confirmation step for every tier and room switch. We only update
 * the HUD labels and button states here — not on click — so the UI always reflects
 * the engine's actual confirmed state, never an assumed or pending state.
 *
 * Also drives scene.js tier and room visual updates, which we could not confirm
 * earlier because Python might have rejected the request.
 *
 * @param {string}  tier       — "mono" | "stereo" | "5.1"
 * @param {string}  room       — "treated" | "wet"
 * @param {boolean} isPlaying  — current playback state
 */
function onStateChanged(tier, room, isPlaying) {
  const tierChanged = tier !== _state.tier;
  const roomChanged = room !== _state.room;

  _state.tier      = tier;
  _state.room      = room;
  _state.isPlaying = isPlaying;

  // Update Play/Pause button.
  if (_el.playPauseBtn) {
    _el.playPauseBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  }

  // Update tier button active states.
  for (const [k, btn] of Object.entries(_el.tierBtns)) {
    const active = k === tier;
    btn.classList.toggle("ht-btn--active", active);
    btn.setAttribute("aria-pressed", String(active));
    btn.disabled = false;  // re-enable after in-flight switch resolves
  }

  // Update room button active states.
  for (const [k, btn] of Object.entries(_el.roomBtns)) {
    const active = k === room;
    btn.classList.toggle("ht-btn--active", active);
    btn.setAttribute("aria-pressed", String(active));
    btn.disabled = false;
  }

  // Disable tier/room controls if now playing (they may have been re-enabled
  // by an optimistic update that Python then confirmed with isPlaying=true).
  _setFormatControlsDisabled(isPlaying);

  // Push confirmed tier/room to scene.js visual layer.
  if (tierChanged && window.htScene?.setActiveTier) {
    window.htScene.setActiveTier(tier);
  }
  if (roomChanged && window.htScene?.setRoomVariant) {
    window.htScene.setRoomVariant(room);
  }

  // Refresh mixer strip active states for the new tier.
  if (tierChanged) _updateMixerStrips(tier);

  _setStatus(`${_tierLabel(tier)} · ${_roomLabel(room)}${isPlaying ? " · Playing" : ""}`);
}

/**
 * onOffAxisReadout — called by scene.js every frame with the latest offset string.
 * Throttling is handled upstream (scene.js calls this inside the render loop, which
 * runs at the render frame rate). We just apply the string — no DOM thrash.
 *
 * @param {string} text — e.g. "14 cm left, 8 cm forward" or "sweet spot"
 */
function onOffAxisReadout(text) {
  if (_el.offAxisReadout && _el.offAxisReadout.textContent !== text) {
    _el.offAxisReadout.textContent = text;
  }
}

// ---------------------------------------------------------------------------
// STEP 13: Internal helpers
// ---------------------------------------------------------------------------

/**
 * _setFormatControlsDisabled — enables or disables all tier and room buttons.
 * Called when playback starts (disable) or pauses (enable).
 *
 * @param {boolean} disabled
 */
function _setFormatControlsDisabled(disabled) {
  for (const btn of Object.values(_el.tierBtns))  btn.disabled = disabled;
  for (const btn of Object.values(_el.roomBtns))  btn.disabled = disabled;
}

/**
 * _updateMixerStrips — dims and disables strips for speakers not active in the
 * current tier. Only active speakers are interactive.
 *
 * Active sets mirror audio_engine.py §5.3 exactly — same mapping as scene.js.
 *
 * @param {string} tier
 */
function _updateMixerStrips(tier) {
  const activeSets = {
    "mono":   new Set(["C"]),
    "stereo": new Set(["L", "R"]),
    "5.1":    new Set(["C", "L", "R", "Ls", "Rs", "LFE"]),
  };

  const active = activeSets[tier] ?? new Set();

  for (const [key, strip] of Object.entries(_el.speakerStrips)) {
    const isActive = active.has(key);
    strip.row.classList.toggle("ht-mixer-strip--inactive", !isActive);
    strip.muteBtn.disabled = !isActive;
    strip.soloBtn.disabled = !isActive;
  }
}

function _setStatus(text) {
  if (_el.statusLine) _el.statusLine.textContent = text;
}

function _tierLabel(tier) {
  return { mono: "Mono", stereo: "Stereo", "5.1": "5.1" }[tier] ?? tier;
}

function _roomLabel(room) {
  return { treated: "Treated", wet: "Wet" }[room] ?? room;
}

/**
 * _el — dual-role helper: DOM element factory AND named-ref store.
 *
 * As a factory (called as a function):
 *   const div = _el("div", "ht-some-class another-class");
 *
 * As a ref store (properties set directly on the function object):
 *   _el.someRef = div;
 *
 * This works because functions are first-class objects in JavaScript.
 * Function declarations are hoisted to the top of their scope, so _el is
 * available as a function throughout this file even though it is declared here
 * at the bottom. The dict-style refs (_el.tierBtns, _el.roomBtns,
 * _el.speakerStrips) are initialised to empty objects immediately below so
 * that Object.values() / Object.entries() calls on them never see undefined.
 *
 * IMPORTANT: do NOT also declare `const _el = { … }` anywhere in this file.
 * A const declaration and a function declaration with the same name in the
 * same scope is a SyntaxError ("Identifier '_el' has already been declared").
 */
function _el(tag, classes) {
  const node = document.createElement(tag);
  if (classes) node.className = classes;
  return node;
}

// Initialise the dict-style ref buckets on the function object.
// All other refs start as undefined (falsy) which is fine — every consumer
// guards with `if (_el.xxx)` before reading them.
_el.tierBtns      = {};
_el.roomBtns      = {};
_el.speakerStrips = {};
