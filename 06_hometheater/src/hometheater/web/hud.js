/**
 * hud.js — HUD construction and control wiring for the Home Theater Spatial Audio Simulator
 *
 * Multimediaprogrammierung Chapter 05 companion — native PySide6 desktop app.
 * Author: Renato Fera P. / TU Ilmenau
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
 *
 * SOLO BEHAVIOUR (multi-solo):
 *   Multiple speakers can be soloed simultaneously. Clicking S on an un-soloed speaker
 *   adds it to the active solo set (muting all non-soloed speakers). Clicking S on an
 *   already-soloed speaker removes it. When the solo set becomes empty, all speakers
 *   are restored to their individually saved mute states.
 *   This matches DAW convention: solo is additive, not exclusive.
 */

// ---------------------------------------------------------------------------
// STEP 1: Signal registry — registered synchronously at parse time
// ---------------------------------------------------------------------------

window.htSignals = {
  onEngineReady,
  onAssetError,
  onStateChanged,
  onOffAxisReadout,
};

// ---------------------------------------------------------------------------
// STEP 2: UI state
// ---------------------------------------------------------------------------

const _state = {
  tier:        "mono",
  room:        "treated",
  isPlaying:   false,
  engineReady: false,
  mixerOpen:   false,
  muteSaved:   { C: false, L: false, R: false, Ls: false, Rs: false, LFE: false },
  soloSet:     new Set(),
  leftOpen:    true,
  rightOpen:   true,
};

// ---------------------------------------------------------------------------
// STEP 3: Build HUD
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", _buildHUD);

function _buildHUD() {
  const root = document.getElementById("ht-hud");
  if (!root) {
    console.error("[hud.js] #ht-hud mount point not found in DOM.");
    return;
  }
  _injectStyles();
  _buildLoadingOverlay(root);
  _buildLeftSidebar(root);
  _buildRightPanel(root);
  _buildFooter(root);
}

// ---------------------------------------------------------------------------
// STEP 4: Styles
// ---------------------------------------------------------------------------

function _injectStyles() {
  const style = document.createElement("style");
  style.textContent = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:          #0d0f14;
      --surface:     #161a22;
      --surface-2:   #1e2430;
      --border:      rgba(255,255,255,.08);
      --accent:      #00c8c8;
      --accent-dim:  rgba(0,200,200,.15);
      --amber:       #f59e0b;
      --amber-dim:   rgba(245,158,11,.15);
      --green:       #22c55e;
      --green-dim:   rgba(34,197,94,.15);
      --red:         #ef4444;
      --red-dim:     rgba(239,68,68,.15);
      --text:        #e2e8f0;
      --text-2:      #94a3b8;
      --text-3:      #475569;
      --radius:      6px;
      --radius-sm:   4px;
      --font:        'Inter', system-ui, -apple-system, sans-serif;
      --font-mono:   'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
      --sidebar-w:   264px;
      --diag-w:      248px;
      --tab-w:       20px;
      --transition:  .22s cubic-bezier(.4,0,.2,1);
    }

    body { font-family: var(--font); color: var(--text); background: transparent; overflow: hidden; }

    #ht-hud { position: fixed; inset: 0; pointer-events: none; z-index: 10; }
    #ht-hud > * { pointer-events: auto; }

    /* ── Overlay ── */
    .ht-overlay {
      position: fixed; inset: 0;
      background: var(--bg);
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px;
      z-index: 100;
      transition: opacity .4s ease;
    }
    .ht-overlay--fade { opacity: 0; pointer-events: none; }
    .ht-overlay-icon  { font-size: 56px; line-height: 1; margin-bottom: 4px; }
    .ht-overlay-title { font-size: 22px; font-weight: 600; letter-spacing: -.3px; }
    .ht-overlay-sub   { font-size: 13px; color: var(--accent); font-weight: 500; }
    .ht-overlay-note  { font-size: 12px; color: var(--text-2); max-width: 440px; text-align: center; line-height: 1.6; }
    .ht-overlay-msg   { font-size: 12px; color: var(--text-3); font-family: var(--font-mono); }

    .ht-start-btn {
      margin-top: 8px; padding: 11px 36px;
      background: var(--accent); color: #000;
      border: none; border-radius: var(--radius);
      font-size: 14px; font-weight: 600; font-family: var(--font);
      cursor: pointer; transition: opacity .15s, transform .1s;
    }
    .ht-start-btn:hover:not(:disabled) { opacity: .88; }
    .ht-start-btn:active:not(:disabled) { transform: scale(.97); }
    .ht-start-btn:disabled { opacity: .3; cursor: not-allowed; }

    .ht-error-panel {
      background: var(--red-dim); border: 1px solid var(--red);
      border-radius: var(--radius); padding: 12px 16px; max-width: 440px;
    }
    .ht-error-title { font-size: 11px; font-weight: 700; color: var(--red); margin-bottom: 4px; letter-spacing: .05em; }
    .ht-error-text  { font-size: 11px; color: var(--text-2); font-family: var(--font-mono); line-height: 1.5; }

    /* ── Left sidebar wrapper ── */
    .ht-left-wrap {
      position: fixed; top: 0; left: 0; bottom: 0;
      display: flex; align-items: stretch;
      z-index: 20;
    }

    .ht-sidebar {
      width: var(--sidebar-w);
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column;
      overflow-y: auto; overflow-x: hidden;
      transition: width var(--transition), opacity var(--transition);
      will-change: width;
    }
    .ht-sidebar--collapsed { width: 0; opacity: 0; border-right: none; }
    .ht-sidebar::-webkit-scrollbar { width: 3px; }
    .ht-sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    /* Collapse tab — always visible, sits to the right of the sidebar */
    .ht-collapse-tab {
      width: var(--tab-w);
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: none;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      color: var(--text-3);
      font-size: 11px;
      align-self: center;
      height: 48px;
      transition: color .15s, background .15s;
      flex-shrink: 0;
      user-select: none;
    }
    .ht-collapse-tab:hover { color: var(--text); background: var(--surface-2); }

    /* ── Right panel wrapper ── */
    .ht-right-wrap {
      position: fixed; top: 0; right: 0; bottom: 0;
      display: flex; align-items: stretch;
      flex-direction: row-reverse;
      z-index: 20;
    }

    .ht-diag-panel {
      width: var(--diag-w);
      background: var(--surface);
      border-left: 1px solid var(--border);
      display: flex; flex-direction: column;
      overflow-y: auto; overflow-x: hidden;
      transition: width var(--transition), opacity var(--transition);
      will-change: width;
    }
    .ht-diag-panel--collapsed { width: 0; opacity: 0; border-left: none; }
    .ht-diag-panel::-webkit-scrollbar { width: 3px; }
    .ht-diag-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .ht-collapse-tab-right {
      width: var(--tab-w);
      background: var(--surface);
      border: 1px solid var(--border);
      border-right: none;
      border-radius: var(--radius-sm) 0 0 var(--radius-sm);
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      color: var(--text-3);
      font-size: 11px;
      align-self: center;
      height: 48px;
      flex-shrink: 0;
      transition: color .15s, background .15s;
      user-select: none;
    }
    .ht-collapse-tab-right:hover { color: var(--text); background: var(--surface-2); }

    /* ── Branding ── */
    .ht-brand {
      padding: 14px 14px 12px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: flex-start; gap: 10px;
      flex-shrink: 0;
    }
    .ht-brand-logo {
      width: 38px; height: 38px;
      border-radius: var(--radius-sm);
      object-fit: contain;
      background: var(--surface-2);
      flex-shrink: 0;
    }
    .ht-brand-logo-fallback {
      width: 38px; height: 38px;
      border-radius: var(--radius-sm);
      background: var(--surface-2);
      flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px; font-weight: 700; color: var(--accent); letter-spacing: .04em;
    }
    .ht-brand-text { flex: 1; min-width: 0; }
    .ht-brand-chapter {
      font-size: 9px; font-weight: 600; letter-spacing: .1em;
      color: var(--text-3); text-transform: uppercase; margin-bottom: 2px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .ht-brand-title {
      font-size: 12px; font-weight: 600; color: var(--text);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .ht-brand-author { font-size: 11px; color: var(--text-2); margin-top: 1px; }
    .ht-linkedin-btn {
      display: inline-flex; align-items: center; gap: 4px;
      margin-top: 5px; padding: 3px 8px;
      background: transparent;
      border: 1px solid rgba(10,102,194,.5);
      border-radius: var(--radius-sm);
      color: #5aa3e0; font-size: 10px; font-weight: 500; font-family: var(--font);
      cursor: pointer; transition: background .15s, border-color .15s;
    }
    .ht-linkedin-btn:hover { background: rgba(10,102,194,.12); border-color: rgba(10,102,194,.8); }
    .ht-linkedin-icon { width: 10px; height: 10px; flex-shrink: 0; }

    /* ── Section ── */
    .ht-section { padding: 16px 14px 12px; flex-shrink: 0; }
    .ht-section + .ht-section { border-top: 1px solid var(--border); }
    .ht-section-label {
      font-size: 9px; font-weight: 700; letter-spacing: .11em;
      color: var(--text-3); text-transform: uppercase; margin-bottom: 10px;
    }

    /* ── Transport ── */
    .ht-transport { display: flex; gap: 7px; }

    .ht-play-btn {
      flex: 1;
      display: flex; align-items: center; justify-content: center; gap: 7px;
      padding: 9px 12px;
      background: var(--green); border: none; border-radius: var(--radius);
      color: #000; font-size: 13px; font-weight: 700; font-family: var(--font);
      cursor: pointer; transition: background .15s, transform .1s;
    }
    .ht-play-btn:hover:not(:disabled) { background: #16a34a; }
    .ht-play-btn:active:not(:disabled) { transform: scale(.97); }
    .ht-play-btn:disabled { opacity: .35; cursor: not-allowed; }
    .ht-play-btn--pause { background: var(--red); }
    .ht-play-btn--pause:hover:not(:disabled) { background: #dc2626; }

    .ht-reset-btn {
      display: flex; align-items: center; justify-content: center;
      padding: 9px 10px;
      background: transparent;
      border: 1px solid var(--border); border-radius: var(--radius);
      color: var(--text-2); font-size: 11px; font-weight: 500; font-family: var(--font);
      cursor: pointer; transition: border-color .15s, color .15s, background .15s;
      white-space: nowrap; flex-shrink: 0;
    }
    .ht-reset-btn:hover { border-color: var(--text-3); color: var(--text); background: var(--surface-2); }

    /* ── Pill groups ── */
    .ht-pill-group { display: flex; gap: 5px; }
    .ht-pill-group--col { flex-direction: column; gap: 5px; }

    .ht-pill-btn {
      flex: 1;
      display: flex; align-items: center; justify-content: space-between; gap: 6px;
      padding: 8px 10px;
      background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius);
      color: var(--text-2); font-size: 12px; font-weight: 500; font-family: var(--font);
      cursor: pointer; transition: border-color .15s, color .15s, background .15s;
      text-align: left; white-space: nowrap;
    }
    .ht-pill-btn:hover:not(:disabled) { border-color: var(--text-3); color: var(--text); background: #232a38; }
    .ht-pill-btn:disabled { opacity: .35; cursor: not-allowed; }
    .ht-pill-btn--active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
    .ht-pill-btn--active:hover:not(:disabled) { background: rgba(0,200,200,.22); }

    .ht-pill-left { display: flex; align-items: center; gap: 7px; min-width: 0; }
    .ht-pill-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--border); flex-shrink: 0; transition: background .15s;
    }
    .ht-pill-btn--active .ht-pill-dot { background: var(--accent); }
    .ht-rt60-chip {
      font-family: var(--font-mono); font-size: 9px; color: var(--text-3);
      flex-shrink: 0; transition: color .15s;
    }
    .ht-pill-btn--active .ht-rt60-chip { color: var(--accent); }

    /* ── Mixer ── */
    .ht-mixer-toggle-btn {
      display: flex; align-items: center; justify-content: space-between;
      /* NOT width:100% — size to content + a reasonable cap */
      width: fit-content; min-width: 120px; max-width: 160px;
      padding: 6px 10px;
      background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius);
      color: var(--text-2); font-size: 11px; font-weight: 500; font-family: var(--font);
      cursor: pointer; transition: border-color .15s, color .15s, background .15s;
      gap: 8px;
    }
    .ht-mixer-toggle-btn:hover { border-color: var(--text-3); color: var(--text); background: #232a38; }
    .ht-mixer-toggle-btn--open { border-color: var(--text-3); color: var(--text); }
    .ht-mixer-caret { font-size: 9px; transition: transform .18s; display: inline-block; flex-shrink: 0; }
    .ht-mixer-toggle-btn--open .ht-mixer-caret { transform: rotate(180deg); }

    .ht-mixer-panel { display: none; flex-direction: column; gap: 1px; margin-top: 8px; }
    .ht-mixer-panel--open { display: flex; }

    .ht-mixer-strip {
      display: flex; align-items: center; gap: 5px;
      padding: 5px 6px; border-radius: var(--radius-sm);
      transition: opacity .15s;
    }
    .ht-mixer-strip--inactive { opacity: .28; pointer-events: none; }
    .ht-mixer-strip--muted .ht-mixer-label { opacity: .4; }
    .ht-mixer-label { flex: 1; font-size: 11px; color: var(--text); font-weight: 500; transition: opacity .15s; }

    .ht-ms-btn {
      width: 26px; height: 22px;
      border-radius: var(--radius-sm); border: 1px solid var(--border);
      background: var(--surface-2); color: var(--text-3);
      font-size: 10px; font-weight: 700; font-family: var(--font);
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      transition: background .12s, border-color .12s, color .12s; flex-shrink: 0;
    }
    .ht-ms-btn:hover:not(:disabled) { border-color: var(--text-3); color: var(--text); }
    .ht-ms-btn:disabled { opacity: .25; cursor: not-allowed; }
    .ht-mute-btn--active { background: var(--red-dim);   border-color: var(--red);   color: var(--red);   }
    .ht-solo-btn--active { background: var(--amber-dim); border-color: var(--amber); color: var(--amber); }

    /* ── Hint bar ── */
    .ht-hint-bar { margin-top: auto; padding: 12px 14px; border-top: 1px solid var(--border); flex-shrink: 0; }
    .ht-hint-label {
      font-size: 9px; font-weight: 700; letter-spacing: .1em;
      color: var(--text-3); text-transform: uppercase; margin-bottom: 7px;
    }
    .ht-hints { display: flex; flex-direction: column; gap: 4px; }
    .ht-hint-row { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
    .ht-kbd {
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 20px; height: 16px; padding: 0 4px;
      background: var(--surface-2); border: 1px solid var(--border); border-radius: 3px;
      font-family: var(--font-mono); font-size: 9px; color: var(--text-2); flex-shrink: 0;
    }
    .ht-hint-desc { font-size: 10px; color: var(--text-3); }

    /* ── Diagnostics panel ── */
    .ht-diag-header {
      padding: 13px 14px; border-bottom: 1px solid var(--border);
      font-size: 9px; font-weight: 700; letter-spacing: .11em;
      color: var(--text-3); text-transform: uppercase; flex-shrink: 0;
    }
    .ht-diag-section { padding: 14px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .ht-diag-section-label {
      font-size: 9px; font-weight: 700; letter-spacing: .1em;
      color: var(--text-3); text-transform: uppercase; margin-bottom: 9px;
    }

    .ht-offaxis-readout {
      font-family: var(--font-mono); font-size: 20px; font-weight: 600;
      color: var(--accent); line-height: 1.15; margin-bottom: 3px; word-break: break-word;
    }
    .ht-offaxis-readout--onspot { color: var(--green); }
    .ht-offaxis-sub { font-size: 10px; color: var(--text-3); }

    .ht-status-badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 10px; border-radius: 20px;
      font-size: 11px; font-weight: 600;
      background: var(--green-dim); border: 1px solid var(--green); color: var(--green);
    }
    .ht-status-badge--error   { background: var(--red-dim);   border-color: var(--red);   color: var(--red);   }
    .ht-status-badge--pending { background: var(--amber-dim); border-color: var(--amber); color: var(--amber); }
    .ht-status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

    .ht-diag-row {
      display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
      padding: 4px 0;
    }
    .ht-diag-row + .ht-diag-row { border-top: 1px solid rgba(255,255,255,.04); }
    .ht-diag-key  { font-size: 11px; color: var(--text-3); flex-shrink: 0; }
    .ht-diag-val  { font-family: var(--font-mono); font-size: 11px; color: var(--text); text-align: right; word-break: break-all; }
    .ht-diag-val--accent { color: var(--accent); }
    .ht-diag-val--amber  { color: var(--amber);  }
    .ht-diag-val--green  { color: var(--green);  }

    /* ── Footer ── */
    .ht-footer {
      position: fixed; bottom: 0;
      left: 0; right: 0;
      padding: 5px 20px;
      text-align: center;
      pointer-events: none;
      z-index: 5;
    }
    .ht-credit { font-size: 10px; color: var(--text-3); font-family: var(--font-mono); }
  `;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// STEP 5: Loading overlay
// ---------------------------------------------------------------------------

function _buildLoadingOverlay(root) {
  const overlay = _el("div", "ht-overlay");

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

  const msg = _el("div", "ht-overlay-msg");
  msg.textContent = "Initialising audio engine…";
  overlay.appendChild(msg);
  _el.overlayMsg = msg;

  const btn = _el("button", "ht-start-btn");
  btn.textContent = "Start";
  btn.disabled = true;
  btn.setAttribute("aria-label", "Start the experience");
  btn.addEventListener("click", _onStartClick);
  overlay.appendChild(btn);
  _el.startBtn = btn;

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
  if (!_el.overlay) return;
  _el.overlay.classList.add("ht-overlay--fade");
  _el.overlay.addEventListener("transitionend", () => {
    _el.overlay?.remove();
    _el.overlay = null;
  }, { once: true });
}

// ---------------------------------------------------------------------------
// STEP 6: Left sidebar
// ---------------------------------------------------------------------------

function _buildLeftSidebar(root) {
  // Wrapper holds both the sidebar and its collapse tab so they move together.
  const wrap = _el("div", "ht-left-wrap");

  const sidebar = _el("div", "ht-sidebar");
  _buildBrand(sidebar);
  _buildPlaybackSection(sidebar);
  _buildTierSection(sidebar);
  _buildRoomSection(sidebar);
  _buildMixerSection(sidebar);
  _buildHintBar(sidebar);
  wrap.appendChild(sidebar);
  _el.sidebar = sidebar;

  // Collapse tab
  const tab = _el("div", "ht-collapse-tab");
  tab.setAttribute("role", "button");
  tab.setAttribute("aria-label", "Toggle left panel");
  tab.setAttribute("tabindex", "0");
  tab.textContent = "‹";
  tab.addEventListener("click", _toggleLeftPanel);
  tab.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") _toggleLeftPanel(); });
  wrap.appendChild(tab);
  _el.leftTab = tab;

  root.appendChild(wrap);
}

function _toggleLeftPanel() {
  _state.leftOpen = !_state.leftOpen;
  _el.sidebar.classList.toggle("ht-sidebar--collapsed", !_state.leftOpen);
  _el.leftTab.textContent = _state.leftOpen ? "‹" : "›";
  _el.leftTab.setAttribute("aria-label", _state.leftOpen ? "Collapse left panel" : "Expand left panel");
}

// ── Branding ────────────────────────────────────────────────────────────────

function _buildBrand(parent) {
  const brand = _el("div", "ht-brand");

  const logoPath = window.htManifest?.logo;
  const logo = _el("img", "ht-brand-logo");
  logo.src = logoPath ? `ht://assets/${logoPath}` : "";
  logo.alt = "AMS Group · TU Ilmenau";
  logo.onerror = () => {
    logo.remove();
    const fallback = _el("div", "ht-brand-logo-fallback");
    fallback.textContent = "AMS";
    brand.insertBefore(fallback, brand.firstChild);
  };
  brand.appendChild(logo);

  const text = _el("div", "ht-brand-text");

  const chapter = _el("div", "ht-brand-chapter");
  chapter.textContent = "Multimediaprogrammierung · Chapter 05";
  text.appendChild(chapter);

  const title = _el("div", "ht-brand-title");
  title.textContent = "Home Theater Spatial Audio";
  text.appendChild(title);

  const author = _el("div", "ht-brand-author");
  author.textContent = "Renato Fera P.";
  text.appendChild(author);

  const linkedIn = _el("button", "ht-linkedin-btn");
  linkedIn.innerHTML = `
    <svg class="ht-linkedin-icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
    </svg>LinkedIn`;
  linkedIn.setAttribute("aria-label", "Open LinkedIn profile");
  linkedIn.addEventListener("click", () => {
    window.htBridge?.open_url("https://www.linkedin.com/in/renato-fera");
  });
  text.appendChild(linkedIn);

  brand.appendChild(text);
  parent.appendChild(brand);
}

// ── Playback section ─────────────────────────────────────────────────────────

function _buildPlaybackSection(parent) {
  const section = _el("div", "ht-section");

  const lbl = _el("div", "ht-section-label");
  lbl.textContent = "Playback";
  section.appendChild(lbl);

  const transport = _el("div", "ht-transport");

  const playBtn = _el("button", "ht-play-btn");
  playBtn.innerHTML = `<span class="ht-play-icon">▶</span><span class="ht-play-label">Start</span>`;
  playBtn.setAttribute("aria-label", "Play or pause");
  playBtn.addEventListener("click", _onPlayPauseClick);
  transport.appendChild(playBtn);
  _el.playPauseBtn = playBtn;

  const resetBtn = _el("button", "ht-reset-btn");
  resetBtn.textContent = "↺ Reset View";
  resetBtn.setAttribute("aria-label", "Reset camera to sweet spot");
  resetBtn.addEventListener("click", _onResetView);
  transport.appendChild(resetBtn);

  section.appendChild(transport);
  parent.appendChild(section);
}

// ── Tier section ─────────────────────────────────────────────────────────────

function _buildTierSection(parent) {
  const section = _el("div", "ht-section");

  const lbl = _el("div", "ht-section-label");
  lbl.textContent = "Surround Format";
  section.appendChild(lbl);

  const group = _el("div", "ht-pill-group");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Surround tier");

  for (const { key, label } of [
    { key: "mono",   label: "Mono"   },
    { key: "stereo", label: "Stereo" },
    { key: "5.1",    label: "5.1"    },
  ]) {
    const btn = _el("button", "ht-pill-btn");
    const dot = _el("span", "ht-pill-dot");
    const txt = document.createTextNode(label);
    btn.appendChild(txt);
    btn.appendChild(dot);
    btn.dataset.tier = key;
    btn.setAttribute("aria-pressed", key === _state.tier ? "true" : "false");
    if (key === _state.tier) btn.classList.add("ht-pill-btn--active");
    btn.addEventListener("click", () => _onTierClick(key));
    group.appendChild(btn);
    _el.tierBtns[key] = btn;
  }

  section.appendChild(group);
  parent.appendChild(section);
}

// ── Room section ──────────────────────────────────────────────────────────────

function _buildRoomSection(parent) {
  const section = _el("div", "ht-section");

  const lbl = _el("div", "ht-section-label");
  lbl.textContent = "Room Acoustics";
  section.appendChild(lbl);

  const group = _el("div", "ht-pill-group ht-pill-group--col");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Room acoustic treatment");

  for (const { key, label, rt60 } of [
    { key: "treated", label: "Dry – Treated",    rt60: "RT60 ≈ 0.3 s" },
    { key: "wet",     label: "Wet – Reflective",  rt60: "RT60 ≈ 1.1 s" },
  ]) {
    const btn = _el("button", "ht-pill-btn");

    const left = _el("span", "ht-pill-left");
    const dot  = _el("span", "ht-pill-dot");
    const txt  = _el("span");
    txt.textContent = label;
    left.appendChild(dot);
    left.appendChild(txt);

    const rt = _el("span", "ht-rt60-chip");
    rt.textContent = rt60;

    btn.appendChild(left);
    btn.appendChild(rt);
    btn.dataset.room = key;
    btn.setAttribute("aria-pressed", key === _state.room ? "true" : "false");
    if (key === _state.room) btn.classList.add("ht-pill-btn--active");
    btn.addEventListener("click", () => _onRoomClick(key));
    group.appendChild(btn);
    _el.roomBtns[key] = btn;
  }

  section.appendChild(group);
  parent.appendChild(section);
}

// ── Mixer section ─────────────────────────────────────────────────────────────

function _buildMixerSection(parent) {
  const section = _el("div", "ht-section");

  const lbl = _el("div", "ht-section-label");
  lbl.textContent = "Speaker Mixer";
  section.appendChild(lbl);

  const toggle = _el("button", "ht-mixer-toggle-btn");
  toggle.innerHTML = `<span>Channels</span><span class="ht-mixer-caret">▾</span>`;
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-controls", "ht-mixer-panel");
  toggle.addEventListener("click", _onMixerToggleClick);
  section.appendChild(toggle);
  _el.mixerToggleBtn = toggle;

  const panel = _el("div", "ht-mixer-panel");
  panel.id = "ht-mixer-panel";
  panel.setAttribute("role", "region");
  panel.setAttribute("aria-label", "Speaker mixer");

  for (const { key, label } of [
    { key: "C",   label: "Center" },
    { key: "L",   label: "Left"   },
    { key: "R",   label: "Right"  },
    { key: "Ls",  label: "Surr L" },
    { key: "Rs",  label: "Surr R" },
    { key: "LFE", label: "LFE"    },
  ]) {
    const row = _el("div", "ht-mixer-strip");
    row.dataset.speakerKey = key;

    const labelEl = _el("span", "ht-mixer-label");
    labelEl.textContent = label;
    row.appendChild(labelEl);

    const muteBtn = _el("button", "ht-ms-btn");
    muteBtn.textContent = "M";
    muteBtn.setAttribute("aria-label", `Mute ${label}`);
    muteBtn.setAttribute("aria-pressed", "false");
    muteBtn.addEventListener("click", () => _onMuteClick(key));
    row.appendChild(muteBtn);

    const soloBtn = _el("button", "ht-ms-btn");
    soloBtn.textContent = "S";
    soloBtn.setAttribute("aria-label", `Solo ${label}`);
    soloBtn.setAttribute("aria-pressed", "false");
    soloBtn.addEventListener("click", () => _onSoloClick(key));
    row.appendChild(soloBtn);

    panel.appendChild(row);
    _el.speakerStrips[key] = { row, muteBtn, soloBtn, label: labelEl };
  }

  section.appendChild(panel);
  _el.mixerPanel = panel;

  parent.appendChild(section);
  _updateMixerStrips(_state.tier);
}

// ── Hint bar ──────────────────────────────────────────────────────────────────

function _buildHintBar(parent) {
  const bar = _el("div", "ht-hint-bar");

  const lbl = _el("div", "ht-hint-label");
  lbl.textContent = "Navigation";
  bar.appendChild(lbl);

  const hints = _el("div", "ht-hints");
  for (const { keys, desc } of [
    { keys: ["W","A","S","D"], desc: "move"        },
    { keys: ["L-drag"],        desc: "look"        },
    { keys: ["R-drag"],        desc: "walk"        },
    { keys: ["Scroll"],        desc: "fwd / back"  },
    { keys: ["Space"],         desc: "start / stop"},
  ]) {
    const row = _el("div", "ht-hint-row");
    for (const k of keys) {
      const kbd = _el("span", "ht-kbd");
      kbd.textContent = k;
      row.appendChild(kbd);
    }
    const d = _el("span", "ht-hint-desc");
    d.textContent = desc;
    row.appendChild(d);
    hints.appendChild(row);
  }

  bar.appendChild(hints);
  parent.appendChild(bar);
}

// ---------------------------------------------------------------------------
// STEP 7: Right diagnostics panel
// ---------------------------------------------------------------------------

function _buildRightPanel(root) {
  const wrap = _el("div", "ht-right-wrap");

  const panel = _el("div", "ht-diag-panel");
  _buildDiagContents(panel);
  wrap.appendChild(panel);
  _el.diagPanel = panel;

  // Collapse tab (on the left edge of the right panel)
  const tab = _el("div", "ht-collapse-tab-right");
  tab.setAttribute("role", "button");
  tab.setAttribute("aria-label", "Toggle right panel");
  tab.setAttribute("tabindex", "0");
  tab.textContent = "›";
  tab.addEventListener("click", _toggleRightPanel);
  tab.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") _toggleRightPanel(); });
  wrap.appendChild(tab);
  _el.rightTab = tab;

  root.appendChild(wrap);
}

function _toggleRightPanel() {
  _state.rightOpen = !_state.rightOpen;
  _el.diagPanel.classList.toggle("ht-diag-panel--collapsed", !_state.rightOpen);
  _el.rightTab.textContent = _state.rightOpen ? "›" : "‹";
  _el.rightTab.setAttribute("aria-label", _state.rightOpen ? "Collapse right panel" : "Expand right panel");
}

function _buildDiagContents(panel) {
  const header = _el("div", "ht-diag-header");
  header.textContent = "Listener Diagnostics";
  panel.appendChild(header);

  // Position
  const posSection = _el("div", "ht-diag-section");
  const posLbl = _el("div", "ht-diag-section-label");
  posLbl.textContent = "Position";
  posSection.appendChild(posLbl);

  const readout = _el("div", "ht-offaxis-readout ht-offaxis-readout--onspot");
  readout.textContent = "sweet spot";
  readout.setAttribute("aria-live", "polite");
  posSection.appendChild(readout);
  _el.offAxisReadout = readout;

  const sub = _el("div", "ht-offaxis-sub");
  sub.textContent = "offset from calibration point";
  posSection.appendChild(sub);
  panel.appendChild(posSection);

  // Engine status
  const statusSection = _el("div", "ht-diag-section");
  const statusLbl = _el("div", "ht-diag-section-label");
  statusLbl.textContent = "Engine";
  statusSection.appendChild(statusLbl);

  const badge = _el("div", "ht-status-badge ht-status-badge--pending");
  badge.innerHTML = `<span class="ht-status-dot"></span><span class="ht-status-text">Initialising…</span>`;
  statusSection.appendChild(badge);
  _el.statusBadge = badge;
  _el.statusText  = badge.querySelector(".ht-status-text");
  panel.appendChild(statusSection);

  // Active state
  const stateSection = _el("div", "ht-diag-section");
  const stateLbl = _el("div", "ht-diag-section-label");
  stateLbl.textContent = "Active State";
  stateSection.appendChild(stateLbl);

  _el.diagTierVal = _diagRow(stateSection, "Format",   "Mono",    "ht-diag-val--accent");
  _el.diagRoomVal = _diagRow(stateSection, "Room",     "Treated", "ht-diag-val--accent");
  _el.diagPlayVal = _diagRow(stateSection, "Playback", "stopped", "ht-diag-val--amber");
  panel.appendChild(stateSection);

  // Channels
  const spkSection = _el("div", "ht-diag-section");
  const spkLbl = _el("div", "ht-diag-section-label");
  spkLbl.textContent = "Channels";
  spkSection.appendChild(spkLbl);

  _el.diagSpkVal  = _diagRow(spkSection, "Active", "1 / 6", "ht-diag-val--green");
  _el.diagSoloVal = _diagRow(spkSection, "Solo",   "none",  "");
  panel.appendChild(spkSection);

  // Content / film credit
  const creditSection = _el("div", "ht-diag-section");
  const creditLbl = _el("div", "ht-diag-section-label");
  creditLbl.textContent = "Content";
  creditSection.appendChild(creditLbl);

  const creditVal = _el("div", "ht-diag-val");
  creditVal.id = "ht-film-credit";
  creditVal.style.cssText = "font-size:10px;color:var(--text-3);font-family:var(--font-mono);line-height:1.5;text-align:left;white-space:normal;";
  creditVal.textContent = _getFilmCredit();
  creditSection.appendChild(creditVal);
  panel.appendChild(creditSection);
}

/** Helper — builds a key/value diagnostics row and returns the value element. */
function _diagRow(parent, key, initialVal, valClass) {
  const row = _el("div", "ht-diag-row");
  const k = _el("span", "ht-diag-key"); k.textContent = key;
  const v = _el("span", `ht-diag-val ${valClass}`); v.textContent = initialVal;
  row.appendChild(k); row.appendChild(v);
  parent.appendChild(row);
  return v;
}

// ---------------------------------------------------------------------------
// STEP 8: Footer — film credit
// ---------------------------------------------------------------------------

function _buildFooter(root) {
  const footer = _el("div", "ht-footer");
  const credit = _el("span", "ht-credit");
  credit.id = "ht-film-credit-footer";
  credit.textContent = _getFilmCredit();
  footer.appendChild(credit);
  root.appendChild(footer);
}

// ---------------------------------------------------------------------------
// STEP 9: Event handlers
// ---------------------------------------------------------------------------

function _onPlayPauseClick() {
  if (!_state.engineReady) return;
  if (_state.isPlaying) {
    window.htBridge?.pause();
    _setPlayBtn(false);
    _setFormatControlsDisabled(false);
  } else {
    window.htBridge?.play();
    _setPlayBtn(true);
    _setFormatControlsDisabled(true);
  }
}

function _onResetView() {
  // Calls the resetView function exposed by scene.js on window.htScene.
  // scene.js zeroes _ctrl.offset, _ctrl.yaw, _ctrl.pitch — see scene.js STEP 3.
  window.htScene?.resetView();
}

function _onTierClick(key) {
  if (_state.isPlaying) return;
  if (key === _state.tier) return;
  for (const btn of Object.values(_el.tierBtns)) btn.disabled = true;
  window.htBridge?.set_tier(key);
}

function _onRoomClick(key) {
  if (_state.isPlaying) return;
  if (key === _state.room) return;
  for (const btn of Object.values(_el.roomBtns)) btn.disabled = true;
  window.htScene?.setRoomVariant(key);
  window.htBridge?.set_room(key);
}

function _onMixerToggleClick() {
  _state.mixerOpen = !_state.mixerOpen;
  _el.mixerPanel.classList.toggle("ht-mixer-panel--open", _state.mixerOpen);
  _el.mixerToggleBtn.setAttribute("aria-expanded", String(_state.mixerOpen));
  _el.mixerToggleBtn.classList.toggle("ht-mixer-toggle-btn--open", _state.mixerOpen);
}

function _onMuteClick(key) {
  _state.muteSaved[key] = !_state.muteSaved[key];
  const muted = _state.muteSaved[key];
  // If this speaker is in the solo set and is now muted, evict it from solo.
  if (muted && _state.soloSet.has(key)) {
    _state.soloSet.delete(key);
    _applySoloState();
    return;
  }
  _applyMuteVisual(key, muted);
  window.htBridge?.set_speaker_mute(key, muted);
}

/**
 * _onSoloClick — multi-solo: adds/removes key from _state.soloSet.
 * When the set is non-empty, speakers outside it are force-muted.
 * When empty, all speakers revert to their individually saved mute states.
 */
function _onSoloClick(key) {
  if (_state.soloSet.has(key)) {
    _state.soloSet.delete(key);
  } else {
    _state.soloSet.add(key);
  }
  _applySoloState();
}

function _applySoloState() {
  const hasSolo = _state.soloSet.size > 0;
  for (const key of Object.keys(_state.muteSaved)) {
    const strip = _el.speakerStrips[key];
    if (!strip) continue;
    const isSoloed      = _state.soloSet.has(key);
    const effectiveMute = hasSolo ? !isSoloed : _state.muteSaved[key];

    strip.soloBtn.setAttribute("aria-pressed", String(isSoloed));
    strip.soloBtn.classList.toggle("ht-solo-btn--active", isSoloed);
    strip.row.classList.toggle("ht-mixer-strip--muted", effectiveMute);
    window.htBridge?.set_speaker_mute(key, effectiveMute);
  }
  // Keep mute-button visuals in sync with saved mute state (independent of solo).
  for (const key of Object.keys(_state.muteSaved)) {
    const strip = _el.speakerStrips[key];
    if (!strip) continue;
    const saved = _state.muteSaved[key];
    strip.muteBtn.setAttribute("aria-pressed", String(saved));
    strip.muteBtn.classList.toggle("ht-mute-btn--active", saved);
  }
  _updateDiagSolo();
}

// ---------------------------------------------------------------------------
// STEP 10: Signal handlers
// ---------------------------------------------------------------------------

function onEngineReady() {
  _state.engineReady = true;
  if (_el.overlayMsg) _el.overlayMsg.textContent = "Engine ready — headphones on?";
  if (_el.startBtn)   { _el.startBtn.disabled = false; _el.startBtn.focus(); }
  const creditEl = document.getElementById("ht-film-credit");
  if (creditEl) creditEl.textContent = _getFilmCredit();
  _setStatus("Ready", "ready");
}

function onAssetError(message) {
  if (_el.overlay) { _el.overlay.classList.remove("ht-overlay--fade"); _el.overlay.hidden = false; }
  if (_el.errorPanel) { _el.errorText.textContent = message; _el.errorPanel.hidden = false; }
  if (_el.startBtn) _el.startBtn.disabled = true;
  _setStatus("Asset error", "error");
  console.error("[hud.js] Asset error:", message);
}

function onStateChanged(tier, room, isPlaying) {
  const tierChanged = tier !== _state.tier;
  const roomChanged = room !== _state.room;
  _state.tier      = tier;
  _state.room      = room;
  _state.isPlaying = isPlaying;

  _setPlayBtn(isPlaying);

  for (const [k, btn] of Object.entries(_el.tierBtns)) {
    const active = k === tier;
    btn.classList.toggle("ht-pill-btn--active", active);
    btn.setAttribute("aria-pressed", String(active));
    btn.disabled = false;
  }
  for (const [k, btn] of Object.entries(_el.roomBtns)) {
    const active = k === room;
    btn.classList.toggle("ht-pill-btn--active", active);
    btn.setAttribute("aria-pressed", String(active));
    btn.disabled = false;
  }

  _setFormatControlsDisabled(isPlaying);

  if (tierChanged && window.htScene?.setActiveTier) window.htScene.setActiveTier(tier);
  if (roomChanged && window.htScene?.setRoomVariant) window.htScene.setRoomVariant(room);
  if (tierChanged) _updateMixerStrips(tier);

  if (_el.diagTierVal) _el.diagTierVal.textContent = _tierLabel(tier);
  if (_el.diagRoomVal) _el.diagRoomVal.textContent = _roomLabel(room);
  if (_el.diagPlayVal) {
    _el.diagPlayVal.textContent = isPlaying ? "playing" : "stopped";
    _el.diagPlayVal.className   = `ht-diag-val ${isPlaying ? "ht-diag-val--green" : "ht-diag-val--amber"}`;
  }

  _setStatus(
    `${_tierLabel(tier)} · ${_roomLabel(room)}${isPlaying ? " · Playing" : ""}`,
    isPlaying ? "ready" : "ready"
  );
}

function onOffAxisReadout(text) {
  if (!_el.offAxisReadout) return;
  if (_el.offAxisReadout.textContent === text) return;
  _el.offAxisReadout.textContent = text;
  const onSpot = text === "sweet spot";
  _el.offAxisReadout.classList.toggle("ht-offaxis-readout--onspot", onSpot);
}

// ---------------------------------------------------------------------------
// STEP 11: Internal helpers
// ---------------------------------------------------------------------------

function _setPlayBtn(playing) {
  if (!_el.playPauseBtn) return;
  const icon  = _el.playPauseBtn.querySelector(".ht-play-icon");
  const label = _el.playPauseBtn.querySelector(".ht-play-label");
  if (playing) {
    if (icon)  icon.textContent  = "■";
    if (label) label.textContent = "Stop";
    _el.playPauseBtn.classList.add("ht-play-btn--pause");
    _el.playPauseBtn.setAttribute("aria-label", "Stop playback");
  } else {
    if (icon)  icon.textContent  = "▶";
    if (label) label.textContent = "Start";
    _el.playPauseBtn.classList.remove("ht-play-btn--pause");
    _el.playPauseBtn.setAttribute("aria-label", "Start playback");
  }
}

function _setFormatControlsDisabled(disabled) {
  for (const btn of Object.values(_el.tierBtns)) btn.disabled = disabled;
  for (const btn of Object.values(_el.roomBtns)) btn.disabled = disabled;
}

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
  if (_el.diagSpkVal) _el.diagSpkVal.textContent = `${active.size} / 6`;
}

function _applyMuteVisual(key, muted) {
  const strip = _el.speakerStrips[key];
  if (!strip) return;
  strip.muteBtn.setAttribute("aria-pressed", String(muted));
  strip.muteBtn.classList.toggle("ht-mute-btn--active", muted);
  strip.row.classList.toggle("ht-mixer-strip--muted", muted);
}

function _updateDiagSolo() {
  if (!_el.diagSoloVal) return;
  if (_state.soloSet.size === 0) {
    _el.diagSoloVal.textContent = "none";
    _el.diagSoloVal.className   = "ht-diag-val";
  } else {
    _el.diagSoloVal.textContent = [..._state.soloSet].join(", ");
    _el.diagSoloVal.className   = "ht-diag-val ht-diag-val--amber";
  }
}

function _setStatus(text, kind = "ready") {
  if (!_el.statusBadge || !_el.statusText) return;
  _el.statusText.textContent = text;
  _el.statusBadge.className  = "ht-status-badge";
  if (kind === "error")   _el.statusBadge.classList.add("ht-status-badge--error");
  if (kind === "pending") _el.statusBadge.classList.add("ht-status-badge--pending");
}

function _tierLabel(tier) {
  return { mono: "Mono", stereo: "Stereo", "5.1": "5.1" }[tier] ?? tier;
}

function _roomLabel(room) {
  return { treated: "Treated", wet: "Wet" }[room] ?? room;
}

function _getFilmCredit() {
  const m = window.htManifest;
  const c = m?.film_credit ?? m?.credit;
  if (!c) return "Film: [loading…] © Blender Foundation, CC-BY 3.0";
  return `Film: "${c.title}" © ${c.author ?? "Blender Foundation"}, ${c.license ?? "CC-BY 3.0"}`;
}

// ---------------------------------------------------------------------------
// STEP 12: _el — DOM element factory + named-ref store
// ---------------------------------------------------------------------------

function _el(tag, classes) {
  const node = document.createElement(tag);
  if (classes) node.className = classes;
  return node;
}

_el.tierBtns      = {};
_el.roomBtns      = {};
_el.speakerStrips = {};
