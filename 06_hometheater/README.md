# Home Theater Spatial Audio Simulator

**Multimediaprogrammierung Chapter 05 — Native Companion App**
TU Ilmenau · <!-- TODO: confirm author credit -->

A first-person binaural home-theater experience. Put on headphones, sit at the
virtual sweet spot, and switch between three historical surround formats —
**Mono → Stereo → 5.1** — hearing the same film scene rendered through each via
genuinely discrete channel content. A second switch toggles **room acoustic
treatment** (treated home-theater vs. untreated "wet" room), changing both wall
materials and real reverberation simultaneously via Qt's native `QAudioRoom`.

All output is **headphone-only** (`QAudioEngine.OutputMode.Headphone`). There is
no real speaker array — "5.1" means five `QSpatialSound` sources placed at ITU-R
BS.775 virtual speaker angles and binaurally rendered. This is the whole point,
not a limitation.

---

## Requirements

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (dependency manager — do **not** use pip
  or poetry directly)
- PySide6 ≥ 6.8 (installed automatically by uv)
- A platform with a working Qt Multimedia backend (see Troubleshooting below)

---

## 1. Get the assets

The app requires five user-supplied files. None are bundled — download them once
and place them at the paths below.

### 1a. Choose a Blender Foundation open movie

All three are CC-BY 3.0, free to use for education. Pick one:

| Film | Video download | 5.1 FLAC download |
|---|---|---|
| Sintel (2010) | [blender.org/films/sintel](https://www.blender.org/films/sintel) | [media.xiph.org/sintel](https://media.xiph.org/sintel/sintel-master.flac) |
| Tears of Steel (2012) | [blender.org/films/tears-of-steel](https://www.blender.org/films/tears-of-steel) | [media.xiph.org/tearsofsteel](https://media.xiph.org/tearsofsteel/) |
| Big Buck Bunny (2008) | [blender.org/films/big-buck-bunny](https://www.blender.org/films/big-buck-bunny) | [media.xiph.org/bbb](https://media.xiph.org/bbb/) |

**You need two files from the same film:**
- The video (`.mp4` or `.webm`) — audio track not needed; it is muted at runtime
- The **discrete 6-channel FLAC master** — not an AC-3 or DTS encode; Xiph.org
  hosts the lossless FLAC masters specifically for this use case

### 1b. Get the 3D models

You need three CC0-licensed GLTF/GLB models:

| Model | Suggested source |
|---|---|
| Sofa | [Sketchfab](https://sketchfab.com) or [Poly Haven](https://polyhaven.com) — search "sofa CC0" |
| TV / flat screen | Same — search "television CC0" or "monitor CC0" |
| Speaker (bookshelf or floor-standing) | Same — search "speaker CC0"; one model is reused for all speaker positions |

Download each as `.glb` (binary GLTF, single file). If a model is only available
as `.gltf + bin + textures`, use a tool such as
[gltf-transform](https://gltf-transform.donmccurdy.com/) to pack it into a
single `.glb` before placing it.

Scale note: the app expects models in real-world metres. A sofa should be roughly
0.9 m tall; a floor-standing speaker roughly 1.0 m. If a downloaded model is
wildly off-scale, adjust with gltf-transform's `--scale` flag.

### 1c. Place the files

```
hometheater-demo/
└── assets/
    ├── manifest.json          ← edit this (see §2)
    ├── models/
    │   ├── sofa.glb           ← your sofa model
    │   ├── tv.glb             ← your TV/screen model
    │   └── speaker.glb        ← your speaker model (reused per position)
    └── media/
        ├── sintel.mp4         ← video (or .webm); exact name set in manifest
        └── sintel-master.flac ← discrete 6-channel FLAC; exact name set in manifest
```

The filenames above are examples. The actual names are whatever you download —
update `manifest.json` to match (see §2).

---

## 2. Edit `assets/manifest.json`

Open `assets/manifest.json` and fill in your actual filenames and the film
credit. A minimal valid manifest:

```json
{
  "film_title":    "Sintel",
  "film_credit":   "© 2010 Blender Foundation | blender.org/films/sintel | CC-BY 3.0",
  "video_file":    "sintel.mp4",
  "audio_51_file": "sintel-master.flac",
  "model_sofa":    "sofa.glb",
  "model_tv":      "tv.glb",
  "model_speaker": "speaker.glb",
  "room_dimensions": {
    "width":  6.0,
    "height": 2.8,
    "depth":  7.5
  }
}
```

**All paths are filenames only** (no directory prefix). The app knows where to
look:
- `video_file` and `audio_51_file` are resolved under `assets/media/`
- `model_*` are resolved under `assets/models/`

`room_dimensions` is in metres (Y-up, same coordinate system as the Three.js
scene). These values set both the visual room geometry and the `QAudioRoom`
acoustic volume simultaneously — change them together here, never in one place
only.

---

## 3. Download the Three.js vendor files

The app runs offline and does not use a CDN. Download these two files once into
`src/hometheater/web/vendor/`:

```
web/vendor/
├── three.module.js       ← Three.js r165 ES module build
└── GLTFLoader.js         ← Three.js r165 GLTFLoader addon
```

**Download URLs (r165):**

```
https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js
https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/GLTFLoader.js
```

Save each file with the exact name shown above (the importmap in `index.html`
references these names verbatim).

---

## 4. Install dependencies and run

```bash
# Install all Python dependencies into a managed virtual environment
uv sync

# Run the app
uv run python -m hometheater
```

On first run, the asset pipeline will split and downmix the 5.1 FLAC into
individual channel WAV files and cache them under `cache/`. This takes a few
seconds for a typical film clip and is skipped on subsequent runs unless the
source FLAC changes.

---

## 5. Using the app

1. **Put on headphones** — the HUD will remind you, but all spatial rendering
   assumes binaural headphone output. Speaker playback will not produce correct
   spatial imaging.

2. **Click Start** — the loading overlay fades once the audio engine is ready
   and all assets are confirmed loadable.

3. **Look around** — left-drag to look, right-drag to pan, scroll wheel or
   W/S to move forward/back. Movement is intentionally clamped to a small
   radius around the calibrated listening sweet spot — this is a teaching point,
   not a bug. The off-axis readout (top-left HUD) shows your distance and
   direction from the sweet spot.

4. **Switch formats** — the **Tier** button cycles Mono → Stereo → 5.1. The
   **Room** button toggles Treated/Wet. Both are disabled during playback;
   pause first, switch, then resume.

5. **Mixer panel** — the mixer button (bottom-right) opens per-speaker
   Mute/Solo strips. Only speakers active in the current tier are shown.

---

## 6. Project layout (for developers)

```
hometheater-demo/
├── pyproject.toml                   ← uv project; PySide6 ≥ 6.8 pinned
├── README.md                        ← this file
├── .gitignore
├── assets/
│   ├── manifest.json
│   ├── models/                      ← user-supplied .glb files
│   └── media/                       ← user-supplied video + FLAC
├── cache/                           ← generated by asset_pipeline.py; gitignored
└── src/hometheater/
    ├── __init__.py
    ├── main.py                      ← QMainWindow, startup sequence, ht:// scheme
    ├── asset_pipeline.py            ← 6-channel FLAC split + ITU downmix + cache
    ├── audio_engine.py              ← SpatialAudioEngine (QAudioEngine wrapper)
    ├── bridge.py                    ← QWebChannel slots/signals
    └── web/
        ├── index.html               ← QWebChannel bootstrap, importmap
        ├── style.css                ← design tokens (IBM Plex, JetBrains Mono)
        ├── scene.js                 ← Three.js scene, GLTF, POV controls
        ├── hud.js                   ← HUD wiring, transport, mixer panel
        └── vendor/
            ├── three.module.js      ← Three.js r165 (download separately)
            └── GLTFLoader.js        ← Three.js r165 addon (download separately)
```

---

## 7. Manual verification checklist

The audio engine cannot be verified programmatically by the build toolchain —
run through this checklist after the first successful launch:

- [ ] **Tier switch while paused** — switching Mono → Stereo → 5.1 while
      paused produces a clean restart from 0:00 with correct channel routing.
      No glitch, no hang.
- [ ] **Tier switch blocked while playing** — the Tier and Room buttons are
      visually disabled during playback. Attempting to call `set_tier` from the
      browser console while playing should log an error and change nothing.
- [ ] **Room switch** — switching Treated ↔ Wet visibly changes wall materials
      (roughness, colour) and is audibly different on the same content — the wet
      room should sound noticeably more reverberant.
- [ ] **Off-axis imaging** — moving away from the sweet spot in any direction
      produces an audible imaging change (not just volume), particularly
      noticeable in 5.1 when the surround sources shift off their calibrated angles.
- [ ] **Missing asset error** — rename one model GLB to trigger a missing-asset
      condition. The HUD should display a readable error naming the missing file
      and expected path. No crash, no silent empty scene.
- [ ] **Missing FLAC error** — remove the FLAC from `assets/media/` and re-run.
      The startup error dialog should name the missing file. No traceback.
- [ ] **Close during playback** — closing the window while audio is playing
      should not produce an audible glitch or error in the terminal.

---

## 8. Troubleshooting

### `from PySide6.QtSpatialAudio import QAudioEngine` fails

This is an environment issue, not a code bug. The platform's Qt Multimedia
backend plugin is likely absent. Try:

```bash
# macOS
brew install qt@6

# Ubuntu / Debian
sudo apt install libqt6multimedia6 libqt6spatialaudioplugin6

# Windows
# Reinstall PySide6 via uv — the Windows wheel bundles the backend.
uv sync --reinstall
```

If the import still fails after installing platform packages, check that the
installed PySide6 version is ≥ 6.8:

```bash
uv run python -c "import PySide6; print(PySide6.__version__)"
```

### No audio output / silent playback

1. Confirm headphones are the default output device — `QAudioEngine` picks the
   system default at construction time and does not rescan.
2. Confirm the FLAC is a true discrete 6-channel file, not a 2-channel stereo
   file renamed `.flac`. The asset pipeline will reject it with a readable
   error if the channel count is wrong.
3. On Linux, PulseAudio/PipeWire must be running. Check with `pactl info`.

### `ht://` resources return 404 in the HUD

The scheme handler is registered before `QApplication` is constructed. If you
are running `main.py` directly (bypassing `python -m hometheater`), ensure
`_register_ht_scheme()` is called before `QApplication(sys.argv)`. Running via
`uv run python -m hometheater` handles this correctly.

### GLB models appear at wrong scale

Adjust the model with gltf-transform before placing it in `assets/models/`:

```bash
npx @gltf-transform/cli transform sofa-original.glb sofa.glb --scale 0.01
```

The scene is authored in metres. A sofa should be ~0.9 m tall; a
bookshelf speaker ~0.5 m; a floor-standing speaker ~1.0 m.

---

## 9. Licence and credits

This application is a companion to the *Multimediaprogrammierung* course at
TU Ilmenau and is intended for educational use.

Film assets (video + audio) are sourced from the Blender Foundation open movie
library under **CC-BY 3.0**. The specific film and credit line are declared in
`assets/manifest.json` and displayed in the application HUD.

3D models are sourced by the user under CC0. No model files are distributed
with this repository.

Three.js (r165) is MIT-licensed. See `web/vendor/` once downloaded.

PySide6 is LGPL-licensed. See [qt.io/licensing](https://www.qt.io/licensing/).
