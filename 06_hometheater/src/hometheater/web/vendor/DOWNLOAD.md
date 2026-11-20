# web/vendor/ — Three.js r165 (download once, not committed to git)

This directory must contain the Two Three.js r165 files listed below.
They are not bundled with the repository because they are large binary
assets; download them once with the commands below.

The app loads Three.js via an ES module importmap in index.html and does
NOT use a CDN — this is a deliberate choice for an offline desktop app
(see the comment at the top of index.html for the full rationale).

---

## Files required

| Filename | Size (approx) | Purpose |
|---|---|---|
| `three.module.js` | ~670 KB | Three.js r165 ES module build |
| `GLTFLoader.js` | ~90 KB | GLTF/GLB loader addon |

---

## Download (pick one method)

### curl

```bash
cd web/vendor

curl -L -o three.module.js \
  https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js

curl -L -o GLTFLoader.js \
  https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/GLTFLoader.js
```

### wget

```bash
cd web/vendor

wget -O three.module.js \
  https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js

wget -O GLTFLoader.js \
  https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/GLTFLoader.js
```

### Browser

Download each URL manually and save with the exact filename shown:

- https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js
  → save as `three.module.js`

- https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/GLTFLoader.js
  → save as `GLTFLoader.js`

---

## Verify

After downloading, this directory should contain exactly:

```
web/vendor/
├── DOWNLOAD.md        ← this file
├── three.module.js    ← ~670 KB
└── GLTFLoader.js      ← ~90 KB
```

If either file is missing, the Three.js scene will not load and the
HUD will display an asset error.

---

## Why r165 specifically?

r165 is pinned in the importmap in index.html and in the GLTF loader
import path. Do not substitute a different version without also updating
the importmap — Three.js's ES module export paths change between minor
versions and a version mismatch will produce silent import failures.
