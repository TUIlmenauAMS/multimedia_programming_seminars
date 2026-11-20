/**
 * scene.js — Three.js 3D scene for the Home Theater Spatial Audio Simulator
 *
 * Multimediaprogrammierung Chapter 05 companion — native PySide6 desktop app.
 * Author: Renato Fera P. / TU Ilmenau  // TODO: confirm author credit block
 *
 * COORDINATE SYSTEM (read this before touching any position/rotation):
 *   - Units: meters, Y-up, right-handed (Three.js default).
 *   - This matches QAudioEngine with distanceScaleMode = DistanceScaleMeter,
 *     and the ITU-R BS.775 speaker circle in audio_engine.py.
 *   - "Forward" (facing the TV) is the −Z direction in world space.
 *   - Listener starts at world origin [0, 1.2, 0] (sweet spot, seated head height).
 *   - All speaker GLB placements use the same azimuth/radius as audio_engine.py —
 *     single source of truth lives there; scene.js reads it from htManifest.room.
 *
 * LOAD ORDER CONTRACT (see index.html):
 *   hud.js is loaded before this file. It registers window.htSignals synchronously.
 *   This file registers window.htAppReady, which index.html calls after the bridge
 *   is open and htManifest is fetched. Do not reorder the script tags.
 *
 * CDN NOTE: Three.js is loaded from web/vendor/ (local copy), not a CDN.
 *   This is a deliberate deviation from the GitHub Pages web demos' CDN pattern.
 *   Reason: this is an offline desktop application — QWebEngineView has no guarantee
 *   of network access at runtime, and a CDN failure would silently break the entire
 *   visual layer. The vendor/ files are pinned to r165 and committed to the repo.
 */

import * as THREE from "three";
import { GLTFLoader }  from "three/addons/GLTFLoader.js";

// ---------------------------------------------------------------------------
// STEP 1: Constants — tunable in one place, never buried in logic below
// ---------------------------------------------------------------------------

/**
 * SWEET_SPOT_BOUNDS — maximum translation the user may apply from the calibrated
 * listening position, in meters.
 *
 * CONCEPT: ITU-R BS.775 defines virtual speaker angles as measured from a *specific*
 * listener position (the "sweet spot"). Moving off that point breaks the stereo image
 * and the binaural decode precision, which is exactly the teaching moment we want the
 * student to feel. The bounds are intentionally tight enough to feel the degradation
 * without letting the student wander into the room geometry or lose the ITU frame
 * entirely. Tune these after first visual test with the real GLB furniture scale.
 *
 * See §7.4 of the build plan.
 */
const SWEET_SPOT_BOUNDS = {
  x: 1.0,   // ± 1.0 m left/right  — wide enough to feel L/R image collapse
  y: 0.3,   // ± 0.3 m up/down     — head bob only
  z: 1.0,   // ± 1.0 m forward/back — enough to feel depth cues change
};

/**
 * SWEET_SPOT_ORIGIN — world position of the calibrated listening position.
 * This must be the exact point that audio_engine.py treats as [0,0,0] for the
 * QAudioListener at session start, and the point from which all ITU speaker angles
 * are measured. It is also the camera start position.
 *
 * Y = 1.2 m ≈ seated ear height.
 */
// Z = +0.8 m: places the listener 2.55 m from the TV screen face — correct
// for a 55" display (ITU recommends ~1.5–3× screen height; 55" height ≈ 0.73 m,
// so 2.55 m ≈ 3.5× — comfortable home-theater distance, 16° vertical subtense).
const SWEET_SPOT_ORIGIN = new THREE.Vector3(0, 1.2, 0.8);

/**
 * SPEAKER_RADIUS — calibration radius from the sweet spot to each positional speaker.
 * All five positional speakers (L, R, C, Ls, Rs) sit on a circle of this radius —
 * equal distance is the ITU calibration requirement. The LFE is non-directional and
 * placed visually near the TV; its distance is cosmetic only.
 *
 * NATIVE ENGINE EQUIVALENT: matches the radius used in audio_engine.py when setting
 * QSpatialSound.position for each source. Keep these numerically identical.
 *
 * See §5.2 of the build plan.
 */
const SPEAKER_RADIUS = 2.0; // meters — matches audio_engine.py

/**
 * POSE_EMIT_HZ — maximum rate at which per-frame camera pose is forwarded to Python
 * via bridge.update_listener_pose(). 60 Hz is fine for most hardware; lower this to
 * 30 if profiling shows the QWebChannel slot call is a bottleneck.
 * See §6 of the build plan.
 */
const POSE_EMIT_HZ = 60;
const POSE_EMIT_INTERVAL_MS = 1000 / POSE_EMIT_HZ;

/**
 * ITU_SPEAKERS — virtual speaker definitions.
 * Azimuth in degrees, 0° = facing TV (−Z), positive = right (counter-clockwise
 * when viewed from above, matching the standard ITU diagram).
 *
 * NATIVE ENGINE EQUIVALENT: these azimuths must mirror the azimuth values used in
 * audio_engine.py to set QSpatialSound.position for each source. If you change one,
 * change the other. The authoritative specification is ITU-R BS.775-3, Table 1.
 *
 * See §5.2 of the build plan.
 */
const ITU_SPEAKERS = [
  { key: "C",   azimuthDeg:    0, radius: SPEAKER_RADIUS, isLFE: false },
  { key: "L",   azimuthDeg:  -30, radius: SPEAKER_RADIUS, isLFE: false },
  { key: "R",   azimuthDeg:   30, radius: SPEAKER_RADIUS, isLFE: false },
  { key: "Ls",  azimuthDeg: -110, radius: SPEAKER_RADIUS, isLFE: false },
  { key: "Rs",  azimuthDeg:  110, radius: SPEAKER_RADIUS, isLFE: false },
  // LFE: non-directional; placed visually near the TV at floor level.
  // CONCEPT: a subwoofer's wavelengths are long enough (~6–17 m) that the human
  // auditory system cannot reliably localize them — placing it off-center would not
  // change the perceived source direction, only the room modal response. The
  // near-TV placement here is a visual convention, not an acoustic claim.
  // See §5.2 build plan note on LFE directionality.
  { key: "LFE", azimuthDeg:    0, radius: 1.2,            isLFE: true  },
];

// ---------------------------------------------------------------------------
// STEP 2: Module-level state — everything the render loop touches
// ---------------------------------------------------------------------------

let renderer, camera, scene;

/** Three.js clock for delta-time, used to throttle pose emission. */
const clock = new THREE.Clock();
let poseAccumulatorMs = 0;

/**
 * speakerMeshes — keyed by ITU_SPEAKERS[i].key → THREE.Object3D.
 * Updated by setActiveTier() to toggle emissive glow.
 */
const speakerMeshes = {};

/**
 * videoTexture — the live THREE.VideoTexture driving the dedicated screen
 * plane in front of the TV (see _attachVideoTexture). THREE.VideoTexture
 * refreshes itself automatically every frame the renderer draws it, so
 * nothing in the render loop needs to poll or tick this reference; it's kept
 * here only so other code (e.g. teardown/dispose) can reach it later.
 */
let videoTexture = null;

/** roomMeshes — keyed by face name ("left","right","front","back","ceiling","floor")
 *  → THREE.Mesh. Updated by setRoomVariant() to swap Three.js materials together
 *  with the acoustic material switch. See §5.4 / §7.3 of the build plan. */
const roomMeshes = {};

/**
 * roomConfig — populated from htManifest.room (loaded in index.html).
 * Holds width/height/depth in meters. Single source of truth for both the
 * Three.js BoxGeometry and the QAudioRoom.dimensions in audio_engine.py.
 * See §5.1 build plan note on shared config.
 */
let roomConfig = null;

// ---------------------------------------------------------------------------
// STEP 3: Entry point — called by index.html after bridge + manifest are ready
// ---------------------------------------------------------------------------

/**
 * htAppReady() is registered on window so index.html's bootstrap script can call it
 * once the QWebChannel is open and htManifest has been fetched.
 *
 * Sequence:
 *   1. Read room config from manifest.
 *   2. Bootstrap Three.js renderer + camera.
 *   3. Build the procedural room box.
 *   4. Load GLTF models (sofa, TV, speakers).
 *   5. Start the render loop.
 *   6. Wire resize handler.
 *
 * CONCEPT: we do not start the render loop until the manifest is confirmed available,
 * because the room dimensions and model paths both come from it. Starting earlier
 * would require placeholder geometry that we'd then swap out — an unnecessary seam.
 */
window.htAppReady = async function htAppReady() {
  const manifest = window.htManifest;

  // Validate the room config block exists before reading dimensions.
  if (!manifest?.room_dimensions) {
    _fatalAssetError(
      "manifest.json is missing the required 'room_dimensions' block.\n" +
      "Expected: { room_dimensions: { width, height, depth } }"
    );
    return;
  }
  roomConfig = manifest.room_dimensions;

  _initRenderer();
  _buildCamera();
  _buildRoom(roomConfig);

  // Load models asynchronously; render loop starts immediately with procedural
  // geometry while models stream in — avoids a blank viewport on slow I/O.
  _loadModels(manifest).catch(err => {
    _fatalAssetError(`Model loading failed: ${err.message}`);
  });

  _startRenderLoop();
  window.addEventListener("resize", _onResize);

  // Expose scene-side controls to hud.js and index.html signal handlers.
  window.htScene = {
    setActiveTier,
    setRoomVariant,
  };
};

// ---------------------------------------------------------------------------
// STEP 4: Renderer and camera setup
// ---------------------------------------------------------------------------

function _initRenderer() {
  renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById("ht-canvas"),
    antialias: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);

  // ReinhardToneMapping is gentler than ACES for a lit interior — ACES crushes
  // shadows too aggressively, making near-black walls invisible.
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 1.8;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene = new THREE.Scene();

  // Slight ambient haze ties the procedural room together before model loading
  // completes. Same dark-grey as --bg in style.css (#0d0e12) for visual continuity.
  scene.background = new THREE.Color(0x0d0e12);
  scene.fog = new THREE.Fog(0x0d0e12, 6, 14);
}

function _buildCamera() {
  camera = new THREE.PerspectiveCamera(
    75,                                        // vertical FOV, degrees
    window.innerWidth / window.innerHeight,    // aspect ratio
    0.05,                                      // near clip — 5 cm, tight for a room-scale scene
    20                                         // far clip — 20 m covers the room + fog fade
  );

  // Start at the ITU calibration sweet spot, facing the TV (−Z).
  camera.position.copy(SWEET_SPOT_ORIGIN);
  camera.lookAt(new THREE.Vector3(0, SWEET_SPOT_ORIGIN.y, -SPEAKER_RADIUS));

  // PointerLock-style look is handled in STEP 7 (controls).
  _initControls();
}

// ---------------------------------------------------------------------------
// STEP 5: Procedural room geometry
// ---------------------------------------------------------------------------

/**
 * _buildRoom — constructs the six interior wall/floor/ceiling planes.
 *
 * CONCEPT: we build the room from individual PlaneGeometry meshes rather than
 * inverting a BoxGeometry, because we need independent material references per face.
 * When the room variant switches (treated ↔ wet), _applyRoomVariant() swaps each
 * face's material and Python's QAudioRoom.Material in a single coordinated call —
 * the two must never be set independently. See §5.4 / §7.3 of the build plan.
 *
 * @param {object} cfg  — { width, height, depth } in meters
 */
function _buildRoom(cfg) {
  const { width: W, height: H, depth: D } = cfg;
  const hw = W / 2, hh = H / 2, hd = D / 2;

  // DoubleSide prevents any plane going invisible if the camera clips its surface.
  const makeMat = (color, roughness = 0.85) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.0, side: THREE.DoubleSide });

  // --- floor --- warm grey-brown carpet
  const floorGeo = new THREE.PlaneGeometry(W, D);
  const floor = new THREE.Mesh(floorGeo, makeMat(0x5a4f3e, 0.95));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  floor.receiveShadow = true;
  scene.add(floor);
  roomMeshes.floor = floor;

  // --- ceiling --- off-white, reflects light back down
  const ceilGeo = new THREE.PlaneGeometry(W, D);
  const ceil = new THREE.Mesh(ceilGeo, makeMat(0xc8c4bc, 0.9));
  ceil.rotation.x = Math.PI / 2;
  ceil.position.y = H;
  scene.add(ceil);
  roomMeshes.ceiling = ceil;

  // --- back wall (behind listener, +Z) --- warm taupe
  const backGeo = new THREE.PlaneGeometry(W, H);
  const back = new THREE.Mesh(backGeo, makeMat(0x7a7060));
  back.position.set(0, hh, hd);
  back.rotation.y = Math.PI;
  scene.add(back);
  roomMeshes.back = back;

  // --- front wall (TV wall, −Z) --- slightly darker, draws eye to screen
  const frontGeo = new THREE.PlaneGeometry(W, H);
  const front = new THREE.Mesh(frontGeo, makeMat(0x4a4540));
  front.position.set(0, hh, -hd);
  scene.add(front);
  roomMeshes.front = front;

  // --- left wall (−X) ---
  const leftGeo = new THREE.PlaneGeometry(D, H);
  const left = new THREE.Mesh(leftGeo, makeMat(0x7a7060));
  left.position.set(-hw, hh, 0);
  left.rotation.y = Math.PI / 2;
  scene.add(left);
  roomMeshes.left = left;

  // --- right wall (+X) ---
  const rightGeo = new THREE.PlaneGeometry(D, H);
  const right = new THREE.Mesh(rightGeo, makeMat(0x7a7060));
  right.position.set(hw, hh, 0);
  right.rotation.y = -Math.PI / 2;
  scene.add(right);
  roomMeshes.right = right;

  // --- 3-point lighting rig ---
  // Key: warm overhead, like recessed ceiling downlights.
  const ambient = new THREE.AmbientLight(0xfff5e0, 1.4);
  scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0xffe8c0, 2.5);
  keyLight.position.set(0, H - 0.1, 0);
  keyLight.target.position.set(0, 0, 0);
  keyLight.castShadow = true;
  scene.add(keyLight);
  scene.add(keyLight.target);

  // Fill: cooler light from behind listener, lifts sofa and back wall.
  const fillLight = new THREE.DirectionalLight(0xd0e8ff, 1.0);
  fillLight.position.set(0, hh, hd - 0.3);
  fillLight.target.position.set(0, hh, -1);
  scene.add(fillLight);
  scene.add(fillLight.target);

  // Rim: subtle warm side-fill to prevent flat look.
  const rimLight = new THREE.DirectionalLight(0xffe0a0, 0.6);
  rimLight.position.set(-hw + 0.5, H - 0.5, 0);
  rimLight.target.position.set(0, 0, 0);
  scene.add(rimLight);
  scene.add(rimLight.target);
}

// ---------------------------------------------------------------------------
// STEP 6: GLTF model loading — sofa, TV (with VideoTexture), speakers
// ---------------------------------------------------------------------------

/**
 * _loadModels — loads all user-supplied GLB files from paths in the manifest.
 *
 * Error policy: if any model fails to load, we call _fatalAssetError() with a
 * human-readable message naming the missing file. We never silently render an
 * empty scene. See §1 and §7.3 of the build plan.
 *
 * @param {object} manifest — the parsed manifest.json object
 */
async function _loadModels(manifest) {
  const loader = new GLTFLoader();

  // Helper: promisify GLTFLoader.load() with a clear error on failure.
  const loadGLB = (url, label) =>
    new Promise((resolve, reject) => {
      loader.load(
        url,
        gltf => resolve(gltf),
        undefined,  // progress callback — not needed here
        err => reject(new Error(`Could not load ${label} ("${url}"): ${err.message || err}`))
      );
    });

  // Sofa — user-supplied CC0 model, path from manifest.
  if (!manifest.models?.sofa) {
    _fatalAssetError("manifest.json is missing models.sofa path.");
    return;
  }
  const sofaGltf = await loadGLB(`ht://assets/${manifest.models.sofa}`, "sofa");

  // Auto-scale the sofa to a real-world size. A standard 3-seat sofa is ~2.1 m wide.
  // We compute the bounding box after load and scale to that target width.
  // This handles whatever unit the GLB was authored in.
  {
    const bbox = new THREE.Box3().setFromObject(sofaGltf.scene);
    const sofaSize = new THREE.Vector3();
    bbox.getSize(sofaSize);
    const sofaTargetWidth = 2.1; // metres — standard 3-seat sofa
    const sofaScale = sofaTargetWidth / sofaSize.x;
    sofaGltf.scene.scale.setScalar(sofaScale);

    // Recompute bbox after scaling to get correct Y-min for floor placement.
    const bbox2 = new THREE.Box3().setFromObject(sofaGltf.scene);
    const yMin = bbox2.min.y;

    // Centre of sofa at sweet-spot Z so the listener sits in it.
    // Offset back by ~0.3 m so the sweet spot is near the front cushion.
    sofaGltf.scene.position.set(0, -yMin, SWEET_SPOT_ORIGIN.z + 0.3);

    // Rotate 180° around Y so the sofa faces the TV (−Z direction).
    // GLB sofas are typically authored facing +Z (toward the camera by default).
    sofaGltf.scene.rotation.y = Math.PI;
  }
  scene.add(sofaGltf.scene);

  // TV — user-supplied CC0 model. After loading, we find the screen face and
  // attach the VideoTexture. The video element is always muted (see index.html).
  if (!manifest.models?.tv) {
    _fatalAssetError("manifest.json is missing models.tv path.");
    return;
  }
  const tvGltf = await loadGLB(`ht://assets/${manifest.models.tv}`, "TV");

  // ---------------------------------------------------------------------------
  // TV CONSOLE TABLE — procedural box, real-world proportions:
  //   Width:  1.80 m  (wider than the 1.218 m TV so it has shelf space either side)
  //   Height: 0.45 m  (standard low AV console / media unit)
  //   Depth:  0.45 m  (deep enough for the center speaker to sit in front)
  //
  // Back face is flush with the front wall (z = -roomConfig.depth/2).
  // Centre X = 0 (matches TV horizontal centre).
  // ---------------------------------------------------------------------------
  const TV_TABLE_W  = 1.80;  // metres
  const TV_TABLE_H  = 0.45;  // metres
  const TV_TABLE_D  = 0.45;  // metres
  const TV_TABLE_Z  = -(roomConfig.depth / 2) + TV_TABLE_D / 2;  // centre-Z of table

  const tableGeo = new THREE.BoxGeometry(TV_TABLE_W, TV_TABLE_H, TV_TABLE_D);
  const tableMat = new THREE.MeshStandardMaterial({
    color:     0x2a2218,   // dark espresso wood
    roughness: 0.6,
    metalness: 0.05,
  });
  const tableMesh = new THREE.Mesh(tableGeo, tableMat);
  tableMesh.position.set(0, TV_TABLE_H / 2, TV_TABLE_Z);
  tableMesh.castShadow    = true;
  tableMesh.receiveShadow = true;
  scene.add(tableMesh);

  // Table legs — four thin boxes, one at each corner.
  // Real media consoles often have thin tapered legs; we use 0.04 m square legs.
  const LEG_W = 0.04, LEG_H = TV_TABLE_H, LEG_D = 0.04;
  const legGeo = new THREE.BoxGeometry(LEG_W, LEG_H, LEG_D);
  const legMat = tableMat; // same wood material
  const legOffsets = [
    [ TV_TABLE_W / 2 - 0.06,  TV_TABLE_D / 2 - 0.06],
    [-TV_TABLE_W / 2 + 0.06,  TV_TABLE_D / 2 - 0.06],
    [ TV_TABLE_W / 2 - 0.06, -TV_TABLE_D / 2 + 0.06],
    [-TV_TABLE_W / 2 + 0.06, -TV_TABLE_D / 2 + 0.06],
  ];
  for (const [lx, lz] of legOffsets) {
    const leg = new THREE.Mesh(legGeo, legMat);
    leg.position.set(lx, LEG_H / 2, TV_TABLE_Z + lz);
    scene.add(leg);
  }

  // ---------------------------------------------------------------------------
  // TV MODEL — led_tv.glb (Sketchfab), raw bbox ≈ 13.0 W × 7.75 H × 2.15 D
  //   TV_SCALE 0.09369 → 1.218 m W × 0.726 m H × 0.202 m D  (≈ 55" panel)
  //
  // Placed on top of the console table:
  //   X: +0.0468 m corrects the mesh origin's horizontal offset from true centre
  //   Y: table surface (TV_TABLE_H) minus the raw Y-min correction (0.0217 m)
  //      so the TV stand base sits flush on the table top
  //   Z: front face of table + small inset so the stand doesn't overhang
  // ---------------------------------------------------------------------------
  // TV_SCALE: raw bbox ≈ 13.0 W × 7.75 H × 2.15 D (GLB units).
  // 0.1327 → 1.725 m W ≈ 75" panel. Adjust if the model looks wrong in-scene.
  const TV_SCALE = 0.1327;
  tvGltf.scene.scale.setScalar(TV_SCALE);
  // Y-min of the TV stand scales proportionally: raw 0.0217 * TV_SCALE / 0.09369
  const TV_Y_MIN = 0.0217 * (TV_SCALE / 0.09369);
  tvGltf.scene.position.set(
    0.0468,                            // X: horizontal centre correction
    TV_TABLE_H - TV_Y_MIN,            // Y: stand base flush on table top
    TV_TABLE_Z - 0.02                 // Z: slight inset so stand fits
  );
  scene.add(tvGltf.scene);
  _attachVideoTexture(tvGltf.scene, manifest.media?.video);

  // Speakers — one GLB reused for all five positional speakers, with an optional
  // distinct subwoofer GLB for the LFE position. See §7.3 of the build plan.
  if (!manifest.models?.speaker) {
    _fatalAssetError("manifest.json is missing models.speaker path.");
    return;
  }
  const speakerGltf = await loadGLB(`ht://assets/${manifest.models.speaker}`, "speaker");
  // Scale speaker to a real bookshelf speaker: ~0.35 m tall.
  _autoScaleGLB(speakerGltf.scene, 0.35, "y");

  // Load center-channel model — manifest key is models.center or legacy model_center.
  // Scale on Y to 0.20 m (realistic height for a horizontal center-channel cabinet).
  // Scaling on X produced a disproportionately giant model when falling back to the
  // bookshelf speaker GLB, because that model is taller than it is wide.
  let centerGltf = speakerGltf;
  const centerPath = manifest.models?.center ?? manifest.model_center;
  if (centerPath) {
    centerGltf = await loadGLB(`ht://assets/${centerPath}`, "center speaker");
  }
  // Always (re-)scale the center on Y — overrides the 0.35 m bookshelf scale if
  // centerGltf fell back to speakerGltf, and correctly sizes a dedicated model too.
  _autoScaleGLB(centerGltf.scene, 0.20, "y");

  // Load subwoofer model; fall back to the standard speaker model if not provided.
  let subGltf = speakerGltf;
  if (manifest.models?.subwoofer) {
    subGltf = await loadGLB(`ht://assets/${manifest.models.subwoofer}`, "subwoofer");
    // Subwoofer: ~0.45 m tall cube.
    _autoScaleGLB(subGltf.scene, 0.45, "y");
  }

  _placeSpeakers(speakerGltf, centerGltf, subGltf);
}

/**
 * TV_SCREEN_INSET — fraction of the TV model's own bounding box that is NOT
 * screen (thin bezel on the sides/top, stand/neck below the panel). These
 * are visual tuning knobs, not physics: if a different TV GLB is ever
 * swapped in, nudge these fractions while looking at the render until the
 * video sits exactly inside the glass and never spills onto the body.
 *
 * Measured by eye against led_tv.glb: the panel is almost edge-to-edge (a
 * very thin bezel on the sides/top), and roughly the bottom tenth of the
 * model's total height is the stand/neck below the panel, not screen.
 */
const TV_SCREEN_INSET = {
  left:   0.015,
  right:  0.015,
  top:    0.02,
  bottom: 0.10,
};

/**
 * TV_SCREEN_FORWARD_OFFSET — how far (metres) the video plane sits in front
 * of the TV model's own front face. Must be > 0 or the two surfaces z-fight;
 * kept tiny so the video still reads as flush with the glass.
 */
const TV_SCREEN_FORWARD_OFFSET = 0.004;

/**
 * _attachVideoTexture — places a dedicated video plane exactly over the TV's
 * screen area, instead of baking the video into the TV model's shared
 * body/bezel/stand material.
 *
 * WHY A SEPARATE PLANE (read this before changing the approach):
 * -----------------------------------------------------------------
 * led_tv.glb is a single mesh with a single material shared by the bezel,
 * stand, back panel, buttons and logo — there is no mesh or material that
 * belongs to just the glass. The previous build tried to paint the video
 * into a hand-measured sub-rectangle of that shared material's texture atlas
 * and then rotate it 90° to compensate for an assumed UV mismatch. That
 * approach is fragile by construction (it depends on exact pixel coordinates
 * that have to be re-derived by hand for this specific model) and is exactly
 * what caused the video to show up rotated, cropped, and offset onto the
 * wrong part of the TV instead of filling the screen.
 *
 * The robust fix is to leave the TV's own material completely untouched —
 * so its PBR bezel/stand/logo detail keeps working exactly as modelled — and
 * lay an independent, unlit video quad directly in front of the screen area:
 *   1. Measure the TV model's own world-space bounding box after it has
 *      already been scaled and positioned (STEP 6 did that just above).
 *   2. Shrink that box by TV_SCREEN_INSET to exclude the bezel and the
 *      stand/neck — the plane can then only ever cover the glass, never the
 *      TV body, satisfying "video only on the screen".
 *   3. Size the plane to the VIDEO's own native aspect ratio ("contain" fit)
 *      once its metadata is known, so the footage is never stretched,
 *      rotated, or cropped — any leftover sliver inside the inset simply
 *      shows the TV's own dark body colour, like unused letterbox bars on a
 *      real set.
 *   4. Use MeshBasicMaterial with toneMapped=false so the screen reads as a
 *      self-lit display rather than being dimmed by the room's Reinhard
 *      tone mapping.
 *   5. Use THREE.VideoTexture, which the renderer refreshes automatically
 *      every frame a new video frame is available — no manual canvas
 *      compositing or per-frame tick() bookkeeping required.
 *
 * @param {THREE.Object3D} tvScene  — the loaded TV GLTF scene graph (already
 *                                    scaled, positioned, and added to `scene`)
 * @param {string|undefined} videoPath — relative video path from manifest
 */
function _attachVideoTexture(tvScene, videoPath) {
  // Make sure world matrices are current — tvScene's position/scale were set
  // moments ago, and matrixWorld is otherwise only refreshed during render().
  tvScene.updateMatrixWorld(true);

  const bbox = new THREE.Box3().setFromObject(tvScene);
  if (bbox.isEmpty()) {
    console.warn("[scene.js] TV model has no geometry — cannot place screen.");
    return;
  }

  const size = new THREE.Vector3();
  bbox.getSize(size);

  // Shrink the full TV bounding box down to just the glass.
  const screenFullW = size.x * (1 - TV_SCREEN_INSET.left - TV_SCREEN_INSET.right);
  const screenFullH = size.y * (1 - TV_SCREEN_INSET.top  - TV_SCREEN_INSET.bottom);

  const centerX = (bbox.min.x + bbox.max.x) / 2;
  // Anchor from the TOP of the model down, since the bottom inset (stand/
  // neck) is larger than the top inset (bezel) — the screen is not
  // vertically centred within the model's full bounding box.
  const screenTopY    = bbox.max.y - size.y * TV_SCREEN_INSET.top;
  const screenCenterY = screenTopY - screenFullH / 2;

  // Front face = the TV model's own max-Z extent. The model already faces
  // the sweet spot in +Z, same as every other prop in this room, so no
  // rotation is applied here — which is exactly what removes the old
  // "rotated video" bug.
  const screenZ = bbox.max.z + TV_SCREEN_FORWARD_OFFSET;

  // Unit plane (1m × 1m) so _fitToScreen() below can simply rescale it per
  // axis instead of rebuilding geometry once the video's real aspect ratio
  // is known.
  const screenGeo = new THREE.PlaneGeometry(1, 1);
  const screenMat = new THREE.MeshBasicMaterial({
    color: 0x001428,   // standby colour: deep "screen is on but idle" blue
    toneMapped: false,  // self-lit display — not subject to room tone mapping
  });
  const screenMesh = new THREE.Mesh(screenGeo, screenMat);
  screenMesh.scale.set(screenFullW, screenFullH, 1);
  screenMesh.position.set(centerX, screenCenterY, screenZ);
  scene.add(screenMesh);

  if (!videoPath) {
    console.warn("[scene.js] No media.video in manifest — TV shows standby colour.");
    return;
  }

  // -------------------------------------------------------------------------
  // Wire the <video> element. Playback is driven by bridge.video_command;
  // we only configure src/attributes here.
  // -------------------------------------------------------------------------
  const videoEl = document.getElementById("ht-video");
  videoEl.src         = `ht://assets/${videoPath}`;
  videoEl.muted       = true;
  videoEl.loop        = true;
  videoEl.playsInline = true;
  videoEl.preload     = "auto";

  const videoTex = new THREE.VideoTexture(videoEl);
  videoTex.colorSpace      = THREE.SRGBColorSpace;
  videoTex.minFilter       = THREE.LinearFilter;
  videoTex.magFilter       = THREE.LinearFilter;
  videoTex.generateMipmaps = false;

  // Fit the video to the screen without ever stretching, rotating, or
  // cropping it: shrink whichever axis would otherwise overshoot the glass,
  // so the full frame is always visible ("contain"), centred on the same
  // point the standby colour occupied.
  function _fitToScreen() {
    const videoAspect  = videoEl.videoWidth / videoEl.videoHeight;
    const screenAspect = screenFullW / screenFullH;
    let w = screenFullW, h = screenFullH;
    if (videoAspect > screenAspect) {
      h = screenFullW / videoAspect;   // wider than the glass → letterbox top/bottom
    } else {
      w = screenFullH * videoAspect;   // narrower than the glass → pillarbox sides
    }
    screenMesh.scale.set(w, h, 1);
  }

  if (videoEl.readyState >= videoEl.HAVE_METADATA) {
    _fitToScreen();
  } else {
    videoEl.addEventListener("loadedmetadata", _fitToScreen, { once: true });
  }

  screenMat.map = videoTex;
  screenMat.color.set(0xffffff);  // let the video texture show at full brightness
  screenMat.needsUpdate = true;

  // Kept on the module-level slot for parity with the rest of the file and
  // in case other code ever needs to reach the live texture (e.g. to dispose
  // of it on teardown); the render loop no longer needs to "tick" it — see
  // STEP 9, where THREE.VideoTexture updates itself automatically.
  videoTexture = videoTex;
}

/**
 * _autoScaleGLB — scales a loaded GLTF scene so one axis matches a real-world
 * target size in metres. Handles Sketchfab models exported in cm, mm, or any
 * arbitrary unit without requiring manual measurement per model.
 *
 * @param {THREE.Object3D} gltfScene  — the gltf.scene to scale in-place
 * @param {number} targetMetres       — desired real-world size on the axis
 * @param {"x"|"y"|"z"} axis          — bounding-box axis to match
 */
function _autoScaleGLB(gltfScene, targetMetres, axis = "y") {
  const bbox = new THREE.Box3().setFromObject(gltfScene);
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const current = size[axis];
  if (current > 0) gltfScene.scale.setScalar(targetMetres / current);
}

/**
 * _getYMin — returns the world-space Y minimum of a (possibly already scaled)
 * GLTF scene object. Used to compute the floor-placement offset: positioning
 * an instance at y = -yMin puts its lowest point exactly on y = 0.
 *
 * GLB model origins are rarely at the mesh bottom, so hardcoding y = 0 or any
 * constant sinks part of the mesh into the floor. This measures the actual bbox.
 *
 * @param {THREE.Object3D} obj — the scene object at its final scale
 * @returns {number} world-space Y minimum of the bounding box
 */
function _getYMin(obj) {
  return new THREE.Box3().setFromObject(obj).min.y;
}

/**
 * _placeSpeakers — instances speaker GLBs at the ITU_SPEAKERS positions and
 * registers each in the speakerMeshes map for tier-driven glow control.
 *
 * Speaker positions are computed relative to SWEET_SPOT_ORIGIN, so the ITU
 * calibration circle is always centred on the actual listening position.
 *
 * CONCEPT: each physical speaker position corresponds to a QSpatialSound source
 * in audio_engine.py at the same world position. The emissive glow mirrors the
 * acoustic reality. See §7.3 of the build plan.
 *
 * @param {GLTF} speakerGltf   — base speaker model (L, R, Ls, Rs)
 * @param {GLTF} centerGltf    — center-channel model (C); may equal speakerGltf
 * @param {GLTF} subGltf       — subwoofer model (LFE); may equal speakerGltf
 */
function _placeSpeakers(speakerGltf, centerGltf, subGltf) {
  // ---------------------------------------------------------------------------
  // Visual placement constants — must stay in sync with the table built in
  // _loadModels. Re-derived here to avoid a shared variable across async scopes.
  // ---------------------------------------------------------------------------
  const TV_TABLE_H = 0.45;   // metres — console table top surface height
  const TV_TABLE_D = 0.45;   // metres — console table depth
  const TV_TABLE_Z = -(roomConfig.depth / 2) + TV_TABLE_D / 2;

  for (const spk of ITU_SPEAKERS) {
    const gltf = spk.isLFE ? subGltf : (spk.key === "C" ? centerGltf : speakerGltf);

    // Deep-clone so each instance has independent materials for emissive glow.
    const instance = gltf.scene.clone(true);

    // ------------------------------------------------------------------
    // Measure the cloned instance's actual bounding box BEFORE positioning.
    // GLB origins are almost never at the mesh bottom, so we compute yMin
    // and use -yMin as the Y offset to put the lowest point at any target
    // surface. Without this, speakers sink into or float above surfaces.
    // ------------------------------------------------------------------
    const yMin = _getYMin(instance);

    let x, y, z;

    if (spk.key === "C") {
      // -----------------------------------------------------------------------
      // CENTER SPEAKER — on the TV console table, horizontally centred.
      // y = TV_TABLE_H - yMin puts the cabinet base flush on the table surface.
      // Z = front edge of table (table centre + half depth - small margin) so
      // the speaker sits at the front of the unit facing the listener.
      // -----------------------------------------------------------------------
      x = 0;
      y = TV_TABLE_H - yMin;
      z = TV_TABLE_Z + (TV_TABLE_D / 2) - 0.08;  // near front edge of table

    } else if (spk.isLFE) {
      // -----------------------------------------------------------------------
      // LFE SUBWOOFER — on the floor beside the right edge of the TV table.
      // y = -yMin puts its base flush on the floor regardless of origin offset.
      // Table right edge is at X = +0.90 m; sub is 15 cm clear of that.
      // -----------------------------------------------------------------------
      x = 0.90 + 0.15;
      y = -yMin;                                   // base flush on floor
      z = -(roomConfig.depth / 2) + 0.25;

    } else {
      // -----------------------------------------------------------------------
      // L, R, Ls, Rs — ITU-R BS.775 circle XZ positions (unchanged).
      // y = -yMin puts the cabinet base on the floor. The model is 0.35 m tall
      // so the tweeter sits at ~0.175 m — these are floor-standing speakers,
      // not bookshelf speakers on stands, which matches what the GLB looks like.
      // -----------------------------------------------------------------------
      const azRad = THREE.MathUtils.degToRad(spk.azimuthDeg);
      x = SWEET_SPOT_ORIGIN.x + spk.radius * Math.sin(azRad);
      z = SWEET_SPOT_ORIGIN.z - spk.radius * Math.cos(azRad);
      y = -yMin;  // base on the floor
    }

    instance.position.set(x, y, z);

    // ------------------------------------------------------------------
    // Orient each speaker to face the sweet spot.
    // lookAt() uses the object's current Y position as the pivot, so we
    // must call it AFTER setting position. We pass a target at the same Y
    // as the instance to prevent any tilt — only the horizontal angle matters.
    // ------------------------------------------------------------------
    if (spk.key === "C") {
      // Center: faces straight toward +Z (listener direction). No tilt needed.
      instance.rotation.set(0, Math.PI, 0);
    } else {
      // Surround and front L/R: face toward the sweet spot horizontally.
      // Construct a target at the same Y as the speaker to avoid tilt.
      const flatTarget = new THREE.Vector3(
        SWEET_SPOT_ORIGIN.x,
        y,                        // same Y as speaker — keeps it upright
        SWEET_SPOT_ORIGIN.z
      );
      instance.lookAt(flatTarget);
    }

    scene.add(instance);
    speakerMeshes[spk.key] = instance;

    // Prepare emissive material references on all child meshes — we toggle
    // emissiveIntensity rather than swapping materials to avoid GC pressure in
    // the render loop.
    instance.traverse(child => {
      if (!child.isMesh) return;
      // Ensure each mesh has its own material instance (clone() may share refs).
      if (!Array.isArray(child.material)) {
        child.material = child.material.clone();
      }
      // Set the emissive colour but start with intensity 0 (all speakers off).
      // LFE uses a different accent colour to visually distinguish it.
      child.material.emissive = spk.isLFE
        ? new THREE.Color(0xf59e0b)   // amber — matches --orange in style.css
        : new THREE.Color(0x22d3ee);  // cyan  — matches --accent in style.css
      child.material.emissiveIntensity = 0.0;
    });
  }
}

// ---------------------------------------------------------------------------
// STEP 7: POV camera controls — mouse-look + WASD, clamped to sweet spot
// ---------------------------------------------------------------------------

/**
 * Controls state — raw input accumulated each frame, applied in _updateControls().
 * We track euler angles manually rather than using PointerLockControls because
 * QWebEngineView intercepts pointer lock requests on some platforms; raw mousemove
 * on the canvas is more portable in a WebEngine context.
 */
const _ctrl = {
  yaw:   0,     // radians, free — no clamp on look direction
  pitch: 0,     // radians, free
  // Translation offset from SWEET_SPOT_ORIGIN, clamped each frame.
  offset: new THREE.Vector3(),
  // Mouse drag state.
  leftDrag:  false,
  rightDrag: false,
  lastX: 0,
  lastY: 0,
  // WASD key state.
  keys: { w: false, a: false, s: false, d: false },
};

function _initControls() {
  const canvas = renderer.domElement;

  // --- Mouse look (left drag) and pan (right drag) ---
  canvas.addEventListener("mousedown", e => {
    if (e.button === 0) { _ctrl.leftDrag  = true; }
    if (e.button === 2) { _ctrl.rightDrag = true; }
    _ctrl.lastX = e.clientX;
    _ctrl.lastY = e.clientY;
  });
  window.addEventListener("mouseup", e => {
    if (e.button === 0) _ctrl.leftDrag  = false;
    if (e.button === 2) _ctrl.rightDrag = false;
  });
  window.addEventListener("mousemove", e => {
    if (!_ctrl.leftDrag && !_ctrl.rightDrag) return;
    const dx = e.clientX - _ctrl.lastX;
    const dy = e.clientY - _ctrl.lastY;
    _ctrl.lastX = e.clientX;
    _ctrl.lastY = e.clientY;

    if (_ctrl.leftDrag) {
      // Left drag = look (yaw + pitch). Standard FPS mouse-look.
      _ctrl.yaw   -= dx * 0.002;
      _ctrl.pitch -= dy * 0.002;
      // Clamp pitch to avoid gimbal flip.
      _ctrl.pitch = THREE.MathUtils.clamp(_ctrl.pitch, -Math.PI / 2.5, Math.PI / 2.5);
    }

    if (_ctrl.rightDrag) {
      // Right drag = walk in the horizontal plane.
      // dy (vertical drag) → move forward/back along camera look direction in XZ.
      // dx (horizontal drag) → strafe left/right perpendicular to look direction.
      const moveSpeed = 0.004;
      const fwdX = -Math.sin(_ctrl.yaw);
      const fwdZ = -Math.cos(_ctrl.yaw);
      // Forward/back: dragging up (negative dy) moves forward toward the TV.
      _ctrl.offset.x += fwdX * (-dy) * moveSpeed;
      _ctrl.offset.z += fwdZ * (-dy) * moveSpeed;
      // Strafe: dragging right moves right.
      _ctrl.offset.x += Math.cos(_ctrl.yaw) * dx * moveSpeed;
      _ctrl.offset.z -= Math.sin(_ctrl.yaw) * dx * moveSpeed;
    }
  });

  // Suppress context menu on right-click so right-drag pan works cleanly.
  canvas.addEventListener("contextmenu", e => e.preventDefault());

  // --- Scroll wheel = move forward/back along camera look direction in XZ ---
  // Moves along sin(yaw)/cos(yaw) so forward always means "toward where you're
  // looking", not a fixed world axis. scrolling up (negative deltaY) = forward.
  canvas.addEventListener("wheel", e => {
    const moveAmt = e.deltaY * 0.001;
    _ctrl.offset.x += -Math.sin(_ctrl.yaw) * (-moveAmt);
    _ctrl.offset.z += -Math.cos(_ctrl.yaw) * (-moveAmt);
  }, { passive: true });

  // --- WASD keys ---
  window.addEventListener("keydown", e => {
    const k = e.key.toLowerCase();
    if (k in _ctrl.keys) _ctrl.keys[k] = true;
  });
  window.addEventListener("keyup", e => {
    const k = e.key.toLowerCase();
    if (k in _ctrl.keys) _ctrl.keys[k] = false;
  });
}

/**
 * _updateControls — called every frame before the render call.
 * Applies WASD movement, clamps the offset to SWEET_SPOT_BOUNDS, then composes
 * the camera's world transform from yaw/pitch euler angles + clamped offset.
 *
 * @param {number} dt — frame delta-time in seconds
 */
function _updateControls(dt) {
  const MOVE_SPEED = 0.4; // m/s — slow enough to feel deliberate, fast enough to explore

  // WASD movement in camera-local XZ (ignoring the Y component of the look direction
  // so "forward" stays horizontal — standard FPS convention).
  const forward = new THREE.Vector3(
    -Math.sin(_ctrl.yaw),
     0,
    -Math.cos(_ctrl.yaw)
  );
  const right = new THREE.Vector3(
    Math.cos(_ctrl.yaw),
    0,
    -Math.sin(_ctrl.yaw)
  );

  if (_ctrl.keys.w) _ctrl.offset.addScaledVector(forward, MOVE_SPEED * dt);
  if (_ctrl.keys.s) _ctrl.offset.addScaledVector(forward, -MOVE_SPEED * dt);
  if (_ctrl.keys.a) _ctrl.offset.addScaledVector(right, -MOVE_SPEED * dt);
  if (_ctrl.keys.d) _ctrl.offset.addScaledVector(right,  MOVE_SPEED * dt);

  // Clamp offset to SWEET_SPOT_BOUNDS — the invisible acoustic fence.
  _ctrl.offset.x = THREE.MathUtils.clamp(_ctrl.offset.x, -SWEET_SPOT_BOUNDS.x, SWEET_SPOT_BOUNDS.x);
  _ctrl.offset.y = THREE.MathUtils.clamp(_ctrl.offset.y, -SWEET_SPOT_BOUNDS.y, SWEET_SPOT_BOUNDS.y);
  _ctrl.offset.z = THREE.MathUtils.clamp(_ctrl.offset.z, -SWEET_SPOT_BOUNDS.z, SWEET_SPOT_BOUNDS.z);

  // Compose camera world position: sweet spot + clamped offset.
  camera.position.copy(SWEET_SPOT_ORIGIN).add(_ctrl.offset);

  // Compose camera orientation from yaw + pitch euler angles.
  // Order: YXZ — yaw around world-Y, then pitch around local-X. Standard FPS.
  const euler = new THREE.Euler(_ctrl.pitch, _ctrl.yaw, 0, "YXZ");
  camera.quaternion.setFromEuler(euler);
}

// ---------------------------------------------------------------------------
// STEP 8: Per-frame bridge pose emission
// ---------------------------------------------------------------------------

/**
 * _emitListenerPose — sends the camera's current world position and orientation
 * to Python via htBridge.update_listener_pose(x, y, z, qx, qy, qz, qw).
 *
 * CONCEPT: Python's QAudioListener position/rotation must track the Three.js camera
 * every frame so the binaural decode reflects where the virtual listener actually is,
 * not where they started. This is the real-time synchronization contract between the
 * visual and audio engines. See §6 and §7.4 of the build plan.
 *
 * NATIVE ENGINE EQUIVALENT:
 *   camera.position → QAudioListener.setPosition(x, y, z)
 *   camera.quaternion → QAudioListener.setRotation(qx, qy, qz, qw)
 *
 * Throttled to POSE_EMIT_HZ to avoid flooding the QWebChannel slot queue.
 * If profiling shows the bridge call is a bottleneck, reduce POSE_EMIT_HZ to 30.
 * See §6 of the build plan for the note on not applying smoothing here.
 */
function _emitListenerPose() {
  const p = camera.position;
  const q = camera.quaternion;
  window.htBridge?.update_listener_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w);
}

/**
 * _computeOffAxisReadout — returns a human-readable string of the current
 * displacement from the sweet spot, with direction components.
 *
 * Example output: "14 cm left, 8 cm forward"
 * Example output at origin: "sweet spot"
 *
 * CONCEPT: the readout turns the movement constraint into a visible teaching
 * point — the student can hear imaging degrade while watching their distance grow.
 * "Direction version" chosen over bare distance because it names which axis is
 * involved, reinforcing the asymmetry between lateral and depth displacement
 * in a 5.1 mix. See §7.4 of the build plan.
 */
function _computeOffAxisReadout() {
  const o = _ctrl.offset;
  const totalCm = Math.round(o.length() * 100);

  if (totalCm < 2) return "sweet spot";  // within 2 cm → on-axis

  const parts = [];

  const xCm = Math.round(Math.abs(o.x) * 100);
  if (xCm >= 2) parts.push(`${xCm} cm ${o.x < 0 ? "left" : "right"}`);

  const yCm = Math.round(Math.abs(o.y) * 100);
  if (yCm >= 2) parts.push(`${yCm} cm ${o.y < 0 ? "down" : "up"}`);

  // Z convention: negative Z is toward the TV (forward in this room).
  const zCm = Math.round(Math.abs(o.z) * 100);
  if (zCm >= 2) parts.push(`${zCm} cm ${o.z < 0 ? "forward" : "back"}`);

  return parts.join(", ");
}

// ---------------------------------------------------------------------------
// STEP 9: Render loop
// ---------------------------------------------------------------------------

function _startRenderLoop() {
  function frame() {
    requestAnimationFrame(frame);

    const dtSec = clock.getDelta();

    // 1. Update controls — WASD + mouse, clamp offset.
    _updateControls(dtSec);

    // 2. Emit listener pose to Python at the throttled rate.
    poseAccumulatorMs += dtSec * 1000;
    if (poseAccumulatorMs >= POSE_EMIT_INTERVAL_MS) {
      _emitListenerPose();
      poseAccumulatorMs -= POSE_EMIT_INTERVAL_MS;
    }

    // 3. Push off-axis readout to the HUD. htSignals is registered by hud.js.
    if (window.htSignals?.onOffAxisReadout) {
      window.htSignals.onOffAxisReadout(_computeOffAxisReadout());
    }

    // 4. Render. THREE.VideoTexture (the screen plane's map — see
    //    _attachVideoTexture) refreshes itself from the <video> element
    //    automatically whenever a new frame is available; nothing to do here.
    renderer.render(scene, camera);
  }

  frame();
}

// ---------------------------------------------------------------------------
// STEP 10: Tier and room variant switches (called by hud.js via window.htScene)
// ---------------------------------------------------------------------------

/**
 * setActiveTier — updates speaker emissive glow to reflect which sources are
 * active in the given tier. Called after Python confirms the switch via
 * state_changed(tier, room, is_playing).
 *
 * CONCEPT: the emissive glow makes the tier concept visible — in Mono, only the
 * center speaker lights up; in Stereo, left and right; in 5.1, all five plus LFE.
 * This directly mirrors the chapter's principle that the visual scene and acoustic
 * scene should change together. See §7.3 of the build plan.
 *
 * NATIVE ENGINE EQUIVALENT:
 *   "speaker glows" ↔ QSpatialSound.volume = 1.0 (active) vs 0.0 (muted)
 *
 * @param {"mono"|"stereo"|"5.1"} tier
 */
function setActiveTier(tier) {
  // Which speaker keys are acoustically active per tier — matches audio_engine.py §5.3.
  const activeSets = {
    "mono":   new Set(["C"]),
    "stereo": new Set(["L", "R"]),
    "5.1":    new Set(["C", "L", "R", "Ls", "Rs", "LFE"]),
  };

  const active = activeSets[tier];
  if (!active) {
    console.warn(`[scene.js] Unknown tier: "${tier}"`);
    return;
  }

  for (const spk of ITU_SPEAKERS) {
    const mesh = speakerMeshes[spk.key];
    if (!mesh) continue;

    const intensity = active.has(spk.key) ? 1.5 : 0.0;
    mesh.traverse(child => {
      if (child.isMesh && child.material?.emissive) {
        child.material.emissiveIntensity = intensity;
      }
    });
  }
}

/**
 * setRoomVariant — updates Three.js wall materials to match the acoustic room
 * variant selected. Must be called in the same event as the Python-side
 * audio_engine.py room switch so visual and acoustic material are never out of sync.
 *
 * CONCEPT: §5.4 of the build plan states that visual and acoustic material must
 * change in a single coordinated call, never independently. The hud.js button
 * handler calls bridge.set_room() (Python) and this function in sequence; Python
 * fires state_changed() back to confirm, at which point this is called again with
 * the confirmed state. The result is: optimistic visual update on click, confirmed
 * on signal. If Python rejects (e.g. called while playing), hud.js reverts.
 *
 * NATIVE ENGINE EQUIVALENT:
 *   wall material change ↔ QAudioRoom.setWallMaterial(wall, QAudioRoom.Material.X)
 *
 * @param {"treated"|"wet"} variant
 */
function setRoomVariant(variant) {
  // Visual material palette per room variant.
  // "treated" — acoustic panels, heavy curtains, carpet: muted warm tones.
  // "wet"     — bare brick, plaster, marble, glass: cooler, harder surfaces.
  const palettes = {
    // "treated" — acoustic panels, heavy curtains, carpet: warm muted tones.
    treated: {
      left:    { color: 0x6b5a4e, roughness: 0.95 }, // heavy curtain — warm dark brown
      right:   { color: 0x6b5a4e, roughness: 0.95 },
      front:   { color: 0x4a4540, roughness: 0.9  }, // fiberglass panel — dark charcoal
      back:    { color: 0x7a7060, roughness: 0.9  }, // treated back wall — warm taupe
      ceiling: { color: 0xb0aca4, roughness: 0.85 }, // acoustic tile — light grey
      floor:   { color: 0x5a4f3e, roughness: 0.95 }, // carpet — warm grey-brown
    },
    // "wet" — bare brick, plaster, marble, glass: cooler, harder, more reflective.
    wet: {
      left:    { color: 0x7a5a4a, roughness: 0.75 }, // bare brick — terracotta-brown
      right:   { color: 0x7a5a4a, roughness: 0.75 },
      front:   { color: 0x6a7080, roughness: 0.5  }, // smooth plaster — cool blue-grey
      back:    { color: 0x6a7080, roughness: 0.5  },
      ceiling: { color: 0xb8bec8, roughness: 0.3  }, // polished concrete — light cool grey
      floor:   { color: 0x909098, roughness: 0.1  }, // marble — high specular, mid-grey
    },
  };

  const palette = palettes[variant];
  if (!palette) {
    console.warn(`[scene.js] Unknown room variant: "${variant}"`);
    return;
  }

  for (const [face, props] of Object.entries(palette)) {
    const mesh = roomMeshes[face];
    if (!mesh) continue;
    mesh.material.color.setHex(props.color);
    mesh.material.roughness = props.roughness;
    // metalness stays 0 for all variants — none of these materials are metallic.
    mesh.material.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// STEP 11: Resize handler
// ---------------------------------------------------------------------------

function _onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// ---------------------------------------------------------------------------
// STEP 12: Error helper
// ---------------------------------------------------------------------------

/**
 * _fatalAssetError — routes an asset-load failure to the HUD's error display
 * via window.htSignals.onAssetError(), which hud.js registers.
 * Never logs only to the browser console — the error must be visible in the UI.
 * See §1 and §6 (asset_error signal) of the build plan.
 *
 * @param {string} message — human-readable, names the missing file and expected path
 */
function _fatalAssetError(message) {
  console.error("[scene.js] Asset error:", message);
  if (window.htSignals?.onAssetError) {
    window.htSignals.onAssetError(message);
  }
}
