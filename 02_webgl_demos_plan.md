# Build Plan: Chapter 02 — WebGL 2 Demos (Multimediaprogrammierung)

> **Audience of this file: Claude, in a fresh session with no prior context.**
> Read the whole file before writing any code. It is a complete specification. Where it is precise (ids, filenames, shader interfaces, header/footer markup), follow it exactly. Where it leaves freedom, choose the simplest solution that satisfies the acceptance criteria.

---

## 0. How this file is used

**The user will upload with this file:**
1. `01_webaudio_oscillator.html`: the HTML **template** from Chapter 01. It is the reference for look and structure (header, footer, info boxes, pills, demo card).
2. `01_multimediaprogrammierung_webaudio.ipynb`: the Chapter 01 notebook (reference only).
3. `02_multimediaprogrammierung_webgl.ipynb`: the Chapter 02 **theory notebook**. The demos must match its terminology, naming conventions and code snippets.

If any of these is missing, everything needed is embedded in Section 3 (page template). Say which file is missing and continue.

**Task:** create **seven standalone HTML demo files** (Section 5), one at a time. After each file, stop, present it, and wait for the user's feedback before building the next one, unless the user says "build all".

**Do not** create exercises, tasks or "try it yourself" sections. The user explicitly does not want them.

---

## 1. Project context

- **Course:** *Multimediaprogrammierung*, bachelor's level, Multimedia Engineering.
- **Institute:** Institut für Medientechnik – Technische Universität Ilmenau (AMS Group).
- **Format:** 90-minute seminars. Chapter 01 was the Web Audio API. **Chapter 02 is WebGL 2.**
- **Instructor:** Renato Fera P. (credits appear in header and footer).
- **Students:** know basic JavaScript and a little linear algebra. They see WebGL for the first time.
- **Pedagogy:** The notebook teaches the theory. The HTML demos are **interactive, tutorial-style code** that students read next to the running page. The demos build on each other and are worked through in order.
- **Repository:** https://github.com/TUIlmenauAMS/multimedia_programming_seminars (files live in the repo root, next to the notebooks).
- **Language:** all page text and code comments in **English**. Credits keep their German titles ("Institut für Medientechnik", "Fachgebietsleiter", "Wissenschaftlicher Mitarbeiter").

---

## 2. Global constraints and conventions

### 2.1 Technology
- **WebGL 2 only** (`canvas.getContext('webgl2')`), shaders in **GLSL ES 3.00** (`#version 300 es` as the very first characters of the source string).
- **Raw WebGL. No libraries**, no CDN scripts, no external fonts, no external images. Each file is **fully self-contained** (one `.html` file with inline `<style>` and `<script>`), and works when opened via `file://`.
- The only external resources allowed are the AMS logo `<img>` from the template (raw.githubusercontent URL) and outbound `<a>` links.
- No `localStorage`/`sessionStorage`; no build step; no frameworks.
- Target: current Chrome, Edge, Firefox, Safari.

### 2.2 Professional code standards (JavaScript)
- A single `<script type="module">` at the end of `<body>` (strict mode by default). No globals.
- `const` by default, `let` only when reassigned. No `var`.
- Small, single-purpose functions with **JSDoc** (`@param`, `@returns`).
- Constants for magic numbers (`const FOV_DEFAULT = 45;`).
- Throw/handle errors: always check `getContext`, shader compile status and program link status, and print the info log.
- Show a friendly in-page error banner (`.error-banner`) if WebGL 2 is unavailable or a shader fails; never leave a silent black canvas.
- Handle `devicePixelRatio` and resizing (`resizeCanvasToDisplaySize`). Redraw on `resize` (via `ResizeObserver` on the canvas or by checking each frame).
- Use `requestAnimationFrame` with the timestamp argument. Never use `setInterval` for rendering.
- Use `performance`-independent animation: derive angles from accumulated time, not frame counts.
- Cache all `getUniformLocation` / `getAttribLocation` results at init. Never look them up per frame.
- Initialization happens once; the render function only sets uniforms, binds the VAO and draws.
- Event listeners via `addEventListener`; DOM lookups via `document.querySelector`/`getElementById`, collected once in a small `ui` object.

### 2.3 Professional code standards (CSS)
- Reuse the template CSS (Section 3.2) and **extend**, don't rewrite. Use CSS custom properties for the palette (the template uses hard-coded values; define `:root` variables with the same values).
- Mobile-friendly: flex/grid, `max-width: 100%`, no horizontal page scroll.
- Class names are lowercase-kebab (`control-row`, `gl-canvas`).

### 2.4 Tutorial-style comments (most important requirement)
The source is meant to be **read by students**. Therefore:
- Open the script with a **block comment** that states: demo title, what it teaches, the matching notebook section(s), and the list of steps in the file.
- Divide the script into numbered sections with banner comments, for example:
  ```js
  // ══════════════════════════════════════════════════════════════
  // STEP 3 — Create the vertex buffer
  // ══════════════════════════════════════════════════════════════
  ```
- Before every non-trivial block, a short comment explains **what** it does and **why** (not just a restatement of the code line).
- Mention the WebGL **state machine** where relevant ("`bindBuffer` sets the current ARRAY_BUFFER; later calls act on it").
- Comment each GLSL line group (shaders are JS template strings, so GLSL `//` comments work).
- Where a helper repeats from an earlier demo, comment `// Same helper as in demo 01_webgl_triangle.html`. Where something is **new** in this demo, mark it `// NEW in this demo:`.
- Do not over-comment trivial lines (`i++`). Aim for clarity, not noise.

### 2.5 Page naming and IDs
- Files (exactly these names):
  1. `02_webgl_triangle.html`
  2. `02_webgl_uniforms.html`
  3. `02_webgl_transform.html`
  4. `02_webgl_cube.html`
  5. `02_webgl_texture.html`
  6. `02_webgl_video.html`
  7. `02_webgl_lighting.html`
- Canvas id is always `glcanvas`. Controls use the ids listed in each demo spec.

### 2.6 Naming in shaders
`a_` attributes, `u_` uniforms, `v_` varyings (matches the notebook). Fragment output is always `outColor`.

---

## 3. Page template

Every demo page has this structure, taken from the Chapter 01 template:

1. **Header** (logo, title, institute, credits). Exact markup in 3.1.
2. **Section 1 — concept** (`h2.section`): 1–3 short paragraphs and an `.info-box` with the pipeline/data flow of this demo.
3. `<hr />`
4. **Section 2 — Interactive Demo** (`h2.section`): one sentence, then `.demo-card` containing the `<canvas id="glcanvas" class="gl-canvas">`, the controls and a small `.stats` read-out.
5. `<hr />`
6. **Section 3 — How It Works**: explains the code structure, referencing the step names in the source. Includes an `.info-box` with a note, and an optional collapsible `<details class="shader-source">` that shows the **live shader sources** (filled by JS from the same strings that are compiled, so they never drift).
7. **Pills** with links (MDN etc.).
8. **Footer**.
9. `<script type="module">`.

Keep prose short. The code comments carry the teaching.

### 3.1 Header and footer markup (use exactly, change only the demo name)

```html
<header class="nb-header">
  <img
    src="https://raw.githubusercontent.com/TUIlmenauAMS/multimedia_programming_seminars/main/ams_group_tui.png"
    alt="AMS Group – TU Ilmenau"
  />
  <div>
    <h1>Multimediaprogrammierung</h1>
    <h2>Chapter 02: WebGL 2 &mdash; Demo: DEMO_NAME</h2>
    <p><strong>Institut für Medientechnik &ndash; Technische Universität Ilmenau</strong><br>
      <a href="https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-elektrotechnik-und-informationstechnik/profil/institute-und-fachgebiete/fachgebiet-angewandte-mediensysteme" target="_blank">
        Institut für Medientechnik &ndash; AMS Group
      </a>
    </p>
    <p>
      <strong>Fachgebietsleiter:</strong> Prof. Dr.-Ing. Gerald Schuller<br>
      <strong>Wissenschaftlicher Mitarbeiter:</strong> Renato Fera P.
      <a href="https://www.linkedin.com/in/renato-profeta/" target="_blank">LinkedIn</a>
    </p>
  </div>
</header>
```

```html
<footer class="nb-footer">
  <p><strong>Multimediaprogrammierung</strong> &middot; Chapter 02: WebGL 2 &middot; Demo: DEMO_NAME<br>
    Institut für Medientechnik &ndash; Technische Universität Ilmenau</p>
  <p>Demo created by <strong>Renato Fera P.</strong> &middot;
    <a href="https://www.linkedin.com/in/renato-profeta/" target="_blank">LinkedIn</a>
  </p>
</footer>
```

`DEMO_NAME` values: Triangle, Uniforms and Animation, 2D Transformations, Rotating Cube, Textures, Video Filters, Lighting.

### 3.2 Base CSS (copied from the template; keep, then extend)

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 16px; line-height: 1.6; color: #212121; background: #fff;
  max-width: 900px; margin: 0 auto; padding: 0 24px 48px;
}

/* Header */
.nb-header { display: flex; align-items: center; gap: 24px; border-bottom: 2px solid #e0e0e0; padding: 16px 0; margin-bottom: 32px; }
.nb-header img { width: 120px; border-radius: 12px; flex-shrink: 0; }
.nb-header h1 { font-size: 1.6rem; margin-bottom: 4px; }
.nb-header h2 { font-size: 1.1rem; font-weight: 400; margin-bottom: 8px; color: #444; }
.nb-header p  { font-size: 0.92rem; margin-bottom: 2px; }
.nb-header a  { color: #1a73e8; text-decoration: none; }
.nb-header a:hover { text-decoration: underline; }

/* Pills */
.pills { margin: 12px 0; display: flex; flex-wrap: wrap; gap: 8px; }
.pill { display: inline-block; padding: 6px 14px; border: 1px solid #1a73e8; border-radius: 20px;
  text-decoration: none; font-weight: 600; font-size: 0.92rem; color: #1a73e8; transition: background 0.15s, color 0.15s; }
.pill:hover { background: #1a73e8; color: #fff; }

/* Sections */
h2.section { font-size: 1.3rem; margin: 40px 0 12px; }
hr { border: none; border-top: 1px solid #e0e0e0; margin: 36px 0; }
p  { margin-bottom: 12px; }

/* Demo card */
.demo-card { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 12px; padding: 28px 32px; margin: 24px 0; }
.demo-card h3 { font-size: 1rem; font-weight: 600; margin-bottom: 20px; color: #444; text-transform: uppercase; letter-spacing: 0.04em; }

/* Controls */
.control-row { display: flex; align-items: center; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }
.control-row label { min-width: 120px; font-weight: 500; font-size: 0.95rem; }
.control-row input[type=range] { flex: 1; min-width: 160px; accent-color: #1a73e8; }
.control-row select { padding: 6px 10px; border: 1px solid #ccc; border-radius: 8px; font-size: 0.95rem; background: #fff; }
.value-badge { min-width: 64px; text-align: right; font-variant-numeric: tabular-nums; font-size: 0.92rem; color: #555; }

/* Buttons */
.btn-row { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.btn { padding: 10px 28px; border: none; border-radius: 24px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity 0.15s, transform 0.1s; }
.btn:active { transform: scale(0.97); }
.btn-start { background: #1a73e8; color: #fff; }
.btn-stop  { background: #e53935; color: #fff; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }

/* Canvas (template value; WebGL demos override height via .gl-canvas) */
canvas { display: block; width: 100%; height: 120px; border-radius: 8px; background: #1e1e2e; margin-top: 20px; }

/* Footer */
.nb-footer { text-align: center; padding: 18px 0 8px; margin-top: 48px; border-top: 2px solid #e0e0e0; font-size: 0.92rem; color: #444; line-height: 1.7; }
.nb-footer a { color: #1a73e8; text-decoration: none; }
.nb-footer a:hover { text-decoration: underline; }

/* Info box */
.info-box { border-left: 5px solid #1a73e8; background: rgba(26,115,232,0.07); border-radius: 6px; padding: 12px 18px; margin: 20px 0; font-size: 0.95rem; }

code { background: #f1f3f4; border-radius: 4px; padding: 1px 6px; font-size: 0.9em; font-family: "SFMono-Regular", Consolas, monospace; }
```

### 3.3 CSS additions needed for WebGL demos

```css
:root { --accent: #1a73e8; --danger: #e53935; --border: #e0e0e0; --canvas-bg: #1e1e2e; }

/* Overrides the template's 120px canvas. Aspect ratio instead of fixed height. */
canvas.gl-canvas { height: auto; aspect-ratio: 16 / 10; margin-top: 0; touch-action: none; }
canvas.gl-canvas.square { aspect-ratio: 1 / 1; max-width: 520px; margin-left: auto; margin-right: auto; }

.canvas-wrap { margin-bottom: 20px; }
.stats { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.82rem; color: #555; margin-top: 10px; }
.control-row input[type=checkbox] { accent-color: var(--accent); width: 18px; height: 18px; }
.control-row input[type=color] { width: 44px; height: 32px; border: 1px solid #ccc; border-radius: 8px; padding: 2px; background: #fff; }
.error-banner { background: #fdecea; border-left: 5px solid var(--danger); border-radius: 6px; padding: 12px 18px; margin: 20px 0; color: #7f1d1d; }
details.shader-source { margin: 16px 0; border: 1px solid var(--border); border-radius: 8px; padding: 8px 14px; background: #fafafa; }
details.shader-source summary { cursor: pointer; font-weight: 600; }
details.shader-source pre { overflow-x: auto; font-size: 0.82rem; line-height: 1.5; margin-top: 10px; font-family: "SFMono-Regular", Consolas, monospace; }
.matrix { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85rem; border-collapse: collapse; }
.matrix td { padding: 2px 10px; text-align: right; border: none; }
@media (max-width: 600px) { .nb-header { flex-direction: column; align-items: flex-start; } .demo-card { padding: 20px 16px; } }
```

### 3.4 Script skeleton (every demo follows this order)

```js
/**
 * ═══════════════════════════════════════════════════════════════
 * Chapter 02 — WebGL 2 · Demo: <name>
 * Teaches: <concepts>.  Notebook: <section numbers>.
 * Steps in this file:  1) … 2) … 3) …
 * ═══════════════════════════════════════════════════════════════
 */

// STEP 0 — Collect DOM references once
// STEP 1 — Get the WebGL 2 context (with friendly error banner)
// STEP 2 — Shader sources (GLSL ES 3.00)
// STEP 3 — Helper functions: createShader, createProgram, resizeCanvasToDisplaySize (+ math helpers if needed)
// STEP 4 — Compile + link, look up attribute/uniform locations (cached)
// STEP 5 — Create buffers and record VAO(s)
// STEP 6 — Wire UI events to state
// STEP 7 — Render function (per frame): resize → viewport → clear → set uniforms → draw
// STEP 8 — Start the loop
```

### 3.5 Shared helpers (copy identically into each file; keep these exact semantics)

```js
/** Compile one shader stage; throw with the driver's info log on failure. */
function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return shader;
}

/** Link a vertex + fragment shader into a program; throw on failure. */
function createProgram(gl, vertexSource, fragmentSource) {
  const vs = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error:\n${log}`);
  }
  return program;
}

/** Match the drawing buffer to the CSS display size × devicePixelRatio. Returns true if it changed. */
function resizeCanvasToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width  = Math.round(canvas.clientWidth  * dpr);
  const height = Math.round(canvas.clientHeight * dpr);
  const changed = canvas.width !== width || canvas.height !== height;
  if (changed) { canvas.width = width; canvas.height = height; }
  return changed;
}
```

**Matrix helpers** (column-major, `Float32Array`; hand-written, no gl-matrix; mention gl-matrix in a comment). Use these exact conventions so all demos agree:

```js
// ---- mat3 (2D transforms, demo 03). All return column-major Float32Array(9). ----
const mat3 = {
  identity:    () => new Float32Array([1,0,0, 0,1,0, 0,0,1]),
  translation: (tx, ty) => new Float32Array([1,0,0, 0,1,0, tx,ty,1]),
  rotation:    (a) => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c,s,0, -s,c,0, 0,0,1]); },
  scaling:     (sx, sy) => new Float32Array([sx,0,0, 0,sy,0, 0,0,1]),
  /** Returns a × b (apply b first, then a). */
  multiply: (a, b) => {
    const out = new Float32Array(9);
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) {
      let s = 0; for (let k = 0; k < 3; k++) s += a[k*3 + r] * b[c*3 + k];
      out[c*3 + r] = s;
    }
    return out;
  },
};

// ---- mat4 (3D, demos 04–07). Column-major Float32Array(16). ----
const mat4 = {
  identity:    () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  translation: (x, y, z) => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]),
  rotationX:   (a) => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); },
  rotationY:   (a) => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); },
  scaling:     (x, y, z) => new Float32Array([x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]),
  perspective: (fovY, aspect, near, far) => {
    const f = 1 / Math.tan(fovY / 2), nf = 1 / (near - far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
  },
  /** Returns a × b (apply b first, then a). */
  multiply: (a, b) => {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k*4 + r] * b[c*4 + k];
      out[c*4 + r] = s;
    }
    return out;
  },
};
```

Upload with `gl.uniformMatrix3fv(loc, false, m)` / `gl.uniformMatrix4fv(loc, false, m)` (transpose is always `false` in WebGL). In the view matrix the camera sits at distance `d` on +z looking at the origin: `view = mat4.translation(0, 0, -d)`.

---

## 4. Teaching progression

| # | Demo | New concept (everything else is repeated and marked "same as before") | Notebook § |
|---|------|------------------------------------------------------------------------|-----------|
| 1 | Triangle | context, shaders, buffer, attributes, VAO, `drawArrays`, varyings | 3, 4, 5, 6 |
| 2 | Uniforms | uniforms, animation loop, quad with `TRIANGLE_STRIP` | 7 |
| 3 | Transform | mat3, composition order, column-major | 8 |
| 4 | Cube | mat4 MVP, perspective, depth test, culling, index buffer | 9 |
| 5 | Texture | textures, UVs, filtering, mipmaps, wrap modes, sampler | 10 |
| 6 | Video | video as a texture, per-frame upload, image-processing fragment shaders | 10 (video) |
| 7 | Lighting | normals, Lambert, normal matrix | 11 |

Each demo's code should be as short as possible so the **new** concept stands out. Reuse the code of the earlier demo and mark additions with `// NEW in this demo:`.

---

## 5. Demo specifications

Conventions for the specs: "Controls" lists element ids. Every `<input type=range>` has a `.value-badge` span whose id is `<controlId>Val`. Every demo also has a `.stats` line (id `stats`) as described.

### 5.1 `02_webgl_triangle.html` — Triangle

**Goal:** First complete WebGL 2 program, from context to pixels. This is the most heavily commented file.

**Geometry:** one triangle, **interleaved** buffer, 5 floats per vertex (`x, y, r, g, b`), stride = `5 * Float32Array.BYTES_PER_ELEMENT`. Vertices: top (0.0, 0.6) red; bottom-left (−0.6, −0.6) green; bottom-right (0.6, −0.6) blue.

**Shaders:**
```glsl
// vertex
#version 300 es
in vec2 a_position;
in vec3 a_color;
out vec3 v_color;
void main() {
  v_color = a_color;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
// fragment
#version 300 es
precision highp float;
in vec3 v_color;
out vec4 outColor;
void main() { outColor = vec4(v_color, 1.0); }
```

**Controls:**
- `wireframe` (checkbox): additionally draws `gl.LINE_LOOP` with the same VAO and a second constant-color pass is **not** needed. Draw the triangle, then draw the outline using a `u_useFlatColor`-free approach: simplest is a second tiny program or a `uniform bool u_outline` that outputs white. Use the `uniform bool u_outline` approach and explain it in a comment.
- `showCorners` (checkbox, default on): no extra GL work; toggles an overlay `<span>` legend ("red / green / blue vertex") under the canvas.

**Stats line:** `WebGL: <gl.getParameter(gl.VERSION)> · Drawing buffer: W×H px · CSS size: w×h px · devicePixelRatio: n`. Updated on resize. This teaches the two canvas sizes (notebook §3).

**Rendering:** render once at start and **again on resize** (no animation loop; a static scene). Explain in a comment why no loop is needed.

**Info box (Section 1):** the pipeline chain `Vertex data → Vertex shader → Rasterization → Fragment shader → Canvas`.

**Pills:** WebGL API – MDN, `drawArrays` – MDN, `createVertexArray` – MDN.

**Acceptance:** colors interpolate smoothly; canvas stays sharp on resize and on high-DPI; wireframe toggle works; error banner appears if the context is `null`.

---

### 5.2 `02_webgl_uniforms.html` — Uniforms and Animation

**Goal:** Drive the GPU from JavaScript with uniforms and a `requestAnimationFrame` loop.

**Geometry:** a quad from 4 vertices drawn with `gl.TRIANGLE_STRIP` (explain the strip order in a comment). Attributes: `a_position` (vec2) and `a_uv` (vec2, passed as varying `v_uv` for a radial effect).

**Uniforms:** `u_time` (float, seconds), `u_color` (vec3), `u_pulse` (float amplitude), `u_aspect` (float, canvas width / height so the shape is not stretched).

**Shaders:** the vertex shader scales the quad by `1.0 + u_pulse * sin(u_time * 2.0)` and corrects `x` by `1/u_aspect`. The fragment shader renders a soft radial glow: `float d = length(v_uv - 0.5) * 2.0; float glow = smoothstep(1.0, 0.0, d); outColor = vec4(u_color * glow * (0.7 + 0.3 * sin(u_time * 3.0)), 1.0);`.

**Controls:** `speed` (range 0–3, step 0.05, default 1), `pulse` (range 0–0.5, step 0.01, default 0.15), `color` (`input type=color`, default `#1a73e8`, convert hex → 0–1 floats), `paused` (checkbox).

**Animation state:** accumulate `time += dt * speed` where `dt` is derived from the rAF timestamp difference. **Pausing and changing speed must not make time jump** (this is the reason to accumulate rather than multiply the raw timestamp, so explain it in a comment).

**Stats line:** `time: 12.34 s · FPS: 60`. FPS via a smoothed average, updated about twice per second.

**Pills:** Uniforms – MDN, `requestAnimationFrame` – MDN.

**Acceptance:** sliders update immediately; pause freezes without jump; color picker changes the glow; resizing keeps the shape circular.

---

### 5.3 `02_webgl_transform.html` — 2D Transformations

**Goal:** Matrices as a composable transformation, and why order matters.

**Canvas:** `class="gl-canvas square"`.

**Geometry:** the classic asymmetric **letter "F"** built from 3 rectangles (6 triangles = 18 vertices, `drawArrays(gl.TRIANGLES)`), centered at the origin, spanning about −0.25…+0.25 so there is room to move. Use a clearly asymmetric shape so rotation and mirroring are visible. Also draw a **reference grid with axes** using `gl.LINES` (vertices for lines at x, y = −1, −0.5, 0, 0.5, 1; the x and y axes in a brighter color) with the identity matrix, in a second VAO and the same program (use a `u_color` vec3 uniform, so the F and the grid just differ in uniform color and matrix).

**Shader:** vertex shader multiplies `u_matrix * vec3(a_position, 1.0)`. Fragment outputs `u_color`.

**Controls:** `tx`, `ty` (range −0.8…0.8, step 0.01), `rot` (range −180…180 degrees, step 1), `sx`, `sy` (range 0.2…2.5, step 0.01, default 1), `order` (select), `animate` (checkbox: continuously rotates), `reset` (button).

`order` options (the matrix is built by multiplying in this order, written left → right as in the formula):
1. `T · R · S`  "scale → rotate → translate" (the standard)
2. `R · T · S`  "scale → translate → rotate"
3. `T · S · R`  "rotate → scale → translate"
4. `S · R · T`  "translate → rotate → scale"

**Matrix read-out:** a `<table class="matrix">` showing the 3×3 matrix **in mathematical (row) notation**, updated live. Remember the array is column-major, so print `m[col*3+row]`. Also show the formula text for the selected order (e.g. `M = T · R · S`).

**Notes in the page:** `.info-box` explaining "read right to left" and "why column-major".

**Pills:** 2D Matrices (webgl2fundamentals), `uniformMatrix` – MDN.

**Acceptance:** with `T·R·S` and tx=0.5, rotating spins the F in place; with `R·T·S` the F orbits the origin. The readout matches what is drawn. The F is **not distorted** (square canvas).

---

### 5.4 `02_webgl_cube.html` — Rotating Cube

**Goal:** First 3D scene. MVP matrices, perspective, depth test, face culling, index buffer.

**Geometry:** a unit cube centered at the origin, **24 vertices** (4 per face), per-face colors (6 distinct colors), **36 indices** (`Uint16Array`) in an `ELEMENT_ARRAY_BUFFER` recorded in the VAO. Attributes: `a_position` (vec3), `a_color` (vec3). Build faces from a small data table in code (face normal + 4 corners) rather than 24 hand-typed lines, if it stays readable.
**Winding:** all triangles **counter-clockwise when seen from outside** (the default front face). This must be verified programmatically (see Section 7).

**Shaders:** `gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0)`; pass `v_color` through.

**Controls:** `autoRotate` (checkbox, default on), `speed` (range 0–3), `fov` (range 20–100 degrees, default 45), `distance` (range 2–10, default 4), `depthTest` (checkbox, default on), `cullFace` (checkbox, default on), plus **mouse/touch drag** (pointer events with `setPointerCapture`) to rotate the cube manually; drag adds to `rotX`/`rotY` (clamp `rotX` to ±90°). A short hint under the canvas: "Drag to rotate".

**Render:** `clear(COLOR | DEPTH)`; apply the `depthTest` / `cullFace` toggles with `gl.enable/disable` **every frame** (cheap and keeps the code simple); model = `rotationY × rotationX`; `drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0)`. Recompute the projection on resize/fov change using the current aspect.

**Info box:** a warning that with the depth test **off** and culling **off** you will see faces painted in the wrong order (explanation of why).

**Stats line:** `rotX: 20° · rotY: 45° · fov: 45° · FPS: 60`.

**Pills:** Model View Projection – MDN, Perspective (webgl2fundamentals), `drawElements` – MDN.

**Acceptance:** cube looks correct with defaults; with depth test off the wrong faces visibly overlap; with culling on and depth test on the result is correct; with culling off it is still correct (the depth test handles it); drag rotation is smooth; fov and distance change the apparent perspective.

---

### 5.5 `02_webgl_texture.html` — Textures

**Goal:** UV coordinates, texture creation and upload, filtering, mipmaps, wrap modes, sampler uniform.

**Geometry:** the same cube as demo 4 but with `a_uv` (vec2) instead of colors: each face maps UV (0,0)…(1,1), 24 vertices, 36 indices. Same MVP setup with auto-rotate and drag.

**Texture (procedural; no file loading):** create it with a 2D canvas (`document.createElement('canvas')`), upload with `texImage2D(..., canvas)`, set `UNPACK_FLIP_Y_WEBGL` to `true` and explain why. Provide two sources (`textureSource` select):
- `checker` — **16 × 16 px** colored checkerboard (low resolution on purpose so magnification differences are visible), with a distinct 1-px marker in one corner so orientation is visible.
- `detail` — **256 × 256 px**: fine 1-px grid lines plus a large letter "F" (shows minification aliasing and what mipmaps fix).

Always call `gl.generateMipmap(gl.TEXTURE_2D)` after upload.

**Shaders:** vertex passes `v_uv = a_uv * u_uvScale`; fragment `outColor = texture(u_texture, v_uv)`. Sampler on texture unit 0 (`gl.activeTexture(gl.TEXTURE0)`, `gl.uniform1i(u_texture, 0)`; explain units in a comment).

**Controls:**
- `textureSource` (select: checker / detail)
- `minFilter` (select: `NEAREST`, `LINEAR`, `NEAREST_MIPMAP_NEAREST`, `LINEAR_MIPMAP_LINEAR`; default `LINEAR_MIPMAP_LINEAR`)
- `magFilter` (select: `NEAREST`, `LINEAR`; default `LINEAR`)
- `wrap` (select: `REPEAT`, `MIRRORED_REPEAT`, `CLAMP_TO_EDGE`; applies to S and T)
- `uvScale` (range 0.5–8, step 0.1, default 1): values > 1 repeat the texture so wrapping is visible and the texture is minified.
- `distance` (range 2–12, default 4): further away = more minification.
- `autoRotate` (checkbox).

Changing a select calls `gl.texParameteri` immediately (state changes on the bound texture). Show the current **parameter values as code** in the stats line, e.g. `TEXTURE_MIN_FILTER = LINEAR_MIPMAP_LINEAR`.

**Info box:** explain that in WebGL 1 mipmaps and `REPEAT` required power-of-two sizes and in WebGL 2 they do not. A 16×16 texture is a power of two anyway; mention it, and optionally make the `detail` texture 200×200 to prove the point (if so, label it).

**Pills:** Using textures – MDN, `texImage2D` – MDN, Textures (webgl2fundamentals).

**Acceptance:** NEAREST + magnified 16×16 texture looks blocky, LINEAR looks smooth; with `detail`, uvScale 8 and large distance `NEAREST` shows shimmering while `LINEAR_MIPMAP_LINEAR` is calm; wrap modes visibly differ when `uvScale` is > 1 (and with UVs outside 0…1, e.g. add a `uvOffset` nudge of 0.25 if CLAMP_TO_EDGE would otherwise show no difference).

---

### 5.6 `02_webgl_video.html` — Video Filters (NEW demo)

**Goal:** Use **video as a texture** and process it **in real time with fragment shaders**: the classic multimedia use case and the clearest demonstration of why GPUs are good at image and video processing. Texture upload per frame is new; everything else is reused from demo 5.

**Geometry:** one full-quad (4 vertices, `TRIANGLE_STRIP`, positions + UVs) filling a letterboxed region of the canvas so that the **video aspect ratio is preserved** (`u_scale` vec2 computed from `videoWidth/videoHeight` vs canvas aspect; area outside the video stays the canvas background color).

**Sources** (`source` select). The page must work with **zero permissions** by default:
1. `pattern` — **default.** A procedurally animated 512×512 2D canvas drawn each frame (e.g. moving colored circles on a hue-shifting gradient, plus a few high-contrast edges so filters like Sobel and threshold look good). No permission, no file, works from `file://`.
2. `file` — an `<input type="file" accept="video/*">` (shown only when this source is selected). Create the URL with `URL.createObjectURL(file)`; set `video.loop = true`, `video.muted = true`, `video.playsInline = true`, then `play()`. Revoke the previous object URL when a new file is chosen.
3. `webcam` — `navigator.mediaDevices.getUserMedia({ video: true })` → `video.srcObject`. **Handle failure gracefully** (permission denied, no camera, insecure context, e.g. when the page is shown through htmlpreview): show a message in an `.error-banner` and fall back to `pattern`. Stop the stream's tracks when switching away.

All three sources are handled by one function `getFrameSource()` that returns either the `<video>` or the pattern `<canvas>`, so the render loop is source-agnostic.

**Per-frame upload:** each frame, if the source is ready (for video: `video.readyState >= video.HAVE_CURRENT_DATA`), call
`gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source)`. Parameters (set once at init): `LINEAR` min/mag, `CLAMP_TO_EDGE` wrap, **no mipmaps** (explain in a comment: mipmaps would have to be regenerated each frame and video is displayed at roughly 1:1). `UNPACK_FLIP_Y_WEBGL = true`. Mention `requestVideoFrameCallback` and `texSubImage2D` as optimizations in comments only.

**Fragment shader — filters via `uniform int u_filter`** (0…6), `uniform float u_amount` (0…1, blend between original and filtered), `uniform bool u_split` + `uniform float u_splitPos` (left of the split line: original; right: filtered). Use `vec2 texel = 1.0 / vec2(textureSize(u_texture, 0));` (GLSL ES 3.00) for neighbor sampling.

| `u_filter` | Name | Implementation hint |
|-----------:|------|---------------------|
| 0 | None | passthrough |
| 1 | Grayscale | luminance `dot(c, vec3(0.2126, 0.7152, 0.0722))` (explain Rec. 709 weights in a comment) |
| 2 | Sepia | 3×3 color matrix |
| 3 | Invert | `1.0 - c` |
| 4 | Threshold | luminance > 0.5 → white else black |
| 5 | Edge detection (Sobel) | 3×3 neighborhood of luminance, `Gx`/`Gy` kernels, magnitude `length(vec2(gx, gy))` |
| 6 | Blur (3×3 box) | average of 9 samples |

Implement each filter as a small GLSL function (`vec3 grayscale(vec3 c)`, `float luma(vec3 c)`, …) and select in `main()` with `if/else if` on the uniform (allowed in GLSL ES 3.00; mention the GPU cost model briefly in a comment).

**Controls:** `source` (select), `videoFile` (file input, conditionally visible), `filter` (select, 7 options above), `amount` (range 0–1, step 0.01, default 1), `split` (checkbox, default **on**), `splitPos` (range 0–1, step 0.01, default 0.5; hidden/disabled when `split` is off), `playPause` (button for the video sources; hidden for `pattern`).

**Visual aid:** draw a thin vertical divider line at the split position (in the fragment shader: where `abs(v_uv.x - u_splitPos) < thickness` output white, evaluated in canvas space, not video space; the letterbox must not shift it incorrectly, so compute the split in canvas-relative x with a `v_canvasUv` varying).

**Stats line:** `source: pattern · frame: 512×512 · filter: Sobel · FPS: 60`.

**Info box (Section 1):** "A video is just a texture whose pixels change every frame. The fragment shader runs once per pixel **in parallel**: a 1080p frame has about 2 million pixels, and the GPU processes them all at the same time." Also note: image-processing filters are **fragment-shader-only** work; the vertex shader just draws a quad.

**Pills:** `texImage2D` – MDN, HTMLVideoElement – MDN, `getUserMedia` – MDN, `textureSize` (GLSL ES 3.00 spec link as in notebook).

**Acceptance:** default state works with no interaction beyond opening the page (animated pattern with split view original | filtered). All 7 filters visibly differ. Switching to a local video file works, the aspect ratio is preserved, and the file is not stretched. Webcam denial produces the error banner and falls back to the pattern, not a broken page. No console errors. Switching sources repeatedly does not leak streams or object URLs.

**Out of scope for this demo** (do not add): audio analysis, recording, multi-pass filtering, framebuffers.

---

### 5.7 `02_webgl_lighting.html` — Lighting

**Goal:** Normals and diffuse (Lambert) lighting computed in the shader. Closes the chapter and is the **optional** demo of the seminar.

**Geometry:** two models selectable with `model`:
- `cube`: 24 vertices, **flat** normals (4 identical normals per face), 36 indices.
- `sphere`: UV sphere (e.g. 32 segments × 24 rings), **smooth** normals (`normal = normalize(position)`), indices as `Uint16Array`. Generate procedurally with a short, commented function.
Two VAOs (one per model) sharing one program. Attributes: `a_position` (vec3), `a_normal` (vec3).

**Shaders:**
```glsl
// vertex
#version 300 es
in vec3 a_position;
in vec3 a_normal;
uniform mat4 u_model, u_view, u_projection;
out vec3 v_normal;
void main() {
  // Normal matrix = inverse transpose of the model matrix's upper-left 3×3
  // (identical to the model rotation unless non-uniform scale is used).
  v_normal = transpose(inverse(mat3(u_model))) * a_normal;
  gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
// fragment
#version 300 es
precision highp float;
in vec3 v_normal;
uniform vec3 u_lightDirection;   // direction pointing TOWARD the light (world space)
uniform vec3 u_baseColor;
uniform float u_ambient;
uniform bool u_showNormals;
out vec4 outColor;
void main() {
  vec3 N = normalize(v_normal);              // interpolation can shorten vectors
  vec3 L = normalize(u_lightDirection);
  float diffuse = max(dot(N, L), 0.0);
  vec3 color = u_showNormals ? N * 0.5 + 0.5
                             : u_baseColor * (u_ambient + diffuse);
  outColor = vec4(color, 1.0);
}
```
The light direction is in **world space**; the model rotates, the light does not (state this in a comment and the info box).

**Light direction from angles:** `azimuth` (0–360°) and `elevation` (−90…90°) → `L = (cos(el)·sin(az), sin(el), cos(el)·cos(az))`.

**Controls:** `model` (select: cube / sphere), `azimuth`, `elevation`, `ambient` (range 0–1, step 0.01, default 0.15), `color` (color input, default `#e8a21a`), `showNormals` (checkbox, normals visualized as RGB), `autoRotate` (checkbox, rotates the **model**, default on), plus drag-to-rotate (same implementation as demo 4). Depth test + culling on.

**Optional small light marker:** a tiny line from the origin toward the light direction is **not** required; skip it unless it is trivial.

**Info box:** the formula `diffuse = max(dot(N, L), 0)` with a one-line explanation. A note on cube (flat) vs sphere (smooth) normals and why the cube looks faceted. Mention in one sentence that specular/Blinn-Phong and PBR build on this (do not implement them).

**Pills:** Directional lighting (webgl2fundamentals), Lighting – MDN.

**Acceptance:** the lit side follows the sliders; the sphere shades smoothly and the cube shades per face; ambient prevents pure black; "show normals" shows a smooth RGB gradient on the sphere and 6 flat colors on the cube; the light stays fixed when the model rotates.

---

## 6. Notebook synchronisation (pending change)

The theory notebook `02_multimediaprogrammierung_webgl.ipynb` was written when there were **six** demos. The video demo adds a **seventh**. When asked (or when the user uploads the notebook and says "sync it"), apply these changes with a small Python script editing the notebook JSON (preserve the template style: markdown cells only, `---` separators, pill links, the same header and footer):

1. **Overview table (cell 1):** change row 12's description to "Triangle, uniforms, transform, cube, texture, video and lighting demos".
2. **Section 10 (Textures):** add a subsection **"Video as a texture"** after "Sampling in the shader":
   - `texImage2D` accepts an `HTMLVideoElement`; a video is a texture whose pixels change every frame, so upload once per frame inside the render loop.
   - Short snippet: check `video.readyState >= video.HAVE_CURRENT_DATA`, then `gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video)`.
   - Short note on **real-time filters in the fragment shader** (grayscale, Sobel…) and a tiny GLSL grayscale example.
   - A note on sources: `<video>` from a file (`URL.createObjectURL`), webcam (`getUserMedia`).
3. **Section 12 (Interactive Demos):** insert **Demo 6 — Video Filters** (link `02_webgl_video.html`, pill label "Open Video Demo") before the lighting demo, and renumber **Lighting to Demo 7 (optional)**. The link base is `https://htmlpreview.github.io/?url=https://raw.githubusercontent.com/TUIlmenauAMS/multimedia_programming_seminars/main/<file>`.
4. **Section 13 (Further Resources):** add a pill/row for MDN "Using video as a texture" / `getUserMedia` only if helpful.

Keep all other cells byte-identical. Cell ids must stay unique.

---

## 7. Quality gates (run before presenting each file)

The container may have no GPU and no network. Verify what can be verified; state clearly what could **not** be tested.

1. **Syntax:** extract the `<script type="module">` body and run `node --check` on it (save as `.mjs`).
2. **Pure-logic unit tests in Node** (copy the helper functions into a temp script):
   - `mat3`/`mat4` helpers: `multiply(identity, M) == M`; `rotationY(π/2)` maps `(1,0,0,1)` to `(0,0,-1,1)`; translation then rotation order differs from rotation then translation; `perspective` maps a point on the near plane center to `z = -1` after the perspective divide.
   - **Cube/sphere geometry:** for every triangle, compute the geometric normal from the winding `(b−a)×(c−a)` and assert it points **outward** (dot product with the vector from the origin to the triangle centroid > 0), and assert all indices are `< vertexCount`. This is the guard against inside-out faces.
   - Sphere: `|normal| ≈ 1` for all vertices.
3. **GLSL review:** re-read each shader for: `#version 300 es` first with no leading whitespace; `precision highp float;` in the fragment shader; every `in`/`out` pair matches by name and type between stages; every uniform used is declared and has a cached location; no GLSL ES 1.00 leftovers (`attribute`, `varying`, `gl_FragColor`, `texture2D`).
4. **No external dependencies:** `grep` for `<script src`, `<link rel="stylesheet"`, `@import`, `fetch(`, `http` inside the script (only the logo `<img>` and `<a href>` may reference URLs).
5. **Template fidelity:** header and footer text identical to 3.1 apart from the demo name; `<title>` is `Web Audio…` style: `WebGL 2 – <Demo name> Demo`.
6. **Optional headless render:** if Chromium/Playwright is available offline, try a screenshot with software GL (`--use-gl=angle --use-angle=swiftshader` or `--enable-unsafe-swiftshader`). If not available, skip and note it. Never claim a visual check you did not run.
7. **Checklist per file:** all control ids from the spec exist and are wired; no console errors expected on load; error banner path exists; resize handling present; tutorial comments present in the style of 2.4; no exercises/tasks anywhere.

---

## 8. Delivery workflow

1. Read this file fully. Confirm the list of files received (Section 0).
2. Build **one demo at a time** in the order of Section 4. Write the file to `/mnt/user-data/outputs/<filename>` (plain `create_file`; these are downloadable course files).
3. Run the quality gates (Section 7). Fix issues.
4. Present the file with `present_files`. Reply with **at most**: one sentence about what the demo shows, a short list of anything that could not be verified (e.g. no GPU in the sandbox), and a question whether to continue with the next demo. No long recap.
5. When all seven exist, offer to apply the notebook synchronisation (Section 6).
6. Do not change the file names, ids or layout without asking.
