# FaceGazeSynth Implementation Plans

---

# Phase 4 Implementation Plan: Physics-Guided Gaze Redirection on Real Photos

## Goal
Take real headshot photos (center gaze) and generate photorealistic versions with gaze redirected to target angles (±5°, ±10°, ±15°, ±20°) using the same corneal refraction physics from Phases 1-3.

## Key Design Decisions

### 1. Approach: Physics-Guided Warping
Rather than using generative AI models (diffusion/GAN) which struggle with precise gaze angle control, we warp the real eye regions using displacement values computed from our validated physics engine. This gives us exact angle control while preserving photorealistic quality from the original photo.

### 2. Detection: MediaPipe FaceLandmarker
MediaPipe's 478-landmark face mesh (with `refine_landmarks=True` for iris) provides iris center, radius, eyelid contours, and eye corners. CPU-only, lightweight (3.7MB model). Iris landmarks 468-477 give 5 points per eye (center + 4 cardinal).

### 3. Physics Bridge
`refraction_corrected_displacement(angle_deg)` from Phase 1 gives iris displacement in mm. We calibrate mm-to-pixel scale from the detected iris size (accounting for ~18% corneal magnification), then convert physics displacement to pixel displacement.

### 4. Warping Strategy
- `cv2.remap()` with inverse mapping: for each destination pixel near the iris, compute the source coordinate by undoing the displacement and foreshortening
- Cosine foreshortening: iris width scales by `cos(angle)` at off-axis gaze
- Soft alpha blending (quadratic falloff) for seamless transition at iris boundary
- Eyelid clipping is natural — the photo's eyelids mask the iris at large angles

## Architecture: `facegazesynth/redirection/`

| File | Purpose |
|------|---------|
| `detection.py` | MediaPipe iris/eye detection → `EyeDetection`, `FaceDetection` dataclasses |
| `physics_mapping.py` | `calibrate_eye()` → mm/px scale; `target_displacement_px()` → pixel shift |
| `warping.py` | `warp_eye_region()` — remap iris with foreshortening + soft blend |
| `inpainting.py` | `inpaint_sclera()` — fill exposed sclera via `cv2.inpaint()` (Telea) |
| `specular.py` | `reposition_specular()` — detect and move Purkinje image |
| `compositing.py` | `redirect_single_eye()`, `redirect_both_eyes()` — orchestrates full pipeline |

## Pipeline: `facegazesynth/pipeline/redirect.py`

- `redirect_gaze(image_path, angle_deg)` → PIL Image
- `redirect_gaze_sweep(image_path, angles)` → labeled grid image
- `redirect_batch(input_dir, output_dir, angles)` → per-person directories + JSON manifest

## CLI: `scripts/redirect_gaze.py`

```bash
# Single angle
.venv/bin/python scripts/redirect_gaze.py --input samples/Dean_Center.png --angle 15

# Full sweep (9 angles with labels)
.venv/bin/python scripts/redirect_gaze.py --input samples/Dean_Center.png --sweep

# Batch all samples
.venv/bin/python scripts/redirect_gaze.py --batch --input-dir samples

# Debug overlay (detection landmarks)
.venv/bin/python scripts/redirect_gaze.py --input samples/Dean_Center.png --angle 10 --debug
```

## Validation

Round-trip test: redirect → re-detect iris → measure displacement → compare to `refraction_corrected_displacement()`.

**Result:** RMS 0.674mm (threshold 1.0mm, PASS). 14 new tests, 82 total passing.

## Dependencies Added
- `mediapipe>=0.10` — face landmark detection with iris
- `opencv-python>=4.8` — warping, inpainting, image processing

---

# Phase 2 Implementation Plan: FLAME Face Mesh Integration

## Goal
Embed the physics-based eyeball renderer into a parametric FLAME face mesh, producing full-face renders with physically accurate gaze.

## Key Design Decisions

### 1. Package Stack
- **`smplx`** for FLAME model (most maintained, pip-installable, supports shape + expression + pose)
- **`trimesh`** for ray-mesh intersection on the face surface (numpy-native, optional `pyembree` for speed)
- **`torch`** as smplx dependency (CPU-only sufficient)

### 2. Coordinate System Alignment
- FLAME outputs vertices in **meters**, Y-up, Z-forward — same forward axis as our eye model
- Our eye model works in **millimeters**, Z-forward
- Strategy: multiply FLAME vertices by 1000 to convert to mm, then position eyes at FLAME's regressed eye joint positions (joints 2 and 3)

### 3. Eye Replacement Strategy
- FLAME includes simple eyeball submeshes (not physically accurate) — we **discard** those vertices
- Position our `EyeballGeometry` at FLAME's eye joint positions (which shift per identity)
- FLAME's `leye_pose`/`reye_pose` are ignored — we use our own `rotate_eye()` with `(theta_h, theta_v)`

### 4. Rendering Architecture
- **Composite ray tracing** with depth buffer: trace rays against both face mesh and eye geometry, keep nearest hit
- Face mesh: `trimesh.ray.intersects_location()` for ray-triangle intersection
- Eyes: existing `render_eye()` logic (cornea refraction → iris plane → sclera)
- Eyelid occlusion happens naturally — if a face mesh triangle is closer than the eye surface, it wins

---

## Implementation Steps

### Step 1: Dependencies & FLAME Model Setup
**Files:** `pyproject.toml`, new `facegazesynth/face_model/__init__.py`

- Add `smplx`, `trimesh`, `torch` (cpu) to dependencies
- Create `face_model/` module
- Add config for FLAME model path (env var or parameter, default `models/flame2023`)

**FLAME model files (already downloaded):**
The FLAME 2023 Open model (CC-BY-4.0) is set up in `models/flame2023/`:
1. `flame2023_Open.pkl` — model weights, downloaded from flame.is.tue.mpg.de
2. `mediapipe_landmark_embedding.npz` — landmark embedding, downloaded from flame.is.tue.mpg.de ("FLAME Mediapipe Landmark Embedding")
3. `flame/FLAME_NEUTRAL.pkl` — symlink to `flame2023_Open.pkl` (required by `smplx` naming convention)
4. `flame/flame_static_embedding.pkl` — converted from the `.npz` to `.pkl` format for `smplx`

Verified working: `smplx.create(model_path='models/flame2023', model_type='flame')` → 5023 vertices, 110 joints.

### Step 2: FLAME Mesh Loader
**File:** `facegazesynth/face_model/flame_mesh.py`

- Wrap `smplx.create(model_type='flame')` with our parameter interface
- `FaceMesh` dataclass: `vertices (N,3)`, `faces (F,3)`, `vertex_normals (N,3)`, `eye_joints (2,3)`, `face_params` (identity/expression metadata)
- `build_face_mesh(betas, expression, jaw_pose, ...) → FaceMesh`
- Convert meters → mm
- Identify and remove FLAME eyeball vertices (connected components analysis on the mesh — eyeballs are disconnected submeshes)
- Extract eye joint positions (joints[2] = left eye, joints[3] = right eye)

### Step 3: Eye-Face Composition
**File:** `facegazesynth/face_model/composition.py`

- `compose_face_with_eyes(face_mesh, theta_h, theta_v, eye_params) → CompositeScene`
- Position two `EyeballGeometry` instances at the FLAME eye joint locations
- Apply gaze rotation to each eye independently
- Handle left/right eye mirroring (as in existing `stereo_pair.py`)
- `CompositeScene` dataclass holds: face trimesh, left eye geom, right eye geom

### Step 4: Face Material / Skin Shader
**File:** `facegazesynth/materials/skin.py`

- Lambertian diffuse with skin-tone base color (parameterizable)
- Subtle color variation across face (forehead vs cheeks vs eyelids) using vertex position
- Per-vertex normal interpolation for smooth shading (barycentric interpolation on triangle hits)
- Keep it simple — no subsurface scattering in Phase 2

### Step 5: Composite Renderer
**File:** `facegazesynth/rendering/composite_renderer.py`

- `render_composite(scene, camera, light) → (H, W, 3) RGB image`
- For each ray:
  1. Test face mesh intersection via trimesh (returns hit points, distances, triangle indices)
  2. Test left eye + right eye via existing `render_eye()` (returns colors + depths)
  3. Depth-compare: keep nearest surface per pixel
  4. Apply face material for face hits, existing eye materials for eye hits
- Background color: configurable (default neutral gray)
- Reuse existing `OrthographicCamera` with wider viewport to frame full face (~200mm wide)

### Step 6: Face Pipeline
**File:** `facegazesynth/pipeline/face_render.py`

- `render_face(theta_h, theta_v, betas, expression, resolution, ...) → PIL.Image`
- High-level entry point analogous to `render_single_eye()`
- Default: neutral identity, neutral expression, frontal pose
- `render_face_sweep(angles, betas, expression, ...) → grid image`
- Sweep gaze angles with face context (same validation as Phase 1 but with face)

### Step 7: Perspective Camera (Optional but Recommended)
**File:** update `facegazesynth/rendering/camera.py`

- Add `PerspectiveCamera` class alongside existing `OrthographicCamera`
- Perspective projection is more realistic for face photos
- Parameters: focal length (mm), sensor size, camera distance
- Generate diverging rays from a single viewpoint instead of parallel rays
- Both camera types implement same interface (`generate_rays() → Ray`)

### Step 8: Tests
**File:** `tests/test_face_model.py`, `tests/test_composite_renderer.py`

- Test FLAME mesh loading and vertex count
- Test eyeball vertex removal (face mesh should have ~4800 vertices after removing ~200 eyeball vertices)
- Test eye joint extraction and positioning
- Test coordinate conversion (meters → mm)
- Test composite rendering produces non-zero image
- Test eyelid occlusion: at extreme gaze angles, parts of the eye should be occluded
- Test that iris displacement validation still passes with face context

### Step 9: Scripts
**Files:** `scripts/render_face.py`, `scripts/render_face_sweep.py`

- CLI scripts analogous to existing `render_single.py` and `render_sweep.py`
- Additional args: `--betas` (identity), `--expression`, `--jaw-pose`
- `render_face_sweep.py` produces grid of face renders at different gaze angles

### Step 10: Update Validation
**File:** update `facegazesynth/validation/`

- Extend validation to work with face-context renders
- Pupil detection may need adjustment (face skin surrounding the eye)
- Verify iris displacement still matches refraction-corrected theory (RMS < 0.5mm)

---

## File Summary

| New File | Purpose |
|---|---|
| `facegazesynth/face_model/__init__.py` | Module init |
| `facegazesynth/face_model/flame_mesh.py` | FLAME wrapper, mesh generation |
| `facegazesynth/face_model/composition.py` | Eye-face composition |
| `facegazesynth/materials/skin.py` | Face skin shader |
| `facegazesynth/rendering/composite_renderer.py` | Multi-geometry ray tracer |
| `facegazesynth/pipeline/face_render.py` | High-level face rendering pipeline |
| `tests/test_face_model.py` | Face model tests |
| `tests/test_composite_renderer.py` | Composite renderer tests |
| `scripts/render_face.py` | CLI: render single face |
| `scripts/render_face_sweep.py` | CLI: render face gaze sweep |

| Modified File | Change |
|---|---|
| `pyproject.toml` | Add smplx, trimesh, torch dependencies |
| `facegazesynth/rendering/camera.py` | Add PerspectiveCamera |
| `facegazesynth/validation/` | Adapt pupil detection for face context |

---

## Risks & Mitigations

1. **FLAME model download required** — ✅ Done. FLAME 2023 Open (CC-BY-4.0) downloaded and verified in `models/flame2023/`.
2. **trimesh ray intersection performance** — For 10k triangles at 512² pixels, could be slow without pyembree. Mitigation: make pyembree an optional accelerator dependency; default resolution 256.
3. **Eye socket fit** — FLAME eye joints may not perfectly match our eye model's anatomical dimensions (12mm sclera radius). Mitigation: add a scale/offset parameter to fine-tune eye positioning per identity.
4. **Eyelid gaps** — After removing FLAME eyeball vertices, there may be holes around the eye socket. Mitigation: the eyeball geometry fills the socket; any small gaps are occluded by the sclera sphere.

## Implementation Order
Steps 1-3 first (get a face mesh with eyes positioned), then Step 5 (composite renderer) to see results, then Steps 4, 6-10 to polish.
