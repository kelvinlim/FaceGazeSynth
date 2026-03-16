# FaceGazeSynth — Plan Progress

> Last updated: 2026-03-16

---

## Phase 1 — Physics-Based Eyeball Model

**Status: Complete**

| # | Milestone | Status | Completed |
|---|-----------|--------|-----------|
| 1.1 | Anatomical geometry (sclera, cornea, iris, pupil, limbus) | Done | 2026-03-15 |
| 1.2 | Eye parameters dataclass with published anatomical values | Done | 2026-03-15 |
| 1.3 | Gaze rotation mechanics (Rodrigues' formula, rotation center) | Done | 2026-03-15 |
| 1.4 | Corneal refraction (Snell's law, Fresnel reflectance) | Done | 2026-03-15 |
| 1.5 | Ray-geometry intersections (sphere, plane, cornea cap) | Done | 2026-03-15 |
| 1.6 | Orthographic camera ray generation | Done | 2026-03-15 |
| 1.7 | Unit tests for geometry, rotation, refraction, intersections (33 tests) | Done | 2026-03-15 |
| 1.8 | Material properties (procedural iris texture, sclera shading, limbus darkening) | Done | 2026-03-15 |
| 1.9 | Corneal specular highlight (Purkinje image, Blinn-Phong + Fresnel) | Done | 2026-03-15 |
| 1.10 | End-to-end rendering pipeline (ray trace → image output) | Done | 2026-03-15 |
| 1.11 | Stereo pair rendering (mirror-image eyes at 63mm IPD) | Done | 2026-03-15 |
| 1.12 | Full gaze sweep at 0°, ±5°, ±10°, ±15°, ±20°, ±30° | Done | 2026-03-15 |
| 1.13 | Validation: iris displacement vs. refraction-corrected theory (RMS 0.386mm, PASS) | Done | 2026-03-15 |
| 1.14 | Diagnostic visualizations (cross-section, ray fan diagram) | Done | 2026-03-15 |

### Key Results
- Custom Python ray tracer with numpy vectorization (512×512 in seconds)
- Corneal refraction confirmed working: measured magnification ratio > 1.0 at all gaze angles
- Validation RMS error 0.386mm vs. refraction-corrected theory (< 0.5mm threshold)
- Iris displacement clearly diverges from naive (no-refraction) prediction

---

## Phase 2 — Face Integration (FLAME 2023 Open)

**Status: Complete**

| # | Milestone | Status | Completed |
|---|-----------|--------|-----------|
| 2.0 | FLAME model setup (download, smplx integration, landmark embedding) | Done | 2026-03-15 |
| 2.1 | Dependencies & face_model module (smplx, trimesh, torch, rtree) | Done | 2026-03-15 |
| 2.2 | FLAME mesh loader (build_face_mesh, eyeball vertex removal, eye joints) | Done | 2026-03-15 |
| 2.3 | Eye-face composition (position eyeballs at FLAME eye joints) | Done | 2026-03-15 |
| 2.4 | Face material / skin shader (Lambertian diffuse, smooth shading) | Done | 2026-03-15 |
| 2.5 | Composite renderer (depth-buffer ray tracing: face mesh + eyes) | Done | 2026-03-15 |
| 2.6 | Face rendering pipeline (render_face, render_face_sweep) | Done | 2026-03-15 |
| 2.7 | Perspective camera (optional, alongside existing orthographic) | Done | 2026-03-15 |
| 2.8 | Tests (mesh loading, eye positioning, composite rendering: 53 total) | Done | 2026-03-15 |
| 2.9 | CLI scripts (render_face.py, render_face_sweep.py) | Done | 2026-03-15 |
| 2.10 | Update validation for face-context renders (RMS 0.381mm, PASS) | Done | 2026-03-15 |

### Key Results
- Full face renders with FLAME 2023 Open mesh + physics-based eyeballs at 512px
- Conjugate gaze: both eyes rotate in the same world-space direction
- Eye-region 4x4 supersampling for smooth iris/sclera detail
- Cornea cap boundary seam fixed (sclera normals used consistently across limbus)
- Iris collarette and limbus darkening softened to remove artifactual rings
- 53 tests passing (33 Phase 1 + 15 face model + 5 composite renderer)
- Face-context iris displacement RMS 0.381mm (< 1.0mm threshold, PASS)
- Perspective camera added (50mm focal length, natural foreshortening)

### Setup Notes
- Using FLAME 2023 Open (CC-BY-4.0) — model files in `models/flame2023/`
- Downloaded: `flame2023_Open.pkl`, `mediapipe_landmark_embedding.npz`
- Verified: `smplx.create()` loads successfully → 5023 vertices, 110 joints

---

## Phase 3 — Diverse Faces with Emotions

**Status: Complete**

| # | Milestone | Status | Completed |
|---|-----------|--------|-----------|
| 3.1 | Data-driven albedo texture (AlbedoMM PCA model, UV mapping, BGR→RGB) | Done | 2026-03-16 |
| 3.2 | Emotion presets (8 emotions: neutral, happy, sad, angry, surprised, disgusted, fearful, contempt) | Done | 2026-03-16 |
| 3.3 | Skin shader updated for albedo textures with procedural fallback | Done | 2026-03-16 |
| 3.4 | Gaze calibration preserved through expression deformation (RMS 0.26mm with surprised) | Done | 2026-03-16 |
| 3.5 | Batch generation pipeline (identity × emotion × gaze, JSON manifest) | Done | 2026-03-16 |

### Key Results
- AlbedoMM PCA model (145 components) for diverse skin appearance
- 8 emotion presets mapped to FLAME expression + jaw_pose coefficients
- Random identity sampling via FLAME betas
- Gaze calibration RMS 0.26mm even with large expression deformation (surprised)
- Batch pipeline generates identity × emotion × gaze grid with manifest.json
- 68 tests passing (33 Phase 1 + 20 Phase 2 + 15 Phase 3)

## Phase 4 — Physics-Guided Gaze Redirection on Real Photos

**Status: Complete**

| # | Milestone | Status | Completed |
|---|-----------|--------|-----------|
| 4.1 | MediaPipe eye/iris detection (478 landmarks) | Done | 2026-03-16 |
| 4.2 | Physics-to-pixel mapping (corneal magnification calibration) | Done | 2026-03-16 |
| 4.3 | Iris warping with cosine foreshortening | Done | 2026-03-16 |
| 4.4 | Sclera inpainting (Telea method) | Done | 2026-03-16 |
| 4.5 | Specular highlight (Purkinje image) repositioning | Done | 2026-03-16 |
| 4.6 | Compositing pipeline (warp → inpaint → specular) | Done | 2026-03-16 |
| 4.7 | Batch processing (6 identities × 9 angles) with CLI | Done | 2026-03-16 |
| 4.8 | Round-trip validation (RMS 0.674mm, threshold 1.0mm) | Done | 2026-03-16 |

### Key Results
- 6 real headshot photos redirected to 9 gaze angles each (±5°, ±10°, ±15°, ±20°)
- Physics-accurate iris displacement via `refraction_corrected_displacement()`
- Round-trip validation RMS 0.674mm (PASS)
- 82 tests passing (33 Phase 1 + 20 Phase 2 + 15 Phase 3 + 14 Phase 4)
