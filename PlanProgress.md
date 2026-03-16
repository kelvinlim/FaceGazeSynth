# FaceGazeSynth — Plan Progress

> Last updated: 2026-03-15

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

**Status: Not started**

| # | Milestone | Status | Completed |
|---|-----------|--------|-----------|
| 3.1 | Identity variation parameters (demographics, skin reflectance) | Not started | — |
| 3.2 | Expression parameters via FACS (Facial Action Coding System) | Not started | — |
| 3.3 | Emotion label mapping (happy, sad, angry, scornful, etc.) to AU blendshapes | Not started | — |
| 3.4 | Verify gaze calibration is preserved through face deformation | Not started | — |
| 3.5 | Batch generation pipeline for diverse face + gaze + emotion combinations | Not started | — |
