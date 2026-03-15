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

## Phase 2 — Face Integration

**Status: Not started**

| # | Milestone | Status | Completed |
|---|-----------|--------|-----------|
| 2.1 | Select and integrate parametric face mesh (FLAME or Basel Face Model) | Not started | — |
| 2.2 | Embed eyeballs at correct interpupillary distance (~63mm) and socket depth | Not started | — |
| 2.3 | Eyelid geometry conforming to eyeball curvature | Not started | — |
| 2.4 | Eye socket shadowing and occlusion | Not started | — |
| 2.5 | Tear film and caruncle details | Not started | — |
| 2.6 | Full face + gaze renders at all target angles | Not started | — |

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
