# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceGazeSynth is a synthetic face generation system that produces realistic eye gaze using physics-based modeling. The goal is to generate faces with controlled gaze angles (0° to ±30°) that look realistic by getting the optics right — particularly corneal refraction, which is why AI-generated gaze typically looks "dead."

## Build and Run Commands

```bash
# Install (use the project .venv)
.venv/bin/pip install -e ".[dev]"

# Run tests
.venv/bin/pytest -v

# Render a single eye
.venv/bin/python scripts/render_single.py --theta-h 20 --resolution 512 --output output/eye.png

# Render full gaze sweep (stereo pairs at all target angles)
.venv/bin/python scripts/render_sweep.py --resolution 256

# Render a face with gaze (orthographic)
.venv/bin/python scripts/render_face.py --theta-h 20 --resolution 512 --output output/face.png

# Render a face with perspective camera
.venv/bin/python scripts/render_face.py --theta-h 10 --resolution 512 --perspective --output output/face_persp.png

# Render face gaze sweep grid (with angle labels)
.venv/bin/python scripts/render_face_sweep.py --resolution 128

# Render custom angle series (e.g., ±20° in 5° steps)
.venv/bin/python scripts/render_face_sweep.py --angles '[-20,-15,-10,-5,0,5,10,15,20]' --resolution 256

# Render with emotion and albedo texture
.venv/bin/python scripts/render_face.py --theta-h 10 --emotion happy --albedo --resolution 512

# Generate batch: 5 identities × 5 emotions × 3 gaze angles
.venv/bin/python scripts/generate_batch.py --n-identities 5 --resolution 256

# Redirect gaze in a real photo (single angle)
.venv/bin/python scripts/redirect_gaze.py --input samples/Dean_Center.png --angle 15

# Redirect gaze sweep (full series with labels)
.venv/bin/python scripts/redirect_gaze.py --input samples/Dean_Center.png --sweep

# Batch redirect all sample photos
.venv/bin/python scripts/redirect_gaze.py --batch --input-dir samples --output-dir output/redirected

# Run validation suite (iris displacement vs. theory)
.venv/bin/python scripts/validate.py --diagnostics

# Run validation with face-context check
.venv/bin/python scripts/validate.py --face-context
```

## Architecture

**Phase 1 — Custom Python Ray Tracer (implemented)**

The renderer traces rays through a composite eyeball geometry with corneal refraction:

- `facegazesynth/eye_model/` — Eyeball geometry and physical constants. `geometry.py` builds the composite model (sclera sphere + cornea cap + iris plane). `parameters.py` holds all physical constants as a frozen dataclass. `rotation.py` rotates geometry around the anatomical rotation center (13.5mm behind corneal apex, NOT the sclera center).

- `facegazesynth/optics/` — Core physics. `intersections.py` handles ray-sphere and ray-plane intersection (vectorized with numpy). `refraction.py` implements 3D Snell's law. All operations are batched (N,3) arrays for performance.

- `facegazesynth/rendering/` — Camera, lighting, and main render loop. `renderer.py` orchestrates: test cornea cap → refract → iris plane; test sclera; apply materials. The cornea cap vs. sclera is determined by whether the hit point falls within the limbus circle (intersection of two spheres).

- `facegazesynth/materials/` — Procedural iris texture (radial fibers, collarette, crypts), sclera shading, corneal specular (Purkinje image), limbus darkening.

- `facegazesynth/pipeline/` — High-level: `single_eye.py`, `stereo_pair.py` (two mirror-image eyes at 63mm IPD), `sweep.py` (render all target angles).

- `facegazesynth/validation/` — Measures apparent iris displacement from rendered images and compares against theoretical predictions with/without corneal refraction.

**Key physics simplification:** Single-surface Gullstrand model — cornea treated as one refracting surface (air n=1.0 → aqueous n=1.336), ignoring 0.5mm cornea thickness.

**Phase 2 — Face Integration (implemented)**

Embeds physics-based eyeballs into a FLAME 2023 Open parametric face mesh:

- `facegazesynth/face_model/` — FLAME mesh loading and eye-face composition. `flame_mesh.py` wraps smplx, removes FLAME's eyeball submeshes (connected component analysis), extracts eye joint positions, converts m→mm. `composition.py` positions two `EyeballGeometry` instances at FLAME eye joints with correct mirroring and conjugate gaze rotation.

- `facegazesynth/rendering/composite_renderer.py` — Depth-buffer compositing: traces rays against face mesh (trimesh) and both eyes (render_eye with return_depth), keeps nearest hit. Eye regions get NxN supersampling for anti-aliased iris/sclera detail.

- `facegazesynth/materials/skin.py` — Lambertian diffuse skin shader with barycentric-interpolated smooth normals and subtle position-based color variation.

- `facegazesynth/pipeline/face_render.py` — High-level `render_face()` and `render_face_sweep()`.

- `facegazesynth/rendering/camera.py` — `OrthographicCamera` and `PerspectiveCamera` (focal length, sensor size, camera distance). Both implement `generate_rays() → Ray` and `mm_per_pixel()`.

- `facegazesynth/validation/face_validation.py` — Face-context iris displacement validation: renders eyes within the face mesh, crops right eye region, measures displacement vs. theory. RMS 0.381mm (PASS).

- `scripts/render_face.py`, `scripts/render_face_sweep.py` — CLI entry points. `--perspective` flag for perspective camera.

**FLAME model setup:** Model files live in `models/flame2023/`. The directory structure expected by `smplx`:
```
models/flame2023/
├── flame2023_Open.pkl                          # Downloaded from flame.is.tue.mpg.de (CC-BY-4.0)
├── mediapipe_landmark_embedding.npz            # Downloaded from flame.is.tue.mpg.de
├── flame/
│   ├── FLAME_NEUTRAL.pkl → ../flame2023_Open.pkl   # Symlink (smplx expects this name)
│   └── flame_static_embedding.pkl                   # Converted from mediapipe_landmark_embedding.npz
```
Load with: `smplx.create(model_path='models/flame2023', model_type='flame')` → 5023 vertices, 110 joints.

**Phase 3 — Diverse Faces with Emotions (implemented)**

- `facegazesynth/materials/albedo.py` — AlbedoMM PCA model loader (145 components), texture sampling with auto-scaled coefficients, UV-mapped per-pixel color lookup. BGR→RGB conversion.

- `facegazesynth/face_model/expressions.py` — 8 emotion presets (neutral, happy, sad, angry, surprised, disgusted, fearful, contempt) mapped to FLAME expression + jaw_pose. `random_identity()` for shape sampling.

- `facegazesynth/pipeline/batch.py` — Batch generation: identity × emotion × gaze combinations with JSON manifest. `scripts/generate_batch.py` CLI.

**Albedo model:** `models/flame2023/albedoModel2020_FLAME_albedoPart.npz` (1.7GB, from AlbedoMM CVPR 2020, academic license). 512×512 texture space, 145 PCA components for diffuse + specular albedo.

**Phase 4 — Physics-Guided Gaze Redirection on Real Photos (implemented)**

Takes real headshot photos and redirects gaze to target angles using the same corneal refraction physics:

- `facegazesynth/redirection/detection.py` — MediaPipe FaceLandmarker (478 landmarks with iris) for eye/iris detection. Returns iris center, radius, eyelid contours, eye corners.

- `facegazesynth/redirection/physics_mapping.py` — Bridges physics and pixel space. Calibrates mm_per_pixel from detected iris size (accounting for corneal magnification), converts `refraction_corrected_displacement()` to pixel displacement.

- `facegazesynth/redirection/warping.py` — Iris region warping with cosine foreshortening. Uses `cv2.remap()` with soft alpha blending falloff.

- `facegazesynth/redirection/inpainting.py` — Fills exposed sclera when iris moves, using `cv2.inpaint()` (Telea method).

- `facegazesynth/redirection/specular.py` — Detects and repositions corneal specular highlight (Purkinje image) based on corneal geometry.

- `facegazesynth/redirection/compositing.py` — Orchestrates single-eye and both-eye redirection (warp → inpaint → specular).

- `facegazesynth/pipeline/redirect.py` — High-level: `redirect_gaze()`, `redirect_gaze_sweep()`, `redirect_batch()`. Batch processes all photos in a directory with JSON manifest.

- `facegazesynth/validation/redirect_validation.py` — Round-trip validation: redirect → re-detect → measure displacement vs. physics prediction. RMS 0.674mm (PASS, threshold 1.0mm).

**MediaPipe model:** `models/face_landmarker.task` (3.7MB, downloaded from Google). Required for iris landmark detection.

## Key Physics

The critical rendering challenge is **corneal refraction**. Camera rays hitting the cornea must be refracted (Snell's law at the curved corneal surface) before intersecting the iris plane. At off-axis gaze angles, refraction is asymmetric. The specular highlight (Purkinje image) on the cornea is also an important gaze direction cue.

Validation: rendered iris displacement vs. gaze angle matches refraction-corrected theory (RMS < 0.5mm) and clearly diverges from the naive no-refraction prediction.
