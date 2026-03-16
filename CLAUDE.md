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

# Run validation suite (iris displacement vs. theory)
.venv/bin/python scripts/validate.py --diagnostics
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

**Phase 2 — Face Integration (in progress)**
Embed eyeballs into FLAME 2023 Open parametric face mesh. See `PLAN.md` for full implementation plan.

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

**Phase 3 — Diverse Faces with Emotions (not yet started)**
Identity variation + expression parameters via FACS.

## Key Physics

The critical rendering challenge is **corneal refraction**. Camera rays hitting the cornea must be refracted (Snell's law at the curved corneal surface) before intersecting the iris plane. At off-axis gaze angles, refraction is asymmetric. The specular highlight (Purkinje image) on the cornea is also an important gaze direction cue.

Validation: rendered iris displacement vs. gaze angle matches refraction-corrected theory (RMS < 0.5mm) and clearly diverges from the naive no-refraction prediction.
