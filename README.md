# FaceGazeSynth

Physics-based synthetic face generation with realistic eye gaze. Generates faces with controlled gaze angles (0° to ±30°) by modeling corneal refraction — the optical effect that makes eyes look alive.

## Quick Start

```bash
# Set up
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Render a single eye
.venv/bin/python scripts/render_single.py --theta-h 20 --resolution 512

# Render full gaze sweep (stereo pairs at all target angles)
.venv/bin/python scripts/render_sweep.py --resolution 256

# Run validation (iris displacement vs. theory)
.venv/bin/python scripts/validate.py --diagnostics

# Run tests
.venv/bin/pytest -v
```

## How It Works

The renderer traces rays through a composite eyeball model:

1. **Geometry** — Sclera sphere (~24mm) with a protruding cornea cap (7.8mm radius dome), iris disc, and pupil aperture
2. **Corneal refraction** — Rays hitting the cornea are refracted via Snell's law (Gullstrand simplified eye model, IOR 1.336), making the iris appear magnified and shifted
3. **Gaze rotation** — Eye rotates around the anatomical rotation center (13.5mm behind corneal apex), producing realistic iris displacement and sclera asymmetry
4. **Materials** — Procedural iris texture (radial fibers, collarette, crypts), Lambertian sclera, corneal specular highlight (Purkinje image)

## Project Status

See [PlanProgress.md](PlanProgress.md) for detailed milestone tracking.

- **Phase 1** — Physics-based eyeball model: **Complete**
- **Phase 2** — Face integration (FLAME/Basel Face Model): Not started
- **Phase 3** — Diverse faces with emotions (FACS): Not started
