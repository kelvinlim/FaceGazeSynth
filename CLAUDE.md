# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceGazeSynth is a synthetic face generation system that produces realistic eye gaze using physics-based modeling. The goal is to generate faces with controlled gaze angles (0° to ±30°) that look realistic by getting the optics right — particularly corneal refraction, which is why AI-generated gaze typically looks "dead."

## Architecture (3-Phase Plan)

**Phase 1 — Physics-Based Eyeball Model (current focus)**
- Two eyeballs rendered at precise gaze angles (0°, ±5°, ±10°, ±15°, ±20°, ±30°)
- Composite geometry: sclera (~24mm sphere), cornea (7.8mm radius dome, protruding ~2.5mm), iris (12mm disc, 3.6mm behind corneal apex), pupil (3–4mm aperture)
- Corneal refraction via Snell's law (refractive index 1.376) — iris is seen *through* the cornea, producing magnification (~4–8%) and asymmetric distortion at off-axis angles
- Eye rotates around a center 13.5mm behind corneal apex
- Two implementation paths considered: (A) custom Python ray tracer (numpy/PIL/matplotlib), (B) Blender Python API (bpy) with Cycles renderer

**Phase 2 — Face Integration**
- Embed eyeballs into parametric face mesh (FLAME or Basel Face Model)
- Eyelid geometry, eye socket shadowing, tear film details
- Interpupillary distance ~63mm

**Phase 3 — Diverse Faces with Emotions**
- Identity variation (demographics) and expression parameters
- Emotion mapping via FACS (Facial Action Coding System)
- Eyeball model stays fixed; face deforms around it

## Key Physics

The critical rendering challenge is **corneal refraction**. Camera rays hitting the cornea must be refracted before intersecting the iris plane. At off-axis gaze angles, refraction is asymmetric (near vs. far iris edge magnified differently). The specular highlight (Purkinje image) on the cornea is also an important gaze direction cue.

Validation target: rendered iris displacement vs. gaze angle should match published empirical curves (Daugman et al.).

## Project Status

This project is in early stages. See [Idea.md](Idea.md) for the full specification and technical rationale.
