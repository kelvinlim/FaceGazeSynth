#!/usr/bin/env python3
"""Render a full face with physics-based eyeballs at a given gaze angle."""

import argparse
import json
from pathlib import Path

import numpy as np

from facegazesynth.pipeline.face_render import render_face


def main():
    parser = argparse.ArgumentParser(description="Render a face with gaze")
    parser.add_argument("--theta-h", type=float, default=0.0, help="Horizontal gaze angle (degrees)")
    parser.add_argument("--theta-v", type=float, default=0.0, help="Vertical gaze angle (degrees)")
    parser.add_argument("--resolution", type=int, default=256, help="Image width in pixels")
    parser.add_argument("--betas", type=str, default=None, help="Identity coefficients as JSON array")
    parser.add_argument("--expression", type=str, default=None, help="Expression coefficients as JSON array")
    parser.add_argument("--jaw-pose", type=str, default=None, help="Jaw rotation as JSON array [rx, ry, rz]")
    parser.add_argument("--emotion", type=str, default=None, help="Emotion preset (happy, sad, angry, etc.)")
    parser.add_argument("--emotion-intensity", type=float, default=1.0, help="Emotion intensity (0-1)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FLAME model directory")
    parser.add_argument("--perspective", action="store_true", help="Use perspective camera")
    parser.add_argument("--focal-length", type=float, default=50.0, help="Focal length in mm (perspective only)")
    parser.add_argument("--camera-distance", type=float, default=500.0, help="Camera distance in mm (perspective only)")
    parser.add_argument("--albedo", action="store_true", help="Use data-driven albedo texture")
    parser.add_argument("--albedo-seed", type=int, default=None, help="Seed for albedo sampling")
    parser.add_argument("--output", type=str, default="output/face.png", help="Output path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    betas = np.array(json.loads(args.betas)) if args.betas else None
    expression = np.array(json.loads(args.expression)) if args.expression else None
    jaw_pose = np.array(json.loads(args.jaw_pose)) if args.jaw_pose else None

    img = render_face(
        theta_h_deg=args.theta_h,
        theta_v_deg=args.theta_v,
        resolution=args.resolution,
        betas=betas,
        expression=expression,
        jaw_pose=jaw_pose,
        emotion=args.emotion,
        emotion_intensity=args.emotion_intensity,
        model_path=args.model_path,
        perspective=args.perspective,
        focal_length=args.focal_length,
        camera_distance=args.camera_distance,
        use_albedo=args.albedo,
        albedo_seed=args.albedo_seed,
    )
    img.save(args.output)
    print(f"Saved to {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
