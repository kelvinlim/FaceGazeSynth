#!/usr/bin/env python3
"""Render a grid of faces at different gaze angles."""

import argparse
import json
from pathlib import Path

import numpy as np

from facegazesynth.pipeline.face_render import render_face_sweep


def main():
    parser = argparse.ArgumentParser(description="Render face gaze sweep")
    parser.add_argument("--resolution", type=int, default=128, help="Width per face in pixels")
    parser.add_argument("--angles", type=str, default=None, help="Gaze angles as JSON array (degrees)")
    parser.add_argument("--betas", type=str, default=None, help="Identity coefficients as JSON array")
    parser.add_argument("--expression", type=str, default=None, help="Expression coefficients as JSON array")
    parser.add_argument("--model-path", type=str, default=None, help="Path to FLAME model directory")
    parser.add_argument("--output", type=str, default="output/face_sweep.png", help="Output path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    angles = json.loads(args.angles) if args.angles else None
    betas = np.array(json.loads(args.betas)) if args.betas else None
    expression = np.array(json.loads(args.expression)) if args.expression else None

    img = render_face_sweep(
        angles=angles,
        resolution=args.resolution,
        betas=betas,
        expression=expression,
        model_path=args.model_path,
    )
    img.save(args.output)
    print(f"Saved to {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
