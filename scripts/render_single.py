#!/usr/bin/env python3
"""Render a single eyeball at a given gaze angle."""

import argparse
from pathlib import Path
from facegazesynth.pipeline.single_eye import render_single_eye


def main():
    parser = argparse.ArgumentParser(description="Render a single eyeball")
    parser.add_argument("--theta-h", type=float, default=0.0, help="Horizontal gaze angle (degrees)")
    parser.add_argument("--theta-v", type=float, default=0.0, help="Vertical gaze angle (degrees)")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--output", type=str, default="output/single_eye.png", help="Output path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    img = render_single_eye(
        theta_h_deg=args.theta_h,
        theta_v_deg=args.theta_v,
        resolution=args.resolution,
    )
    img.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
