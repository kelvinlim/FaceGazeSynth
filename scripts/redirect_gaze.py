#!/usr/bin/env python3
"""Redirect gaze in real photos using physics-guided warping."""

import argparse
from pathlib import Path

from facegazesynth.pipeline.redirect import (
    redirect_gaze,
    redirect_gaze_sweep,
    redirect_batch,
)


def main():
    parser = argparse.ArgumentParser(description="Physics-guided gaze redirection")
    parser.add_argument("--input", type=str, help="Input photo path")
    parser.add_argument("--angle", type=float, help="Target gaze angle in degrees")
    parser.add_argument("--sweep", action="store_true", help="Generate full angle sweep")
    parser.add_argument("--batch", action="store_true", help="Process all photos in input-dir")
    parser.add_argument("--input-dir", type=str, default="samples", help="Input directory for batch")
    parser.add_argument("--output-dir", type=str, default="output/redirected", help="Output directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path (single image)")
    parser.add_argument("--debug", action="store_true", help="Save debug detection overlay")
    args = parser.parse_args()

    if args.batch:
        redirect_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        return

    if not args.input:
        parser.error("--input is required unless using --batch")

    if args.debug:
        import numpy as np
        from PIL import Image
        from facegazesynth.redirection.detection import detect_eyes, draw_debug_overlay

        img = np.array(Image.open(args.input).convert("RGB"))
        det = detect_eyes(img)
        debug = draw_debug_overlay(img, det)
        out = args.output or "output/debug_detection.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(debug).save(out)
        print(f"Debug overlay saved to {out}")
        return

    if args.sweep:
        out = args.output or str(
            Path(args.output_dir) / (Path(args.input).stem + "_sweep.png")
        )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        img = redirect_gaze_sweep(args.input)
        img.save(out)
        print(f"Sweep saved to {out} ({img.width}x{img.height})")
        return

    if args.angle is None:
        parser.error("--angle is required for single-image mode")

    out = args.output
    if out is None:
        stem = Path(args.input).stem
        label = "Center" if args.angle == 0 else f"{abs(int(args.angle))}{'L' if args.angle < 0 else 'R'}"
        out = f"output/{stem}_{label}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    img = redirect_gaze(args.input, args.angle)
    img.save(out)
    print(f"Saved to {out} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
