#!/usr/bin/env python3
"""Render full gaze angle sweep with stereo pairs."""

import argparse
from facegazesynth.pipeline.sweep import render_sweep


def main():
    parser = argparse.ArgumentParser(description="Render gaze angle sweep")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution per eye")
    parser.add_argument("--output-dir", type=str, default="output/sweep", help="Output directory")
    args = parser.parse_args()

    print("Rendering gaze sweep...")
    render_sweep(resolution=args.resolution, output_dir=args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
