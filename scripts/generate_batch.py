#!/usr/bin/env python3
"""Generate a batch of diverse face renders with varied identity, emotion, and gaze."""

import argparse
from facegazesynth.pipeline.batch import generate_batch


def main():
    parser = argparse.ArgumentParser(description="Generate diverse face batch")
    parser.add_argument("--n-identities", type=int, default=5, help="Number of identities")
    parser.add_argument("--emotions", type=str, default=None,
                        help="Comma-separated emotions (default: neutral,happy,sad,angry,surprised)")
    parser.add_argument("--gaze-angles", type=str, default=None,
                        help="Comma-separated gaze angles in degrees (default: 0,10,20)")
    parser.add_argument("--resolution", type=int, default=256, help="Image width in pixels")
    parser.add_argument("--output-dir", type=str, default="output/batch", help="Output directory")
    parser.add_argument("--no-albedo", action="store_true", help="Disable albedo textures")
    parser.add_argument("--perspective", action="store_true", help="Use perspective camera")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    emotions = args.emotions.split(",") if args.emotions else None
    gaze_angles = [float(a) for a in args.gaze_angles.split(",")] if args.gaze_angles else None

    generate_batch(
        n_identities=args.n_identities,
        emotions=emotions,
        gaze_angles=gaze_angles,
        resolution=args.resolution,
        output_dir=args.output_dir,
        use_albedo=not args.no_albedo,
        base_seed=args.seed,
        perspective=args.perspective,
    )


if __name__ == "__main__":
    main()
