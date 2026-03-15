#!/usr/bin/env python3
"""Run the validation suite: measure iris displacement and compare to theory."""

import argparse
from facegazesynth.validation.compare import run_validation
from facegazesynth.validation.diagnostics import plot_cross_section, plot_ray_fan


def main():
    parser = argparse.ArgumentParser(description="Run validation suite")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    parser.add_argument("--output", type=str, default="output/validation_curve.png")
    parser.add_argument("--diagnostics", action="store_true", help="Also generate diagnostic plots")
    args = parser.parse_args()

    if args.diagnostics:
        print("Generating diagnostic plots...")
        plot_cross_section()
        plot_ray_fan()
        print()

    print("Running iris displacement validation...")
    results = run_validation(resolution=args.resolution, output_path=args.output)

    # Pass/fail check
    if results["rms_error"] < 0.5:
        print(f"\n  PASS: RMS error {results['rms_error']:.4f} mm < 0.5 mm threshold")
    else:
        print(f"\n  WARNING: RMS error {results['rms_error']:.4f} mm >= 0.5 mm threshold")


if __name__ == "__main__":
    main()
