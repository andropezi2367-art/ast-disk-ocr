"""Command-line interface for a single AST disk image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .core import load_catalog, recognize_disk_text
from .preprocess import preprocess_disk_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative OCR for one AST disk")
    parser.add_argument("image", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--center-x", type=float)
    parser.add_argument("--center-y", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise SystemExit("--minimum-confidence must be between 0 and 1")
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    supplied = [args.center_x is not None, args.center_y is not None, args.radius is not None]
    if any(supplied) and not all(supplied):
        raise SystemExit("--center-x, --center-y, and --radius must be supplied together")
    cx = args.center_x if args.center_x is not None else image.shape[1] / 2.0
    cy = args.center_y if args.center_y is not None else image.shape[0] / 2.0
    radius = args.radius if args.radius is not None else min(image.shape[:2]) * 0.45

    try:
        from rapidocr import RapidOCR
    except ImportError as error:
        raise SystemExit('Install the optional runtime: pip install -e ".[rapidocr]"') from error

    prepared, baseline = preprocess_disk_text(image, cx, cy, radius)
    result = recognize_disk_text(
        RapidOCR(), prepared, load_catalog(args.catalog), args.minimum_confidence
    )
    payload = result.to_dict() | {
        "orientation": {
            "baseline_found": baseline.found,
            "angle_degrees": baseline.angle_degrees,
        }
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

