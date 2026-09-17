"""Draw random Ultralytics OBB labels for visual inspection."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COLORS = [(0, 0, 255), (0, 180, 0), (255, 80, 0), (180, 0, 180), (0, 165, 255)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--names", nargs="*", default=["bridge", "harbor", "storage_tank"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_files = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    random.Random(args.seed).shuffle(image_files)
    saved = 0

    for image_path in image_files:
        relative = image_path.relative_to(args.images).with_suffix(".txt")
        label_path = args.labels / relative
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 9:
                continue
            class_id = int(float(parts[0]))
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
            points = np.stack([coords[0::2] * width, coords[1::2] * height], axis=1).astype(np.int32)
            color = COLORS[class_id % len(COLORS)]
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            name = args.names[class_id] if class_id < len(args.names) else str(class_id)
            anchor = tuple(points[0])
            cv2.putText(image, name, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        output_path = args.output / f"{saved:03d}_{image_path.name}"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write {output_path}")
        saved += 1
        if saved >= args.count:
            break

    print(f"Saved {saved} visualizations to {args.output}")


if __name__ == "__main__":
    main()
