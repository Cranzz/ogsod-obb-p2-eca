"""Validate an Ultralytics YOLO OBB dataset before training.

Expected label format (normalized corner coordinates):
    class x1 y1 x2 y2 x3 y3 x4 y4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Dataset root containing images/ and labels/.")
    parser.add_argument("--images", type=Path, help="Single image directory. Overrides --data-root.")
    parser.add_argument("--labels", type=Path, help="Single label directory. Overrides --data-root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_check"))
    parser.add_argument("--edge-padding", type=float, default=0.002)
    return parser.parse_args()


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_side_lengths(points: list[tuple[float, float]]) -> list[float]:
    lengths = []
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        lengths.append(math.hypot(x2 - x1, y2 - y1))
    return lengths


def resolve_directories(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.images or args.labels:
        if not (args.images and args.labels):
            raise ValueError("--images and --labels must be provided together.")
        return [("single", args.images, args.labels)]

    if not args.data_root:
        raise ValueError("Provide --data-root, or provide both --images and --labels.")

    directories = []
    for split in args.splits:
        images = args.data_root / "images" / split
        labels = args.data_root / "labels" / split
        if images.exists() and labels.exists():
            directories.append((split, images, labels))
    if not directories:
        raise FileNotFoundError(f"No images/<split> and labels/<split> pairs found under {args.data_root}.")
    return directories


def inspect_split(
    split: str,
    image_dir: Path,
    label_dir: Path,
    num_classes: int,
    edge_padding: float,
) -> tuple[dict, list[dict]]:
    image_files = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    label_files = set(label_dir.rglob("*.txt"))
    class_counts: Counter[int] = Counter()
    errors: Counter[str] = Counter()
    error_examples: dict[str, list[str]] = defaultdict(list)
    object_count = 0
    empty_label_files = 0
    matched_label_files: set[Path] = set()
    image_sizes: Counter[tuple[int, int]] = Counter()
    areas: list[float] = []
    short_sides: list[float] = []
    long_sides: list[float] = []

    def add_error(kind: str, sample: str) -> None:
        errors[kind] += 1
        if len(error_examples[kind]) < 10:
            error_examples[kind].append(sample)

    for image_path in image_files:
        relative = image_path.relative_to(image_dir).with_suffix(".txt")
        label_path = label_dir / relative
        if not label_path.exists():
            add_error("missing_label", str(image_path))
            continue

        matched_label_files.add(label_path)
        try:
            with Image.open(image_path) as image:
                image_sizes[image.size] += 1
        except Exception as exc:  # pragma: no cover - depends on broken files
            add_error("unreadable_image", f"{image_path}: {exc}")
            continue

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_label_files += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()
            where = f"{label_path}:{line_number}"
            if len(parts) != 9:
                add_error("invalid_field_count", f"{where}: {len(parts)} fields")
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                add_error("non_numeric_value", f"{where}: {line}")
                continue
            if not 0 <= class_id < num_classes:
                add_error("class_out_of_range", f"{where}: class={class_id}")
                continue

            points = list(zip(coords[0::2], coords[1::2]))
            if any(value < -edge_padding or value > 1.0 + edge_padding for value in coords):
                add_error("coordinate_out_of_range", f"{where}: {line}")
                continue

            area = polygon_area(points)
            sides = sorted(polygon_side_lengths(points))
            if area <= 1e-8 or sides[0] <= 1e-8:
                add_error("degenerate_polygon", f"{where}: area={area}")
                continue

            class_counts[class_id] += 1
            object_count += 1
            areas.append(area)
            short_sides.append(sides[0])
            long_sides.append(sides[-1])

    unmatched_labels = sorted(label_files - matched_label_files)
    for label_path in unmatched_labels:
        add_error("missing_image", str(label_path))

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    summary = {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "images": len(image_files),
        "labels": len(label_files),
        "matched_pairs": len(image_files) - errors["missing_label"],
        "empty_label_files": empty_label_files,
        "objects": object_count,
        "class_counts": {str(index): class_counts[index] for index in range(num_classes)},
        "errors": dict(errors),
        "error_examples": dict(error_examples),
        "image_sizes_top": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in image_sizes.most_common(10)
        ],
        "normalized_area": {
            "min": min(areas) if areas else None,
            "p25": percentile(areas, 0.25),
            "median": percentile(areas, 0.5),
            "p75": percentile(areas, 0.75),
            "max": max(areas) if areas else None,
        },
        "normalized_short_side": {
            "min": min(short_sides) if short_sides else None,
            "median": percentile(short_sides, 0.5),
            "max": max(short_sides) if short_sides else None,
        },
        "normalized_long_side": {
            "min": min(long_sides) if long_sides else None,
            "median": percentile(long_sides, 0.5),
            "max": max(long_sides) if long_sides else None,
        },
    }
    return summary, [{"split": split, "class_id": index, "instances": class_counts[index]} for index in range(num_classes)]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    rows = []
    for split, image_dir, label_dir in resolve_directories(args):
        summary, split_rows = inspect_split(split, image_dir, label_dir, args.num_classes, args.edge_padding)
        summaries.append(summary)
        rows.extend(split_rows)
        print(
            f"[{split}] images={summary['images']} labels={summary['labels']} "
            f"objects={summary['objects']} errors={sum(summary['errors'].values())}"
        )

    report_path = args.output / "dataset_summary.json"
    report_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=True), encoding="utf-8")
    csv_path = args.output / "class_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_id", "instances"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
