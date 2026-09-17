"""Train one experiment in the OGSOD YOLOv8s-OBB ablation series."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.modules import register_custom_modules  # noqa: E402

register_custom_modules()

from ultralytics import YOLO  # noqa: E402


EXPERIMENTS = {
    "baseline": PROJECT_ROOT / "models" / "yolov8s_obb_baseline.yaml",
    "p2": PROJECT_ROOT / "models" / "yolov8s_obb_p2.yaml",
    "eca": PROJECT_ROOT / "models" / "yolov8s_obb_eca.yaml",
    "p2_eca": PROJECT_ROOT / "models" / "yolov8s_obb_p2_eca.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the OGSOD OBB data.yaml.")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--pretrained", default=None, help="Optional compatible checkpoint, such as yolov8s-obb.pt.")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    model_path = EXPERIMENTS[args.experiment]
    model = YOLO(str(model_path))
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.exists():
            raise FileNotFoundError(pretrained_path)
        model.load(str(pretrained_path))

    run_name = args.name or f"yolov8s_obb_ogsod_{args.experiment}"
    print(f"Experiment: {args.experiment}")
    print(f"Model config: {model_path}")
    print(f"Data config: {data_path}")
    print(f"Pretrained: {args.pretrained or 'none'}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=run_name,
        plots=True,
        save_period=10,
    )

    best_model = YOLO(str(Path(results.save_dir) / "weights" / "best.pt"))
    metrics = best_model.val(
        data=str(data_path),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )
    print(f"Best model: {results.save_dir}")
    print(f"{args.eval_split} Precision: {metrics.box.mp:.4f}")
    print(f"{args.eval_split} Recall: {metrics.box.mr:.4f}")
    print(f"{args.eval_split} mAP50: {metrics.box.map50:.4f}")
    print(f"{args.eval_split} mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
