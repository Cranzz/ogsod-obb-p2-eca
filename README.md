# OGSOD-1.0 SAR YOLOv8s-OBB + P2 + ECA

Reproducible code for single-modality SAR oriented object detection on
OGSOD-1.0. The baseline is YOLOv8s-OBB. The ablation contains Baseline, P2,
ECA, and P2+ECA variants.

## Scope

- SAR images only; optical images are not used.
- No multimodal fusion, DETR/Transformer, distillation, or semi-supervised learning.
- Local machines are used for code checks and data conversion only.
- Full training is performed in Google Colab with a T4 GPU.
- Datasets, checkpoints, and runs are stored outside GitHub.

## Project Layout

```text
models/
  modules/eca.py
  yolov8s_obb_baseline.yaml
  yolov8s_obb_p2.yaml
  yolov8s_obb_eca.yaml
  yolov8s_obb_p2_eca.yaml
scripts/
  check_obb_dataset.py
  convert_ogsod_obb.py
  visualize_obb.py
  train_obb.py
  train_colab.ipynb
data/
  ogsod_obb.example.yaml
```

## Data Format

The public OGSOD-1.0 rotated label package uses DOTA-style lines:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

The conversion script creates normalized Ultralytics OBB labels:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Class mapping:

```text
0: bridge
1: harbor
2: storage_tank
```

Invalid zero-area boxes are removed and written to `removed_labels.csv`.

## Convert Data

```bash
python scripts/convert_ogsod_obb.py \
  --source-root /path/to/OGSOD-1.0 \
  --obb-root "/path/to/OGSOD-1.0(obb_label) " \
  --output /path/to/OGSOD-1.0-yolo-obb \
  --val-fraction 0.1 \
  --seed 42 \
  --images-mode symlink \
  --image-size 256 256 \
  --progress-every 1000
```

## Check Data

```bash
python scripts/check_obb_dataset.py \
  --data-root /path/to/OGSOD-1.0-yolo-obb \
  --splits train test \
  --num-classes 3 \
  --output reports/ogsod_obb_check
```

## Build Check Without Training

```bash
python - <<'PY'
import torch
from models.modules import register_custom_modules
register_custom_modules()
from ultralytics import YOLO

for config in (
    "models/yolov8s_obb_baseline.yaml",
    "models/yolov8s_obb_p2.yaml",
    "models/yolov8s_obb_eca.yaml",
    "models/yolov8s_obb_p2_eca.yaml",
):
    model = YOLO(config)
    model.model.eval()
    with torch.no_grad():
        model.model(torch.randn(1, 3, 640, 640))
    print(config, "OK")
PY
```

## Cloud Training

Use `scripts/train_obb.py` in Colab after mounting prepared data from Google
Drive. Run the baseline first, then P2, ECA, and P2+ECA with identical seeds,
input sizes, epochs, optimizer settings, and evaluation splits.

```bash
python scripts/train_obb.py \
  --data /content/drive/MyDrive/OGSOD-1.0-yolo-obb/data.yaml \
  --experiment baseline \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --seed 42
```

For a strict structural ablation, do not load pretrained weights for any
variant. A pretrained practical-performance experiment should be reported
separately.
