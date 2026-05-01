from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover - explicit runtime dependency error
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> Dict:
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must define a mapping: {config_path}")

    return cfg


def create_dataset_yaml(dataset_root: Path, class_name: str = "pine_cone") -> Path:
    """Create YOLO dataset.yaml that points to tiled train/val/test image folders."""
    dataset_root = dataset_root.resolve()
    images_root = dataset_root / "images"

    required_dirs = [images_root / "train", images_root / "val", images_root / "test"]
    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset split folders. Run preprocess first. Missing: " + ", ".join(missing)
        )

    dataset_yaml = dataset_root / "dataset.yaml"
    data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": [class_name],
    }

    with dataset_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    return dataset_yaml


def _resolve_model_checkpoint(model_size: str) -> str:
    model_size = model_size.lower().strip()
    mapping = {
        "n": "yolov8n.pt",
        "nano": "yolov8n.pt",
        "yolov8n": "yolov8n.pt",
        "yolo26n": "yolo26n.pt",
        "s": "yolov8s.pt",
        "small": "yolov8s.pt",
        "m": "yolov8m.pt",
        "medium": "yolov8m.pt",
    }
    if model_size not in mapping:
        raise ValueError("model_size must be one of: n, nano, yolov8n, yolo26n, s, small, m, medium")
    return mapping[model_size]


def train_yolo(
    dataset_yaml: Path,
    model_size: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project_dir: Path,
    run_name: str,
    workers: int,
    seed: int,
    use_wandb: bool,
) -> Dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - explicit runtime dependency error
        raise ImportError("Ultralytics is required. Install it with: pip install ultralytics") from exc

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", "pine-cone-yolo")
        os.environ.setdefault("WANDB_NAME", run_name)

    model = YOLO(_resolve_model_checkpoint(model_size))

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project_dir),
        name=run_name,
        seed=seed,
        pretrained=True,
        cache=False,
        val=True,
        save=True,
        plots=True,
        verbose=True,
        # augmentations
        degrees=180.0,
        translate=0.10,
        scale=0.50,
        shear=5.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.03,
        hsv_s=0.70,
        hsv_v=0.50,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.0,
        close_mosaic=10,
    )

    save_dir = getattr(results, "save_dir", None)
    return {
        "run_dir": str(save_dir) if save_dir is not None else "",
        "dataset_yaml": str(dataset_yaml),
        "model": _resolve_model_checkpoint(model_size),
        "batch": batch,
        "imgsz": imgsz,
        "epochs": epochs,
        "wandb_enabled": use_wandb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on tiled pine-cone dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Path to tiled YOLO dataset root",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="s",
        choices=["n", "nano", "yolov8n", "yolo26n", "s", "small", "m", "medium"],
        help="YOLOv8 backbone size",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size")
    parser.add_argument("--batch", type=int, default=None, help="Training batch size")
    parser.add_argument("--device", type=str, default=None, help="Device id, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--run-name", type=str, default=None, help="Training run name")
    parser.add_argument(
        "--class-name",
        type=str,
        default="pine_cone",
        help="Single-class name stored in dataset.yaml",
    )
    parser.add_argument(
        "--log-backend",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Logging backend preference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})

    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 150))
    imgsz = args.imgsz if args.imgsz is not None else int(train_cfg.get("imgsz", 640))
    batch = args.batch if args.batch is not None else int(train_cfg.get("batch", 16))
    device = args.device if args.device is not None else str(train_cfg.get("device", "0"))
    workers = args.workers if args.workers is not None else int(train_cfg.get("workers", 8))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    run_name = args.run_name if args.run_name is not None else str(train_cfg.get("run_name", "yolov8_finetune"))

    project_dir = Path(train_cfg.get("project_dir", "runs/yolo"))

    dataset_yaml = create_dataset_yaml(args.dataset_root, class_name=args.class_name)
    summary = train_yolo(
        dataset_yaml=dataset_yaml,
        model_size=args.model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project_dir=project_dir,
        run_name=run_name,
        workers=workers,
        seed=seed,
        use_wandb=(args.log_backend == "wandb"),
    )

    print(
        "Training launched/completed | "
        f"model={summary['model']} imgsz={summary['imgsz']} batch={summary['batch']} epochs={summary['epochs']} | "
        f"dataset_yaml={summary['dataset_yaml']} | run_dir={summary['run_dir']} | "
        f"wandb={summary['wandb_enabled']}"
    )


if __name__ == "__main__":
    main()
