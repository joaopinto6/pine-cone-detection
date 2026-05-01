from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - explicit runtime dependency error
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.roi import canopy_roi_bgr


@dataclass(frozen=True)
class ManifestEntry:
    image_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ImageAnnotations:
    image_id: str
    image_path: Path
    mask_path: Path
    image_width: int
    image_height: int
    bboxes_abs_xywh: List[Tuple[int, int, int, int]]
    yolo_boxes: List[YoloBox]


@dataclass
class SplitStats:
    source_images: int = 0
    total_windows: int = 0
    kept_tiles: int = 0
    discarded_by_roi: int = 0
    total_labels: int = 0


def _resolve_manifest_path(path_text: str, manifest_path: Path, project_root: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate

    from_project_root = (project_root / candidate).resolve()
    if from_project_root.exists():
        return from_project_root

    # Handle manifest paths relative to the CSV.
    return (manifest_path.parent / candidate).resolve()


def load_manifest_entries(manifest_csv: Path, project_root: Path | None = None) -> List[ManifestEntry]:
    """Load image/mask pairs from a manifest CSV file."""
    manifest_csv = manifest_csv.resolve()
    project_root = project_root.resolve() if project_root else Path.cwd().resolve()

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    entries: List[ManifestEntry] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"id", "image_path", "mask_path"}
        if reader.fieldnames is None or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Manifest must contain columns {sorted(required_fields)}; got {reader.fieldnames}"
            )

        for row in reader:
            image_id = row["id"].strip()
            image_path = _resolve_manifest_path(row["image_path"], manifest_csv, project_root)
            mask_path = _resolve_manifest_path(row["mask_path"], manifest_csv, project_root)

            if not image_path.exists():
                raise FileNotFoundError(f"Image path from manifest does not exist: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask path from manifest does not exist: {mask_path}")

            entries.append(
                ManifestEntry(
                    image_id=image_id,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

    return entries


def mask_to_bboxes(mask_path: Path, min_contour_area: float = 1.0) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    """Extract contour bounding boxes from a binary mask image."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask image: {mask_path}")

    image_height, image_width = mask.shape[:2]

    # Convert mask to binary format.
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        bboxes.append((x, y, w, h))

    return bboxes, image_width, image_height


def xywh_to_yolo(x: int, y: int, w: int, h: int, image_width: int, image_height: int, class_id: int = 0) -> YoloBox:
    """Convert an absolute XYWH box to normalized YOLO format."""
    x_center = (x + (w / 2.0)) / float(image_width)
    y_center = (y + (h / 2.0)) / float(image_height)
    box_width = w / float(image_width)
    box_height = h / float(image_height)

    return YoloBox(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=box_width,
        height=box_height,
    )


def build_annotations_in_memory(
    manifest_csv: Path,
    class_id: int = 0,
    min_contour_area: float = 1.0,
    project_root: Path | None = None,
) -> List[ImageAnnotations]:
    """Build per-image absolute and YOLO-normalized annotations from mask contours."""
    entries = load_manifest_entries(manifest_csv=manifest_csv, project_root=project_root)

    annotations: List[ImageAnnotations] = []
    for entry in entries:
        bboxes_abs, image_width, image_height = mask_to_bboxes(
            mask_path=entry.mask_path,
            min_contour_area=min_contour_area,
        )

        yolo_boxes = [
            xywh_to_yolo(
                x=x,
                y=y,
                w=w,
                h=h,
                image_width=image_width,
                image_height=image_height,
                class_id=class_id,
            )
            for (x, y, w, h) in bboxes_abs
        ]

        annotations.append(
            ImageAnnotations(
                image_id=entry.image_id,
                image_path=entry.image_path,
                mask_path=entry.mask_path,
                image_width=image_width,
                image_height=image_height,
                bboxes_abs_xywh=bboxes_abs,
                yolo_boxes=yolo_boxes,
            )
        )

    return annotations


def yolo_boxes_to_array(boxes: Sequence[YoloBox]) -> np.ndarray:
    """Convert in-memory YOLO boxes to a numeric Nx5 array for downstream steps."""
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)

    return np.asarray(
        [[b.class_id, b.x_center, b.y_center, b.width, b.height] for b in boxes],
        dtype=np.float32,
    )


def load_config(config_path: Path) -> Dict:
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must define a mapping: {config_path}")

    return cfg


def _compute_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _iter_tile_windows(image_width: int, image_height: int, tile_size: int, overlap_ratio: float) -> Iterable[Tuple[int, int]]:
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(f"overlap_ratio must be in [0, 1): {overlap_ratio}")

    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    x_starts = _compute_starts(image_width, tile_size, stride)
    y_starts = _compute_starts(image_height, tile_size, stride)

    for y in y_starts:
        for x in x_starts:
            yield x, y


def _bbox_center_inside_tile(x: int, y: int, w: int, h: int, tile_x: int, tile_y: int, tile_size: int) -> bool:
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    return (tile_x <= cx < (tile_x + tile_size)) and (tile_y <= cy < (tile_y + tile_size))


def _clip_bbox_to_tile(x: int, y: int, w: int, h: int, tile_x: int, tile_y: int, tile_size: int) -> Tuple[int, int, int, int] | None:
    x1 = max(x, tile_x)
    y1 = max(y, tile_y)
    x2 = min(x + w, tile_x + tile_size)
    y2 = min(y + h, tile_y + tile_size)

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    local_x = x1 - tile_x
    local_y = y1 - tile_y
    return local_x, local_y, clipped_w, clipped_h


def split_entries_image_level(
    entries: List[ManifestEntry],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[ManifestEntry]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    shuffled = list(entries)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_entries = shuffled[:train_count]
    val_entries = shuffled[train_count : train_count + val_count]
    test_entries = shuffled[train_count + val_count : train_count + val_count + test_count]

    return {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }


def _write_yolo_label_file(label_path: Path, boxes: Sequence[YoloBox]) -> None:
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _process_entry_to_tiles(
    entry: ManifestEntry,
    split_name: str,
    split_dirs: Dict[str, Path],
    cfg: Dict,
    class_id: int,
    min_contour_area: float,
) -> Tuple[List[Dict], SplitStats]:
    tile_size = int(cfg["preprocess"]["tile_size"])
    overlap_ratio = float(cfg["preprocess"]["tile_overlap"])
    canopy_min_ratio = float(cfg["preprocess"]["canopy_min_ratio"])

    image_bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {entry.image_path}")

    image_height, image_width = image_bgr.shape[:2]
    canopy_mask = canopy_roi_bgr(image_bgr, cfg)
    bboxes_abs, _, _ = mask_to_bboxes(entry.mask_path, min_contour_area=min_contour_area)

    images_dir = split_dirs["images"]
    labels_dir = split_dirs["labels"]
    metadata_rows: List[Dict] = []

    stats = SplitStats(source_images=1)
    for tile_x, tile_y in _iter_tile_windows(image_width, image_height, tile_size, overlap_ratio):
        stats.total_windows += 1

        tile_roi = canopy_mask[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
        canopy_ratio = float(np.count_nonzero(tile_roi)) / float(tile_size * tile_size)
        if canopy_ratio < canopy_min_ratio:
            stats.discarded_by_roi += 1
            continue

        tile_image = image_bgr[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
        tile_stem = f"{entry.image_id}_x{tile_x}_y{tile_y}"
        tile_image_path = images_dir / f"{tile_stem}.jpg"
        tile_label_path = labels_dir / f"{tile_stem}.txt"

        tile_boxes: List[YoloBox] = []
        for x, y, w, h in bboxes_abs:
            if not _bbox_center_inside_tile(x, y, w, h, tile_x, tile_y, tile_size):
                continue

            clipped = _clip_bbox_to_tile(x, y, w, h, tile_x, tile_y, tile_size)
            if clipped is None:
                continue

            local_x, local_y, local_w, local_h = clipped
            tile_boxes.append(
                xywh_to_yolo(
                    x=local_x,
                    y=local_y,
                    w=local_w,
                    h=local_h,
                    image_width=tile_size,
                    image_height=tile_size,
                    class_id=class_id,
                )
            )

        cv2.imwrite(str(tile_image_path), tile_image)
        _write_yolo_label_file(tile_label_path, tile_boxes)

        stats.kept_tiles += 1
        stats.total_labels += len(tile_boxes)

        metadata_rows.append(
            {
                "split": split_name,
                "image_id": entry.image_id,
                "tile_stem": tile_stem,
                "tile_image": str(tile_image_path),
                "tile_label": str(tile_label_path),
                "canopy_ratio": f"{canopy_ratio:.6f}",
                "num_labels": str(len(tile_boxes)),
            }
        )

    return metadata_rows, stats


def build_tiled_yolo_dataset(
    manifest_csv: Path,
    config_path: Path,
    output_root: Path,
    class_id: int = 0,
    min_contour_area: float = 1.0,
) -> Dict:
    cfg = load_config(config_path)
    entries = load_manifest_entries(manifest_csv)

    preprocess_cfg = cfg.get("preprocess", {})
    split_map = split_entries_image_level(
        entries=entries,
        train_ratio=float(preprocess_cfg.get("train_ratio", 0.70)),
        val_ratio=float(preprocess_cfg.get("val_ratio", 0.15)),
        test_ratio=float(preprocess_cfg.get("test_ratio", 0.15)),
        seed=int(preprocess_cfg.get("split_seed", 42)),
    )

    output_root = output_root.resolve()
    images_root = output_root / "images"
    labels_root = output_root / "labels"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_rows: List[Dict[str, str]] = []
    summary: Dict[str, SplitStats] = {
        "train": SplitStats(),
        "val": SplitStats(),
        "test": SplitStats(),
    }

    for split_name in ("train", "val", "test"):
        split_images_dir = images_root / split_name
        split_labels_dir = labels_root / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        split_entries = split_map[split_name]
        (output_root / f"{split_name}_ids.txt").write_text(
            "\n".join(entry.image_id for entry in split_entries),
            encoding="utf-8",
        )

        for entry in split_entries:
            rows, entry_stats = _process_entry_to_tiles(
                entry=entry,
                split_name=split_name,
                split_dirs={"images": split_images_dir, "labels": split_labels_dir},
                cfg=cfg,
                class_id=class_id,
                min_contour_area=min_contour_area,
            )
            metadata_rows.extend(rows)

            split_acc = summary[split_name]
            split_acc.source_images += entry_stats.source_images
            split_acc.total_windows += entry_stats.total_windows
            split_acc.kept_tiles += entry_stats.kept_tiles
            split_acc.discarded_by_roi += entry_stats.discarded_by_roi
            split_acc.total_labels += entry_stats.total_labels

    metadata_csv = output_root / "tiles_manifest.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "image_id", "tile_stem", "tile_image", "tile_label", "canopy_ratio", "num_labels"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    return {
        "output_root": str(output_root),
        "metadata_csv": str(metadata_csv),
        "total_source_images": len(entries),
        "train_images": len(split_map["train"]),
        "val_images": len(split_map["val"]),
        "test_images": len(split_map["test"]),
        "train_tiles": summary["train"].kept_tiles,
        "val_tiles": summary["val"].kept_tiles,
        "test_tiles": summary["test"].kept_tiles,
        "train_labels": summary["train"].total_labels,
        "val_labels": summary["val"].total_labels,
        "test_labels": summary["test"].total_labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess masks and build a tiled YOLO dataset")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"), help="Path to manifest CSV")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to config YAML")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Output dataset root directory",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class id for YOLO labels")
    parser.add_argument("--min-contour-area", type=float, default=1.0, help="Minimum contour area in mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_tiled_yolo_dataset(
        manifest_csv=args.manifest,
        config_path=args.config,
        output_root=args.output_root,
        class_id=args.class_id,
        min_contour_area=args.min_contour_area,
    )
    print(
        "Preprocess complete | "
        f"images={summary['total_source_images']} "
        f"(train/val/test={summary['train_images']}/{summary['val_images']}/{summary['test_images']}) | "
        f"tiles={summary['train_tiles'] + summary['val_tiles'] + summary['test_tiles']} | "
        f"labels={summary['train_labels'] + summary['val_labels'] + summary['test_labels']} | "
        f"metadata={summary['metadata_csv']}"
    )


if __name__ == "__main__":
    main()
