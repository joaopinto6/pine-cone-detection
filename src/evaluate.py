from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import load_manifest_entries, mask_to_bboxes


@dataclass(frozen=True)
class PredBox:
    xyxy: Tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class MetricsRow:
    conf_threshold: float
    iou_match_threshold: float
    nms_iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float


def _read_test_ids(test_ids_path: Path) -> List[str]:
    if not test_ids_path.exists():
        raise FileNotFoundError(f"Test ids file not found: {test_ids_path}")

    ids = [line.strip() for line in test_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No ids found in: {test_ids_path}")
    return ids


def _load_test_entries(manifest_csv: Path, test_ids_path: Path) -> List:
    entries = load_manifest_entries(manifest_csv)
    by_id = {entry.image_id: entry for entry in entries}

    test_ids = _read_test_ids(test_ids_path)
    missing_ids = [image_id for image_id in test_ids if image_id not in by_id]
    if missing_ids:
        raise ValueError(f"These test ids are missing from manifest: {missing_ids}")

    # Use only the requested test IDs in the specified order.
    return [by_id[image_id] for image_id in test_ids]


def _load_entries_by_scope(manifest_csv: Path, dataset_scope: str, test_ids_path: Path) -> List:
    if dataset_scope == "all":
        return load_manifest_entries(manifest_csv)
    if dataset_scope == "test":
        return _load_test_entries(manifest_csv=manifest_csv, test_ids_path=test_ids_path)
    raise ValueError(f"Unsupported dataset_scope: {dataset_scope}")


def _to_xyxy_from_gt(x: int, y: int, w: int, h: int) -> Tuple[float, float, float, float]:
    return float(x), float(y), float(x + w), float(y + h)


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def _greedy_match(
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    gt_boxes: Sequence[Tuple[float, float, float, float]],
    iou_match_threshold: float,
) -> Tuple[int, int, int, List[float]]:
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, []

    pairs: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            iou = _box_iou(pred, gt)
            if iou >= iou_match_threshold:
                pairs.append((iou, pred_idx, gt_idx))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_preds = set()
    matched_gts = set()
    matched_ious: List[float] = []

    for iou, pred_idx, gt_idx in pairs:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matched_ious.append(iou)

    tp = len(matched_preds)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_ious


def _extract_sahi_prediction_boxes(prediction_result) -> List[PredBox]:
    boxes: List[PredBox] = []

    for obj_pred in prediction_result.object_prediction_list:
        score = float(obj_pred.score.value)

        # Handle SAHI bbox format variants.
        if hasattr(obj_pred.bbox, "to_xyxy"):
            x1, y1, x2, y2 = obj_pred.bbox.to_xyxy()
        else:
            x1 = float(obj_pred.bbox.minx)
            y1 = float(obj_pred.bbox.miny)
            x2 = float(obj_pred.bbox.maxx)
            y2 = float(obj_pred.bbox.maxy)

        boxes.append(PredBox(xyxy=(float(x1), float(y1), float(x2), float(y2)), score=score))

    return boxes


def run_sahi_inference_on_test(
    model_path: Path,
    manifest_csv: Path,
    test_ids_path: Path,
    dataset_scope: str,
    slice_size: int,
    overlap_ratio: float,
    device: str,
    nms_iou_threshold: float,
) -> Dict[str, Dict[str, List]]:
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:  # pragma: no cover - explicit runtime dependency error
        raise ImportError("SAHI is required. Install it with: pip install sahi") from exc

    eval_entries = _load_entries_by_scope(
        manifest_csv=manifest_csv,
        dataset_scope=dataset_scope,
        test_ids_path=test_ids_path,
    )

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=0.001,
        device=device,
    )

    per_image = {}
    for entry in eval_entries:
        gt_bboxes_abs, _, _ = mask_to_bboxes(entry.mask_path, min_contour_area=1.0)
        gt_xyxy = [_to_xyxy_from_gt(x, y, w, h) for (x, y, w, h) in gt_bboxes_abs]

        prediction_result = get_sliced_prediction(
            image=str(entry.image_path),
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=nms_iou_threshold,
            verbose=0,
        )
        pred_boxes = _extract_sahi_prediction_boxes(prediction_result)

        per_image[entry.image_id] = {
            "gt_xyxy": gt_xyxy,
            "pred_boxes": pred_boxes,
            "image_path": str(entry.image_path),
        }

    return per_image


def evaluate_predictions(
    per_image_predictions: Dict[str, Dict[str, List]],
    conf_thresholds: Sequence[float],
    iou_match_thresholds: Sequence[float],
    nms_iou_threshold: float,
) -> List[MetricsRow]:
    rows: List[MetricsRow] = []

    for iou_match_threshold in iou_match_thresholds:
        for conf_th in conf_thresholds:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            all_matched_ious: List[float] = []

            for image_data in per_image_predictions.values():
                gt_xyxy = image_data["gt_xyxy"]
                pred_boxes = [box.xyxy for box in image_data["pred_boxes"] if box.score >= conf_th]

                tp, fp, fn, matched_ious = _greedy_match(
                    pred_boxes=pred_boxes,
                    gt_boxes=gt_xyxy,
                    iou_match_threshold=iou_match_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_matched_ious.extend(matched_ious)

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

            rows.append(
                MetricsRow(
                    conf_threshold=float(conf_th),
                    iou_match_threshold=float(iou_match_threshold),
                    nms_iou_threshold=float(nms_iou_threshold),
                    true_positives=total_tp,
                    false_positives=total_fp,
                    false_negatives=total_fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    mean_iou=mean_iou,
                )
            )

    return rows


def write_metrics_csv(output_csv: Path, rows: Sequence[MetricsRow]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "conf_threshold",
                "iou_match_threshold",
                "nms_iou_threshold",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "mean_iou",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row.conf_threshold:.4f}",
                    f"{row.iou_match_threshold:.4f}",
                    f"{row.nms_iou_threshold:.4f}",
                    row.true_positives,
                    row.false_positives,
                    row.false_negatives,
                    f"{row.precision:.6f}",
                    f"{row.recall:.6f}",
                    f"{row.f1:.6f}",
                    f"{row.mean_iou:.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model with SAHI on test images only")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("fine-tuned-models/yolov8small-best.pt"),
        help="Path to trained YOLO weights (best.pt)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--test-ids",
        type=Path,
        default=Path("data/yolo_dataset/test_ids.txt"),
        help="Path to test image ids file",
    )
    parser.add_argument(
        "--dataset-scope",
        type=str,
        default="all",
        choices=["test", "all"],
        help="Evaluate only test ids or all entries in manifest",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=640,
        help="SAHI slice size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="SAHI slice overlap ratio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        help="NMS IoU thresholds used by SAHI postprocess",
    )
    parser.add_argument(
        "--iou-match-threshold",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        help="IoU thresholds for TP/FP/FN matching",
    )
    parser.add_argument(
        "--conf-thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        help="Confidence thresholds for metrics",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV with metrics",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_csv = args.output_csv
    if output_csv is None:
        if args.dataset_scope == "all":
            output_csv = Path("runs/eval-final/v8_metrics_global.csv")
        else:
            output_csv = Path("runs/eval-final/v8_metrics_test.csv")

    rows: List[MetricsRow] = []
    num_eval_images = 0
    for nms_iou_threshold in args.nms_iou_threshold:
        per_image_predictions = run_sahi_inference_on_test(
            model_path=args.model_path,
            manifest_csv=args.manifest,
            test_ids_path=args.test_ids,
            dataset_scope=args.dataset_scope,
            slice_size=args.slice_size,
            overlap_ratio=args.overlap,
            device=args.device,
            nms_iou_threshold=nms_iou_threshold,
        )
        num_eval_images = len(per_image_predictions)

        rows.extend(
            evaluate_predictions(
                per_image_predictions=per_image_predictions,
                conf_thresholds=args.conf_thresholds,
                iou_match_thresholds=args.iou_match_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
        )

    write_metrics_csv(output_csv, rows)

    for row in rows:
        print(
            f"conf={row.conf_threshold:.2f} | "
            f"nms_iou={row.nms_iou_threshold:.2f} | "
            f"TP={row.true_positives} FP={row.false_positives} FN={row.false_negatives} | "
            f"Precision={row.precision:.4f} Recall={row.recall:.4f} F1={row.f1:.4f} "
            f"MeanIoU={row.mean_iou:.4f}"
        )

    print(f"Saved metrics CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
