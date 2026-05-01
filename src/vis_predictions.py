from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import _extract_sahi_prediction_boxes, _to_xyxy_from_gt
from src.preprocess import load_manifest_entries, mask_to_bboxes

# Visualization color palette
COLOR_TP = (0, 255, 0)  # Green
COLOR_FP = (0, 0, 255)  # Red
COLOR_FN = (0, 255, 255)  # Yellow
THICKNESS_FINAL = 8


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


def _greedy_match_indices(
	pred_boxes: Sequence[Tuple[float, float, float, float]],
	gt_boxes: Sequence[Tuple[float, float, float, float]],
	iou_match_threshold: float,
) -> Tuple[set[int], set[int], List[float]]:
	pairs: List[Tuple[float, int, int]] = []
	for pred_idx, pred in enumerate(pred_boxes):
		for gt_idx, gt in enumerate(gt_boxes):
			iou = _box_iou(pred, gt)
			if iou >= iou_match_threshold:
				pairs.append((iou, pred_idx, gt_idx))

	pairs.sort(key=lambda x: x[0], reverse=True)

	matched_preds: set[int] = set()
	matched_gts: set[int] = set()
	matched_ious: List[float] = []

	for iou, pred_idx, gt_idx in pairs:
		if pred_idx in matched_preds or gt_idx in matched_gts:
			continue
		matched_preds.add(pred_idx)
		matched_gts.add(gt_idx)
		matched_ious.append(iou)

	return matched_preds, matched_gts, matched_ious


def _draw_boxes_xyxy(image, boxes_xyxy: Sequence[Tuple[float, float, float, float]], color, thickness: int) -> None:
	for x1, y1, x2, y2 in boxes_xyxy:
		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def _overlay_stats_panel(image, tp: int, fp: int, fn: int, precision: float, recall: float, f1: float) -> None:
	info_text = f"F1: {f1:.2f} (P:{precision:.2f}, R:{recall:.2f})"
	cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
	cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

	legend_y = image.shape[0] - 15
	x1 = image.shape[1] - 170
	x2 = image.shape[1] - 5
	cv2.rectangle(image, (x1, legend_y - 95), (x2, legend_y + 10), (0, 0, 0), -1)
	cv2.putText(image, f"TP ({tp})", (x1 + 10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TP, 2)
	cv2.putText(image, f"FP ({fp})", (x1 + 10, legend_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_FP, 2)
	cv2.putText(image, f"FN ({fn})", (x1 + 10, legend_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_FN, 2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize SAHI predictions on all manifest images")
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
		help="Path to manifest CSV (all rows will be processed)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("figures/final_detection"),
		help="Directory for final visualization images",
	)
	parser.add_argument(
		"--json-dir",
		type=Path,
		default=Path("runs/eval/per_image"),
		help="Directory for optional per-image JSON summaries",
	)
	parser.add_argument("--save-json", action="store_true", help="Save TP/FP/FN results per image as JSON")
	parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size")
	parser.add_argument("--overlap", type=float, default=0.20, help="SAHI overlap ratio")
	parser.add_argument("--conf-threshold", type=float, default=0.45, help="Confidence threshold for drawing/eval")
	parser.add_argument("--iou-match-threshold", type=float, default=0.1, help="IoU threshold for TP/FP/FN")
	parser.add_argument("--nms-iou-threshold", type=float, default=0.2, help="SAHI NMS IoU threshold")
	parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g. cuda:0 or cpu")
	parser.add_argument("--max-images", type=int, default=None, help="Optional max number of images to process")
	parser.add_argument(
		"--summary-csv",
		type=Path,
		default=Path("runs/eval-final-v8s/per_image_summary.csv"),
		help="Output CSV with one summary row per image",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	try:
		from sahi import AutoDetectionModel
		from sahi.predict import get_sliced_prediction
	except ImportError as exc:  # pragma: no cover - explicit runtime dependency error
		raise ImportError("SAHI is required. Install it with: pip install sahi") from exc

	entries = load_manifest_entries(args.manifest)
	if args.max_images is not None:
		entries = entries[: args.max_images]

	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	json_dir = args.json_dir.resolve()
	if args.save_json:
		json_dir.mkdir(parents=True, exist_ok=True)

	detection_model = AutoDetectionModel.from_pretrained(
		model_type="ultralytics",
		model_path=str(args.model_path),
		confidence_threshold=0.001,
		device=args.device,
	)

	total_tp = 0
	total_fp = 0
	total_fn = 0
	per_image_rows: List[Dict[str, str]] = []

	print(f"Visualizing predictions on {len(entries)} images...")
	for entry in entries:
		image = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
		if image is None:
			print(f"[WARN] Skipping {entry.image_id}: could not read image {entry.image_path}")
			continue

		gt_bboxes_abs, _, _ = mask_to_bboxes(entry.mask_path, min_contour_area=1.0)
		gt_xyxy = [_to_xyxy_from_gt(x, y, w, h) for (x, y, w, h) in gt_bboxes_abs]

		prediction_result = get_sliced_prediction(
			image=str(entry.image_path),
			detection_model=detection_model,
			slice_height=args.slice_size,
			slice_width=args.slice_size,
			overlap_height_ratio=args.overlap,
			overlap_width_ratio=args.overlap,
			postprocess_type="NMS",
			postprocess_match_metric="IOU",
			postprocess_match_threshold=args.nms_iou_threshold,
			verbose=0,
		)
		pred_boxes = _extract_sahi_prediction_boxes(prediction_result)
		pred_xyxy = [box.xyxy for box in pred_boxes if box.score >= args.conf_threshold]

		matched_preds, matched_gts, _ = _greedy_match_indices(
			pred_boxes=pred_xyxy,
			gt_boxes=gt_xyxy,
			iou_match_threshold=args.iou_match_threshold,
		)

		tp_boxes = [pred_xyxy[i] for i in sorted(matched_preds)]
		fp_boxes = [pred_xyxy[i] for i in range(len(pred_xyxy)) if i not in matched_preds]
		fn_boxes = [gt_xyxy[i] for i in range(len(gt_xyxy)) if i not in matched_gts]

		tp = len(tp_boxes)
		fp = len(fp_boxes)
		fn = len(fn_boxes)
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

		# Same draw order as provided style: FN, FP, TP
		_draw_boxes_xyxy(image, fn_boxes, COLOR_FN, THICKNESS_FINAL)
		_draw_boxes_xyxy(image, fp_boxes, COLOR_FP, THICKNESS_FINAL)
		_draw_boxes_xyxy(image, tp_boxes, COLOR_TP, THICKNESS_FINAL)
		_overlay_stats_panel(image, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

		out_path = output_dir / f"{entry.image_id}_detection_vis.jpg"
		cv2.imwrite(str(out_path), image)

		per_image_rows.append(
			{
				"image_id": entry.image_id,
				"num_gt": str(len(gt_xyxy)),
				"num_pred": str(len(pred_xyxy)),
				"tp": str(tp),
				"fp": str(fp),
				"fn": str(fn),
				"precision": f"{precision:.6f}",
				"recall": f"{recall:.6f}",
				"f1": f"{f1:.6f}",
			}
		)

		if args.save_json:
			payload = {
				"image_id": entry.image_id,
				"image_path": str(entry.image_path),
				"mask_path": str(entry.mask_path),
				"conf_threshold": args.conf_threshold,
				"iou_match_threshold": args.iou_match_threshold,
				"detections_tp": tp_boxes,
				"detections_fp": fp_boxes,
				"ground_truth_fn": fn_boxes,
				"metrics": {
					"tp": tp,
					"fp": fp,
					"fn": fn,
					"precision": precision,
					"recall": recall,
					"f1": f1,
				},
			}
			(json_dir / f"{entry.image_id}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

		total_tp += tp
		total_fp += fp
		total_fn += fn
		print(f"  - {entry.image_id}: TP={tp} FP={fp} FN={fn} | F1={f1:.3f}")

	global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
	global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
	global_f1 = (
		(2.0 * global_precision * global_recall) / (global_precision + global_recall)
		if (global_precision + global_recall) > 0
		else 0.0
	)

	args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
	with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
		fieldnames = [
			"image_id",
			"num_gt",
			"num_pred",
			"tp",
			"fp",
			"fn",
			"precision",
			"recall",
			"f1",
		]
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(per_image_rows)

	print("\n[OK] Visualization complete")
	print(f"Output images: {output_dir}")
	print(f"Per-image CSV: {args.summary_csv.resolve()}")
	if args.save_json:
		print(f"Per-image JSON: {json_dir}")
	print(
		f"Global metrics at conf={args.conf_threshold:.2f}, IoU={args.iou_match_threshold:.2f} | "
		f"TP={total_tp} FP={total_fp} FN={total_fn} | "
		f"P={global_precision:.4f} R={global_recall:.4f} F1={global_f1:.4f}"
	)


if __name__ == "__main__":
	main()
