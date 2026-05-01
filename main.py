from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_predictions, run_sahi_inference_on_test, write_metrics_csv
from src.index import build_manifest
from src.preprocess import build_tiled_yolo_dataset
from src.train import create_dataset_yaml, load_config, train_yolo


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pine cone detection pipeline controller")
	subparsers = parser.add_subparsers(dest="command", required=True)

	index_parser = subparsers.add_parser(
		"index",
		help="Pair raw images with masks and write a manifest CSV",
	)
	index_parser.add_argument(
		"--images-dir",
		type=Path,
		default=Path("data/raw"),
		help="Directory containing original aerial images",
	)
	index_parser.add_argument(
		"--masks-dir",
		type=Path,
		default=Path("data/masks"),
		help="Directory containing binary mask images",
	)
	index_parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/manifest.csv"),
		help="Output CSV manifest path",
	)


	preprocess_parser = subparsers.add_parser(
		"preprocess",
		help="Build tiled YOLO dataset with ROI filtering and image-level split",
	)
	preprocess_parser.add_argument(
		"--manifest",
		type=Path,
		default=Path("data/manifest.csv"),
		help="Path to image/mask manifest CSV",
	)
	preprocess_parser.add_argument(
		"--config",
		type=Path,
		default=Path("configs/default.yaml"),
		help="Path to YAML config",
	)
	preprocess_parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("data/yolo_dataset"),
		help="Output root directory for tiled dataset",
	)
	preprocess_parser.add_argument(
		"--class-id",
		type=int,
		default=0,
		help="YOLO class id",
	)
	preprocess_parser.add_argument(
		"--min-contour-area",
		type=float,
		default=1.0,
		help="Minimum contour area when converting masks to boxes",
	)


	train_parser = subparsers.add_parser(
		"train",
		help="Fine-tune YOLOv8 on tiled dataset",
	)
	train_parser.add_argument(
		"--config",
		type=Path,
		default=Path("configs/default.yaml"),
		help="Path to YAML config",
	)
	train_parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("data/yolo_dataset"),
		help="Path to tiled YOLO dataset root",
	)
	train_parser.add_argument(
		"--model-size",
		type=str,
		default="s",
		choices=["n", "nano", "yolov8n", "yolo26n", "s", "small", "m", "medium"],
		help="YOLOv8 model size",
	)
	train_parser.add_argument("--epochs", type=int, default=None)
	train_parser.add_argument("--imgsz", type=int, default=None)
	train_parser.add_argument("--batch", type=int, default=None)
	train_parser.add_argument("--device", type=str, default=None)
	train_parser.add_argument("--workers", type=int, default=None)
	train_parser.add_argument("--seed", type=int, default=None)
	train_parser.add_argument("--run-name", type=str, default=None)
	train_parser.add_argument("--class-name", type=str, default="pine_cone")
	train_parser.add_argument(
		"--log-backend",
		type=str,
		default="tensorboard",
		choices=["tensorboard", "wandb"],
	)


	eval_parser = subparsers.add_parser(
		"evaluate",
		help="Run SAHI sliced inference and compute metrics on test images",
	)
	eval_parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("fine-tuned-models/yolov8small-best.pt"),
		help="Path to trained YOLO weights (best.pt)",
	)
	eval_parser.add_argument(
		"--manifest",
		type=Path,
		default=Path("data/manifest.csv"),
		help="Path to manifest CSV",
	)
	eval_parser.add_argument(
		"--test-ids",
		type=Path,
		default=Path("data/yolo_dataset/test_ids.txt"),
		help="Path to test image ids file",
	)
	eval_parser.add_argument(
		"--dataset-scope",
		type=str,
		default="test",
		choices=["test", "all"],
		help="Evaluate only test ids or all entries in manifest",
	)
	eval_parser.add_argument("--slice-size", type=int, default=640)
	eval_parser.add_argument("--overlap", type=float, default=0.20)
	eval_parser.add_argument("--device", type=str, default="cpu")
	eval_parser.add_argument("--nms-iou-threshold", type=float, nargs="+", default=[0.30, 0.50, 0.70])
	eval_parser.add_argument("--iou-match-threshold", type=float, nargs="+", default=[0.30, 0.50, 0.70])
	eval_parser.add_argument("--conf-thresholds", type=float, nargs="+", default=[0.10, 0.25, 0.40, 0.50, 0.60])
	eval_parser.add_argument("--output-csv", type=Path, default=None)

	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.command == "index":
		summary = build_manifest(
			images_dir=args.images_dir,
			masks_dir=args.masks_dir,
			output_csv=args.output,
		)
		print(
			"Created manifest: "
			f"{summary['output_csv']} | pairs={summary['paired']} "
			f"| missing_masks={summary['missing_masks']} "
			f"| missing_images={summary['missing_images']}"
		)

	if args.command == "preprocess":
		summary = build_tiled_yolo_dataset(
			manifest_csv=args.manifest,
			config_path=args.config,
			output_root=args.output_root,
			class_id=args.class_id,
			min_contour_area=args.min_contour_area,
		)
		print(
			"Preprocess complete: "
			f"images={summary['total_source_images']} "
			f"(train/val/test={summary['train_images']}/{summary['val_images']}/{summary['test_images']}) "
			f"| tiles={summary['train_tiles'] + summary['val_tiles'] + summary['test_tiles']} "
			f"| labels={summary['train_labels'] + summary['val_labels'] + summary['test_labels']}"
		)

	if args.command == "train":
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
			"Train complete: "
			f"model={summary['model']} imgsz={summary['imgsz']} batch={summary['batch']} epochs={summary['epochs']} "
			f"| run_dir={summary['run_dir']} | dataset_yaml={summary['dataset_yaml']}"
		)

	if args.command == "evaluate":
		output_csv = args.output_csv
		if output_csv is None:
			if args.dataset_scope == "all":
				output_csv = Path("runs/eval/metrics_global.csv")
			else:
				output_csv = Path("runs/eval/metrics_test.csv")

		rows = []
		num_images = 0
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
			num_images = len(per_image_predictions)
			rows.extend(
				evaluate_predictions(
					per_image_predictions=per_image_predictions,
					conf_thresholds=args.conf_thresholds,
					iou_match_thresholds=args.iou_match_threshold,
					nms_iou_threshold=nms_iou_threshold,
				)
			)
		write_metrics_csv(output_csv, rows)
		print(
			f"Evaluate complete: dataset_scope={args.dataset_scope} "
			f"| num_images={num_images} | metrics_csv={output_csv}"
		)


if __name__ == "__main__":
	main()
