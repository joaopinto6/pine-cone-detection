from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import load_manifest_entries


def _should_sync_cuda(device: str) -> bool:
    device = str(device).lower()
    return device.startswith("cuda") or device.isdigit()


def _cuda_sync_if_needed(device: str) -> None:
    if not _should_sync_cuda(device):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        # If torch/cuda sync is not available, continue without hard failure.
        pass


def _timed_predict(
    detection_model,
    image_path: Path,
    device: str,
    slice_size: int,
    overlap: float,
    nms_iou_threshold: float,
) -> float:
    from sahi.predict import get_sliced_prediction

    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    get_sliced_prediction(
        image=str(image_path),
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=nms_iou_threshold,
        verbose=False,
    )
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _build_stats(latencies_ms: List[float]) -> Dict[str, float]:
    avg_ms = statistics.fmean(latencies_ms)
    std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
    }


def benchmark_model(
    model_name: str,
    model_path: str,
    image_paths: List[Path],
    device: str,
    slice_size: int,
    conf: float,
    overlap: float,
    nms_iou_threshold: float,
    warmup: int,
) -> Dict[str, float]:
    try:
        from sahi import AutoDetectionModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SAHI is required. Install it with: pip install sahi") from exc

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    if not image_paths:
        raise ValueError("No images to benchmark")

    warmup_image = image_paths[0]
    for _ in range(max(0, warmup)):
        _timed_predict(
            detection_model,
            warmup_image,
            device=device,
            slice_size=slice_size,
            overlap=overlap,
            nms_iou_threshold=nms_iou_threshold,
        )

    latencies_ms: List[float] = []
    for image_path in image_paths:
        latencies_ms.append(
            _timed_predict(
                detection_model,
                image_path=image_path,
                device=device,
                slice_size=slice_size,
                overlap=overlap,
                nms_iou_threshold=nms_iou_threshold,
            )
        )

    stats = _build_stats(latencies_ms)
    stats["num_images"] = float(len(image_paths))
    stats["model_name"] = model_name
    stats["model_path"] = model_path
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark average inference speed per image: YOLOv8n vs YOLOv8s (SAHI sliced inference)")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"), help="Path to manifest CSV")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (cpu, 0, cuda:0)")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.20, help="SAHI overlap ratio")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.25, help="SAHI NMS IoU threshold")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per model")
    parser.add_argument("--model-n", type=str, default="fine-tuned-models/yolov8nano-best.pt", help="YOLOv8n model path")
    parser.add_argument("--model-s", type=str, default="fine-tuned-models/yolov8small-best.pt", help="YOLOv8s model path")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/bench/inference_speed_comparison.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = load_manifest_entries(args.manifest)
    if args.max_images is not None:
        entries = entries[: args.max_images]

    image_paths = [entry.image_path for entry in entries]
    if not image_paths:
        raise ValueError("No images found from manifest")

    print(
        f"Benchmarking on {len(image_paths)} images | device={args.device} | "
        f"slice={args.slice_size} | overlap={args.overlap:.2f}"
    )

    results = [
        benchmark_model(
            model_name="yolov8n",
            model_path=args.model_n,
            image_paths=image_paths,
            device=args.device,
            slice_size=args.slice_size,
            conf=args.conf,
            overlap=args.overlap,
            nms_iou_threshold=args.nms_iou_threshold,
            warmup=args.warmup,
        ),
        benchmark_model(
            model_name="yolov8s",
            model_path=args.model_s,
            image_paths=image_paths,
            device=args.device,
            slice_size=args.slice_size,
            conf=args.conf,
            overlap=args.overlap,
            nms_iou_threshold=args.nms_iou_threshold,
            warmup=args.warmup,
        ),
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "model_path",
                "num_images",
                "slice_size",
                "overlap",
                "nms_iou_threshold",
                "avg_ms",
                "std_ms",
                "min_ms",
                "max_ms",
                "fps",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["model_name"],
                    row["model_path"],
                    int(row["num_images"]),
                    args.slice_size,
                    f"{args.overlap:.4f}",
                    f"{args.nms_iou_threshold:.4f}",
                    f"{row['avg_ms']:.4f}",
                    f"{row['std_ms']:.4f}",
                    f"{row['min_ms']:.4f}",
                    f"{row['max_ms']:.4f}",
                    f"{row['fps']:.4f}",
                ]
            )

    speedup = results[1]["avg_ms"] / results[0]["avg_ms"] if results[0]["avg_ms"] > 0 else float("inf")

    for row in results:
        print(
            f"{row['model_name']}: avg={row['avg_ms']:.2f} ms | std={row['std_ms']:.2f} | "
            f"min={row['min_ms']:.2f} | max={row['max_ms']:.2f} | fps={row['fps']:.2f}"
        )

    print(f"Relative latency (yolov8s / yolov8n): {speedup:.3f}x")
    print(f"Saved benchmark CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
