from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _list_images(folder: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _index_by_stem(paths: Iterable[Path]) -> Dict[str, Path]:
    indexed: Dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if stem in indexed:
            raise ValueError(
                f"Duplicate file stem '{stem}' found: '{indexed[stem]}' and '{path}'. "
                "Ensure unique base names before indexing."
            )
        indexed[stem] = path
    return indexed


def _to_manifest_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def build_manifest(images_dir: Path, masks_dir: Path, output_csv: Path) -> dict:
    """Pair images and masks by filename stem and write a CSV manifest."""
    images_dir = images_dir.resolve()
    masks_dir = masks_dir.resolve()
    output_csv = output_csv.resolve()
    project_root = Path.cwd().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    images_by_stem = _index_by_stem(_list_images(images_dir, recursive=True))
    masks_by_stem = _index_by_stem(_list_images(masks_dir, recursive=True))

    shared_stems = sorted(set(images_by_stem) & set(masks_by_stem))
    image_only_stems = sorted(set(images_by_stem) - set(masks_by_stem))
    mask_only_stems = sorted(set(masks_by_stem) - set(images_by_stem))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_path", "mask_path"])
        for stem in shared_stems:
            image_path = images_by_stem[stem]
            mask_path = masks_by_stem[stem]
            writer.writerow(
                [
                    stem,
                    _to_manifest_path(image_path, project_root),
                    _to_manifest_path(mask_path, project_root),
                ]
            )

    return {
        "output_csv": str(output_csv),
        "paired": len(shared_stems),
        "missing_masks": len(image_only_stems),
        "missing_images": len(mask_only_stems),
    }
