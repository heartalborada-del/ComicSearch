from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def discover_pack_roots(input_root: Path) -> list[Path]:
    first_level_dirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
    if not first_level_dirs:
        return [input_root]
    return first_level_dirs


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    expand_w = int(width * expand_ratio * 0.5)
    expand_h = int(height * expand_ratio * 0.5)
    h, w = image_shape
    return (
        max(0, x1 - expand_w),
        max(0, y1 - expand_h),
        min(w, x2 + expand_w),
        min(h, y2 + expand_h),
    )


def detect_faces(
    image: np.ndarray,
    detector: YOLO,
    confidence_threshold: float,
    max_detections_per_image: int,
    class_ids: list[int] | None,
) -> list[Detection]:
    detections: list[Detection] = []
    results = detector.predict(
        source=image,
        conf=confidence_threshold,
        classes=class_ids,
        verbose=False,
    )
    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None:
        return detections

    for box in boxes:
        coords = box.xyxy[0].tolist()
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = (int(max(0, round(value))) for value in coords)
        if x2 <= x1 or y2 <= y1:
            continue
        if box.conf is None or int(box.conf.numel()) == 0:
            continue
        score_value = float(box.conf[0].item())
        detections.append(Detection((x1, y1, x2, y2), score_value))

    detections.sort(key=lambda detection: detection.score, reverse=True)
    return detections[:max_detections_per_image]


def parse_yolo_classes(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid --yolo-classes value '{token}': expected integers like '0,1'") from exc
    return values if values else None


def build_face_crops(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")

    pack_roots = discover_pack_roots(input_root)
    output_crop_root = Path(args.output_crop_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_crop_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    yolo_model_path = Path(args.yolo_model).resolve()
    if not yolo_model_path.exists():
        raise FileNotFoundError(
            f"YOLO model file not found: {yolo_model_path} (expected a local .pt or .onnx model file)"
        )
    detector = YOLO(str(yolo_model_path))
    class_ids = parse_yolo_classes(args.yolo_classes)

    written = 0
    scanned = 0
    bbox_expand_ratio = float(args.bbox_expand_ratio)
    with output_manifest.open("w", encoding="utf-8") as manifest_fp:
        for pack_root in pack_roots:
            for image_path in iter_image_paths(pack_root):
                scanned += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                detections = detect_faces(
                    image=image,
                    detector=detector,
                    confidence_threshold=float(args.confidence_threshold),
                    max_detections_per_image=int(args.max_detections_per_image),
                    class_ids=class_ids,
                )
                if not detections:
                    continue

                # Keep crop output aligned with the original directory hierarchy under input_root.
                relative = image_path.relative_to(input_root)
                stem = relative.with_suffix("")
                for index, detection in enumerate(detections):
                    expanded = expand_bbox(detection.bbox, image.shape[:2], bbox_expand_ratio)
                    x1, y1, x2, y2 = expanded
                    if min(x2 - x1, y2 - y1) < int(args.min_crop_size):
                        continue
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_path = output_crop_root / stem.parent / f"{stem.name}_face_crop_{index:02d}.jpg"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(crop_path), crop):
                        continue

                    manifest_fp.write(
                        json.dumps(
                            {
                                "original_image_path": str(image_path),
                                "crop_image_path": str(crop_path),
                                "bbox": [x1, y1, x2, y2],
                                "score": round(detection.score, 6),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_crop_root": str(output_crop_root),
                "output_manifest": str(output_manifest),
                "images_scanned": scanned,
                "crops_written": written,
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build face/head crop subset from manga page images.")
    parser.add_argument("--input-root", required=True, help="Root directory containing source page images.")
    parser.add_argument("--output-crop-root", required=True, help="Output root directory for cropped face images.")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--confidence-threshold", type=float, default=0.35, help="Detection score threshold.")
    parser.add_argument("--min-crop-size", type=int, default=48, help="Minimum crop side length in pixels.")
    parser.add_argument(
        "--max-detections-per-image",
        type=int,
        default=6,
        help="Maximum number of detections kept per image.",
    )
    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.15,
        help="Bounding box expansion ratio before cropping.",
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to YOLO model file (.pt/.onnx).",
    )
    parser.add_argument(
        "--yolo-classes",
        default=None,
        help="Optional comma-separated class ids to keep (e.g. '0,1'). Defaults to all classes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_face_crops(parse_args())
