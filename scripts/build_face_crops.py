from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


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
    gray_image: np.ndarray,
    detector: cv2.CascadeClassifier,
    confidence_threshold: float,
    max_detections_per_image: int,
) -> list[Detection]:
    detections: list[Detection] = []
    if hasattr(detector, "detectMultiScale3"):
        rects, _, level_weights = detector.detectMultiScale3(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=3,
            outputRejectLevels=True,
        )
        for (x, y, w, h), score in zip(rects, level_weights):
            score_value = float(score)
            if score_value < confidence_threshold:
                continue
            detections.append(Detection((int(x), int(y), int(x + w), int(y + h)), score_value))
    else:
        rects = detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(24, 24),
        )
        detections.extend(
            Detection((int(x), int(y), int(x + w), int(y + h)), 1.0)
            for (x, y, w, h) in rects
        )

    detections.sort(key=lambda detection: detection.score, reverse=True)
    return detections[:max_detections_per_image]


def build_face_crops(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_crop_root = Path(args.output_crop_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_crop_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"failed to load haarcascade detector: {cascade_path}")

    written = 0
    scanned = 0
    with output_manifest.open("w", encoding="utf-8") as manifest_fp:
        for image_path in iter_image_paths(input_root):
            scanned += 1
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = detect_faces(
                gray_image=gray,
                detector=detector,
                confidence_threshold=float(args.confidence_threshold),
                max_detections_per_image=int(args.max_detections_per_image),
            )
            if not detections:
                continue

            relative = image_path.relative_to(input_root)
            stem = relative.with_suffix("")
            for index, detection in enumerate(detections):
                expanded = expand_bbox(detection.bbox, gray.shape, float(args.bbox_expand_ratio))
                x1, y1, x2, y2 = expanded
                if min(x2 - x1, y2 - y1) < int(args.min_crop_size):
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_path = output_crop_root / f"{stem.as_posix()}__face_{index:02d}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_path), crop):
                    continue

                manifest_fp.write(
                    json.dumps(
                        {
                            "original_image_path": str(image_path),
                            "crop_image_path": str(crop_path),
                            "bbox": [x1, y1, x2, y2],
                            "score": round(float(detection.score), 6),
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
    return parser.parse_args()


if __name__ == "__main__":
    build_face_crops(parse_args())
