from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedder_onnx import OnnxImageEmbedder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class IndexItem:
    point_id: str
    image_path: Path
    payload: dict[str, Any]


def parse_int_from_tokens(path_value: str, token: str) -> int | None:
    match = re.search(rf"{re.escape(token)}[_-]?(\d+)", path_value, re.IGNORECASE)
    if match:
        return int(match.group(1))
    numeric_matches = re.findall(r"\d+", path_value)
    return int(numeric_matches[0]) if numeric_matches else None


def parse_page_no(path_value: str) -> int | None:
    page_match = re.search(r"(?:page|p)[_-]?(\d+)", path_value, re.IGNORECASE)
    if page_match:
        return int(page_match.group(1))
    stem_matches = re.findall(r"\d+", Path(path_value).stem)
    return int(stem_matches[-1]) if stem_matches else None


def make_point_id(source_type: str, image_path: Path, extra: str = "") -> str:
    digest = hashlib.sha1(f"{source_type}:{image_path}:{extra}".encode("utf-8")).hexdigest()
    return digest


def iter_page_items(dataset_roots: list[Path], keyword_map: dict[str, list[int]]) -> Iterable[IndexItem]:
    for dataset_root in dataset_roots:
        for image_path in dataset_root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            path_str = str(image_path)
            manga_id = parse_int_from_tokens(path_str, "manga")
            pack_id = parse_int_from_tokens(path_str, "pack")
            page_no = parse_page_no(path_str)
            payload = {
                "manga_id": manga_id,
                "pack_id": pack_id,
                "keyword_ids": keyword_map.get(path_str, []),
                "cover_thumb_path": "",
                "page_no": page_no,
                "page_path": path_str,
                "source_type": "page",
                "crop_bbox": None,
                "crop_score": None,
                "crop_original_path": None,
            }
            yield IndexItem(
                point_id=make_point_id("page", image_path),
                image_path=image_path,
                payload=payload,
            )


def iter_crop_items(crop_manifest_path: Path, keyword_map: dict[str, list[int]]) -> Iterable[IndexItem]:
    if not crop_manifest_path.exists():
        return
    with crop_manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            crop_image_path = Path(row["crop_image_path"]).resolve()
            original_image_path = Path(row["original_image_path"]).resolve()
            original_path_str = str(original_image_path)
            payload = {
                "manga_id": parse_int_from_tokens(original_path_str, "manga"),
                "pack_id": parse_int_from_tokens(original_path_str, "pack"),
                "keyword_ids": keyword_map.get(original_path_str, []),
                "cover_thumb_path": "",
                "page_no": parse_page_no(original_path_str),
                "page_path": original_path_str,
                "source_type": "face_crop",
                "crop_bbox": row.get("bbox"),
                "crop_score": row.get("score"),
                "crop_original_path": original_path_str,
            }
            yield IndexItem(
                point_id=make_point_id("face_crop", crop_image_path, extra=json.dumps(row.get("bbox"))),
                image_path=crop_image_path,
                payload=payload,
            )


def load_keyword_map(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"keyword map file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("keyword map must be a JSON object: {\"/abs/path.jpg\": [1,2]}")
    normalized: dict[str, list[int]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        normalized[str(Path(key).resolve())] = [int(v) for v in value]
    return normalized


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    processed = data.get("processed_ids", [])
    return set(processed) if isinstance(processed, list) else set()


def save_state(path: Path, processed_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump({"processed_ids": sorted(processed_ids)}, fp, ensure_ascii=False)


def batch_iter(items: list[IndexItem], batch_size: int) -> Iterable[list[IndexItem]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def embed_batch(embedder: OnnxImageEmbedder, batch: list[IndexItem]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in batch:
        with Image.open(item.image_path) as image:
            vector = embedder.embed_pil(image)
        vectors.append(vector.tolist())
    return vectors


def run_indexing(args: argparse.Namespace) -> None:
    dataset_roots = [Path(root).resolve() for root in args.datasets_root]
    keyword_map = load_keyword_map(Path(args.keyword_map).resolve() if args.keyword_map else None)
    state_path = Path(args.resume_state).resolve()
    processed_ids = set() if args.reset_state else load_state(state_path)

    all_items: list[IndexItem] = []
    all_items.extend(iter_page_items(dataset_roots, keyword_map))
    if args.face_crops_manifest:
        all_items.extend(iter_crop_items(Path(args.face_crops_manifest).resolve(), keyword_map))
    all_items = [item for item in all_items if item.point_id not in processed_ids and item.image_path.exists()]

    if not all_items:
        print(json.dumps({"status": "noop", "reason": "no new items to index"}, ensure_ascii=False))
        return

    embedder = OnnxImageEmbedder(
        onnx_path=str(Path(args.onnx_model).resolve()),
        input_size=int(args.embedder_input_size),
        intra_threads=int(args.embedder_intra_threads),
    )
    client = QdrantClient(url=args.qdrant_url)

    indexed = 0
    for embed_batch_items in batch_iter(all_items, int(args.embed_batch_size)):
        vectors = embed_batch(embedder, embed_batch_items)
        points = [
            qm.PointStruct(id=item.point_id, vector=vector, payload=item.payload)
            for item, vector in zip(embed_batch_items, vectors)
        ]

        for upsert_items in batch_iter(points, int(args.upsert_batch_size)):
            client.upsert(collection_name=args.collection, points=upsert_items, wait=True)

        for item in embed_batch_items:
            processed_ids.add(item.point_id)
        save_state(state_path, processed_ids)
        indexed += len(embed_batch_items)

    print(
        json.dumps(
            {
                "status": "ok",
                "indexed_count": indexed,
                "collection": args.collection,
                "qdrant_url": args.qdrant_url,
                "resume_state": str(state_path),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index all page datasets and optional face-crop subset to Qdrant.")
    parser.add_argument(
        "--datasets-root",
        action="append",
        required=True,
        help="Dataset root path (repeatable). Recursively indexes image pages under each root.",
    )
    parser.add_argument("--face-crops-manifest", default=None, help="Optional JSONL manifest from build_face_crops.py.")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333", help="Qdrant base URL.")
    parser.add_argument("--collection", default="pages", help="Qdrant collection name.")
    parser.add_argument("--onnx-model", required=True, help="ONNX image encoder path.")
    parser.add_argument("--embedder-input-size", type=int, default=224, help="Embedder input image size.")
    parser.add_argument("--embedder-intra-threads", type=int, default=4, help="ONNX intra-op threads.")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embedding.")
    parser.add_argument("--upsert-batch-size", type=int, default=512, help="Batch size for Qdrant upsert.")
    parser.add_argument(
        "--resume-state",
        default=".cache/index_all_datasets_state.json",
        help="State file path for resumable indexing.",
    )
    parser.add_argument("--reset-state", action="store_true", help="Ignore previous state and re-index all items.")
    parser.add_argument(
        "--keyword-map",
        default=None,
        help="Optional JSON map of absolute page path to keyword id list.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_indexing(parse_args())
