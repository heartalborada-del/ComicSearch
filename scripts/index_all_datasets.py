from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedder_onnx import OnnxImageEmbedder
from app.models import Base, Keyword, Manga, Pack, PackKeyword


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PAGE_PATTERN = re.compile(r"(?:page|p)[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class IndexItem:
    point_id: str
    image_path: Path
    payload: dict[str, Any]


@lru_cache(maxsize=16)
def token_id_pattern(token: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(token)}[_-]?(\d+)", re.IGNORECASE)


def parse_int_from_tokens(path_value: str, token: str) -> int | None:
    match = token_id_pattern(token).search(path_value)
    if match:
        return int(match.group(1))
    numeric_matches = re.findall(r"\d+", path_value)
    return int(numeric_matches[0]) if numeric_matches else None


def parse_page_no(path_value: str) -> int | None:
    page_match = PAGE_PATTERN.search(path_value)
    if page_match:
        return int(page_match.group(1))
    stem_matches = re.findall(r"\d+", Path(path_value).stem)
    return int(stem_matches[-1]) if stem_matches else None


def make_point_id(source_type: str, image_path: Path, extra: str = "") -> str:
    resolved_path = image_path.resolve()
    digest = hashlib.sha256(f"{source_type}:{resolved_path}:{extra}".encode("utf-8")).hexdigest()
    return digest


def load_tag_id_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"tag id map file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError('tag-id map must be a JSON object: {"tag": 1}')
    result: dict[str, int] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        try:
            result[key] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def load_dataset_metadata(path: Path | None, dataset_roots: list[Path]) -> tuple[dict[str, list[str]], list[str]]:
    """Load metadata json.

    Supported formats:
    1) {"tags": ["a", "b"]}                        -> global tags for all pages
    2) {"/abs/page.jpg": {"tags": ["a"]}, ...}     -> per-page tags
    3) {"relative/page.jpg": {"tags": ["a"]}, ...} -> resolved under --datasets-root
    """
    if path is None:
        return {}, []
    if not path.exists():
        raise FileNotFoundError(f"dataset metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError('dataset metadata must be a JSON object')
    if isinstance(data.get("tags"), list):
        return {}, [str(tag) for tag in data["tags"] if isinstance(tag, (str, int, float))]

    metadata_tags_by_path: dict[str, list[str]] = {}
    for key, value in data.items():
        tags: list[str] = []
        if isinstance(value, dict) and isinstance(value.get("tags"), list):
            tags = [str(tag) for tag in value["tags"] if isinstance(tag, (str, int, float))]
        elif isinstance(value, list):
            tags = [str(tag) for tag in value if isinstance(tag, (str, int, float))]
        if not tags:
            continue

        raw_path = Path(key)
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
            metadata_tags_by_path[str(resolved)] = tags
            continue
        matched = False
        for root in dataset_roots:
            candidate = (root / raw_path).resolve()
            if candidate.exists():
                metadata_tags_by_path[str(candidate)] = tags
                matched = True
                break
        if not matched:
            metadata_tags_by_path[str(raw_path.resolve())] = tags
    return metadata_tags_by_path, []


def build_keyword_ids(
    page_path: str,
    keyword_map: dict[str, list[int]],
    metadata_tags_by_path: dict[str, list[str]],
    global_tags: list[str],
    tag_id_map: dict[str, int],
) -> list[int]:
    ids = {int(v) for v in keyword_map.get(page_path, [])}
    tags = metadata_tags_by_path.get(page_path, global_tags)
    for tag in tags:
        keyword_id = tag_id_map.get(tag)
        if keyword_id is not None:
            ids.add(int(keyword_id))
    return sorted(ids)


def iter_page_items(
    dataset_roots: list[Path],
    keyword_map: dict[str, list[int]],
    metadata_tags_by_path: dict[str, list[str]],
    global_tags: list[str],
    tag_id_map: dict[str, int],
) -> Iterable[IndexItem]:
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
                "keyword_ids": build_keyword_ids(path_str, keyword_map, metadata_tags_by_path, global_tags, tag_id_map),
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


def iter_crop_items(
    crop_manifest_path: Path,
    keyword_map: dict[str, list[int]],
    metadata_tags_by_path: dict[str, list[str]],
    global_tags: list[str],
    tag_id_map: dict[str, int],
) -> Iterable[IndexItem]:
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
                "keyword_ids": build_keyword_ids(
                    original_path_str, keyword_map, metadata_tags_by_path, global_tags, tag_id_map
                ),
                "cover_thumb_path": "",
                "page_no": parse_page_no(original_path_str),
                "page_path": original_path_str,
                "source_type": "face_crop",
                "crop_bbox": row.get("bbox"),
                "crop_score": row.get("score"),
                "crop_original_path": original_path_str,
            }
            yield IndexItem(
                point_id=make_point_id("face_crop", crop_image_path, extra=str(tuple(row.get("bbox", [])))),
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


def upsert_db_records(
    session: Session,
    items: list[IndexItem],
    keyword_names_by_id: dict[int, str],
) -> None:
    manga_ids: set[int] = set()
    pack_manga: dict[int, int] = {}
    pack_keywords: set[tuple[int, int]] = set()

    for item in items:
        payload = item.payload
        manga_id = payload.get("manga_id")
        pack_id = payload.get("pack_id")
        if manga_id is not None:
            manga_ids.add(int(manga_id))
        if manga_id is not None and pack_id is not None:
            pack_manga.setdefault(int(pack_id), int(manga_id))
        if pack_id is not None:
            for keyword_id in payload.get("keyword_ids") or []:
                pack_keywords.add((int(pack_id), int(keyword_id)))

    for manga_id in sorted(manga_ids):
        if session.get(Manga, manga_id) is None:
            session.add(Manga(manga_id=manga_id, title=f"manga_{manga_id}"))
    session.flush()

    for pack_id, manga_id in sorted(pack_manga.items()):
        if session.get(Pack, pack_id) is None:
            session.add(Pack(pack_id=pack_id, manga_id=manga_id, cover_thumb_path=None, title=None))
    session.flush()

    keyword_ids = {keyword_id for _, keyword_id in pack_keywords}
    for keyword_id in sorted(keyword_ids):
        if session.get(Keyword, keyword_id) is None:
            keyword_name = keyword_names_by_id.get(keyword_id, f"keyword_{keyword_id}")
            session.add(Keyword(id=keyword_id, name=keyword_name))
    session.flush()

    for pack_id, keyword_id in sorted(pack_keywords):
        if session.get(PackKeyword, {"pack_id": pack_id, "keyword_id": keyword_id}) is None:
            session.add(PackKeyword(pack_id=pack_id, keyword_id=keyword_id))

    session.commit()


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
    tag_id_map = load_tag_id_map(Path(args.tag_id_map).resolve() if args.tag_id_map else None)
    metadata_tags_by_path, global_tags = load_dataset_metadata(
        Path(args.dataset_metadata).resolve() if args.dataset_metadata else None,
        dataset_roots,
    )
    keyword_names_by_id = {int(keyword_id): tag for tag, keyword_id in tag_id_map.items()}
    state_path = Path(args.resume_state).resolve()
    processed_ids = set() if args.reset_state else load_state(state_path)

    all_items: list[IndexItem] = []
    for item in iter_page_items(dataset_roots, keyword_map, metadata_tags_by_path, global_tags, tag_id_map):
        if item.point_id in processed_ids or not item.image_path.exists():
            continue
        all_items.append(item)
    if args.face_crops_manifest:
        for item in iter_crop_items(
            Path(args.face_crops_manifest).resolve(),
            keyword_map,
            metadata_tags_by_path,
            global_tags,
            tag_id_map,
        ):
            if item.point_id in processed_ids or not item.image_path.exists():
                continue
            all_items.append(item)

    if not all_items:
        print(json.dumps({"status": "noop", "reason": "no new items to index"}, ensure_ascii=False))
        return

    db_url = args.db_url or os.getenv("DATABASE_URL", "sqlite:///./comicsearch.db")
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    DbSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)
    with DbSession() as db:
        upsert_db_records(db, all_items, keyword_names_by_id)

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

        for qdrant_point_batch in batch_iter(points, int(args.upsert_batch_size)):
            client.upsert(collection_name=args.collection, points=qdrant_point_batch, wait=True)

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
                "db_url": db_url,
                "resume_state": str(state_path),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index all page datasets and optional face-crop subset to Qdrant and SQL database."
    )
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
    parser.add_argument(
        "--dataset-metadata",
        default=None,
        help='Optional metadata JSON. Supports {"tags": [...]} or {"<path>": {"tags": [...]}}.',
    )
    parser.add_argument(
        "--tag-id-map",
        default=None,
        help='Optional tag-to-id JSON map: {"tag": id}. Used with --dataset-metadata tags.',
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy DB URL. Defaults to DATABASE_URL or sqlite:///./comicsearch.db.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_indexing(parse_args())
