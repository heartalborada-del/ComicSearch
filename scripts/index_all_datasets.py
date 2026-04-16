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
from app.models import Base, Keyword, Manga, Pack, PackKeyword, TagIdMap


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PAGE_PATTERN = re.compile(r"(?:page|p)[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class IndexItem:
    point_id: str
    image_path: Path
    payload: dict[str, Any]
    pack_title: str | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    title: str
    tags: list[str]


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


def discover_dataset_roots(root_args: list[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root_arg in root_args:
        base_root = Path(root_arg).resolve()
        if not base_root.exists() or not base_root.is_dir():
            raise FileNotFoundError(f"datasets root not found: {base_root}")

        candidates: list[Path] = []
        if (base_root / "metadata.json").exists():
            candidates.append(base_root)
        for child in sorted(base_root.iterdir()):
            if child.is_dir() and (child / "metadata.json").exists():
                candidates.append(child.resolve())

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)

    if not discovered:
        raise ValueError("no dataset roots found: expected metadata.json in each dataset root or its first-level subdirectories")
    return discovered


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
            result[key.lower()] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def load_dataset_metadata(dataset_roots: list[Path]) -> dict[Path, DatasetMetadata]:
    """Load per-root metadata.json files.

    Expected format for each dataset root:
    {"title": "album title", "tags": ["a", "b"]}
    """

    metadata_by_root: dict[Path, DatasetMetadata] = {}
    for dataset_root in dataset_roots:
        metadata_path = dataset_root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"dataset metadata file not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"dataset metadata must be a JSON object: {metadata_path}")

        title = data.get("title")
        if not isinstance(title, (str, int, float)):
            raise ValueError(f'dataset metadata must contain a string "title": {metadata_path}')

        tags_value = data.get("tags", [])
        if not isinstance(tags_value, list):
            raise ValueError(f'dataset metadata "tags" must be a list: {metadata_path}')

        metadata_by_root[dataset_root.resolve()] = DatasetMetadata(
            title=str(title),
            tags=[str(tag) for tag in tags_value if isinstance(tag, (str, int, float))],
        )
    return metadata_by_root


def find_dataset_root(path: Path, dataset_roots: list[Path]) -> Path | None:
    resolved_path = path.resolve()
    matched_roots = [root for root in dataset_roots if resolved_path.is_relative_to(root)]
    if not matched_roots:
        return None
    return max(matched_roots, key=lambda root: len(root.parts))


def build_keyword_ids(metadata_tags: list[str], tag_id_map: dict[str, int], context_path: str) -> list[int]:
    ids: set[int] = set()
    for tag in metadata_tags:
        normalized_tag = tag.lower()
        keyword_id = tag_id_map.get(normalized_tag)
        if keyword_id is None:
            raise ValueError(f'tag "{tag}" used in metadata for {context_path} is missing in tag-id map and database')
        ids.add(int(keyword_id))
    return sorted(ids)


def load_db_tag_id_map(session: Session) -> tuple[dict[str, int], dict[int, str]]:
    tag_to_id: dict[str, int] = {}
    id_to_tag: dict[int, str] = {}
    for row in session.query(TagIdMap).all():
        normalized = row.tag.lower()
        tag_to_id[normalized] = int(row.keyword_id)
        id_to_tag[int(row.keyword_id)] = row.tag
    return tag_to_id, id_to_tag


def resolve_effective_tag_maps(
    map_tag_to_id: dict[str, int],
    db_tag_to_id: dict[str, int],
    db_id_to_tag: dict[int, str],
) -> tuple[dict[str, int], dict[int, str]]:
    effective_tag_to_id = dict(db_tag_to_id)
    effective_id_to_tag = dict(db_id_to_tag)

    for tag, keyword_id in map_tag_to_id.items():
        existing_id_for_tag = effective_tag_to_id.get(tag)
        if existing_id_for_tag is not None:
            # DB mapping is authoritative when map conflicts with an existing tag.
            continue

        existing_tag_for_id = effective_id_to_tag.get(int(keyword_id))
        if existing_tag_for_id is not None and existing_tag_for_id.lower() != tag:
            if int(keyword_id) in db_id_to_tag:
                # Duplicate id found but DB already has this id mapping, so keep DB mapping.
                continue
            raise ValueError(
                f'duplicate keyword id {keyword_id} for tags "{existing_tag_for_id}" and "{tag}" without DB mapping'
            )

        effective_tag_to_id[tag] = int(keyword_id)
        effective_id_to_tag[int(keyword_id)] = tag

    return effective_tag_to_id, effective_id_to_tag


def validate_used_keyword_ids(items: list[IndexItem], keyword_names_by_id: dict[int, str]) -> None:
    used_keyword_ids: set[int] = set()
    for item in items:
        for keyword_id in item.payload.get("keyword_ids") or []:
            used_keyword_ids.add(int(keyword_id))

    missing = sorted(keyword_id for keyword_id in used_keyword_ids if keyword_id not in keyword_names_by_id)
    if missing:
        raise ValueError(
            "keyword ids used but missing in tag-id map and database: " + ", ".join(str(keyword_id) for keyword_id in missing)
        )


def iter_page_items(
    dataset_roots: list[Path],
    metadata_by_root: dict[Path, DatasetMetadata],
    tag_id_map: dict[str, int],
) -> Iterable[IndexItem]:
    for dataset_root in dataset_roots:
        dataset_metadata = metadata_by_root[dataset_root.resolve()]
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
                "keyword_ids": build_keyword_ids(dataset_metadata.tags, tag_id_map, path_str),
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
                pack_title=dataset_metadata.title,
            )


def iter_crop_items(
    crop_manifest_path: Path,
    dataset_roots: list[Path],
    metadata_by_root: dict[Path, DatasetMetadata],
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
            dataset_root = find_dataset_root(original_image_path, dataset_roots)
            dataset_metadata = metadata_by_root.get(dataset_root.resolve()) if dataset_root is not None else None
            metadata_tags = dataset_metadata.tags if dataset_metadata is not None else []
            payload = {
                "manga_id": parse_int_from_tokens(original_path_str, "manga"),
                "pack_id": parse_int_from_tokens(original_path_str, "pack"),
                "keyword_ids": build_keyword_ids(metadata_tags, tag_id_map, original_path_str),
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
                pack_title=dataset_metadata.title if dataset_metadata is not None else None,
            )


def upsert_db_records(
    session: Session,
    items: list[IndexItem],
    keyword_names_by_id: dict[int, str],
) -> None:
    manga_ids: set[int] = set()
    pack_manga: dict[int, int] = {}
    pack_titles: dict[int, str] = {}
    pack_keywords: set[tuple[int, int]] = set()

    for item in items:
        payload = item.payload
        manga_id = payload.get("manga_id")
        pack_id = payload.get("pack_id")
        if manga_id is not None:
            manga_ids.add(int(manga_id))
        if manga_id is not None and pack_id is not None:
            pack_manga.setdefault(int(pack_id), int(manga_id))
        if pack_id is not None and item.pack_title:
            pack_titles.setdefault(int(pack_id), item.pack_title)
        if pack_id is not None:
            for keyword_id in payload.get("keyword_ids") or []:
                pack_keywords.add((int(pack_id), int(keyword_id)))

    for manga_id in sorted(manga_ids):
        if session.get(Manga, manga_id) is None:
            session.add(Manga(manga_id=manga_id, title=f"manga_{manga_id}"))
    session.flush()

    for pack_id, manga_id in sorted(pack_manga.items()):
        pack = session.get(Pack, pack_id)
        pack_title = pack_titles.get(pack_id)
        if pack is None:
            session.add(Pack(pack_id=pack_id, manga_id=manga_id, cover_thumb_path=None, title=pack_title))
        elif pack_title and pack.title != pack_title:
            pack.title = pack_title
    session.flush()

    keyword_ids = {keyword_id for _, keyword_id in pack_keywords}
    for keyword_id in sorted(keyword_ids):
        if session.get(Keyword, keyword_id) is None:
            keyword_name = keyword_names_by_id.get(keyword_id, f"keyword_{keyword_id}")
            session.add(Keyword(id=keyword_id, name=keyword_name))
    session.flush()

    for keyword_id in sorted(keyword_ids):
        tag_name = keyword_names_by_id.get(keyword_id)
        if not tag_name:
            continue
        normalized_tag = tag_name.lower()
        tag_row = session.get(TagIdMap, normalized_tag)
        if tag_row is None:
            session.add(TagIdMap(tag=normalized_tag, keyword_id=keyword_id))
        elif int(tag_row.keyword_id) != int(keyword_id):
            tag_row.keyword_id = int(keyword_id)
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
    dataset_roots = discover_dataset_roots(args.datasets_root)
    input_tag_id_map = load_tag_id_map(Path(args.tag_id_map).resolve() if args.tag_id_map else None)
    metadata_by_root = load_dataset_metadata(dataset_roots)
    state_path = Path(args.resume_state).resolve()
    processed_ids = set() if args.reset_state else load_state(state_path)

    db_url = args.db_url or os.getenv("DATABASE_URL", "sqlite:///./comicsearch.db")
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    DbSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)

    all_items: list[IndexItem] = []
    with DbSession() as db:
        db_tag_to_id, db_id_to_tag = load_db_tag_id_map(db)
        tag_id_map, keyword_names_by_id = resolve_effective_tag_maps(input_tag_id_map, db_tag_to_id, db_id_to_tag)

        for item in iter_page_items(dataset_roots, metadata_by_root, tag_id_map):
            if item.point_id in processed_ids or not item.image_path.exists():
                continue
            all_items.append(item)
        if args.face_crops_manifest:
            for item in iter_crop_items(
                Path(args.face_crops_manifest).resolve(),
                dataset_roots,
                metadata_by_root,
                tag_id_map,
            ):
                if item.point_id in processed_ids or not item.image_path.exists():
                    continue
                all_items.append(item)

        if not all_items:
            print(json.dumps({"status": "noop", "reason": "no new items to index"}, ensure_ascii=False))
            return

        validate_used_keyword_ids(all_items, keyword_names_by_id)
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
        "--tag-id-map",
        default=None,
        help='Optional tag-to-id JSON map: {"tag": id}. Used with per-root metadata tags.',
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy DB URL. Defaults to DATABASE_URL or sqlite:///./comicsearch.db.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_indexing(parse_args())
