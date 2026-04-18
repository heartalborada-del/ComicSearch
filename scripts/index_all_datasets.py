from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Iterable, TypeVar

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG_MAP_OUTPUT = PROJECT_ROOT / "tag_id_map.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedder_onnx import OnnxImageEmbedder
from app.models import Base, Keyword, Pack, PackKeyword, TagIdMap
from app.natural_sort import NaturalComparator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COMICINFO_FILENAME = "ComicInfo.xml"
TRAILING_TITLE_SINGLE_TAG_PATTERN = re.compile(r"\s*[\[［【][^\]］】]+[\]］】]\s*$")

T = TypeVar("T")


@dataclass(frozen=True)
class IndexItem:
    point_id: int | str
    image_path: Path
    payload: dict[str, Any]
    pack_title: str | None = None
    pack_source: str | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    title: str
    tags: list[str]
    source: str | None = None


def parse_int_from_tokens(path_value: str, token: str) -> int | None:
    match = re.search(rf"{re.escape(token)}[_-]?(\d+)", path_value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_page_no(target_name: str, all_names: list[str]) -> int | None:
    """Calculate 1-based page number from natural-sorted file list + target name."""
    if not all_names:
        return None
    sorted_names = sorted(all_names, key=cmp_to_key(NaturalComparator.compare))
    try:
        return sorted_names.index(target_name) + 1
    except ValueError:
        return None


def build_dataset_image_name_lists(dataset_roots: list[Path]) -> dict[Path, list[str]]:
    result: dict[Path, list[str]] = {}
    for dataset_root in dataset_roots:
        names: list[str] = []
        for image_path in dataset_root.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                names.append(str(image_path.relative_to(dataset_root.resolve())))
        result[dataset_root.resolve()] = names
    return result


def make_point_id(source_type: str, image_path: Path, extra: str = "") -> int:
    resolved_path = image_path.resolve()
    digest = hashlib.sha256(f"{source_type}:{resolved_path}:{extra}".encode("utf-8")).digest()
    # 转换为无符号整数（取前 8 字节）
    return int.from_bytes(digest[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def discover_dataset_roots(root_args: list[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root_arg in root_args:
        base_root = Path(root_arg).resolve()
        if not base_root.exists() or not base_root.is_dir():
            raise FileNotFoundError(f"datasets root not found: {base_root}")

        candidates: list[Path] = []
        if (base_root / COMICINFO_FILENAME).exists():
            candidates.append(base_root)
        for child in sorted(base_root.iterdir()):
            if child.is_dir() and (child / COMICINFO_FILENAME).exists():
                candidates.append(child.resolve())

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)

    if not discovered:
        raise ValueError(
            f"no dataset roots found: expected {COMICINFO_FILENAME} in each dataset root or its first-level subdirectories"
        )
    return discovered


def normalize_tag(tag: str) -> str:
    value = str(tag).strip().lower()
    if not value:
        return ""
    prefix, sep, suffix = value.partition(":")
    if not sep:
        return "_".join(value.split())
    return f"{prefix}:{'_'.join(suffix.strip().split())}"


def normalize_title(title: str) -> str:
    """Normalize display title by removing trailing bracketed release tags.

    Example: "some title [英訳] [DL版]" -> "some title"
    """
    normalized = title.strip()
    while True:
        match = TRAILING_TITLE_SINGLE_TAG_PATTERN.search(normalized)
        if match is None:
            break
        prefix = normalized[: match.start()].strip()
        if not prefix:
            # Keep fully bracketed titles like "【推しの皮】".
            break
        normalized = prefix
    return normalized


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
        normalized = normalize_tag(key)
        if not normalized:
            continue
        try:
            result[normalized] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def load_dataset_metadata(dataset_roots: list[Path]) -> dict[Path, DatasetMetadata]:
    """Load per-root ComicInfo.xml files.

    Expected fields for each dataset root:
    <Title>, <Tags>, optional <Web> / <URL>.
    """

    metadata_by_root: dict[Path, DatasetMetadata] = {}
    for dataset_root in dataset_roots:
        metadata_path = dataset_root / COMICINFO_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"dataset metadata file not found: {metadata_path}")

        try:
            tree = ET.parse(metadata_path)
        except ET.ParseError as exc:
            raise ValueError(f"invalid ComicInfo.xml: {metadata_path} ({exc})") from exc

        root = tree.getroot()

        def _clean_text(value: str | None) -> str | None:
            if value is None:
                return None
            cleaned = value.strip()
            return cleaned if cleaned else None

        title = _clean_text(root.findtext("Title"))
        if not title:
            raise ValueError(f'dataset metadata must contain a non-empty "Title": {metadata_path}')
        title = normalize_title(title)
        if not title:
            raise ValueError(f'dataset metadata title became empty after normalization: {metadata_path}')

        tags_text = _clean_text(root.findtext("Tags"))
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()] if tags_text else []
        source = _clean_text(root.findtext("Web")) or _clean_text(root.findtext("URL"))

        metadata_by_root[dataset_root.resolve()] = DatasetMetadata(
            title=title,
            tags=tags,
            source=source,
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
        normalized_tag = normalize_tag(tag)
        if not normalized_tag:
            continue
        keyword_id = tag_id_map.get(normalized_tag)
        if keyword_id is None:
            raise ValueError(f'tag "{tag}" used in metadata for {context_path} is missing in tag-id map and database')
        ids.add(int(keyword_id))
    return sorted(ids)


def load_db_tag_id_map(session: Session) -> tuple[dict[str, int], dict[int, str]]:
    tag_to_id: dict[str, int] = {}
    id_to_tag: dict[int, str] = {}
    for tag, keyword_id in session.query(TagIdMap.tag, TagIdMap.keyword_id).all():
        normalized = normalize_tag(tag)
        if not normalized:
            continue
        keyword_id_int = int(keyword_id)
        tag_to_id[normalized] = keyword_id_int
        id_to_tag[keyword_id_int] = normalized
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
    image_name_lists_by_root: dict[Path, list[str]],
) -> Iterable[IndexItem]:
    for dataset_root in dataset_roots:
        dataset_root_resolved = dataset_root.resolve()
        dataset_metadata = metadata_by_root[dataset_root_resolved]
        all_names = image_name_lists_by_root.get(dataset_root_resolved, [])

        image_paths = [
            image_path
            for image_path in dataset_root.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_paths.sort(
            key=cmp_to_key(
                lambda a, b: NaturalComparator.compare(
                    str(a.relative_to(dataset_root_resolved)),
                    str(b.relative_to(dataset_root_resolved)),
                )
            )
        )

        for image_path in image_paths:
            path_str = str(image_path)
            relative_name = str(image_path.relative_to(dataset_root_resolved))
            pack_id = parse_int_from_tokens(path_str, "pack")
            page_no = parse_page_no(relative_name, all_names)
            payload = {
                "pack_id": pack_id,
                "keyword_ids": build_keyword_ids(dataset_metadata.tags, tag_id_map, path_str),
                "page_no": page_no,
                "page_path": path_str,
                "source_type": "page",
            }
            yield IndexItem(
                point_id=make_point_id("page", image_path),
                image_path=image_path,
                payload=payload,
                pack_title=dataset_metadata.title,
                pack_source=dataset_metadata.source,
            )


def iter_crop_items(
    crop_manifest_path: Path,
    dataset_roots: list[Path],
    metadata_by_root: dict[Path, DatasetMetadata],
    tag_id_map: dict[str, int],
    image_name_lists_by_root: dict[Path, list[str]],
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
            original_pack_id = parse_int_from_tokens(original_path_str, "pack")
            dataset_root = find_dataset_root(original_image_path, dataset_roots)
            if dataset_root is None:
                continue
            dataset_metadata = metadata_by_root.get(dataset_root.resolve())
            if dataset_metadata is None:
                continue
            metadata_tags = dataset_metadata.tags if dataset_metadata is not None else []
            original_page_no: int | None = None
            if dataset_root is not None:
                dataset_root_resolved = dataset_root.resolve()
                relative_name = str(original_image_path.relative_to(dataset_root_resolved))
                all_names = image_name_lists_by_root.get(dataset_root_resolved, [])
                original_page_no = parse_page_no(relative_name, all_names)
            payload = {
                "pack_id": original_pack_id,
                "keyword_ids": build_keyword_ids(metadata_tags, tag_id_map, original_path_str),
                "page_no": original_page_no,
                "page_path": original_path_str,
                "source_type": "face_crop",
            }
            yield IndexItem(
                point_id=make_point_id("face_crop", crop_image_path, extra=str(tuple(row.get("bbox", [])))),
                image_path=crop_image_path,
                payload=payload,
                pack_title=dataset_metadata.title if dataset_metadata is not None else None,
                pack_source=dataset_metadata.source if dataset_metadata is not None else None,
            )


def upsert_db_records(
    session: Session,
    items: list[IndexItem],
    keyword_names_by_id: dict[int, str],
) -> None:
    title_to_pack_id: dict[str, int] = {}
    source_by_pack_id: dict[int, str] = {}
    for row in session.query(Pack.pack_id, Pack.title).all():
        if not row.title:
            continue
        existing = title_to_pack_id.get(row.title)
        if existing is None or int(row.pack_id) < existing:
            title_to_pack_id[row.title] = int(row.pack_id)

    # Backfill missing pack ids using stable title mapping, creating auto-increment rows when needed.
    for item in items:
        payload = item.payload
        if payload.get("pack_id") is not None:
            if item.pack_title:
                title_to_pack_id.setdefault(item.pack_title, int(payload["pack_id"]))
                if item.pack_source:
                    source_by_pack_id.setdefault(int(payload["pack_id"]), item.pack_source)
            continue

        if not item.pack_title:
            continue

        pack_id = title_to_pack_id.get(item.pack_title)
        if pack_id is None:
            pack = Pack(title=item.pack_title, source=item.pack_source)
            session.add(pack)
            session.flush()
            pack_id = int(pack.pack_id)
            title_to_pack_id[item.pack_title] = pack_id
        elif item.pack_source:
            source_by_pack_id.setdefault(pack_id, item.pack_source)

        payload["pack_id"] = pack_id

    pack_titles: dict[int, str] = {}
    pack_keywords: set[tuple[int, int]] = set()

    for item in items:
        payload = item.payload
        pack_id = payload.get("pack_id")
        if pack_id is not None and item.pack_title:
            pack_titles.setdefault(int(pack_id), item.pack_title)
        if pack_id is not None:
            for keyword_id in payload.get("keyword_ids") or []:
                pack_keywords.add((int(pack_id), int(keyword_id)))

    for pack_id, title in sorted(pack_titles.items()):
        pack = session.get(Pack, pack_id)
        if pack is None:
            session.add(Pack(pack_id=pack_id, title=title, source=source_by_pack_id.get(pack_id)))
        elif pack.title != title:
            pack.title = title
            source_value = source_by_pack_id.get(pack_id)
            if source_value and not pack.source:
                pack.source = source_value
        else:
            source_value = source_by_pack_id.get(pack_id)
            if source_value and not pack.source:
                pack.source = source_value
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
        normalized_tag = normalize_tag(tag_name)
        if not normalized_tag:
            continue
        tag_row = session.get(TagIdMap, normalized_tag)
        if tag_row is None:
            session.add(TagIdMap(tag=normalized_tag, keyword_id=keyword_id))
        elif int(getattr(tag_row, "keyword_id")) != int(keyword_id):
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


def batch_iter(items: list[T], batch_size: int) -> Iterable[list[T]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def embed_batch(embedder: OnnxImageEmbedder, batch: list[IndexItem]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in batch:
        with Image.open(item.image_path) as image:
            vector = embedder.embed_pil(image)
        vectors.append(vector.tolist())
    return vectors


def ensure_metadata_tags_mapped(
    metadata_by_root: dict[Path, DatasetMetadata],
    tag_to_id: dict[str, int],
    id_to_tag: dict[int, str],
) -> int:
    added_count = 0
    next_keyword_id = (max(id_to_tag.keys()) + 1) if id_to_tag else 0

    for metadata in metadata_by_root.values():
        for raw_tag in metadata.tags:
            normalized_tag = normalize_tag(raw_tag)
            if not normalized_tag or normalized_tag in tag_to_id:
                continue
            while next_keyword_id in id_to_tag:
                next_keyword_id += 1
            tag_to_id[normalized_tag] = next_keyword_id
            id_to_tag[next_keyword_id] = normalized_tag
            next_keyword_id += 1
            added_count += 1

    return added_count


def upsert_tag_registry(session: Session, keyword_names_by_id: dict[int, str]) -> None:
    for keyword_id, tag_name in sorted(keyword_names_by_id.items()):
        normalized_tag = normalize_tag(tag_name)
        if not normalized_tag:
            continue

        keyword = session.get(Keyword, int(keyword_id))
        if keyword is None:
            session.add(Keyword(id=int(keyword_id), name=normalized_tag))
        elif keyword.name != normalized_tag:
            keyword.name = normalized_tag

        tag_row = session.get(TagIdMap, normalized_tag)
        if tag_row is None:
            session.add(TagIdMap(tag=normalized_tag, keyword_id=int(keyword_id)))
        elif int(getattr(tag_row, "keyword_id")) != int(keyword_id):
            tag_row.keyword_id = int(keyword_id)

    session.flush()


def export_tag_id_map(path: Path, tag_to_id: dict[str, int]) -> None:
    ordered_map = {
        tag: int(keyword_id)
        for tag, keyword_id in sorted(tag_to_id.items(), key=lambda item: (int(item[1]), item[0]))
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(ordered_map, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def run_indexing(args: argparse.Namespace) -> None:
    dataset_roots = discover_dataset_roots(args.datasets_root)
    metadata_by_root = load_dataset_metadata(dataset_roots)
    image_name_lists_by_root = build_dataset_image_name_lists(dataset_roots)
    state_path = Path(args.resume_state).resolve()
    tag_map_output_path = Path(args.tag_map_output).resolve()
    processed_ids = set() if args.reset_state else load_state(state_path)

    db_url = args.db_url or os.getenv("DATABASE_URL", "sqlite:///./comicsearch.db")
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    DbSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)

    all_items: list[IndexItem] = []
    auto_added_tag_count = 0
    with DbSession() as db:
        db_tag_to_id, db_id_to_tag = load_db_tag_id_map(db)
        tag_id_map = dict(db_tag_to_id)
        keyword_names_by_id = dict(db_id_to_tag)
        auto_added_tag_count = ensure_metadata_tags_mapped(metadata_by_root, tag_id_map, keyword_names_by_id)
        upsert_tag_registry(db, keyword_names_by_id)

        for item in iter_page_items(dataset_roots, metadata_by_root, tag_id_map, image_name_lists_by_root):
            if item.point_id in processed_ids or not item.image_path.exists():
                continue
            all_items.append(item)
        if args.face_crops_manifest:
            for item in iter_crop_items(
                Path(args.face_crops_manifest).resolve(),
                dataset_roots,
                metadata_by_root,
                tag_id_map,
                image_name_lists_by_root,
            ):
                if item.point_id in processed_ids or not item.image_path.exists():
                    continue
                all_items.append(item)

        if not all_items:
            db.commit()
            export_tag_id_map(tag_map_output_path, tag_id_map)
            print(
                json.dumps(
                    {
                        "status": "noop",
                        "reason": "no new items to index",
                        "tag_map_output": str(tag_map_output_path),
                        "auto_added_tags": auto_added_tag_count,
                    },
                    ensure_ascii=False,
                )
            )
            return

        validate_used_keyword_ids(all_items, keyword_names_by_id)
        upsert_db_records(db, all_items, keyword_names_by_id)

    export_tag_id_map(tag_map_output_path, tag_id_map)

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
                "tag_map_output": str(tag_map_output_path),
                "auto_added_tags": auto_added_tag_count,
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
        "--tag-map-output",
        default=str(DEFAULT_TAG_MAP_OUTPUT),
        help="Path to export effective tag-id map JSON. Defaults to project root tag_id_map.json.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy DB URL. Defaults to DATABASE_URL or sqlite:///./comicsearch.db.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_indexing(parse_args())
