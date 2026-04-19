from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import aiohttp
import cv2
import numpy as np
from PIL import Image
from qdrant_client.http import models as qm
from sqlalchemy import select
from sqlalchemy.orm import Session
from ultralytics import YOLO

from app.config import AppSettings
from app.downloader.provider.ehentai import EHentaiProvider
from app.downloader.provider.models import ArchiveInformation, ComicInformation
from app.embedder_onnx import OnnxImageEmbedder
from app.models import EHentaiComicInfo, Keyword, Pack, PackKeyword, TagIdMap
from app.search_service import SearchService
from app.task_manager import TaskCancelledError
from scripts.build_face_crops import detect_faces, expand_bbox


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
logger = logging.getLogger("uvicorn.error")


def normalize_tag(tag: str) -> str:
    value = str(tag).strip().lower()
    if not value:
        return ""
    prefix, sep, suffix = value.partition(":")
    if not sep:
        return "_".join(value.split())
    return f"{prefix}:{'_'.join(suffix.strip().split())}"


def _stable_point_id(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def _image_bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image bytes")
    return image


@lru_cache(maxsize=4)
def _load_yolo_detector(model_path: str) -> YOLO:
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YOLO model file not found: {resolved}")
    return YOLO(str(resolved))


async def _fetch_bytes(session: aiohttp.ClientSession, url: str, retries: int = 3) -> bytes:
    delay_seconds = 1.0
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            async with session.get(url, allow_redirects=True) as response:
                if response.status == 200:
                    body = await response.read()
                    if not body:
                        raise RuntimeError(f"empty response body for {url}")
                    return body

                if response.status in {429} or response.status >= 500:
                    await response.release()
                    if attempt < retries - 1:
                        await asyncio.sleep(delay_seconds)
                        delay_seconds *= 2
                        continue

                raise RuntimeError(f"failed to download {url}: status={response.status}")
        except Exception as exc:
            last_error = exc
            if attempt >= retries - 1:
                break
            await asyncio.sleep(delay_seconds)
            delay_seconds *= 2

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to download {url}")


def _sanitize_file_component(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return normalized or "archive"


def _archive_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name
    if not candidate:
        return "resample.zip"
    if "." not in candidate:
        return f"{candidate}.zip"
    return _sanitize_file_component(candidate)


def _extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"


def _safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            member_name = member.filename.replace("\\", "/")
            if member_name.endswith("/"):
                continue
            target = (output_dir / member_name).resolve()
            if output_dir.resolve() not in target.parents and target != output_dir.resolve():
                raise ValueError(f"unsafe archive entry path: {member_name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)


def _iter_image_paths(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path.relative_to(root)).lower(),
    )


@dataclass(frozen=True)
class EhentaiIngestResult:
    pack_id: int
    title: str
    source: str
    gid: int
    token: str
    page_count: int
    indexed_points: int
    crop_points: int
    keyword_ids: list[int]


class EhentaiIngestService:
    def __init__(
        self,
        settings: AppSettings,
        embedder: OnnxImageEmbedder,
        search_service: SearchService,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._search_service = search_service

    def _gallery_source(self, gid: int, token: str) -> str:
        base_url = "https://exhentai.org" if self._settings.ehentai.is_exhentai else "https://e-hentai.org"
        return f"{base_url}/g/{int(gid)}/{str(token)}/"

    def _find_existing_pack_for_gid(self, session: Session, gid: int) -> Pack | None:
        base_url = "https://exhentai.org" if self._settings.ehentai.is_exhentai else "https://e-hentai.org"
        source_prefix = f"{base_url}/g/{int(gid)}/%"
        row = (
            session.query(Pack)
            .filter(Pack.source.like(source_prefix))
            .order_by(Pack.pack_id.asc())
            .first()
        )
        return cast(Pack | None, row)

    def _provider(self) -> EHentaiProvider:
        return EHentaiProvider(
            proxy=self._settings.ehentai.proxy,
            isExhentai=self._settings.ehentai.is_exhentai,
            cookies=self._settings.ehentai.cookies or None,
        )

    def _session_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._settings.ehentai.proxy:
            kwargs["proxy"] = self._settings.ehentai.proxy
        if self._settings.ehentai.cookies:
            kwargs["cookies"] = self._settings.ehentai.cookies
        return kwargs

    @staticmethod
    def _check_cancel(should_cancel: Any) -> None:
        if should_cancel is not None and bool(should_cancel()):
            raise TaskCancelledError("task was cancelled")

    def _delete_gallery_points(self, gid: int) -> None:
        points_selector = qm.Filter(
            must=[qm.FieldCondition(key="gallery_gid", match=qm.MatchValue(value=int(gid)))],
        )
        self._search_service.qdrant.delete(
            collection_name=self._search_service.collection_name,
            points_selector=points_selector,
            wait=True,
        )

    def _dataset_root(self) -> Path:
        return Path(self._settings.ehentai.archive_extract_root).resolve()

    def _cleanup_gid_datasets(self, gid: int, keep_token: str) -> None:
        root = self._dataset_root()
        if not root.exists():
            return
        keep_name = f"{int(gid)}-{_sanitize_file_component(keep_token)}"
        for child in root.glob(f"{int(gid)}-*"):
            if child.name == keep_name:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    def _dataset_dir(self, gid: int, token: str) -> Path:
        return self._dataset_root() / f"{int(gid)}-{_sanitize_file_component(token)}"

    @staticmethod
    def _select_resample_archive(archives: list[ArchiveInformation]) -> ArchiveInformation:
        for archive in archives:
            if archive.name.lower() == "resample":
                return archive
        raise ValueError("Resample archive is not available for this gallery")

    @staticmethod
    def _write_comic_info_xml(dataset_dir: Path, comic_info: ComicInformation[tuple[str, str]], source_url: str) -> None:
        root = ET.Element("ComicInfo")
        ET.SubElement(root, "Title").text = comic_info.title
        ET.SubElement(root, "Tags").text = ", ".join(comic_info.tags)
        ET.SubElement(root, "Web").text = source_url
        tree = ET.ElementTree(root)
        tree.write(dataset_dir / "ComicInfo.xml", encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _upsert_ehentai_comic_info(
        session: Session,
        comic_info: ComicInformation[tuple[str, str]],
        existing_gallery: EHentaiComicInfo | None = None,
    ) -> None:
        current_gid = int(comic_info.id[0])
        current_token = str(comic_info.id[1])
        old_gid = int(comic_info.old[0])
        old_token = str(comic_info.old[1])

        if existing_gallery is None:
            session.add(
                EHentaiComicInfo(
                    current_gid=current_gid,
                    current_token=current_token,
                    old_gid=old_gid,
                    old_token=old_token,
                )
            )
            return

        if int(existing_gallery.current_gid) == current_gid and str(existing_gallery.current_token) == current_token:
            existing_gallery.old_gid = old_gid
            existing_gallery.old_token = old_token
            return

        previous_current_gid = int(existing_gallery.current_gid)
        previous_current_token = str(existing_gallery.current_token)
        existing_gallery.current_gid = current_gid
        existing_gallery.current_token = current_token
        existing_gallery.old_gid = previous_current_gid
        existing_gallery.old_token = previous_current_token

    async def _download_archive_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        output_path: Path,
        retries: int = 3,
    ) -> None:
        delay_seconds = 1.0
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with output_path.open("wb") as fp:
                            async for chunk in response.content.iter_chunked(1024 * 1024):
                                if chunk:
                                    fp.write(chunk)
                        if output_path.stat().st_size <= 0:
                            raise RuntimeError("downloaded archive is empty")
                        return
                    if response.status in {429} or response.status >= 500:
                        await response.release()
                        if attempt < retries - 1:
                            await asyncio.sleep(delay_seconds)
                            delay_seconds *= 2
                            continue
                    raise RuntimeError(f"failed to download archive {url}: status={response.status}")
            except Exception as exc:
                last_error = exc
                if attempt >= retries - 1:
                    break
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"failed to download archive {url}")

    async def _prepare_archive_dataset(
        self,
        provider: EHentaiProvider,
        comic: tuple[str, str],
        comic_info: ComicInformation[tuple[str, str]],
        source_url: str,
    ) -> list[Path]:
        try:
            archives = await provider.getArchiveInformation(comic)
        except Exception as exc:
            raise RuntimeError(f"archive step=getArchiveInformation failed gid={comic[0]} token={comic[1]}: {exc}") from exc

        selected_archive = self._select_resample_archive(archives)

        try:
            download_url = await provider.getArchiveDownloadURL(comic, selected_archive)
        except Exception as exc:
            raise RuntimeError(f"archive step=getArchiveDownloadURL failed gid={comic[0]} token={comic[1]}: {exc}") from exc

        dataset_dir = self._dataset_dir(int(comic_info.id[0]), str(comic_info.id[1]))
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ehentai_archive_") as tmpdir:
            tmp_root = Path(tmpdir)
            archive_path = tmp_root / _archive_filename_from_url(download_url)
            timeout = aiohttp.ClientTimeout(total=float(self._settings.ehentai.download_timeout_seconds))
            async with aiohttp.ClientSession(timeout=timeout, **self._session_kwargs()) as session:
                try:
                    await self._download_archive_file(session=session, url=download_url, output_path=archive_path)
                except Exception as exc:
                    raise RuntimeError(f"archive step=downloadArchive failed url={download_url}: {exc}") from exc

            if not zipfile.is_zipfile(archive_path):
                raise RuntimeError(f"downloaded archive is not a valid zip file: {archive_path.name}")

            extracted_root = tmp_root / "extracted"
            try:
                _safe_extract_zip(archive_path, extracted_root)
            except Exception as exc:
                raise RuntimeError(f"archive step=extractZip failed file={archive_path.name}: {exc}") from exc

            image_paths = _iter_image_paths(extracted_root)
            if not image_paths:
                raise RuntimeError("resample archive contains no supported image files")

            shutil.move(str(extracted_root), str(dataset_dir))

        self._write_comic_info_xml(dataset_dir=dataset_dir, comic_info=comic_info, source_url=source_url)
        return _iter_image_paths(dataset_dir)

    @staticmethod
    def _ensure_keyword_registry(session: Session, tags: list[str]) -> dict[str, int]:
        tag_to_id: dict[str, int] = {}
        id_to_tag: dict[int, str] = {}

        for tag, keyword_id in session.execute(select(TagIdMap.tag, TagIdMap.keyword_id)).all():
            normalized_tag = normalize_tag(tag)
            tag_to_id[normalized_tag] = int(keyword_id)
            id_to_tag[int(keyword_id)] = normalized_tag

        for keyword_id, keyword_name in session.execute(select(Keyword.id, Keyword.name)).all():
            normalized_name = normalize_tag(keyword_name)
            id_to_tag[int(keyword_id)] = normalized_name
            tag_to_id.setdefault(normalized_name, int(keyword_id))

        next_keyword_id = (max(id_to_tag.keys()) + 1) if id_to_tag else 1
        for raw_tag in tags:
            normalized_tag = normalize_tag(raw_tag)
            if not normalized_tag or normalized_tag in tag_to_id:
                continue
            while next_keyword_id in id_to_tag:
                next_keyword_id += 1
            tag_to_id[normalized_tag] = next_keyword_id
            id_to_tag[next_keyword_id] = normalized_tag
            next_keyword_id += 1

        for keyword_id, keyword_name in sorted(id_to_tag.items()):
            keyword = session.get(Keyword, int(keyword_id))
            if keyword is None:
                session.add(Keyword(id=int(keyword_id), name=keyword_name))
            elif keyword.name != keyword_name:
                keyword.name = keyword_name

            tag_row = session.get(TagIdMap, keyword_name)
            if tag_row is None:
                session.add(TagIdMap(tag=keyword_name, keyword_id=int(keyword_id)))
            else:
                current_keyword_id = int(cast(int, getattr(tag_row, "keyword_id")))
                if current_keyword_id != int(keyword_id):
                    tag_row.keyword_id = int(keyword_id)

        session.flush()
        return tag_to_id

    @staticmethod
    def _upsert_pack_metadata(
        session: Session,
        comic_info: ComicInformation[tuple[str, str]],
        source: str,
        keyword_ids: list[int],
        existing_pack_id: int | None = None,
    ) -> int:
        pack = session.get(Pack, int(existing_pack_id)) if existing_pack_id is not None else None
        if pack is None:
            pack = session.execute(select(Pack).where(Pack.source == source)).scalar_one_or_none()
        if pack is None:
            pack = Pack(title=comic_info.title, source=source)
            session.add(pack)
            session.flush()
        else:
            if pack.title != comic_info.title:
                pack.title = comic_info.title
            if pack.source != source:
                pack.source = source

        pack_id_value = int(cast(Any, pack).pack_id)

        # Keep pack_id stable, but refresh keyword links to reflect latest metadata.
        session.query(PackKeyword).filter(PackKeyword.pack_id == pack_id_value).delete(synchronize_session=False)

        for keyword_id in keyword_ids:
            if session.get(Keyword, int(keyword_id)) is None:
                session.add(Keyword(id=int(keyword_id), name=f"keyword_{int(keyword_id)}"))
            existing_pack_keyword = session.get(PackKeyword, (pack_id_value, int(keyword_id)))
            if existing_pack_keyword is None:
                session.add(PackKeyword(pack_id=pack_id_value, keyword_id=int(keyword_id)))

        session.flush()
        return int(cast(Any, pack).pack_id)

    def _build_payload(
        self,
        *,
        pack_id: int,
        keyword_ids: list[int],
        source: str,
        gid: int,
        token: str,
        page_no: int,
        source_type: str,
        title: str,
        category: str,
        crop_bbox: list[int] | None = None,
        crop_score: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pack_id": int(pack_id),
            "keyword_ids": [int(keyword_id) for keyword_id in keyword_ids],
            "source": source,
            "gallery_gid": int(gid),
            "gallery_token": token,
            "page_no": int(page_no),
            "source_type": source_type,
            "title": title,
            "category": category,
        }
        if crop_bbox is not None:
            payload["crop_bbox"] = [int(value) for value in crop_bbox]
        if crop_score is not None:
            payload["crop_score"] = round(float(crop_score), 6)
        return payload

    def _upsert_points(self, points: list[qm.PointStruct]) -> None:
        if not points:
            return
        self._search_service.qdrant.upsert(collection_name=self._search_service.collection_name, points=points, wait=True)

    @staticmethod
    def _point_struct(*, point_id: Any, vector: Any, payload: Any) -> Any:
        point_factory = cast(Any, qm.PointStruct)
        return point_factory(id=point_id, vector=vector, payload=payload)

    async def ingest_url(
        self,
        url: str,
        db: Session,
        crop_faces: bool = True,
        should_cancel: Any = None,
    ) -> dict[str, Any]:
        self._check_cancel(should_cancel)
        provider = self._provider()
        comic = provider.parseURL(url)
        gid = int(comic[0])
        token = str(comic[1])

        comic_info = await provider.getComicInformation(comic)
        resolved_gid = int(comic_info.id[0])
        resolved_token = str(comic_info.id[1])
        resolved_old_gid = int(comic_info.old[0])
        resolved_old_token = str(comic_info.old[1])
        source = self._gallery_source(resolved_gid, resolved_token)

        existing_gallery = cast(EHentaiComicInfo | None, db.get(EHentaiComicInfo, resolved_gid))
        if existing_gallery is not None and existing_gallery.current_token == resolved_token:
            pack = db.execute(select(Pack.pack_id, Pack.title, Pack.source).where(Pack.source == source)).one_or_none()
            return {
                "status": "duplicate",
                "pack_id": int(pack.pack_id) if pack is not None else None,
                "title": pack.title if pack is not None else comic_info.title,
                "source": source,
                "gid": resolved_gid,
                "token": resolved_token,
                "page_count": int(comic_info.page_count),
                "indexed_points": 0,
                "crop_points": 0,
                "keyword_ids": [],
                "gallerykey_changed": False,
            }

        gallerykey_changed = existing_gallery is not None and existing_gallery.current_token != resolved_token
        existing_pack = self._find_existing_pack_for_gid(db, resolved_gid)
        existing_pack_id = int(existing_pack.pack_id) if existing_pack is not None else None
        if gallerykey_changed:
            self._delete_gallery_points(resolved_gid)
            self._cleanup_gid_datasets(resolved_gid, keep_token=resolved_token)
            db.commit()

        archive_attempted = False
        archive_used = False
        archive_error: str | None = None
        archive_dataset_dir: str | None = None

        archive_image_paths: list[Path] | None = None
        if hasattr(provider, "getArchiveInformation") and hasattr(provider, "getArchiveDownloadURL"):
            archive_attempted = True
            try:
                archive_image_paths = await self._prepare_archive_dataset(
                    provider=provider,
                    comic=(str(resolved_gid), resolved_token),
                    comic_info=comic_info,
                    source_url=url,
                )
                archive_used = True
                archive_dataset_dir = str(self._dataset_dir(resolved_gid, resolved_token))
                assert archive_image_paths is not None
                archive_image_count = len(archive_image_paths)
                logger.info("ehentai archive extracted gid=%s token=%s dir=%s images=%s", resolved_gid, resolved_token, archive_dataset_dir, archive_image_count)
            except Exception as exc:
                archive_error = str(exc)
                logger.warning("ehentai archive flow failed gid=%s token=%s error=%s", resolved_gid, resolved_token, archive_error)
                archive_image_paths = None
                if not self._settings.ehentai.allow_archive_fallback:
                    raise RuntimeError(f"archive ingest failed and fallback is disabled: {archive_error}") from exc

        page_urls: dict[int, str] = {}
        if archive_image_paths is None:
            dataset_dir = self._dataset_dir(resolved_gid, resolved_token)
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            page_numbers = list(range(1, int(comic_info.page_count) + 1))
            page_urls = await provider.getTargetPageImageURL((str(resolved_gid), resolved_token), page_numbers)
            archive_dataset_dir = str(dataset_dir)

        self._check_cancel(should_cancel)
        keyword_map = self._ensure_keyword_registry(db, comic_info.tags)
        keyword_ids = list(
            dict.fromkeys(keyword_map[normalize_tag(tag)] for tag in comic_info.tags if normalize_tag(tag) in keyword_map)
        )
        pack_id = self._upsert_pack_metadata(
            db,
            comic_info,
            source,
            keyword_ids,
            existing_pack_id=existing_pack_id,
        )
        self._upsert_ehentai_comic_info(
            db,
            ComicInformation(
                id=(str(resolved_gid), resolved_token),
                old=(str(resolved_old_gid), resolved_old_token),
                title=comic_info.title,
                subtitle=comic_info.subtitle,
                tags=comic_info.tags,
                page_count=comic_info.page_count,
                category=comic_info.category,
                cover_url=comic_info.cover_url,
                rating=comic_info.rating,
                uploader=comic_info.uploader,
                uploadTimestamp=comic_info.uploadTimestamp,
                size=comic_info.size,
            ),
            existing_gallery=existing_gallery,
        )
        db.commit()

        detector = None
        if crop_faces and self._settings.ehentai.face_crop_model:
            detector = _load_yolo_detector(self._settings.ehentai.face_crop_model)

        timeout = aiohttp.ClientTimeout(total=float(self._settings.ehentai.download_timeout_seconds))
        points: list[qm.PointStruct] = []
        indexed_points = 0
        crop_points = 0

        async with aiohttp.ClientSession(timeout=timeout, **self._session_kwargs()) as session:
            page_entries: list[tuple[int, bytes]] = []
            if archive_image_paths is not None:
                for page_no, image_path in enumerate(archive_image_paths, start=1):
                    page_entries.append((page_no, image_path.read_bytes()))
            else:
                for page_no in sorted(page_urls):
                    page_url = page_urls[page_no]
                    image_bytes = await _fetch_bytes(session, page_url)
                    output_ext = _extension_from_url(page_url)
                    output_path = Path(str(archive_dataset_dir)) / f"page_{int(page_no):04d}{output_ext}"
                    output_path.write_bytes(image_bytes)
                    page_entries.append((page_no, image_bytes))

        if archive_image_paths is None and archive_dataset_dir is not None:
            self._write_comic_info_xml(
                dataset_dir=Path(archive_dataset_dir),
                comic_info=comic_info,
                source_url=url,
            )
            logger.info("ehentai fallback dataset written gid=%s token=%s dir=%s", resolved_gid, resolved_token, archive_dataset_dir)

        for page_no, image_bytes in page_entries:
            self._check_cancel(should_cancel)
            full_vector = self._embedder.embed_bytes(image_bytes)
            full_vector_list = [float(value) for value in np.asarray(full_vector, dtype=np.float32).tolist()]
            points.append(
                self._point_struct(
                    point_id=_stable_point_id("ehentai", resolved_gid, resolved_token, page_no, "page"),
                    vector=full_vector_list,
                    payload=self._build_payload(
                        pack_id=pack_id,
                        keyword_ids=keyword_ids,
                        source=source,
                        gid=resolved_gid,
                        token=resolved_token,
                        page_no=page_no,
                        source_type="ehentai_page",
                        title=comic_info.title,
                        category=str(comic_info.category),
                    ),
                )
            )
            indexed_points += 1

            if detector is None:
                if len(points) >= 64:
                    self._upsert_points(points)
                    points.clear()
                continue

            cv_image = _image_bytes_to_cv2(image_bytes)
            detections = detect_faces(
                image=cv_image,
                detector=detector,
                confidence_threshold=float(self._settings.ehentai.face_crop_confidence_threshold),
                max_detections_per_image=int(self._settings.ehentai.face_crop_max_detections_per_image),
                class_ids=[0],
                device=self._settings.ehentai.face_crop_device,
            )
            for crop_index, detection in enumerate(detections):
                self._check_cancel(should_cancel)
                x1, y1, x2, y2 = expand_bbox(
                    detection.bbox,
                    cv_image.shape[:2],
                    float(self._settings.ehentai.face_crop_expand_ratio),
                )
                if min(x2 - x1, y2 - y1) < int(self._settings.ehentai.face_crop_min_size):
                    continue
                crop = cv_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_vector = self._embedder.embed_pil(crop_image)
                crop_vector_list = [float(value) for value in np.asarray(crop_vector, dtype=np.float32).tolist()]
                points.append(
                    self._point_struct(
                        point_id=_stable_point_id("ehentai", resolved_gid, resolved_token, page_no, "crop", crop_index, x1, y1, x2, y2),
                        vector=crop_vector_list,
                        payload=self._build_payload(
                            pack_id=pack_id,
                            keyword_ids=keyword_ids,
                            source=source,
                            gid=resolved_gid,
                            token=resolved_token,
                            page_no=page_no,
                            source_type="ehentai_face_crop",
                            title=comic_info.title,
                            category=str(comic_info.category),
                            crop_bbox=[x1, y1, x2, y2],
                            crop_score=float(detection.score),
                        ),
                    )
                )
                crop_points += 1

            if len(points) >= 64:
                self._upsert_points(points)
                points.clear()

        self._upsert_points(points)

        return {
            "status": "ok",
            "pack_id": pack_id,
            "title": comic_info.title,
            "source": source,
            "gid": resolved_gid,
            "token": resolved_token,
            "page_count": int(comic_info.page_count),
            "indexed_points": indexed_points,
            "crop_points": crop_points,
            "keyword_ids": keyword_ids,
            "gallerykey_changed": gallerykey_changed,
            "archive_attempted": archive_attempted,
            "archive_used": archive_used,
            "archive_dataset_dir": archive_dataset_dir,
            "archive_error": archive_error,
        }

