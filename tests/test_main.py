import asyncio
import json
import os
import tempfile
import time
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
import numpy as np

from app.config import load_settings
from app.db import get_db
from app.downloader.provider.models import ComicInformation
from app.ehentai_ingest import EhentaiIngestService
from app.models import Base, EHentaiComicInfo, Keyword, Pack, PackKeyword
from app.main import create_app, parse_keyword_ids


class ParseKeywordIdsTests(unittest.TestCase):
    def test_parse_json_array(self):
        self.assertEqual(parse_keyword_ids("[1,2,3]"), [1, 2, 3])

    def test_parse_rejects_comma_separated_format(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_keyword_ids("1,2,3")

    def test_empty(self):
        self.assertEqual(parse_keyword_ids(None), [])
        self.assertEqual(parse_keyword_ids(""), [])

    def test_parse_limit_and_int32_validation(self):
        with self.assertRaises(ValueError):
            parse_keyword_ids("[" + ",".join(str(i) for i in range(1, 22)) + "]")
        with self.assertRaises(ValueError):
            parse_keyword_ids("[2147483648]")
        with self.assertRaises(ValueError):
            parse_keyword_ids("[0]")


class ConfigLoadingTests(unittest.TestCase):
    def test_load_settings_reads_toml_and_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "comicsearch.toml"
            config_path.write_text(
                """
[embedder]
onnx_path = "models/model.onnx"
input_size = 256
intra_threads = 8

[qdrant]
host = "file-host"
port = 6334
collection = "file-pages"

[database]
url = "sqlite:///./from-config.db"

[search_defaults]
robust_partial = false
include_corners = false
include_contrast = true
per_view_limit = 120
top_k_manga = 12

[ehentai]
proxy = "http://127.0.0.1:7890"
is_exhentai = true
face_crop_model = "models/yolov8n.pt"
face_crop_device = "cpu"
face_crop_confidence_threshold = 0.4
face_crop_expand_ratio = 0.2
face_crop_min_size = 40
face_crop_max_detections_per_image = 8
download_timeout_seconds = 45

[ehentai.cookies]
ipb_member_id = "1"
ipb_pass_hash = "2"
igneous = "3"
sk = "4"
hath_perks = "5"
""".strip(),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"QDRANT_HOST": "env-host", "DATABASE_URL": "sqlite:///./override.db"},
                clear=True,
            ):
                settings = load_settings(str(config_path))

        self.assertEqual(settings.embedder.onnx_path, str((config_path.parent / "models/model.onnx").resolve()))
        self.assertEqual(settings.embedder.input_size, 256)
        self.assertEqual(settings.embedder.intra_threads, 8)
        self.assertEqual(settings.qdrant.host, "env-host")
        self.assertEqual(settings.qdrant.port, 6334)
        self.assertEqual(settings.qdrant.collection, "file-pages")
        self.assertEqual(settings.database.url, "sqlite:///./override.db")
        self.assertFalse(settings.search.robust_partial)
        self.assertFalse(settings.search.include_corners)
        self.assertTrue(settings.search.include_contrast)
        self.assertEqual(settings.search.per_view_limit, 120)
        self.assertEqual(settings.search.top_k_manga, 12)
        self.assertEqual(settings.ehentai.proxy, "http://127.0.0.1:7890")
        self.assertTrue(settings.ehentai.is_exhentai)
        self.assertEqual(settings.ehentai.cookies["ipb_member_id"], "1")
        self.assertEqual(settings.ehentai.cookies["sk"], "4")
        self.assertEqual(settings.ehentai.cookies["hath_perks"], "5")
        self.assertEqual(settings.ehentai.face_crop_confidence_threshold, 0.4)
        self.assertEqual(settings.ehentai.face_crop_expand_ratio, 0.2)
        self.assertEqual(settings.ehentai.face_crop_min_size, 40)
        self.assertEqual(settings.ehentai.face_crop_max_detections_per_image, 8)
        self.assertEqual(settings.ehentai.download_timeout_seconds, 45.0)


class SearchValidationTests(unittest.TestCase):
    @staticmethod
    def _build_client() -> TestClient:
        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        app = create_app(embedder=embedder, search_service=search_service)
        return TestClient(app)

    def test_search_rejects_unsupported_image_type(self):
        client = self._build_client()
        response = client.post(
            "/search",
            files={"image": ("a.txt", b"123", "text/plain")},
            data={"keyword_ids": "[1]"},
        )
        self.assertEqual(response.status_code, 415)

    def test_search_rejects_oversized_image(self):
        client = self._build_client()
        response = client.post(
            "/search",
            files={"image": ("a.png", b"a" * (10 * 1024 * 1024 + 1), "image/png")},
            data={"keyword_ids": "[1]"},
        )
        self.assertEqual(response.status_code, 413)

    def test_search_rejects_out_of_range_top_k(self):
        client = self._build_client()
        response = client.post(
            "/search",
            files={"image": ("a.png", b"123", "image/png")},
            data={"keyword_ids": "[1]", "top_k_manga": "51"},
        )
        self.assertEqual(response.status_code, 422)

    def test_search_rejects_too_many_keyword_ids(self):
        client = self._build_client()
        keyword_ids = "[" + ",".join(str(i) for i in range(1, 22)) + "]"
        response = client.post(
            "/search",
            files={"image": ("a.png", b"123", "image/png")},
            data={"keyword_ids": keyword_ids},
        )
        self.assertEqual(response.status_code, 400)

    def test_search_defaults_include_contrast_to_false(self):
        calls: list[dict[str, bool]] = []

        def _multi_views(image_bytes, include_corners, include_contrast):
            _ = image_bytes
            calls.append({"include_corners": include_corners, "include_contrast": include_contrast})
            return [[0.1]]

        embedder = SimpleNamespace(
            multi_views=_multi_views,
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        client = TestClient(create_app(embedder=embedder, search_service=search_service))

        response = client.post(
            "/search",
            files={"image": ("a.png", b"123", "image/png")},
            data={},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["include_contrast"])


class EhentaiImportEndpointTests(unittest.TestCase):
    def test_sync_import_endpoint_is_removed(self):
        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(upsert=lambda **kwargs: None),
            collection_name="pages",
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        app = create_app(embedder=embedder, search_service=search_service)
        client = TestClient(app)

        response = client.post("/ehentai/import", json={"url": "https://e-hentai.org/g/123/abc/", "crop_faces": True})

        self.assertEqual(response.status_code, 404)

    def test_import_task_submit_and_query_status(self):
        calls: list[dict[str, object]] = []

        async def _ingest_url(url: str, db, crop_faces: bool, should_cancel=None):
            _ = db
            _ = should_cancel
            calls.append({"url": url, "crop_faces": crop_faces})
            await asyncio.sleep(0.01)
            return {"status": "ok", "source": url, "crop_faces": crop_faces}

        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(upsert=lambda **kwargs: None),
            collection_name="pages",
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        ingest_service = SimpleNamespace(ingest_url=_ingest_url)
        app = create_app(
            embedder=embedder,
            search_service=search_service,
            ehentai_ingest_service=ingest_service,
        )
        client = TestClient(app)

        submit_response = client.post(
            "/ehentai/import/tasks",
            json={"url": "https://e-hentai.org/g/123/abc/", "crop_faces": False},
        )
        self.assertEqual(submit_response.status_code, 202)
        task_id = submit_response.json()["task_id"]
        self.assertTrue(task_id)

        final_payload = None
        for _ in range(60):
            status_response = client.get(f"/tasks/{task_id}")
            self.assertEqual(status_response.status_code, 200)
            payload = status_response.json()
            if payload["status"] in {"success", "failed"}:
                final_payload = payload
                break
            time.sleep(0.01)

        self.assertIsNotNone(final_payload)
        assert final_payload is not None
        self.assertEqual(final_payload["status"], "success")
        self.assertEqual(final_payload["task_type"], "ehentai_import")
        self.assertEqual(final_payload["result"]["status"], "ok")
        self.assertEqual(calls, [{"url": "https://e-hentai.org/g/123/abc/", "crop_faces": False}])

    def test_task_list_and_cancel(self):
        started = threading.Event()
        release = threading.Event()

        async def _ingest_url(url: str, db, crop_faces: bool, should_cancel=None):
            _ = url, db, crop_faces
            started.set()
            while not release.is_set():
                if should_cancel is not None and should_cancel():
                    raise RuntimeError("cancelled by test")
                await asyncio.sleep(0.01)
            return {"status": "ok"}

        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(upsert=lambda **kwargs: None),
            collection_name="pages",
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        ingest_service = SimpleNamespace(ingest_url=_ingest_url)
        app = create_app(embedder=embedder, search_service=search_service, ehentai_ingest_service=ingest_service)
        client = TestClient(app)

        submit_response = client.post(
            "/ehentai/import/tasks",
            json={"url": "https://e-hentai.org/g/222/def/", "crop_faces": False},
        )
        task_id = submit_response.json()["task_id"]
        self.assertTrue(started.wait(timeout=1.0))

        list_response = client.get("/tasks", params={"limit": 10})
        self.assertEqual(list_response.status_code, 200)
        task_ids = [row["task_id"] for row in list_response.json()]
        self.assertIn(task_id, task_ids)

        cancel_response = client.post(f"/tasks/{task_id}/cancel")
        self.assertEqual(cancel_response.status_code, 200)
        self.assertTrue(cancel_response.json()["cancel_requested"])

        release.set()
        final_payload = None
        for _ in range(80):
            status_response = client.get(f"/tasks/{task_id}")
            payload = status_response.json()
            if payload["status"] in {"cancelled", "failed", "success"}:
                final_payload = payload
                break
            time.sleep(0.01)
        self.assertIsNotNone(final_payload)
        assert final_payload is not None
        self.assertIn(final_payload["status"], {"cancelled", "failed"})

    def test_task_status_returns_404_for_unknown_task(self):
        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(upsert=lambda **kwargs: None),
            collection_name="pages",
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        app = create_app(embedder=embedder, search_service=search_service, ehentai_ingest_service=SimpleNamespace())
        client = TestClient(app)

        response = client.get("/tasks/not_exists")
        self.assertEqual(response.status_code, 404)

    def test_submit_same_payload_reuses_active_task(self):
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        async def _ingest_url(url: str, db, crop_faces: bool, should_cancel=None):
            _ = db, crop_faces, should_cancel
            calls.append(url)
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return {"status": "ok", "source": url}

        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(upsert=lambda **kwargs: None),
            collection_name="pages",
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        ingest_service = SimpleNamespace(ingest_url=_ingest_url)
        app = create_app(embedder=embedder, search_service=search_service, ehentai_ingest_service=ingest_service)
        client = TestClient(app)

        first = client.post(
            "/ehentai/import/tasks",
            json={"url": "https://e-hentai.org/g/333/ghi/", "crop_faces": True},
        )
        self.assertEqual(first.status_code, 202)
        first_payload = first.json()
        self.assertEqual(first_payload["status"], "pending")
        self.assertTrue(started.wait(timeout=1.0))

        second = client.post(
            "/ehentai/import/tasks",
            json={"url": "https://e-hentai.org/g/333/ghi/", "crop_faces": True},
        )
        self.assertEqual(second.status_code, 202)
        second_payload = second.json()

        self.assertEqual(second_payload["task_id"], first_payload["task_id"])
        self.assertEqual(second_payload["status"], "duplicate")
        self.assertEqual(len(calls), 1)

        release.set()

    def test_submit_multiple_urls_returns_ordered_items_with_duplicate_flag(self):
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        async def _ingest_url(url: str, db, crop_faces: bool, should_cancel=None):
            _ = db, crop_faces, should_cancel
            calls.append(url)
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return {"status": "ok", "source": url}

        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(upsert=lambda **kwargs: None),
            collection_name="pages",
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        ingest_service = SimpleNamespace(ingest_url=_ingest_url)
        app = create_app(embedder=embedder, search_service=search_service, ehentai_ingest_service=ingest_service)
        client = TestClient(app)

        unique_suffix = str(time.time_ns())
        first_url = f"https://e-hentai.org/g/444{unique_suffix}/jkl/"
        second_url = f"https://e-hentai.org/g/555{unique_suffix}/mno/"

        response = client.post(
            "/ehentai/import/tasks",
            json={
                "urls": [
                    first_url,
                    second_url,
                    first_url,
                ],
                "crop_faces": True,
            },
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertIn("items", payload)
        items = payload["items"]
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["url"], first_url)
        self.assertEqual(items[1]["url"], second_url)
        self.assertEqual(items[2]["url"], first_url)
        self.assertFalse(items[0]["is_duplicate"])
        self.assertFalse(items[1]["is_duplicate"])
        self.assertTrue(items[2]["is_duplicate"])
        self.assertEqual(items[2]["task_id"], items[0]["task_id"])
        self.assertEqual(items[2]["status"], "duplicate")

        self.assertTrue(started.wait(timeout=1.0))
        for _ in range(80):
            if len(calls) >= 2:
                break
            time.sleep(0.01)
        self.assertEqual(len(calls), 2)
        release.set()


class EhentaiIngestServiceTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        self.session_cls = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)
        self.settings = load_settings()

    def _make_service(self, provider, upserts: list[dict[str, object]]):
        embedder = SimpleNamespace(
            embed_bytes=lambda image_bytes: np.array([0.1], dtype=np.float32),
            embed_pil=lambda image: np.array([0.1], dtype=np.float32),
        )
        search_service = SimpleNamespace(
            qdrant=SimpleNamespace(
                upsert=lambda **kwargs: upserts.append({"op": "upsert", **kwargs}),
                delete=lambda **kwargs: upserts.append({"op": "delete", **kwargs}),
            ),
            collection_name="pages",
        )
        service = EhentaiIngestService(self.settings, cast(Any, embedder), cast(Any, search_service))
        service._provider = lambda: provider  # type: ignore[method-assign]
        return service

    def test_ingest_url_skips_when_gid_and_token_are_unchanged(self):
        class _Provider:
            def __init__(self):
                self.comic_calls = 0
                self.page_calls = 0

            def parseURL(self, url: str):
                _ = url
                return ("123", "abc")

            async def getComicInformation(self, comic):
                _ = comic
                self.comic_calls += 1
                return SimpleNamespace(
                    id=("123", "abc"),
                    old=("123", "abc"),
                    title="demo",
                    subtitle=None,
                    tags=["tag1"],
                    page_count=1,
                    category=ComicInformation.Category.Manga,
                    cover_url="https://example.com/cover.jpg",
                    rating=4.0,
                    uploader="u",
                    uploadTimestamp=1,
                    size=None,
                )

            async def getTargetPageImageURL(self, comic, page):
                _ = comic, page
                self.page_calls += 1
                raise AssertionError("should not download pages for duplicate galleries")

        provider = _Provider()
        upserts: list[dict[str, object]] = []
        service = self._make_service(provider, upserts)

        with self.session_cls() as db:
            db.add(EHentaiComicInfo(current_gid=123, current_token="abc", old_gid=123, old_token="abc"))
            db.commit()

            result = asyncio.run(service.ingest_url("https://e-hentai.org/g/123/abc/", db, crop_faces=False))

        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(result["gid"], 123)
        self.assertEqual(result["token"], "abc")
        self.assertEqual(provider.comic_calls, 1)
        self.assertEqual(provider.page_calls, 0)
        self.assertEqual(upserts, [])

    def test_ingest_url_stores_current_and_old_on_initial_import(self):
        class _Provider:
            def __init__(self):
                self.comic_calls = 0
                self.page_calls = 0

            def parseURL(self, url: str):
                _ = url
                return ("123", "new")

            async def getComicInformation(self, comic):
                _ = comic
                self.comic_calls += 1
                return SimpleNamespace(
                    id=("123", "new"),
                    old=("123", "old"),
                    title="demo",
                    subtitle=None,
                    tags=["tag1"],
                    page_count=1,
                    category=ComicInformation.Category.Manga,
                    cover_url="https://example.com/cover.jpg",
                    rating=4.0,
                    uploader="u",
                    uploadTimestamp=1,
                    size=None,
                )

            async def getTargetPageImageURL(self, comic, page):
                _ = comic, page
                self.page_calls += 1
                return {1: "https://example.com/page.jpg"}

        provider = _Provider()
        upserts: list[dict[str, object]] = []
        service = self._make_service(provider, upserts)

        async def _fetch_bytes(_session, _url, retries=3):
            _ = retries
            return b"not-an-image"

        from app import ehentai_ingest as ingest_module

        original_fetch_bytes = ingest_module._fetch_bytes
        ingest_module._fetch_bytes = _fetch_bytes  # type: ignore[assignment]
        try:
            with self.session_cls() as db:
                result = asyncio.run(service.ingest_url("https://e-hentai.org/g/123/new/", db, crop_faces=False))

                stored = db.get(EHentaiComicInfo, 123)
                self.assertIsNotNone(stored)
                assert stored is not None
                self.assertEqual(stored.current_gid, 123)
                self.assertEqual(stored.current_token, "new")
                self.assertEqual(stored.old_gid, 123)
                self.assertEqual(stored.old_token, "old")
        finally:
            ingest_module._fetch_bytes = original_fetch_bytes  # type: ignore[assignment]

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["gallerykey_changed"])
        self.assertEqual(result["gid"], 123)
        self.assertEqual(result["token"], "new")
        self.assertEqual(provider.page_calls, 1)
        self.assertGreaterEqual(len(upserts), 1)

    def test_ingest_url_updates_gallerykey_when_token_changes(self):
        class _Provider:
            def __init__(self):
                self.comic_calls = 0
                self.page_calls = 0

            def parseURL(self, url: str):
                _ = url
                return ("123", "new")

            async def getComicInformation(self, comic):
                _ = comic
                self.comic_calls += 1
                return SimpleNamespace(
                    id=("123", "new"),
                    old=("123", "old"),
                    title="demo",
                    subtitle=None,
                    tags=["tag1"],
                    page_count=1,
                    category=ComicInformation.Category.Manga,
                    cover_url="https://example.com/cover.jpg",
                    rating=4.0,
                    uploader="u",
                    uploadTimestamp=1,
                    size=None,
                )

            async def getTargetPageImageURL(self, comic, page):
                _ = comic, page
                self.page_calls += 1
                return {1: "https://example.com/page.jpg"}

        provider = _Provider()
        upserts: list[dict[str, object]] = []
        service = self._make_service(provider, upserts)

        async def _fetch_bytes(_session, _url, retries=3):
            _ = retries
            return b"not-an-image"

        from app import ehentai_ingest as ingest_module

        original_fetch_bytes = ingest_module._fetch_bytes
        ingest_module._fetch_bytes = _fetch_bytes  # type: ignore[assignment]
        try:
            with self.session_cls() as db:
                db.add(EHentaiComicInfo(current_gid=123, current_token="old", old_gid=123, old_token="prev"))
                db.commit()

                result = asyncio.run(service.ingest_url("https://e-hentai.org/g/123/new/", db, crop_faces=False))

                stored = db.get(EHentaiComicInfo, 123)
                self.assertIsNotNone(stored)
                assert stored is not None
                self.assertEqual(stored.current_gid, 123)
                self.assertEqual(stored.current_token, "new")
                self.assertEqual(stored.old_gid, 123)
                self.assertEqual(stored.old_token, "old")
        finally:
            ingest_module._fetch_bytes = original_fetch_bytes  # type: ignore[assignment]

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["gallerykey_changed"])
        self.assertEqual(result["gid"], 123)
        self.assertEqual(result["token"], "new")
        self.assertEqual(provider.page_calls, 1)
        self.assertGreaterEqual(len(upserts), 1)
        self.assertEqual(upserts[0].get("op"), "delete")

    def test_search_uses_config_defaults_when_form_fields_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "comicsearch.toml"
            config_path.write_text(
                """
[search_defaults]
robust_partial = false
include_corners = false
include_contrast = true
per_view_limit = 15
top_k_manga = 3
""".strip(),
                encoding="utf-8",
            )

            calls: list[str] = []
            search_calls: list[int] = []
            top_k_calls: list[int] = []

            def _multi_views(image_bytes, include_corners, include_contrast):
                _ = image_bytes, include_corners, include_contrast
                calls.append("multi_views")
                return [[0.1]]

            def _embed_bytes(image_bytes):
                _ = image_bytes
                calls.append("embed_bytes")
                return [0.1]

            embedder = SimpleNamespace(
                multi_views=_multi_views,
                embed_bytes=_embed_bytes,
            )

            def _search_pages_multi_view(vectors, keyword_ids, per_view_limit):
                _ = vectors, keyword_ids
                search_calls.append(per_view_limit)
                return []

            def _aggregate_manga(points, top_k):
                _ = points
                top_k_calls.append(top_k)
                return []

            search_service = SimpleNamespace(
                search_pages_multi_view=_search_pages_multi_view,
                aggregate_manga=_aggregate_manga,
                aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
                confidence=lambda candidates: "low",
            )

            client = TestClient(create_app(config_path=str(config_path), embedder=embedder, search_service=search_service))
            response = client.post(
                "/search",
                files={"image": ("a.png", b"123", "image/png")},
                data={},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, ["embed_bytes"])
        self.assertEqual(search_calls, [15])
        self.assertEqual(top_k_calls, [3])


class PackInfoEndpointTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        self.session_cls = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)

        with self.session_cls() as session:
            session.add(Pack(pack_id=11, title="demo pack", source="https://example.com/demo-pack"))
            session.add(Keyword(id=100, name="action"))
            session.add(Keyword(id=101, name="romance"))
            session.add(PackKeyword(pack_id=11, keyword_id=101))
            session.add(PackKeyword(pack_id=11, keyword_id=100))
            session.commit()

        embedder = SimpleNamespace(
            multi_views=lambda image_bytes, include_corners, include_contrast: [[0.1]],
            embed_bytes=lambda image_bytes: [0.1],
        )
        search_service = SimpleNamespace(
            search_pages_multi_view=lambda vectors, keyword_ids, per_view_limit: [],
            aggregate_manga=lambda points, top_k: [],
            aggregate_packs_for_manga=lambda points, pack_id, top_k: [],
            confidence=lambda candidates: "low",
        )
        app = create_app(embedder=embedder, search_service=search_service)

        def _override_get_db():
            db = self.session_cls()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _override_get_db
        self.client = TestClient(app)

    def test_info_returns_pack_details(self):
        response = self.client.get("/info/11")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "pack_id": 11,
                "title": "demo pack",
                "source": "https://example.com/demo-pack",
                "keyword_ids": [100, 101],
                "keywords": [
                    {"id": 100, "name": "action"},
                    {"id": 101, "name": "romance"},
                ],
            },
        )

    def test_info_query_returns_pack_details(self):
        response = self.client.get("/info", params={"id": 11})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pack_id"], 11)

    def test_info_returns_404_when_pack_missing(self):
        response = self.client.get("/info/999")
        self.assertEqual(response.status_code, 404)

    def test_info_query_returns_404_when_pack_missing(self):
        response = self.client.get("/info", params={"id": 999})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
