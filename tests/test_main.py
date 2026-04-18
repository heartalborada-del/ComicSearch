import unittest
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.models import Base, Keyword, Pack, PackKeyword
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
