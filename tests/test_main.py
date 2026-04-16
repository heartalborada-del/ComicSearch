import unittest
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app, parse_keyword_ids


class ParseKeywordIdsTests(unittest.TestCase):
    def test_parse_json_array(self):
        self.assertEqual(parse_keyword_ids("[1,2,3]"), [1, 2, 3])

    def test_parse_reject_non_json_format(self):
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
            aggregate_packs_for_manga=lambda points, manga_id, top_k: [],
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
            data={"keyword_ids": "[1]", "top_k_packs": "101"},
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


if __name__ == "__main__":
    unittest.main()
