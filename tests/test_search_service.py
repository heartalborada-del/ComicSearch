import unittest
from types import SimpleNamespace

from qdrant_client import QdrantClient
from qdrant_client.qdrant_remote import QdrantRemote

from app.search_service import SearchService


class SearchServiceTests(unittest.TestCase):
    def test_aggregate_and_confidence(self):
        service = SearchService(qdrant_client=None)
        points = [
            SimpleNamespace(id=1, score=0.9, payload={"pack_id": 11, "cover_thumb_path": "a.jpg"}),
            SimpleNamespace(id=2, score=0.87, payload={"pack_id": 11, "cover_thumb_path": "a.jpg"}),
            SimpleNamespace(id=3, score=0.85, payload={"pack_id": 12, "cover_thumb_path": "b.jpg"}),
            SimpleNamespace(id=4, score=0.8, payload={"pack_id": 21, "cover_thumb_path": "c.jpg"}),
        ]

        manga = service.aggregate_manga(points)
        self.assertEqual(manga[0]["pack_id"], 11)
        self.assertIn(service.confidence(manga), {"medium", "high"})

        packs = service.aggregate_packs_for_manga(points, pack_id=11)
        self.assertEqual(packs[0]["pack_id"], 11)
        self.assertEqual(packs[0]["cover_thumb_path"], "a.jpg")

    def test_merge_max_score_per_point(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def search(self, **kwargs):
                _ = kwargs
                self.calls += 1
                if self.calls == 1:
                    return [
                        SimpleNamespace(id=1, score=0.6, payload={"pack_id": 1}),
                        SimpleNamespace(id=2, score=0.7, payload={"pack_id": 2}),
                    ]
                return [
                    SimpleNamespace(id=1, score=0.9, payload={"pack_id": 1}),
                ]

        service = SearchService(FakeClient())
        merged = service.search_pages_multi_view(vectors=[[0.0], [1.0]], keyword_ids=[1])
        best = {point.id: point.score for point in merged}
        self.assertEqual(best[1], 0.9)
        self.assertEqual(best[2], 0.7)


if __name__ == "__main__":
    unittest.main()
