import unittest
from types import SimpleNamespace

from qdrant_client import QdrantClient
from qdrant_client.qdrant_remote import QdrantRemote

from app.search_service import SearchService


class SearchServiceTests(unittest.TestCase):
    def test_aggregate_and_confidence(self):
        service = SearchService(qdrant_client=None)
        points = [
            SimpleNamespace(id=1, score=0.9, payload={"pack_id": 11, "page_no": 7, "cover_thumb_path": "a.jpg"}),
            SimpleNamespace(id=2, score=0.87, payload={"pack_id": 11, "page_no": 4, "cover_thumb_path": "a.jpg"}),
            SimpleNamespace(id=3, score=0.85, payload={"pack_id": 12, "page_no": 8, "cover_thumb_path": "b.jpg"}),
            SimpleNamespace(id=4, score=0.8, payload={"pack_id": 21, "page_no": 2, "cover_thumb_path": "c.jpg"}),
        ]

        manga = service.aggregate_manga(points)
        self.assertEqual(manga[0]["pack_id"], 11)
        self.assertEqual(manga[0]["top_page_no"], 7)
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

    def test_aggregate_prefers_strong_top_match_over_many_mediocre_hits(self):
        service = SearchService(qdrant_client=None)
        points = [
            SimpleNamespace(id=1, score=0.86, payload={"pack_id": 101}),
            SimpleNamespace(id=2, score=0.55, payload={"pack_id": 101}),
            SimpleNamespace(id=3, score=0.79, payload={"pack_id": 202}),
            SimpleNamespace(id=4, score=0.79, payload={"pack_id": 202}),
            SimpleNamespace(id=5, score=0.79, payload={"pack_id": 202}),
            SimpleNamespace(id=6, score=0.79, payload={"pack_id": 202}),
            SimpleNamespace(id=7, score=0.79, payload={"pack_id": 202}),
        ]

        ranked = service.aggregate_manga(points, top_k=2)
        self.assertEqual(ranked[0]["pack_id"], 101)

    def test_aggregate_top_page_no_none_when_missing(self):
        service = SearchService(qdrant_client=None)
        points = [SimpleNamespace(id=1, score=0.9, payload={"pack_id": 1})]

        ranked = service.aggregate_manga(points, top_k=1)
        self.assertIsNone(ranked[0]["top_page_no"])

    def test_confidence_high_for_single_hit_when_top1_reaches_high_threshold(self):
        service = SearchService(qdrant_client=None)
        candidates = [
            {
                "pack_id": 1,
                "score": 0.91,
                "hits": 1,
                "top1_score": 0.995,
            }
        ]

        self.assertEqual(service.confidence(candidates), "high")

    def test_confidence_medium_for_single_hit_when_top1_reaches_medium_threshold(self):
        service = SearchService(qdrant_client=None)
        candidates = [
            {
                "pack_id": 1,
                "score": 0.89,
                "hits": 1,
                "top1_score": 0.985,
            }
        ]

        self.assertEqual(service.confidence(candidates), "medium")

    def test_confidence_falls_back_to_margin_logic_below_top1_thresholds(self):
        service = SearchService(qdrant_client=None)
        candidates = [
            {
                "pack_id": 1,
                "score": 0.96,
                "hits": 3,
                "top1_score": 0.98,
            },
            {
                "pack_id": 2,
                "score": 0.90,
                "hits": 2,
                "top1_score": 0.89,
            },
        ]

        self.assertEqual(service.confidence(candidates), "high")


if __name__ == "__main__":
    unittest.main()
