from __future__ import annotations

import logging
import sys
from collections import defaultdict
from logging import info
from math import log
from typing import Any

import numpy as np
import uvicorn.main
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

class SearchService:
    """Qdrant vector search and manga-first aggregation service."""
    # Log hit-count bonus weight used when scoring aggregated manga candidates.
    HITS_BONUS_WEIGHT = 0.03

    def __init__(self, qdrant_client: QdrantClient, collection_name: str = "pages") -> None:
        self.qdrant = qdrant_client
        self.collection_name = collection_name

    def _and_keyword_filter(self, keyword_ids: list[int] | None):
        if not keyword_ids:
            return None
        return qm.Filter(
            must=[qm.FieldCondition(key="keyword_ids", match=qm.MatchValue(value=int(keyword_id))) for keyword_id in keyword_ids]
        )

    def _query_points(self, vector_value: list[float] | Any, query_filter: Any, per_view_limit: int) -> list[Any]:
        base_kwargs = {
            "collection_name": self.collection_name,
            "query_filter": query_filter,
            "limit": per_view_limit,
            "with_payload": True,
            "with_vectors": False,
        }

        if hasattr(self.qdrant, "query_points"):
            result = self.qdrant.query_points(query=vector_value, **base_kwargs)
            return list(getattr(result, "points", result))

        if hasattr(self.qdrant, "search"):
            # Backward compatibility with older qdrant-client releases.
            try:
                result = self.qdrant.search(query_vector=vector_value, **base_kwargs)
            except (AssertionError, TypeError) as exc:
                if "query_vector" not in str(exc):
                    raise
                result = self.qdrant.search(query=vector_value, **base_kwargs)
            return list(getattr(result, "points", result))

        raise AttributeError("Qdrant client must provide query_points() or search()")

    def search_pages_multi_view(
        self,
        vectors: list[list[float] | Any],
        keyword_ids: list[int] | None,
        per_view_limit: int = 80,
    ) -> list[Any]:
        query_filter = self._and_keyword_filter(keyword_ids)
        best_by_point_id: dict[Any, Any] = {}

        for vector in vectors:
            vector_value = vector.tolist() if isinstance(vector, np.ndarray) else vector
            points = self._query_points(vector_value=vector_value, query_filter=query_filter, per_view_limit=per_view_limit)
            for point in points:
                if point.id not in best_by_point_id or float(point.score) > float(best_by_point_id[point.id].score):
                    best_by_point_id[point.id] = point

        return list(best_by_point_id.values())

    def aggregate_manga(self, points: list[Any], top_k: int = 10) -> list[dict[str, Any]]:
        grouped: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        for point in points:
            payload = point.payload or {}
            pack_id = payload.get("pack_id")
            if pack_id is None:
                continue
            grouped[int(pack_id)].append((float(point.score), payload))

        ranked: list[dict[str, Any]] = []
        for pack_id, items in grouped.items():
            items.sort(key=lambda item: item[0], reverse=True)
            top_scores = [score for score, _ in items[:5]]
            avg_top = sum(top_scores) / len(top_scores)
            hits = len(items)
            score = avg_top + self.HITS_BONUS_WEIGHT * log(1 + hits)
            ranked.append(
                {
                    "pack_id": pack_id,
                    "score": round(score, 6),
                    "hits": hits,
                    "top1_score": round(top_scores[0], 6),
                }
            )

        ranked.sort(key=lambda row: (-row["score"], -row["hits"], row["pack_id"]))
        return ranked[:top_k]

    def aggregate_packs_for_manga(self, points: list[Any], pack_id: int, top_k: int = 30) -> list[dict[str, Any]]:
        grouped: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        for point in points:
            payload = point.payload or {}
            if int(payload.get("pack_id", -1)) != int(pack_id):
                continue
            if "pack_id" not in payload:
                continue
            grouped[int(payload["pack_id"])].append((float(point.score), payload))

        ranked: list[dict[str, Any]] = []
        for pack_id, items in grouped.items():
            items.sort(key=lambda item: item[0], reverse=True)
            top3 = [score for score, _ in items[:3]]
            score = sum(top3) / len(top3)
            ranked.append(
                {
                    "pack_id": pack_id,
                    "score": round(score, 6),
                    "hits": len(items),
                    "cover_thumb_path": items[0][1].get("cover_thumb_path"),
                }
            )

        ranked.sort(key=lambda row: (-row["score"], -row["hits"], row["pack_id"]))
        return ranked[:top_k]

    def confidence(self, manga_candidates: list[dict[str, Any]]) -> str:
        if not manga_candidates:
            return "low"
        if len(manga_candidates) == 1:
            return "high" if manga_candidates[0]["hits"] >= 2 else "medium"

        first, second = manga_candidates[0], manga_candidates[1]
        margin = first["score"] - second["score"]
        if first["hits"] >= 3 and margin >= 0.05:
            return "high"
        if first["hits"] >= 2 and margin >= 0.02:
            return "medium"
        return "low"
