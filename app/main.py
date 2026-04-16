from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.embedder_onnx import OnnxImageEmbedder
from app.search_service import SearchService


class _AppRuntime:
    embedder: Any = None
    search_service: Any = None


def parse_keyword_ids(raw_keyword_ids: str | None) -> list[int]:
    if raw_keyword_ids is None or raw_keyword_ids.strip() == "":
        return []

    candidate = raw_keyword_ids.strip()
    if candidate.startswith("["):
        parsed = json.loads(candidate)
        if not isinstance(parsed, list):
            raise ValueError("keyword_ids must be an integer array")
        return [int(item) for item in parsed]

    return [int(item.strip()) for item in candidate.split(",") if item.strip()]


def create_app(embedder: Any | None = None, search_service: Any | None = None) -> FastAPI:
    app = FastAPI(title="ComicSearch API")
    runtime = _AppRuntime()
    runtime.embedder = embedder
    runtime.search_service = search_service
    app.state.runtime = runtime

    @app.on_event("startup")
    def _startup() -> None:
        if app.state.runtime.embedder is None:
            model_path = os.getenv("ONNX_MODEL_PATH", "models/clip_image_encoder.onnx")
            input_size = int(os.getenv("EMBEDDER_INPUT_SIZE", "224"))
            intra_threads = int(os.getenv("EMBEDDER_INTRA_THREADS", "4"))
            app.state.runtime.embedder = OnnxImageEmbedder(
                onnx_path=model_path,
                input_size=input_size,
                intra_threads=intra_threads,
            )

        if app.state.runtime.search_service is None:
            from qdrant_client import QdrantClient

            qdrant_host = os.getenv("QDRANT_HOST", "127.0.0.1")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            collection_name = os.getenv("QDRANT_COLLECTION", "pages")
            qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            app.state.runtime.search_service = SearchService(qdrant, collection_name=collection_name)

    @app.post("/search")
    async def search(
        image: UploadFile = File(...),
        keyword_ids: str | None = Form(default=None),
        robust_partial: bool = Form(default=True),
        include_corners: bool = Form(default=True),
        include_contrast: bool = Form(default=True),
        per_view_limit: int = Form(default=80),
        top_k_manga: int = Form(default=10),
        top_k_packs: int = Form(default=30),
    ) -> dict[str, Any]:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty image")

        try:
            parsed_keyword_ids = parse_keyword_ids(keyword_ids)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="keyword_ids must be int array JSON or comma-separated ints") from exc

        if robust_partial:
            vectors = app.state.runtime.embedder.multi_views(
                image_bytes,
                include_corners=include_corners,
                include_contrast=include_contrast,
            )
        else:
            vectors = [app.state.runtime.embedder.embed_bytes(image_bytes)]

        points = app.state.runtime.search_service.search_pages_multi_view(
            vectors=vectors,
            keyword_ids=parsed_keyword_ids,
            per_view_limit=max(1, int(per_view_limit)),
        )

        candidate_manga = app.state.runtime.search_service.aggregate_manga(points, top_k=max(1, int(top_k_manga)))
        best_manga = candidate_manga[0] if candidate_manga else None
        top_packs: list[dict[str, Any]] = []
        if best_manga is not None:
            top_packs = app.state.runtime.search_service.aggregate_packs_for_manga(
                points,
                manga_id=best_manga["manga_id"],
                top_k=max(1, int(top_k_packs)),
            )

        return {
            "best_manga": best_manga,
            "confidence": app.state.runtime.search_service.confidence(candidate_manga),
            "candidate_manga": candidate_manga,
            "top_packs_in_best_manga": top_packs,
        }

    return app


app = create_app()
