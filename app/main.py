from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.embedder_onnx import OnnxImageEmbedder
from app.search_service import SearchService


class _AppRuntime:
    embedder: Any = None
    search_service: Any = None


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_KEYWORD_IDS = 20
INT32_MAX = 2_147_483_647


def parse_keyword_ids(raw_keyword_ids: str | None) -> list[int]:
    if raw_keyword_ids is None or raw_keyword_ids.strip() == "":
        return []

    candidate = raw_keyword_ids.strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, list):
        raise ValueError("keyword_ids must be a JSON integer array")
    if len(parsed) > MAX_KEYWORD_IDS:
        raise ValueError(f"keyword_ids must contain at most {MAX_KEYWORD_IDS} items")

    keyword_ids: list[int] = []
    for item in parsed:
        if type(item) is not int:
            raise ValueError("keyword_ids entries must be integers")
        if item <= 0 or item > INT32_MAX:
            raise ValueError("keyword_ids entries must be positive int32 values")
        keyword_ids.append(item)
    return keyword_ids


def create_app(embedder: Any | None = None, search_service: Any | None = None) -> FastAPI:
    runtime = _AppRuntime()
    runtime.embedder = embedder
    runtime.search_service = search_service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime.embedder is None:
            model_path = os.getenv("ONNX_MODEL_PATH", "models/clip_image_encoder.onnx")
            input_size = int(os.getenv("EMBEDDER_INPUT_SIZE", "224"))
            intra_threads = int(os.getenv("EMBEDDER_INTRA_THREADS", "4"))
            runtime.embedder = OnnxImageEmbedder(
                onnx_path=model_path,
                input_size=input_size,
                intra_threads=intra_threads,
            )

        if runtime.search_service is None:
            from qdrant_client import QdrantClient

            qdrant_host = os.getenv("QDRANT_HOST", "127.0.0.1")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            collection_name = os.getenv("QDRANT_COLLECTION", "pages")
            qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            runtime.search_service = SearchService(qdrant, collection_name=collection_name)
        yield

    app = FastAPI(title="ComicSearch API", lifespan=lifespan)
    app.state.runtime = runtime

    @app.post("/search")
    async def search(
        image: UploadFile = File(...),
        keyword_ids: str | None = Form(default=None),
        robust_partial: bool = Form(default=True),
        include_corners: bool = Form(default=True),
        include_contrast: bool = Form(default=True),
        per_view_limit: int = Form(default=80, ge=10, le=300),
        top_k_manga: int = Form(default=10, ge=1, le=50),
        top_k_packs: int = Form(default=30, ge=1, le=100),
    ) -> dict[str, Any]:
        if image.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="unsupported image content type; allowed: image/jpeg, image/png, image/webp",
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty image")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image too large; max size is 10MB")

        try:
            parsed_keyword_ids = parse_keyword_ids(keyword_ids)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid keyword_ids '{keyword_ids}': expected JSON int array ({exc})",
            ) from exc

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
            per_view_limit=int(per_view_limit),
        )

        candidate_manga = app.state.runtime.search_service.aggregate_manga(points, top_k=int(top_k_manga))
        best_manga = candidate_manga[0] if candidate_manga else None
        top_packs: list[dict[str, Any]] = []
        if best_manga is not None:
            top_packs = app.state.runtime.search_service.aggregate_packs_for_manga(
                points,
                manga_id=best_manga["manga_id"],
                top_k=int(top_k_packs),
            )

        return {
            "best_manga": best_manga,
            "confidence": app.state.runtime.search_service.confidence(candidate_manga),
            "candidate_manga": candidate_manga,
            "top_packs_in_best_manga": top_packs,
        }

    return app


app = create_app()
