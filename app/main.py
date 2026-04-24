from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Query, UploadFile, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import AppSettings, load_settings
from app.db import create_db_session, get_db, get_engine, init_db
from app.ehentai_ingest import EhentaiIngestService
from app.embedder_onnx import OnnxImageEmbedder
from app.models import Keyword, Pack, PackKeyword
from app.search_service import SearchService
from app.task_manager import TaskManager


class _AppRuntime:
    embedder: Any = None
    search_service: Any = None
    ehentai_ingest_service: Any = None
    task_manager: TaskManager | None = None
    task_db_session_factory: Callable[[], Any] | None = None
    settings: AppSettings | None = None


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_KEYWORD_IDS = 20
INT32_MAX = 2_147_483_647
logger = logging.getLogger("uvicorn.error")


class EhentaiImportRequest(BaseModel):
    url: HttpUrl | None = None
    urls: list[HttpUrl] | None = None
    crop_faces: bool = True


class EhentaiImportTaskSubmitItemResponse(BaseModel):
    url: str
    task_id: str
    status: str
    is_duplicate: bool


class EhentaiImportTaskSubmitResponse(BaseModel):
    task_id: str | None = None
    status: str | None = None
    items: list[EhentaiImportTaskSubmitItemResponse]


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    cancel_requested: bool = False
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


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


def create_app(
    config_path: str | None = None,
    embedder: Any | None = None,
    search_service: Any | None = None,
    ehentai_ingest_service: Any | None = None,
    task_manager: TaskManager | None = None,
    task_db_session_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    runtime = _AppRuntime()
    runtime.embedder = embedder
    runtime.search_service = search_service
    runtime.ehentai_ingest_service = ehentai_ingest_service
    runtime.task_manager = task_manager
    runtime.task_db_session_factory = task_db_session_factory or create_db_session
    runtime.settings = load_settings(config_path)

    from app.db import configure_database

    configure_database(runtime.settings.database.url)
    init_db()

    if runtime.task_manager is None:
        task_session_factory = runtime.task_db_session_factory
        assert task_session_factory is not None
        runtime.task_manager = TaskManager(task_session_factory, get_engine())

    def _register_ehentai_task_handler() -> None:
        if runtime.task_manager is None or runtime.ehentai_ingest_service is None:
            return

        async def _ehentai_import_handler(payload: dict[str, Any], db: Session, should_cancel: Callable[[], bool]) -> dict[str, Any]:
            return await runtime.ehentai_ingest_service.ingest_url(
                url=str(payload["url"]),
                db=db,
                crop_faces=bool(payload.get("crop_faces", True)),
                should_cancel=should_cancel,
            )

        runtime.task_manager.register_handler("ehentai_import", _ehentai_import_handler)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime.embedder is None:
            runtime.embedder = OnnxImageEmbedder(
                onnx_path=runtime.settings.embedder.onnx_path,
                input_size=runtime.settings.embedder.input_size,
                intra_threads=runtime.settings.embedder.intra_threads,
            )

        if runtime.search_service is None:
            from qdrant_client import QdrantClient

            qdrant = QdrantClient(host=runtime.settings.qdrant.host, port=runtime.settings.qdrant.port)
            runtime.search_service = SearchService(qdrant, collection_name=runtime.settings.qdrant.collection)

        if (
            runtime.ehentai_ingest_service is None
            and runtime.embedder is not None
            and runtime.search_service is not None
            and hasattr(runtime.search_service, "qdrant")
        ):
            settings = runtime.settings
            assert settings is not None
            runtime.ehentai_ingest_service = EhentaiIngestService(
                settings=settings,
                embedder=runtime.embedder,
                search_service=runtime.search_service,
            )

        if runtime.task_manager is not None and runtime.ehentai_ingest_service is not None:
            _register_ehentai_task_handler()
            runtime.task_manager.resume_unfinished()
        yield

    app = FastAPI(title="ComicSearch API", lifespan=lifespan)
    app.state.runtime = runtime

    def _task_record_response(record: Any) -> TaskStatusResponse:
        return TaskStatusResponse(
            task_id=record.task_id,
            task_type=record.task_type,
            status=record.status,
            cancel_requested=bool(record.cancel_requested),
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            result=record.result if isinstance(record.result, dict) else None,
            error=record.error,
        )

    def _pack_info_response(pack_id: int, db: Session) -> dict[str, Any]:
        pack_row = db.execute(select(Pack.pack_id, Pack.title, Pack.source).where(Pack.pack_id == int(pack_id))).one_or_none()
        if pack_row is None:
            raise HTTPException(status_code=404, detail=f"pack not found: {pack_id}")
        pack_id_value, pack_title, pack_source = pack_row

        rows = (
            db.query(Keyword.id, Keyword.name)
            .join(PackKeyword, PackKeyword.keyword_id == Keyword.id)
            .filter(PackKeyword.pack_id == int(pack_id))
            .order_by(Keyword.id.asc())
            .all()
        )
        keywords = [{"id": int(keyword_id), "name": str(keyword_name)} for keyword_id, keyword_name in rows]

        return {
            "pack_id": int(pack_id_value),
            "title": pack_title,
            "source": pack_source,
            "keyword_ids": [keyword["id"] for keyword in keywords],
            "keywords": keywords,
        }

    @app.post("/search")
    async def search(
        image: UploadFile = File(...),
        keyword_ids: str | None = Form(default=None),
        robust_partial: bool | None = Form(default=None),
        include_corners: bool | None = Form(default=None),
        include_contrast: bool | None = Form(default=None),
        per_view_limit: int | None = Form(default=None),
        top_k_manga: int | None = Form(default=None),
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

        search_defaults = app.state.runtime.settings.search
        robust_partial = search_defaults.robust_partial if robust_partial is None else robust_partial
        include_corners = search_defaults.include_corners if include_corners is None else include_corners
        include_contrast = search_defaults.include_contrast if include_contrast is None else include_contrast
        per_view_limit = search_defaults.per_view_limit if per_view_limit is None else int(per_view_limit)
        top_k_manga = search_defaults.top_k_manga if top_k_manga is None else int(top_k_manga)

        if per_view_limit < 10 or per_view_limit > 300:
            raise HTTPException(status_code=422, detail="per_view_limit must be in [10, 300]")
        if top_k_manga < 1 or top_k_manga > 50:
            raise HTTPException(status_code=422, detail="top_k_manga must be in [1, 50]")

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
            per_view_limit=per_view_limit,
        )

        candidate_manga = app.state.runtime.search_service.aggregate_manga(points, top_k=top_k_manga)
        best_manga = candidate_manga[0] if candidate_manga else None

        return {
            "best_manga": best_manga,
            "confidence": app.state.runtime.search_service.confidence(candidate_manga),
            "candidate_manga": candidate_manga,
        }

    @app.post("/ehentai/import/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=EhentaiImportTaskSubmitResponse)
    async def submit_ehentai_import_task(payload: EhentaiImportRequest) -> EhentaiImportTaskSubmitResponse:
        if app.state.runtime.ehentai_ingest_service is None:
            raise HTTPException(status_code=503, detail="ehentai ingest service is not available")

        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        _register_ehentai_task_handler()

        request_urls: list[str] = []
        if payload.url is not None:
            request_urls.append(str(payload.url))
        if payload.urls is not None:
            request_urls.extend(str(url) for url in payload.urls)
        if len(request_urls) == 0:
            raise HTTPException(status_code=400, detail="either url or urls must be provided")

        items: list[EhentaiImportTaskSubmitItemResponse] = []
        for request_url in request_urls:
            submit_result = manager.submit_or_get_existing(
                task_type="ehentai_import",
                payload={"url": request_url, "crop_faces": bool(payload.crop_faces)},
            )
            is_duplicate = not bool(submit_result.created)
            response_status = "duplicate" if is_duplicate else submit_result.status
            logger.info(
                "submit ehentai import task id=%s status=%s created=%s url=%s crop_faces=%s",
                submit_result.task_id,
                response_status,
                submit_result.created,
                request_url,
                bool(payload.crop_faces),
            )
            items.append(
                EhentaiImportTaskSubmitItemResponse(
                    url=request_url,
                    task_id=submit_result.task_id,
                    status=response_status,
                    is_duplicate=is_duplicate,
                )
            )

        first_item = items[0]
        return EhentaiImportTaskSubmitResponse(
            task_id=first_item.task_id,
            status=first_item.status,
            items=items,
        )

    @app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(task_id: str = Path(..., title="Task ID")) -> TaskStatusResponse:
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        record = manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")

        return _task_record_response(record)

    @app.get("/tasks", response_model=list[TaskStatusResponse])
    async def list_tasks(
        limit: int = Query(default=50, ge=1, le=500),
        status_value: str | None = Query(default=None, alias="status"),
    ) -> list[TaskStatusResponse]:
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        records = manager.list_tasks(limit=int(limit), status_filter=status_value)
        return [_task_record_response(record) for record in records]

    @app.post("/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
    async def cancel_task(task_id: str = Path(..., title="Task ID")) -> TaskStatusResponse:
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        record = manager.cancel(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")

        return _task_record_response(record)

    @app.get("/info/{id}")
    async def info(
        id: int = Path(..., title="Pack ID"),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _pack_info_response(pack_id=int(id), db=db)

    @app.get("/info")
    async def info_by_query(
        id: int = Query(..., title="Pack ID"),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _pack_info_response(pack_id=int(id), db=db)

    return app


app = create_app()
