from __future__ import annotations

import json
import logging
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, FastAPI, File, Form, HTTPException, Path, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import AppSettings, load_settings
from app.db import create_db_session, get_db, get_engine, init_db
from app.ehentai_ingest import EhentaiIngestService
from app.embedder_onnx import OnnxImageEmbedder
from app.models import ImportTask, Keyword, LoginRecord, Pack, PackKeyword, SearchUsage, User
from app.search_service import SearchService
from app.task_manager import TaskManager
from app.auth import (
    consume_search_quota,
    create_access_token,
    decode_access_token,
    get_current_user_optional,
    get_quota_info,
    get_today_usage,
    hash_password,
    require_admin,
    require_auth,
    require_search_quota,
    verify_password,
    verify_turnstile_token,
)


def _build_frontend(source_dir: FilePath, dist_dir: FilePath) -> bool:
    """Build the frontend if npm is available and source exists.

    Returns True if the dist directory is ready after this call.
    """
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        return True

    if not source_dir.exists() or not (source_dir / "package.json").exists():
        logger.warning("frontend source not found at %s, skipping build", source_dir)
        return False

    npm_cmd = shutil.which("npm") or shutil.which("npm.cmd")
    if npm_cmd is None:
        logger.warning("npm not found in PATH, cannot auto-build frontend")
        return False

    node_modules = source_dir / "node_modules"
    if not node_modules.exists():
        logger.info("installing frontend dependencies...")
        install_result = subprocess.run(
            [npm_cmd, "install"], cwd=str(source_dir), capture_output=True, text=True, timeout=300,
        )
        if install_result.returncode != 0:
            logger.error("npm install failed:\n%s", install_result.stderr)
            return False

    logger.info("building frontend...")
    build_result = subprocess.run(
        [npm_cmd, "run", "build"], cwd=str(source_dir), capture_output=True, text=True, timeout=300,
    )
    if build_result.returncode != 0:
        logger.error("npm run build failed:\n%s", build_result.stderr)
        return False

    return dist_dir.exists() and (dist_dir / "index.html").exists()


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
    payload: dict[str, Any] | None = None


# ---- Auth Pydantic models ----

class RegisterRequest(BaseModel):
    username: str
    password: str
    turnstile_token: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str
    turnstile_token: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class UserInfoResponse(BaseModel):
    id: int
    username: str
    is_admin: bool
    created_at: str


class QuotaResponse(BaseModel):
    auth_enabled: bool
    daily_quota: int
    used_today: int
    remaining: int
    is_admin: bool
    quota_reset_at: str | None = None


class AuthStatusResponse(BaseModel):
    auth_enabled: bool
    turnstile_site_key: str | None
    logged_in: bool
    user: UserInfoResponse | None


class AdminUserItem(BaseModel):
    id: int
    username: str
    is_admin: bool
    is_active: bool
    created_at: str
    daily_quota: int | None
    used_today: int
    registration_ip: str | None
    last_login_ips: list[str]


class BanUserRequest(BaseModel):
    user_id: int | None = None
    user_ids: list[int] | None = None


class SetQuotaRequest(BaseModel):
    user_id: int | None = None
    user_ids: list[int] | None = None
    daily_quota: int = 0  # -1 = unlimited, 0 = use global default


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

    # API routes are mounted under /api prefix via a router.
    api_router = APIRouter(prefix="/api")

    # CORS middleware — allows frontend (dev or production) to access the API.
    # In development, Vite proxy handles cross-origin; in production, this
    # middleware allows configured origins. Defaults to permissive settings.
    cors_settings = getattr(runtime.settings, "cors", None)
    allow_origins = ["*"]
    if cors_settings is not None:
        allow_origins = list(getattr(cors_settings, "allow_origins", ["*"]) or ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            payload=record.payload if isinstance(record.payload, dict) else None,
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

    @api_router.post("/search")
    async def search(
        request: Request,
        image: UploadFile = File(...),
        keyword_ids: str | None = Form(default=None),
        robust_partial: bool | None = Form(default=None),
        include_corners: bool | None = Form(default=None),
        include_contrast: bool | None = Form(default=None),
        per_view_limit: int | None = Form(default=None),
        top_k_manga: int | None = Form(default=None),
        auth_user: User = Depends(require_search_quota),
        db: Session = Depends(get_db),
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

        # Consume quota after successful search
        consume_search_quota(auth_user, db, app.state.runtime.settings.auth, request=request)

        return {
            "best_manga": best_manga,
            "confidence": app.state.runtime.search_service.confidence(candidate_manga),
            "candidate_manga": candidate_manga,
        }

    @api_router.post(
        "/ehentai/import/tasks",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=EhentaiImportTaskSubmitResponse,
    )
    async def submit_ehentai_import_task(
        payload: EhentaiImportRequest,
        auth_user: User = Depends(require_auth),
    ) -> EhentaiImportTaskSubmitResponse:
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
        require_review = app.state.runtime.settings.ehentai.require_review
        needs_review = require_review and not auth_user.is_admin

        for request_url in request_urls:
            submit_result = manager.submit_or_get_existing(
                task_type="ehentai_import",
                payload={
                    "url": request_url,
                    "crop_faces": bool(payload.crop_faces),
                    "user_id": auth_user.id,
                },
                dedup_statuses=("pending", "running") if not needs_review else ("pending", "running", "pending_review"),
                initial_status="pending_review" if needs_review else "pending",
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

    @api_router.get("/tasks/review", response_model=list[TaskStatusResponse])
    async def list_review_tasks(
        limit: int = Query(default=50, ge=1, le=500),
        auth_user: User = Depends(require_admin),
    ) -> list[TaskStatusResponse]:
        """Admin: list tasks pending review."""
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")
        records = manager.list_tasks(limit=int(limit), status_filter="pending_review")
        return [_task_record_response(record) for record in records]

    @api_router.post("/tasks/{task_id}/approve", response_model=TaskStatusResponse)
    async def approve_task(
        task_id: str = Path(..., title="Task ID"),
        auth_user: User = Depends(require_admin),
    ) -> TaskStatusResponse:
        """Admin: approve a pending_review task."""
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")
        record = manager.approve(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"task not found or not pending review: {task_id}")
        return _task_record_response(record)

    @api_router.post("/tasks/{task_id}/reject", response_model=TaskStatusResponse)
    async def reject_task(
        task_id: str = Path(..., title="Task ID"),
        auth_user: User = Depends(require_admin),
    ) -> TaskStatusResponse:
        """Admin: reject a pending_review task."""
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")
        record = manager.reject(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"task not found or not pending review: {task_id}")
        return _task_record_response(record)

    @api_router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(
        task_id: str = Path(..., title="Task ID"),
        auth_user: User = Depends(require_auth),
        db: Session = Depends(get_db),
    ) -> TaskStatusResponse:
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        record = manager.get(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")

        # Non-admins can only view their own tasks
        if not auth_user.is_admin:
            task_row = db.execute(
                select(ImportTask).where(ImportTask.task_id == task_id)
            ).scalar_one_or_none()
            if task_row is not None and task_row.user_id is not None and task_row.user_id != auth_user.id:
                raise HTTPException(status_code=403, detail="can only view your own tasks")

        return _task_record_response(record)

    @api_router.get("/tasks", response_model=list[TaskStatusResponse])
    async def list_tasks(
        limit: int = Query(default=50, ge=1, le=500),
        status_value: str | None = Query(default=None, alias="status"),
        auth_user: User = Depends(require_auth),
        db: Session = Depends(get_db),
    ) -> list[TaskStatusResponse]:
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        records = manager.list_tasks(limit=int(limit), status_filter=status_value)

        # Non-admins: only show their own tasks
        if not auth_user.is_admin:
            task_rows = db.execute(
                select(ImportTask.task_id, ImportTask.user_id)
                .where(ImportTask.task_id.in_([r.task_id for r in records]))
            ).all()
            user_task_ids = {
                row.task_id for row in task_rows
                if row.user_id is None or row.user_id == auth_user.id
            }
            records = [r for r in records if r.task_id in user_task_ids]

        return [_task_record_response(record) for record in records]

    @api_router.post("/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
    async def cancel_task(
        task_id: str = Path(..., title="Task ID"),
        auth_user: User = Depends(require_auth),
        db: Session = Depends(get_db),
    ) -> TaskStatusResponse:
        manager = app.state.runtime.task_manager
        if manager is None:
            raise HTTPException(status_code=503, detail="task manager is not available")

        # Non-admins can only cancel their own tasks
        if not auth_user.is_admin:
            task_row = db.execute(
                select(ImportTask).where(ImportTask.task_id == task_id)
            ).scalar_one_or_none()
            if task_row is None:
                raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
            if task_row.user_id is not None and task_row.user_id != auth_user.id:
                raise HTTPException(status_code=403, detail="can only cancel your own tasks")

        record = manager.cancel(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")

        return _task_record_response(record)

    @api_router.get("/tag/search")
    async def search_tags(
        q: str = Query(default="", title="Search query"),
        limit: int = Query(default=30, ge=1, le=200),
        auth_user: User = Depends(require_auth),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        """Search tags by name (prefix match, case-insensitive). Requires login.
        Returns matching tags with their associated pack count.
        """
        query = q.strip()
        if not query or len(query) < 2:
            return []

        pattern = f"{query}%"
        tags = db.execute(
            select(Keyword.id, Keyword.name)
            .where(Keyword.name.ilike(pattern))
            .order_by(Keyword.name.asc())
            .limit(int(limit))
        ).all()

        if not tags:
            return []

        keyword_ids = [t.id for t in tags]
        from sqlalchemy import func
        counts = db.execute(
            select(PackKeyword.keyword_id, func.count(PackKeyword.pack_id))
            .where(PackKeyword.keyword_id.in_(keyword_ids))
            .group_by(PackKeyword.keyword_id)
        ).all()
        count_map: dict[int, int] = {kw_id: int(cnt) for kw_id, cnt in counts}

        return [
            {"id": int(kw_id), "name": str(name), "pack_count": count_map.get(kw_id, 0)}
            for kw_id, name in tags
        ]

    @api_router.get("/info/{id}")
    async def info(
        id: int = Path(..., title="Pack ID"),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _pack_info_response(pack_id=int(id), db=db)

    @api_router.get("/info")
    async def info_by_query(
        id: int = Query(..., title="Pack ID"),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        return _pack_info_response(pack_id=int(id), db=db)

    # ---- Auth endpoints ----

    @api_router.get("/auth/status", response_model=AuthStatusResponse)
    async def auth_status(
        request: Request,
        auth_user: User | None = Depends(get_current_user_optional),
    ) -> AuthStatusResponse:
        """Get current auth configuration status and login state."""
        auth_settings = app.state.runtime.settings.auth
        logged_in = auth_user is not None
        user_info: UserInfoResponse | None = None
        if auth_user is not None:
            user_info = UserInfoResponse(
                id=auth_user.id,
                username=auth_user.username,
                is_admin=auth_user.is_admin,
                created_at=auth_user.created_at,
            )
        return AuthStatusResponse(
            auth_enabled=auth_settings.enabled,
            turnstile_site_key=auth_settings.turnstile_site_key,
            logged_in=logged_in,
            user=user_info,
        )

    @api_router.post("/auth/register", status_code=status.HTTP_201_CREATED, response_model=TokenResponse)
    async def register(
        payload: RegisterRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> TokenResponse:
        """Register a new user account."""
        auth_settings = app.state.runtime.settings.auth
        if not auth_settings.enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="account system is not enabled",
            )

        # Verify Turnstile token if configured
        if auth_settings.turnstile_secret_key and payload.turnstile_token:
            client_ip = request.client.host if request.client else None
            await verify_turnstile_token(payload.turnstile_token, auth_settings, client_ip)
        elif auth_settings.turnstile_secret_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="captcha verification required",
            )

        # Validate username
        username = payload.username.strip()
        if len(username) < 2 or len(username) > 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="username must be 2-32 characters",
            )
        if not username.isalnum() and "_" not in username and "-" not in username:
            # Allow alphanumeric, underscore and dash
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
            if not set(username).issubset(allowed):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="username can only contain letters, numbers, underscores and dashes",
                )

        # Validate password
        if len(payload.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="password must be at least 6 characters",
            )

        # Check if username exists
        existing = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="username already exists",
            )

        # Create user
        now = datetime.now(timezone.utc).isoformat()
        client_ip = request.client.host if request.client else None
        user = User(
            username=username,
            password_hash=hash_password(payload.password),
            is_active=True,
            is_admin=False,
            registration_ip=client_ip,
            created_at=now,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Generate token
        token = create_access_token(user.id, user.username, auth_settings)
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user={
                "id": user.id,
                "username": user.username,
                "is_admin": user.is_admin,
                "created_at": user.created_at,
            },
        )

    @api_router.post("/auth/login", response_model=TokenResponse)
    async def login(
        request: Request,
        payload: LoginRequest = Body(...),
        db: Session = Depends(get_db),
    ) -> TokenResponse:
        """Login with username and password."""
        auth_settings = app.state.runtime.settings.auth
        if not auth_settings.enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="account system is not enabled",
            )

        # Verify Turnstile token if configured
        if auth_settings.turnstile_secret_key and payload.turnstile_token:
            client_ip = request.client.host if request.client else None
            await verify_turnstile_token(payload.turnstile_token, auth_settings, client_ip)
        elif auth_settings.turnstile_secret_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="captcha verification required",
            )

        # Find user
        user = db.execute(select(User).where(User.username == payload.username.strip())).scalar_one_or_none()
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid username or password",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="account is disabled",
            )

        # Generate token
        token = create_access_token(user.id, user.username, auth_settings)

        # Record login IP
        client_ip = request.client.host if request.client else "0.0.0.0"
        db.add(LoginRecord(user_id=user.id, ip_address=client_ip, logged_at=datetime.now(timezone.utc).isoformat()))
        db.commit()

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user={
                "id": user.id,
                "username": user.username,
                "is_admin": user.is_admin,
                "created_at": user.created_at,
            },
        )

    @api_router.get("/auth/me", response_model=UserInfoResponse)
    async def me(
        auth_user: User = Depends(require_auth),
    ) -> UserInfoResponse:
        """Get current logged-in user info."""
        return UserInfoResponse(
            id=auth_user.id,
            username=auth_user.username,
            is_admin=auth_user.is_admin,
            created_at=auth_user.created_at,
        )

    @api_router.get("/auth/quota", response_model=QuotaResponse)
    async def quota(
        auth_user: User = Depends(require_auth),
        db: Session = Depends(get_db),
    ) -> QuotaResponse:
        """Get current user's search quota status."""
        quota_info = get_quota_info(auth_user, db, app.state.runtime.settings.auth)
        return QuotaResponse(**quota_info)

    @api_router.get("/auth/users", response_model=list[AdminUserItem])
    async def list_users(
        auth_user: User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> list[AdminUserItem]:
        """Admin: list all users with quota usage and IP info."""
        users = db.execute(
            select(User).order_by(User.id.asc())
        ).scalars().all()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        usage_rows = db.execute(
            select(SearchUsage).where(SearchUsage.usage_date == today)
        ).scalars().all()
        usage_map: dict[int, int] = {row.user_id: row.count for row in usage_rows}

        # Fetch last 3 distinct login IPs per user
        user_ids = [u.id for u in users]
        login_ip_map: dict[int, list[str]] = {uid: [] for uid in user_ids}
        if user_ids:
            seen: dict[int, set[str]] = {uid: set() for uid in user_ids}
            login_rows = (
                db.query(LoginRecord)
                .filter(LoginRecord.user_id.in_(user_ids))
                .order_by(LoginRecord.logged_at.desc())
                .all()
            )
            for lr in login_rows:
                uid = lr.user_id
                if len(seen[uid]) >= 3:
                    continue
                if lr.ip_address not in seen[uid]:
                    seen[uid].add(lr.ip_address)
                    login_ip_map[uid].append(lr.ip_address)

        return [
            AdminUserItem(
                id=u.id,
                username=u.username,
                is_admin=u.is_admin,
                is_active=u.is_active,
                created_at=u.created_at,
                daily_quota=u.daily_quota,
                used_today=usage_map.get(u.id, 0),
                registration_ip=u.registration_ip,
                last_login_ips=login_ip_map.get(u.id, []),
            )
            for u in users
        ]

    def _resolve_user_ids(payload: BanUserRequest | SetQuotaRequest, db: Session) -> list[User]:
        if payload.user_ids:
            users = db.execute(
                select(User).where(User.id.in_(payload.user_ids)).order_by(User.id.asc())
            ).scalars().all()
            return [u for u in users if not u.is_admin]
        if payload.user_id is not None:
            target = db.get(User, payload.user_id)
            if target is None:
                raise HTTPException(status_code=404, detail=f"user not found: {payload.user_id}")
            return [] if target.is_admin else [target]
        raise HTTPException(status_code=400, detail="user_id or user_ids required")

    @api_router.post("/auth/users/ban")
    async def ban_user(
        payload: BanUserRequest = Body(...),
        auth_user: User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Admin: ban one or many users."""
        users = _resolve_user_ids(payload, db)
        count = 0
        for u in users:
            u.is_active = False
            count += 1
        db.commit()
        logger.info("admin %s banned %s users", auth_user.username, count)
        return {"count": count}

    @api_router.post("/auth/users/unban")
    async def unban_user(
        payload: BanUserRequest = Body(...),
        auth_user: User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Admin: unban one or many users."""
        users = _resolve_user_ids(payload, db)
        count = 0
        for u in users:
            u.is_active = True
            count += 1
        db.commit()
        logger.info("admin %s unbanned %s users", auth_user.username, count)
        return {"count": count}

    @api_router.post("/auth/quota/set")
    async def set_user_quota(
        payload: SetQuotaRequest = Body(...),
        auth_user: User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Admin: set per-user daily search quota for one or many users."""
        users = _resolve_user_ids(payload, db)
        value = None if payload.daily_quota <= 0 else payload.daily_quota
        count = 0
        for u in users:
            u.daily_quota = value
            count += 1
        db.commit()
        logger.info("admin %s set quota for %s users to %s", auth_user.username, count, value)
        return {"count": count}

    # Include API router with /api prefix
    app.include_router(api_router)

    # --- Frontend static file serving ---
    frontend_settings = runtime.settings.frontend
    if frontend_settings.enabled:
        dist_dir = FilePath(frontend_settings.dist_dir)
        source_dir = FilePath(frontend_settings.source_dir)

        dist_ready = dist_dir.exists() and (dist_dir / "index.html").exists()
        if not dist_ready and frontend_settings.auto_build:
            dist_ready = _build_frontend(source_dir, dist_dir)

        if dist_ready:
            index_html = dist_dir / "index.html"
            assets_dir = dist_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

            # Serve other static files at root level (favicon, etc.)
            for static_file in dist_dir.iterdir():
                if static_file.is_file() and static_file.name != "index.html":
                    static_name = static_file.name

                    def _make_static_response(filepath: FilePath) -> Callable[..., Any]:
                        async def _serve() -> FileResponse:
                            return FileResponse(str(filepath))
                        return _serve

                    app.add_api_route(f"/{static_name}", _make_static_response(static_file), methods=["GET"])

            # SPA middleware: intercept browser navigation requests (Accept: text/html)
            # and serve index.html, while letting API calls (Accept: application/json)
            # pass through to the API routes.
            @app.middleware("http")
            async def spa_middleware(request: Request, call_next: Callable[..., Any]) -> Any:
                accept = request.headers.get("accept", "")
                # Only intercept browser navigation (not API calls, not static assets)
                if "text/html" in accept and not request.url.path.startswith("/assets") and not request.url.path.startswith("/api"):
                    # Check if it's a static file request first
                    candidate = dist_dir / request.url.path.lstrip("/")
                    if candidate.is_file():
                        return FileResponse(str(candidate))
                    # SPA fallback: serve index.html for all browser navigation
                    return FileResponse(str(index_html))
                return await call_next(request)

            logger.info("frontend served from %s", dist_dir)
        else:
            logger.warning(
                "frontend enabled but dist not ready at %s "
                "(auto_build=%s); frontend will not be served",
                dist_dir,
                frontend_settings.auto_build,
            )

    return app


app = create_app()
