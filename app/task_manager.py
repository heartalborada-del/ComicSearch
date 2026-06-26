from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import Base, ImportTask

TaskHandler = Callable[[dict[str, Any], Session, Callable[[], bool]], Awaitable[dict[str, Any]]]


class TaskCancelledError(Exception):
    pass


@dataclass
class TaskRecord:
    task_id: str
    task_type: str
    status: str
    payload: dict[str, Any]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False


@dataclass
class SubmitResult:
    task_id: str
    status: str
    created: bool


class TaskManager:
    def __init__(self, session_factory: Callable[[], Any], engine: Any) -> None:
        self._session_factory = session_factory
        self._engine = engine
        self._handlers: dict[str, TaskHandler] = {}
        self._running_task_ids: set[str] = set()
        self._lock = Lock()
        Base.metadata.create_all(bind=engine, tables=cast(Any, [ImportTask.__table__]))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _loads_json(raw: str | None) -> dict[str, Any] | None:
        if raw is None or raw.strip() == "":
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None

    def register_handler(self, task_type: str, handler: TaskHandler) -> None:
        with self._lock:
            self._handlers[task_type] = handler

    def submit(self, task_type: str, payload: dict[str, Any]) -> str:
        task_id = uuid4().hex
        created_at = self._now_iso()
        user_id = payload.pop("user_id", None)

        with self._session_factory() as db:
            db.add(
                ImportTask(
                    task_id=task_id,
                    task_type=task_type,
                    status="pending",
                    user_id=int(user_id) if user_id is not None else None,
                    payload_json=json.dumps(payload, ensure_ascii=False),
                    created_at=created_at,
                )
            )
            db.commit()

        self._start_worker(task_id)
        return task_id

    def submit_or_get_existing(
        self,
        task_type: str,
        payload: dict[str, Any],
        dedup_statuses: tuple[str, ...] = ("pending", "running"),
        initial_status: str = "pending",
    ) -> SubmitResult:
        task_id: str | None = None
        status: str | None = None
        created = False
        user_id = payload.pop("user_id", None)

        with self._lock:
            with self._session_factory() as db:
                if dedup_statuses:
                    rows = (
                        db.query(ImportTask)
                        .filter(ImportTask.task_type == task_type)
                        .filter(ImportTask.status.in_(list(dedup_statuses)))
                        .order_by(ImportTask.created_at.desc())
                        .all()
                    )
                    for row in rows:
                        row_payload = self._loads_json(row.payload_json) or {}
                        if row_payload == payload:
                            task_id = row.task_id
                            status = row.status
                            created = False
                            break

                if task_id is None:
                    task_id = uuid4().hex
                    status = initial_status
                    created = True
                    db.add(
                        ImportTask(
                            task_id=task_id,
                            task_type=task_type,
                            status=initial_status,
                            user_id=int(user_id) if user_id is not None else None,
                            payload_json=json.dumps(payload, ensure_ascii=False),
                            created_at=self._now_iso(),
                        )
                    )
                    db.commit()

        assert task_id is not None
        assert status is not None
        if created and initial_status == "pending":
            self._start_worker(task_id)
        return SubmitResult(task_id=task_id, status=status, created=created)

    def approve(self, task_id: str) -> TaskRecord | None:
        """Approve a pending_review task → set to pending and start execution."""
        with self._session_factory() as db:
            row = db.get(ImportTask, task_id)
            if row is None or row.status != "pending_review":
                return None
            row.status = "pending"
            db.commit()
        self._start_worker(task_id)
        return self.get(task_id)

    def reject(self, task_id: str) -> TaskRecord | None:
        """Reject a pending_review task → set to cancelled."""
        with self._session_factory() as db:
            row = db.get(ImportTask, task_id)
            if row is None or row.status != "pending_review":
                return None
            row.status = "cancelled"
            row.finished_at = self._now_iso()
            db.commit()
        return self.get(task_id)

    def _start_worker(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._running_task_ids:
                return
            self._running_task_ids.add(task_id)
        worker = Thread(target=self._run_task_in_thread, args=(task_id,), daemon=True)
        worker.start()

    def _run_task_in_thread(self, task_id: str) -> None:
        try:
            asyncio.run(self._run_task(task_id=task_id))
        finally:
            with self._lock:
                self._running_task_ids.discard(task_id)

    async def _run_task(self, task_id: str) -> None:
        with self._session_factory() as db:
            row = db.get(ImportTask, task_id)
            if row is None:
                return
            if row.status in {"success", "failed", "cancelled"}:
                return
            if int(row.cancel_requested or 0) == 1:
                row.status = "cancelled"
                row.finished_at = self._now_iso()
                db.commit()
                return

            row.status = "running"
            row.started_at = row.started_at or self._now_iso()
            payload = self._loads_json(row.payload_json) or {}
            task_type = row.task_type
            db.commit()

        with self._lock:
            handler = self._handlers.get(task_type)
        if handler is None:
            with self._session_factory() as db:
                row = db.get(ImportTask, task_id)
                if row is None:
                    return
                row.status = "failed"
                row.error = f"no handler registered for task type: {task_type}"
                row.finished_at = self._now_iso()
                db.commit()
            return

        _cancel_cache: dict[str, Any] = {}

        def _should_cancel() -> bool:
            # Throttle cancel checks: this callback is invoked per-page and
            # per-face-crop during long imports. Hitting the DB every call can
            # exhaust the connection pool. Cache the result for 1 second.
            now = time.monotonic()
            cached = _cancel_cache.get("ts")
            if cached is not None and now - cached < 1.0:
                return _cancel_cache.get("value", False)
            # Use a raw engine connection instead of a full ORM session to
            # avoid the overhead and pool pressure of session creation.
            with self._engine.connect() as conn:
                row = conn.execute(
                    text("SELECT cancel_requested FROM import_task WHERE task_id = :tid"),
                    {"tid": task_id},
                ).first()
            cancelled = row is None or int(row[0] or 0) == 1
            _cancel_cache["ts"] = now
            _cancel_cache["value"] = cancelled
            return cancelled

        try:
            with self._session_factory() as work_db:
                result = await handler(payload, work_db, _should_cancel)
        except TaskCancelledError:
            with self._session_factory() as db:
                row = db.get(ImportTask, task_id)
                if row is None:
                    return
                row.status = "cancelled"
                row.finished_at = self._now_iso()
                db.commit()
            return
        except Exception as exc:
            with self._session_factory() as db:
                row = db.get(ImportTask, task_id)
                if row is None:
                    return
                row.status = "failed"
                row.error = str(exc)
                row.finished_at = self._now_iso()
                db.commit()
            return

        with self._session_factory() as db:
            row = db.get(ImportTask, task_id)
            if row is None:
                return
            if int(row.cancel_requested or 0) == 1:
                row.status = "cancelled"
            else:
                row.status = "success"
                row.result_json = json.dumps(result, ensure_ascii=False)
            row.finished_at = self._now_iso()
            db.commit()

    def get(self, task_id: str) -> TaskRecord | None:
        with self._session_factory() as db:
            row = db.get(ImportTask, task_id)
            if row is None:
                return None
            return TaskRecord(
                task_id=row.task_id,
                task_type=row.task_type,
                status=row.status,
                payload=self._loads_json(row.payload_json) or {},
                created_at=row.created_at,
                started_at=row.started_at,
                finished_at=row.finished_at,
                result=self._loads_json(row.result_json),
                error=row.error,
                cancel_requested=int(row.cancel_requested or 0) == 1,
            )

    def list_tasks(self, limit: int = 50, offset: int = 0, status_filter: str | None = None) -> list[TaskRecord]:
        with self._session_factory() as db:
            query = db.query(ImportTask)
            if status_filter is not None:
                query = query.filter(ImportTask.status == status_filter)
            rows = query.order_by(ImportTask.created_at.desc()).offset(int(offset)).limit(int(limit)).all()
            return [
                TaskRecord(
                    task_id=row.task_id,
                    task_type=row.task_type,
                    status=row.status,
                    payload=self._loads_json(row.payload_json) or {},
                    created_at=row.created_at,
                    started_at=row.started_at,
                    finished_at=row.finished_at,
                    result=self._loads_json(row.result_json),
                    error=row.error,
                    cancel_requested=int(row.cancel_requested or 0) == 1,
                )
                for row in rows
            ]

    def count_tasks(self, status_filter: str | None = None) -> int:
        with self._session_factory() as db:
            query = db.query(ImportTask)
            if status_filter is not None:
                query = query.filter(ImportTask.status == status_filter)
            return int(query.count())

    def cancel(self, task_id: str) -> TaskRecord | None:
        with self._session_factory() as db:
            row = db.get(ImportTask, task_id)
            if row is None:
                return None
            row.cancel_requested = 1
            if row.status == "pending":
                row.status = "cancelled"
                row.finished_at = row.finished_at or self._now_iso()
            db.commit()

        return self.get(task_id)

    def resume_unfinished(self) -> None:
        with self._session_factory() as db:
            rows = (
                db.query(ImportTask.task_id)
                .filter(ImportTask.status.in_(["pending", "running"]))
                .all()
            )
        for row in rows:
            self._start_worker(str(row.task_id))
