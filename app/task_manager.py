from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

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

        with self._session_factory() as db:
            db.add(
                ImportTask(
                    task_id=task_id,
                    task_type=task_type,
                    status="pending",
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
    ) -> SubmitResult:
        task_id: str | None = None
        status: str | None = None
        created = False

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
                    status = "pending"
                    created = True
                    db.add(
                        ImportTask(
                            task_id=task_id,
                            task_type=task_type,
                            status="pending",
                            payload_json=json.dumps(payload, ensure_ascii=False),
                            created_at=self._now_iso(),
                        )
                    )
                    db.commit()

        assert task_id is not None
        assert status is not None
        if created:
            self._start_worker(task_id)
        return SubmitResult(task_id=task_id, status=status, created=created)

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

        def _should_cancel() -> bool:
            with self._session_factory() as cancel_db:
                cancel_row = cancel_db.get(ImportTask, task_id)
                return cancel_row is None or int(cancel_row.cancel_requested or 0) == 1

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

    def list_tasks(self, limit: int = 50, status_filter: str | None = None) -> list[TaskRecord]:
        with self._session_factory() as db:
            query = db.query(ImportTask)
            if status_filter is not None:
                query = query.filter(ImportTask.status == status_filter)
            rows = query.order_by(ImportTask.created_at.desc()).limit(int(limit)).all()
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
