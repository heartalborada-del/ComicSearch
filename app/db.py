from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import load_settings
from app.models import Base

_lock = RLock()
_engine: Engine | None = None
SessionLocal = sessionmaker(autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)
DATABASE_URL = None


def configure_database(database_url: str | None = None) -> None:
    global DATABASE_URL, SessionLocal, _engine

    resolved_database_url = database_url or load_settings().database.url
    with _lock:
        if DATABASE_URL == resolved_database_url and _engine is not None and SessionLocal is not None:
            return

        DATABASE_URL = resolved_database_url
        engine_kwargs = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
        _engine = create_engine(DATABASE_URL, **engine_kwargs)
        SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def get_engine() -> Engine:
    if _engine is None:
        configure_database()
    assert _engine is not None
    return _engine


def get_db():
    if SessionLocal is None:
        configure_database()
    assert SessionLocal is not None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def create_db_session():
    if SessionLocal is None:
        configure_database()
    assert SessionLocal is not None

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(config_path: str | None = None) -> None:
    if config_path is not None:
        settings = load_settings(config_path)
        configure_database(settings.database.url)
    else:
        configure_database()

    Base.metadata.create_all(bind=get_engine())


configure_database()
