from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


class Pack(Base):
    __tablename__ = "pack"

    pack_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)

    keywords: Mapped[list[PackKeyword]] = relationship(back_populates="pack", cascade="all, delete-orphan")


class Keyword(Base):
    __tablename__ = "keyword"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class TagIdMap(Base):
    __tablename__ = "tag_id_map"

    tag: Mapped[str] = mapped_column(String, primary_key=True)
    keyword_id: Mapped[int] = mapped_column(ForeignKey("keyword.id", ondelete="CASCADE"), unique=True, nullable=False)


class PackKeyword(Base):
    __tablename__ = "pack_keyword"

    pack_id: Mapped[int] = mapped_column(ForeignKey("pack.pack_id", ondelete="CASCADE"), primary_key=True)
    keyword_id: Mapped[int] = mapped_column(ForeignKey("keyword.id", ondelete="CASCADE"), primary_key=True)

    pack: Mapped[Pack] = relationship(back_populates="keywords")

    __table_args__ = (
        UniqueConstraint("pack_id", "keyword_id", name="uq_pack_keyword"),
        Index("idx_pack_keyword_keyword_id", "keyword_id"),
    )

class EHentaiComicInfo(Base):
    __tablename__ = "ehentai_comic_info"

    current_gid: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    current_token: Mapped[str] = mapped_column(String, nullable=False)
    old_gid: Mapped[int] = mapped_column(Integer, nullable=False)
    old_token: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        Index("idx_ehentai_current_gid_token", "current_gid", "current_token"),
        Index("idx_ehentai_old_gid_token", "old_gid", "old_token"),
    )


class ImportTask(Base):
    __tablename__ = "import_task"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    cancel_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        Index("idx_import_task_status_created", "status", "created_at"),
    )
