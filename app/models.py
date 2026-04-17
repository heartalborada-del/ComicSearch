from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


class Pack(Base):
    __tablename__ = "pack"

    pack_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)

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
