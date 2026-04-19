from dataclasses import dataclass
from typing import Generic, TypeVar
from enum import Enum


T = TypeVar("T")

@dataclass
class ComicInformation(Generic[T]):
    class Category(Enum):
        Doujinshi = "doujinshi"
        Manga = "manga"
        Artist_CG = "artist cg"
        Game_CG = "game cg"
        Non_H = "non-h"
        Image_Set = "image set"
        Western = "western"
        Misc = "misc"
        Cosplay = "cosplay"
        Asian_Porn = "asian porn"
        Private = "private"
        def __str__(self) -> str:
            return self.value
        @classmethod
        def from_str(cls, s: str) -> "ComicInformation.Category":
            # ignore case when matching category
            s = s.lower()
            for category in cls:
                if category.value == s:
                    return category
            raise ValueError(f"Unknown category: {s}")
    id: T
    title: str
    subtitle: str | None
    tags: list[str]
    page_count: int
    category: Category
    cover_url: str
    rating: float
    uploader: str
    uploadTimestamp: int
    size: int | None = None

@dataclass
class ArchiveInformation:
    name: str
    size: str | None
    cost: str | None
    parameter: str | None

class EHGallery:
    @dataclass
    class GMetadata:
        category: ComicInformation.Category
        filecount: int
        filesize: int
        posted: int
        rating: float
        tags: list[str]
        thumb: str
        title: str
        title_jpn: str
        uploader: str
        current_gid: int | None = None
        current_key: str | None = None
        first_gid: int | None = None
        first_key: str | None = None

        @classmethod
        def from_json(cls, data: dict) -> "EHGallery.GMetadata":
            return cls(
                category=ComicInformation.Category.from_str(data["category"]),
                filecount=data["filecount"],
                filesize=data["filesize"],
                posted=data["posted"],
                rating=data["rating"],
                tags=data["tags"],
                thumb=data["thumb"],
                title=data["title"],
                title_jpn=data["title_jpn"],
                uploader=data["uploader"],
                current_gid=data.get("current_gid"),
                current_key=data.get("current_key"),
                first_gid=data.get("first_gid"),
                first_key=data.get("first_key")
            )