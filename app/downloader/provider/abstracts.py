from abc import abstractmethod, ABC
from typing import Generic, TypeVar

from app.downloader.provider.models import ArchiveInformation, ComicInformation

T = TypeVar("T")

class AbstractComicProvider(Generic[T], ABC):
    @abstractmethod
    def parseURL(self, url: str) -> T:
        pass
    @abstractmethod
    async def getComicInformation(self, comic: T) -> ComicInformation[T]:
        pass
    @abstractmethod
    async def getArchiveInformation(self, comic: T) -> list[ArchiveInformation]:
        pass
    @abstractmethod
    async def getArchiveDownloadURL(self, comic: T, archive: ArchiveInformation) -> str:
        pass
    @abstractmethod
    async def getTargetPageImageURL(self, comic: T, page: list[int]) -> dict[int, str]:
        pass
    @abstractmethod
    async def getAllPagesURLs(self, comic: T) -> dict[int, str]:
        pass