import asyncio
import json
import re
from urllib.parse import urlparse

from aiohttp import ClientHandlerType, ClientRequest, ClientResponse, ClientSession
import aiohttp
from bs4 import BeautifulSoup

from app.downloader.provider.abstracts import AbstractComicProvider
from app.downloader.provider.models import ArchiveInformation, ComicInformation, EHGallery
from app.downloader.provider.utils import precheck_response


def _isCopyrightGallery(document: BeautifulSoup) -> bool:
    msg = document.select_one(".d p")
    if msg is not None and msg.find("copyright") != -1:
        return True
    return False

class EHentaiProvider(AbstractComicProvider[tuple[str, str]]):
    class __Middleware:
        '''Middleware for E-Hentai requests, including setting headers and retrying on failure.'''
        def __init__(self, retryCount: int = 3, retryDelay: int = 500, retryDelayWhenRateLimited: int = 3000):
            self.__retryCount = retryCount
            self.__retryDelay = retryDelay
            self.__retryDelayWhenRateLimited = retryDelayWhenRateLimited

        async def requestPreMiddleware(self, req: ClientRequest, handler: ClientHandlerType) -> ClientResponse:
            req.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            req.headers["Referer"] = str(req.url)
            req.headers["Host"] = str(req.url.host)
            req.headers["Accept"] = "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
            req.headers["Accept-Language"] = "en-US,en;q=0.9"
            req.headers["Accept-Encoding"] = "gzip, deflate"
            return await handler(req)
        
        async def requestRetryMiddleware(self, req: ClientRequest, handler: ClientHandlerType) -> ClientResponse:
            for i in range(self.__retryCount):
                try:
                    res = await handler(req)
                    if res.status == 429: # Too Many Requests
                        await res.release()
                        await asyncio.sleep(self.__retryDelayWhenRateLimited / 1000)
                    elif res.status >= 500: # Server Error
                        await res.release()
                        await asyncio.sleep(self.__retryDelay / 1000)
                    else:
                        return res
                except Exception as e:
                    if i == self.__retryCount - 1:
                        raise e
                    else:
                        await asyncio.sleep(self.__retryDelay / 1000)
            raise Exception("Failed to fetch after retries")
    
    __middlewares = __Middleware(retryCount=10, retryDelay=500, retryDelayWhenRateLimited=5000)
    __galleryURLRegex = r"(https?://(exhentai|e-hentai).org/|)g/([0-9a-zA-Z]+)/([0-9a-zA-Z]+)"

    def __init__(self, isExhentai: bool = False, proxy: str | None = None, cookies: dict[str, str] | None = None):
        super().__init__()
        self.__baseURL = "https://exhentai.org" if isExhentai else "https://e-hentai.org"
        self.__apiURL = self.__baseURL + "/api.php"
        self.__proxy = proxy
        self.__cookies = cookies

    def parseURL(self, url):
        match = re.match(self.__galleryURLRegex, url)
        if match:
            return match.group(3), match.group(4)
        else:
            raise ValueError("Invalid URL")
        
    #@lru_cache(maxsize=128)
    async def getComicInformation(self, comic):
        '''
        Get comic information from E-Hentai API. 
        If the comic has been updated, 
        it will return the latest information.
        The comic parameter is a tuple of (gid, token).
        '''
        async with ClientSession(
            proxy=self.__proxy,
            cookies=self.__cookies,
            middlewares=[
                self.__middlewares.requestPreMiddleware, 
                self.__middlewares.requestRetryMiddleware
                ]) as session:
            async with session.post(self.__apiURL, json={
                "method": "gdata",
                "gidlist": [comic],
                "namespace": 1
            }) as res:
                body = await precheck_response(res)
                payload = json.loads(body)
                metadata_list = payload.get("gmetadata")
                if not isinstance(metadata_list, list):
                    raise ValueError(f"Invalid E-Hentai API response for {comic[0]}/{comic[1]}: missing gmetadata")

                for data in metadata_list:
                    api_error = data.get("error")
                    if api_error:
                        raise ValueError(f"E-Hentai API error for {comic[0]}/{comic[1]}: {api_error}")
                    if "category" not in data:
                        raise ValueError(
                            f"Invalid E-Hentai metadata response for {comic[0]}/{comic[1]}: missing category"
                        )

                    gallery = EHGallery.GMetadata.from_json(data)
                    # get latest gallery information
                    if gallery.current_gid is not None and gallery.current_key is not None:
                        current_comic = (str(gallery.current_gid), str(gallery.current_key))
                        if current_comic != (str(comic[0]), str(comic[1])):
                            return await self.getComicInformation(current_comic)
                    # filter out language and advertisement tags, as they are not useful for searching and may cause issues with some comics that have a large number of tags
                    gallery.tags = [tag for tag in gallery.tags if not (tag.startswith("language:") or tag == "other:extraneous ads")]
                    if gallery.first_gid is not None and gallery.first_key is not None:
                        old = (str(gallery.first_gid), str(gallery.first_key))
                    else:
                        old = (str(comic[0]), str(comic[1]))
                    # always use the Japanese title if available.
                    return ComicInformation(
                        id=(str(comic[0]), str(comic[1])),
                        old=old,
                        title=gallery.title_jpn if gallery.title_jpn else gallery.title,
                        subtitle="",
                        tags=gallery.tags,
                        page_count=gallery.filecount,
                        category=gallery.category,
                        cover_url=gallery.thumb,
                        rating=gallery.rating,
                        uploader=gallery.uploader,
                        uploadTimestamp=gallery.posted, 
                        size=gallery.filesize
                    )
                raise ValueError("Comic not found")
    
    async def getAllPagesURLs(self, comic):
        async with ClientSession(
            proxy=self.__proxy,
            cookies=self.__cookies,
            middlewares=[
                self.__middlewares.requestPreMiddleware, 
                self.__middlewares.requestRetryMiddleware
                ]) as session:
            async with session.get(f"{self.__baseURL}/g/{comic[0]}/{comic[1]}") as res:
                body = await precheck_response(res)
                document = BeautifulSoup(body, "lxml")
                if _isCopyrightGallery(document):
                    raise ValueError("Copyright gallery, cannot download")
                expected_page_count = -1
                length_cell = document.find("td", class_="gdt1", string="Length:")
                if length_cell:
                    length_text = length_cell.find_next_sibling("td", class_="gdt2").text.strip()
                    expected_page_count = int(re.search(r"\d+", length_text).group())

                last_gallery_page_no = int(document.select_one(".ptb td:nth-last-child(2)").get_text())
                gallery_page_no = 1
                page_urls: dict[int, str] = {}
                page_no = 1
                while gallery_page_no <= last_gallery_page_no:
                    if gallery_page_no == 1:
                        current_page = document
                    else:
                        async with session.get(f"{self.__baseURL}/g/{comic[0]}/{comic[1]}?p={gallery_page_no-1}") as res:
                            body = await precheck_response(res)
                            current_page = BeautifulSoup(body, "lxml")
                    page_links = current_page.select("#gdt a")
                    for link in page_links:
                        href = link.get("href")
                        if href:
                            page_urls[page_no] = str(href)
                            page_no += 1
                    gallery_page_no += 1

                collected_page_count = page_no - 1
                if expected_page_count != collected_page_count:
                    raise ValueError(
                        f"Page count mismatch, expected {expected_page_count}, got {collected_page_count}"
                    )
                return page_urls

    async def getTargetPageImageURL(self, comic, page):
        pages = {}
        urls = await self.getAllPagesURLs(comic)
        for p in page:
            async with ClientSession(
                proxy=self.__proxy,
                cookies=self.__cookies,
                middlewares=[
                    self.__middlewares.requestPreMiddleware, 
                    self.__middlewares.requestRetryMiddleware
                    ]) as session:
                if not urls.get(p):
                    continue
                async with session.get(urls[p]) as res:
                    body = await precheck_response(res)
                    document = BeautifulSoup(body, "lxml")
                    real_url = document.select_one('img#img').get('src')
                    pages[p] = real_url
        return pages

    async def getArchiveInformation(self, comic):
        if not await self.__checkLogin():
            raise ValueError("Not logged in, cannot download archive")
        async with ClientSession(
            proxy=self.__proxy,
            cookies=self.__cookies,
            middlewares=[
                self.__middlewares.requestPreMiddleware, 
                self.__middlewares.requestRetryMiddleware
                ]) as session:
             async with session.get(f"{self.__baseURL}/archiver.php?gid={comic[0]}&token={comic[1]}") as res:
                archives = []
                body = await precheck_response(res)
                document = BeautifulSoup(body, "lxml")
                # typically there are two archives
                # the original one and the resampled one
                # we will return both of them if available
                forms = document.select("form:has(input[name=dltype])")
                for form in forms:
                    type = form.select_one("input[name=dltype]")
                    if type is None:
                        continue
                    type = type.get("value")
                    name = "Unknown"
                    if type == "org":
                        name = "Original"
                    elif type == "res":
                        name = "Resample"
                    if form.parent is None:
                        continue
                    sizeElement = form.parent.select_one("p:-soup-contains('Size:') > strong")
                    costElement = form.parent.select_one("div:-soup-contains('Cost:') > strong")
                    if sizeElement is None or costElement is None:
                        continue
                    archives.append(ArchiveInformation(
                        name=name,
                        size=sizeElement.get_text().strip(),
                        cost=costElement.get_text().strip(),
                        parameter=None
                    ))
                return archives
        
    async def getArchiveDownloadURL(self, comic, archive):
        if not await self.__checkLogin():
            raise ValueError("Not logged in, cannot download archive")
        async with ClientSession(
            proxy=self.__proxy,
            cookies=self.__cookies,
            middlewares=[
                self.__middlewares.requestPreMiddleware, 
                self.__middlewares.requestRetryMiddleware
                ]) as session:
            data = aiohttp.FormData()
            if archive.name == "Original":
                data.add_field("dltype", "org")
                data.add_field("dlcheck", "Download+Original+Archive")
            elif archive.name == "Resample":
                data.add_field("dltype", "res")
                data.add_field("dlcheck", "Download+Resample+Archive")
            async with session.post(f"{self.__baseURL}/archiver.php?gid={comic[0]}&token={comic[1]}", data=data, allow_redirects=True) as res:
                body = await precheck_response(res)
                document = BeautifulSoup(body, "lxml")
                al = document.select("a")
                if len(al) == 0:
                    raise ValueError("Download link not found")
                href = str(al[0].get("href", None))
                if href is None or not urlparse(href).hostname.endswith("hath.network"):
                    raise ValueError("Download link not found")
                return href+"?start=1"
    
    async def __checkLogin(self):
        async with ClientSession(
            proxy=self.__proxy,
            cookies=self.__cookies,
            middlewares=[
                self.__middlewares.requestPreMiddleware, 
                self.__middlewares.requestRetryMiddleware
                ]) as session:
            async with session.get(f"{self.__baseURL}/home.php", allow_redirects=True, cookies=self.__cookies) as res:
                await res.read()
                if res.url.path != "/home.php":
                    return False
                else:
                    return True