from aiohttp import ClientResponse
from bs4 import BeautifulSoup


async def precheck_response(res: ClientResponse) -> str:
    if res.status != 200:
        raise RuntimeError(f"Invalid status code: {res.status}")

    body = await res.text()
    if not body or not body.strip():
        raise RuntimeError("Failed to load page")
    return body

def ehentai_document_precheck(body: str) -> BeautifulSoup:
    text = body.lstrip()
    if not text.startswith("<"):
        if "IP" in text:
            raise RuntimeError("The IP address has been banned")
        raise RuntimeError("Failed to load page")
    return BeautifulSoup(text, "html.parser")