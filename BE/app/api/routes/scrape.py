import base64
"""
Module: app.api.routes.scrape
Description: Endpoint nhÃ¯Â»ÂÃ¯Â­Â¦ÃÂ­n dÃ¯Â»ÂÃ¯Â­Â¨Ã¯ÂºÂ liÃ¯Â»ÂÃ¯Â­Â¨ÃÂu tÃ¯Â»ÂÃ¯Â­Â¨ÃÂ Chrome Extension (test phase).

/scrape/test  Ã¯Â»ÂÃÂÃÂ NhÃ¯Â»ÂÃ¯Â­Â¦ÃÂ­n metadata + image URLs, filter Ã¯Â»ÂÃ¯Â­Â¦ÃÂ²nh manga thÃ¯Â»ÂÃ¯Â­Â¦ÃÂ­t,
               tÃ¯ÂºÂ£ÃÂm/tÃ¯Â»ÂÃ¯Â­Â¦ÃÂ°o manga trong DB, dÃ¯Â»ÂÃ¯Â­Â¨ÃÂch toÃ¯ÂºÂ£ÃÂ n chapter.
"""

import re
import logging
import uuid
import asyncio
import httpx
from datetime import datetime, timezone
from urllib.parse import urlparse
from pathlib import Path

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scrape", tags=["scrape"])

# Thu muc luu ket qua dich (CWD = BE/ sau khi run.py thuc hien chdir)
PREVIEW_DIR = Path("../library")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory Job Store
# Resets on server restart Ã¯Â»ÂÃÂÃÂ dÃ¯ÂºÂ£Ã¯ÂºÂng Redis/DB nÃ¯Â»ÂÃ¯Â­Â¦Ã¯ÂºÂu cÃ¯Â»ÂÃ¯Â­Â¦ÃÂ¶n persist.
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}


def _make_job(
    job_id: str,
    manga_id: str,
    manga_name: str,
    chapter: int,
    total: int,
    content_type: str = "manga",   # "manga" | "novel"
) -> dict:
    reader_url = (
        f"http://localhost:8000/reader"
        f"?job_id={job_id}&manga_id={manga_id}&chapter={chapter}"
        f"&title={manga_name.replace(' ', '+')}"
        f"&type={content_type}"
    )
    return {
        "job_id":            job_id,
        "content_type":      content_type,       # "manga" | "novel"
        "status":            "queued",            # queued | downloading | translating | done | error
        "manga_id":          manga_id,
        "manga_name":        manga_name,
        "chapter_number":    chapter,
        "total_pages":       total,
        "translated_pages":  0,
        "pages":             [],                  # manga: filled page-by-page; novel: empty
        "novel_text":        None,                # novel: full translated text; manga: None
        "error":             None,
        "reader_url":        reader_url,
    }

# Domain quÃ¯Â»ÂÃ¯Â­Â¦ÃÂ²ng cÃ¯ÂºÂ£ÃÂ°o phÃ¯Â»ÂÃ¯Â­Â¨ÃÂ biÃ¯Â»ÂÃ¯Â­Â¦Ã¯ÂºÂn cÃ¯Â»ÂÃ¯Â­Â¦ÃÂ¶n bÃ¯Â»ÂÃ¯Â­Â¨ÃÂ qua
AD_DOMAINS = {
    "pubadx", "googlesyndication", "doubleclick", "adsystem",
    "amazon-adsystem", "adnxs", "rubiconproject", "openx",
    "pubmatic", "criteo", "taboola", "outbrain",
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ImageInfo(BaseModel):
    url:    str
    index:  int
    width:  int | None = None
    height: int | None = None
    alt:    str | None = None


class MangaMetadata(BaseModel):
    title:        str
    chapter:      int
    chapter_raw:  str
    url:          str
    domain:       str
    extracted_at: str


class ScrapePayload(BaseModel):
    metadata:         MangaMetadata
    image_urls:       list[ImageInfo] = []  # Manga mode: danh sach anh
    text_paragraphs:  list[str]       = []  # Novel mode: danh sach doan van
    total_pages:      int              = 0
    cookies:          str | None       = None
    user_agent:       str | None       = None
    # Tuy chon tu Setup Panel cua extension
    mode:             str              = "async"                  # "async" | "sync" | "rich" | "novel"
    model:            str              = "gemini-3.1-flash-lite"  # Chi dung khi mode=sync/rich/novel
    genre_list:       list[str]        = []                       # The loai de nap vao prompt


class UrlScrapePayload(BaseModel):
    """Payload tu mobile client -- chi can URL, server tu fetch va parse HTML."""
    url:        str
    mode:       str       = ""                     # rong = tu detect; hoac "novel"/"async"/"rich"
    model:      str       = "gemini-3.1-flash-lite"
    genre_list: list[str] = []


# ---------------------------------------------------------------------------
# Title utilities — clean_scraped_title là hàm chuẩn hóa duy nhất cho toàn hệ thống
# ---------------------------------------------------------------------------

# Chapter pattern: "Chapter 300", "Chap 300", "Ch.300", "Tập 300", "Episode 12"
_CHAPTER_PATTERN = re.compile(
    r"(?:第\s*|(?:chapter|chap|ch\.?|tap|t\u1eadp|ep(?:isode)?)[\s.\-#]*)"
    r"(?P<num>\d+)(?:\s*[\u8a71\u7ae0\u5dfb\u8a71])?"  # optional 話/章/巻 suffix
    ,
    re.IGNORECASE,
)

# Site name suffix + mọi thứ phía sau: "- Mangapill", "| MangaClouds", "@ xyz.com"
_SITE_NAME_RE = re.compile(
    r"\s*[-\u2013\u2014|\xb7@]\s*"
    r"(?:mangapill|mangaclouds|mangakakalot|mangadex|mangaplus|mangahere|"
    r"readmanga|manganelo|mangafox|mangastream|mangatown|mangapark|"
    r"webtoon|tapas|comicwalker|viz(?:\s*media)?|crunchyroll|"
    r"read\s+manga\s+online|manga\s+online|online\s+free|"
    r"sm\b"
    r"|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef]+)"  # CJK site names
    r".*$",
    re.IGNORECASE,
)

# Suffix "… Manga Online", "… Manhwa Online Free" nếu còn sót
_ONLINE_SUFFIX_RE = re.compile(
    r"\s*\b(?:manga|manhwa|manhua|webtoon)\s+online\b.*$",
    re.IGNORECASE,
)

# Prefix "Read " đứng đầu: "Read Kill Blue EN Manga Online…"
_READ_PREFIX_RE = re.compile(r"^read\s+", re.IGNORECASE)

# Language suffix: " EN", " JP", " (KR)", "(EN)"
_LANG_SUFFIX_RE = re.compile(
    r"\s+\(?(?:en|jp|kr|cn|vi|eng)\)?\s*$",
    re.IGNORECASE,
)

# File extension còn sót: ".html", ".html.html"
_FILE_EXT_RE = re.compile(r"(?:\.html?)+$", re.IGNORECASE)

# Đặc ký cần normalize: × → x, · → khoảng trắng
_SPECIAL_CHARS_RE = re.compile(r"[×·•]")

# Tiểu từ giữ lowercase trong Title Case (trừ từ đầu tiên)
_SMALL_TITLE_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "nor", "so", "yet", "as", "if",
    "up", "vs", "vs.", "de", "la", "el",
}


def clean_scraped_title(raw: str) -> str:
    """
    Hàm chuẩn hóa title duy nhất cho toàn hệ thống.

    Input  : raw page title từ bất kỳ trang web manga nào.
    Output : display name sạch, Title Case, dùng cho folder name và DB.

    Dùng trước normalize_manga_title() (→ lookup key lowercase)
    và make_library_path() (→ folder name).

    Ví dụ:
      "Munou na Nana Chapter 100 - Mangapill"              → "Munou na Nana"
      "Hunter X Hunter EN - MangaClouds"                   → "Hunter X Hunter"
      "Read Kill Blue EN Manga Online - Read Online Free"  → "Kill Blue"
      "hunter x hunter en chapter 400.html.html"          → "Hunter X Hunter"
      "One Piece - Read Manga Online Free"                 → "One Piece"
      "Hunter x Hunter Hunter x Hunter EN"                 → "Hunter X Hunter"
    """
    s = raw.strip()

    # 1. Xóa file extension còn sót (.html, .html.html)
    s = _FILE_EXT_RE.sub("", s)

    # 2. Xóa prefix "Read "
    s = _READ_PREFIX_RE.sub("", s)

    # 3. Xóa site name + mọi thứ phía sau
    s = _SITE_NAME_RE.sub("", s)

    # 4. Xóa suffix "Manga Online…"
    s = _ONLINE_SUFFIX_RE.sub("", s)

    # 5. Xóa chapter info: "Chapter 100", "Ch.400"
    s = _CHAPTER_PATTERN.sub("", s)

    # 6. Xóa language suffix: " EN", " (JP)"
    s = _LANG_SUFFIX_RE.sub("", s)

    # 7. Normalize đặc ký: × → x, · → space
    s = _SPECIAL_CHARS_RE.sub(lambda m: "x" if m.group() == "×" else " ", s)

    # 8. Strip separators đầu/cuối, collapse whitespace
    s = re.sub(r"[\s\-_:,|.]+$", "", s)
    s = re.sub(r"^[\s\-_:,|.]+", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    # 9. Dedup repeated segments: "Hunter x Hunter Hunter x Hunter" → "Hunter x Hunter"
    words = s.split()
    n     = len(words)
    for half in range(n // 2, 0, -1):
        if [w.lower() for w in words[:half]] == [w.lower() for w in words[half:half * 2]]:
            words = words[:half]
            s     = " ".join(words)
            break

    # 10. Title Case
    result = []
    for i, w in enumerate(words):
        if i == 0 or w.lower() not in _SMALL_TITLE_WORDS:
            result.append(w[0].upper() + w[1:] if w else "")
        else:
            result.append(w.lower())
    return " ".join(result)


def normalize_manga_title(title: str) -> str:
    """
    Chuẩn hóa title để dùng so sánh / lookup DB (lowercase).

    Input có thể là raw hoặc đã qua clean_scraped_title().
    Output luôn là lowercase để so sánh fuzzy.
    """
    title = _CHAPTER_PATTERN.sub("", title)
    title = re.sub(r"[\-_]+$", "", title.strip())
    return title.strip().lower()


def parse_title_and_chapter(title: str, chapter: int) -> tuple[str, int]:
    """
    Trích xuất tên truyện và số chapter từ raw title, sau đó chuẩn hóa tên qua
    clean_scraped_title() để đảm bảo output luôn sạch dù input có dạng gì.

    Ví dụ:
      parse_title_and_chapter("Hunter x Hunter Chapter 300", 0)
        → ("Hunter X Hunter", 300)
      parse_title_and_chapter("One Piece", 1111)
        → ("One Piece", 1111)   ← chapter không đổi, giữ nguyên
    """
    match = _CHAPTER_PATTERN.search(title)

    if match:
        detected_chapter = int(match.group("num"))
        raw_name = _CHAPTER_PATTERN.sub("", title)
        raw_name = re.sub(r"[\-_\s]+$", "", raw_name).strip()
        chapter_out = detected_chapter if chapter == 0 else chapter
    else:
        raw_name    = title
        chapter_out = chapter

    return clean_scraped_title(raw_name), chapter_out


def is_manga_page(img: ImageInfo) -> bool:
    """TrÃ¯Â»ÂÃ¯Â­Â¦ÃÂ² True nÃ¯Â»ÂÃ¯Â­Â¦Ã¯ÂºÂu Ã¯ÂºÂ¥ÃÂÃ¯ÂºÂ£ÃÂ±y cÃ¯ÂºÂ£Ã¯ÂºÂ khÃ¯Â»ÂÃ¯Â­Â¦ÃÂ² nÃ¯ÂºÂ¥ÃÂng lÃ¯ÂºÂ£ÃÂ  Ã¯Â»ÂÃ¯Â­Â¦ÃÂ²nh trang truyÃ¯Â»ÂÃ¯Â­Â¨ÃÂn thÃ¯Â»ÂÃ¯Â­Â¦ÃÂ­t."""
    url = img.url.lower()

    # 1. BÃ¯Â»ÂÃ¯Â­Â¨ÃÂ quÃ¯Â»ÂÃ¯Â­Â¦ÃÂ²ng cÃ¯ÂºÂ£ÃÂ°o theo domain
    for ad in AD_DOMAINS:
        if ad in url:
            return False

    # 2. BÃ¯Â»ÂÃ¯Â­Â¨ÃÂ Ã¯Â»ÂÃ¯Â­Â¦ÃÂ²nh data URI
    if url.startswith("data:"):
        return False

    # 3. BÃ¯Â»ÂÃ¯Â­Â¨ÃÂ Ã¯Â»ÂÃ¯Â­Â¦ÃÂ²nh nhÃ¯Â»ÂÃ¯Â­Â¨ÃÂ theo kÃ¯ÂºÂ£ÃÂ­ch thÃ¯ÂºÂ©Ã¯ÂºÂÃ¯Â»ÂÃ¯Â­Â¨ÃÂc (logo, icon, avatar)
    w = img.width or 0
    h = img.height or 0
    if w > 0 and h > 0:
        if w < 200 or h < 200:     # quÃ¯ÂºÂ£ÃÂ° nhÃ¯Â»ÂÃ¯Â­Â¨ÃÂ
            return False
        if w > h * 3:              # quÃ¯ÂºÂ£ÃÂ° ngang Ã¯Â»ÂÃÂÃÂ banner quÃ¯Â»ÂÃ¯Â­Â¦ÃÂ²ng cÃ¯ÂºÂ£ÃÂ°o
            return False

    # 4. BÃ¯Â»ÂÃ¯Â­Â¨ÃÂ cÃ¯ÂºÂ£ÃÂ°c pattern URL rÃ¯ÂºÂ£Ã¯Â­Â rÃ¯ÂºÂ£ÃÂ ng lÃ¯ÂºÂ£ÃÂ  khÃ¯ÂºÂ£Ã¯ÂºÂng phÃ¯Â»ÂÃ¯Â­Â¦ÃÂ²i trang truyÃ¯Â»ÂÃ¯Â­Â¨ÃÂn
    skip_keywords = ["logo", "avatar", "banner", "icon", "thumbnail",
                     "favicon", "assets/images", "cover", "background"]
    for kw in skip_keywords:
        if kw in url:
            return False

    # 5. Ã¯ÂºÂ©Ã¯ÂºÂu tiÃ¯ÂºÂ£ÃÂ¹n Ã¯Â»ÂÃ¯Â­Â¦ÃÂ²nh cÃ¯ÂºÂ£Ã¯ÂºÂ pattern rÃ¯ÂºÂ£Ã¯Â­Â lÃ¯ÂºÂ£ÃÂ  trang truyÃ¯Â»ÂÃ¯Â­Â¨ÃÂn
    manga_keywords = [
        "chapter", "chap", "storage", "uploads", "pages",
        "001.", "002.", "003.", "/0", "page", "scan",
    ]
    has_manga_signal = any(kw in url for kw in manga_keywords)

    # 6. NÃ¯Â»ÂÃ¯Â­Â¦Ã¯ÂºÂu Ã¯Â»ÂÃ¯Â­Â¦ÃÂ²nh lÃ¯Â»ÂÃ¯Â­Â¨ÃÂn (manga page thÃ¯ÂºÂ©Ã¯ÂºÂÃ¯Â»ÂÃ¯Â­Â¨ÃÂng >= 800px chiÃ¯Â»ÂÃ¯Â­Â¨ÃÂu cao)
    is_large = (h >= 600) or (h == 0 and w == 0)  # unknown size Ã¯Â»ÂÃÂÃÂ giÃ¯Â»ÂÃ¯Â­Â¨Ã¯ÂºÂ lÃ¯Â»ÂÃ¯Â­Â¦ÃÂ°i

    return has_manga_signal or is_large

# ---------------------------------------------------------------------------
# Download images
# ---------------------------------------------------------------------------

async def download_previews(
    imgs: list[ImageInfo],
    session_id: str,
    cookies: str | None,
    user_agent: str | None,
    page_url: str | None = None,
    max_pages: int = 5,
) -> list[dict]:
    """Download at most max_pages images to local. page_url used as Referer."""
    session_dir = PREVIEW_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": user_agent or "Mozilla/5.0",
        "Referer":    page_url or (imgs[0].url if imgs else ""),
        "Accept":     "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    if cookies:
        headers["Cookie"] = cookies

    results = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for i, img in enumerate(imgs[:max_pages]):
            try:
                resp = await client.get(img.url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"[Preview] Skip {img.url[:60]} -> HTTP {resp.status_code}")
                    continue
                ct = resp.headers.get("content-type", "")
                if "png" in ct or img.url.endswith(".png"):
                    ext = "png"
                elif "webp" in ct or img.url.endswith(".webp"):
                    ext = "webp"
                else:
                    ext = "jpg"
                filename = f"page_{i+1:03d}.{ext}"
                filepath = session_dir / filename
                filepath.write_bytes(resp.content)
                size_kb = len(resp.content) // 1024
                logger.info(f"[Preview] -> {filename} ({size_kb} KB)")
                results.append({
                    "page":        i + 1,
                    "filename":    filename,
                    "preview_url": f"/previews/{session_id}/{filename}",
                    "source_url":  img.url,
                    "size_kb":     size_kb,
                })
            except Exception as e:
                logger.warning(f"[Preview] Download fail {img.url[:60]}: {e}")
    return results


async def download_all_images(
    imgs: list[ImageInfo],
    *,
    cookies: str | None,
    user_agent: str | None,
    page_url: str | None = None,
) -> list[bytes]:
    """
    Download ALL images (no page limit).
    page_url is used as Referer -- required by CDNs with hotlink protection.
    Returns list[bytes]; failed images are skipped (logged as warning).
    """
    headers = {
        "User-Agent": user_agent or "Mozilla/5.0",
        "Referer":    page_url or (imgs[0].url if imgs else ""),
        "Accept":     "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    if cookies:
        headers["Cookie"] = cookies

    results: list[bytes] = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for i, img in enumerate(imgs):
            # Inline base64 data: URL (blob URLs converted by extension)
            if img.url.startswith("data:"):
                try:
                    _, b64 = img.url.split(",", 1)
                    img_bytes = base64.b64decode(b64)
                    results.append(img_bytes)
                    logger.debug(f"[Download] [{i+1}/{len(imgs)}] inline data: OK ({len(img_bytes)//1024} KB)")
                except Exception as e:
                    logger.warning(f"[Download] Skip [{i}]: failed decode inline data - {e}")
                continue
            if not img.url.startswith(("http://", "https://")):
                logger.warning(f"[Download] Skip [{i}]: missing scheme - {img.url[:80]!r}")
                continue
            try:
                resp = await client.get(img.url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"[Download] Skip [{i}] HTTP {resp.status_code}: {img.url[:60]}")
                    continue
                results.append(resp.content)
                size_kb = len(resp.content) // 1024
                logger.debug(f"[Download] [{i+1}/{len(imgs)}] {size_kb} KB OK")
            except Exception as e:
                logger.warning(f"[Download] Fail [{i}]: {e}")
    return results


# ---------------------------------------------------------------------------
# /scrape/test
# ---------------------------------------------------------------------------

@router.post("/test")
async def scrape_test(payload: ScrapePayload) -> JSONResponse:
    """
    Test endpoint: nhan du lieu tu Chrome Extension.
    Flow: filter anh -> tim/tao manga DB -> download tat ca anh -> dich.
    """
    from app.services.db_service import AsyncSessionLocal
    from app.api.routes.translate import translate_chapter

    meta     = payload.metadata
    all_imgs = payload.image_urls

    manga_imgs   = [img for img in all_imgs if is_manga_page(img)]
    filtered_out = len(all_imgs) - len(manga_imgs)

    logger.info("=" * 60)
    logger.info("[Scrape] Nhan du lieu tu Extension")
    logger.info(f"  Truyen   : {meta.title}")
    logger.info(f"  Chapter  : {meta.chapter} | Domain: {meta.domain}")
    logger.info(f"  Anh      : {len(all_imgs)} total -> {len(manga_imgs)} sau filter (loai {filtered_out})")

    if not manga_imgs:
        return JSONResponse(content={"status": "no_images", "message": "Khong tim thay anh manga sau khi filter"})

    clean_title, clean_chapter = parse_title_and_chapter(meta.title, meta.chapter)
    logger.info(f"  Title parse: {meta.title!r} -> title={clean_title!r}, ch={clean_chapter}")

    from app.services.anilist_service import resolve_manga_id
    async with AsyncSessionLocal() as session:
        manga_id, canonical_name = await resolve_manga_id(clean_title, session)
    logger.info(f"  [=] manga_id={manga_id!r} -> {canonical_name!r}")

    logger.info(f"  [v] Dang download {len(manga_imgs)} anh...")
    images_bytes = await download_all_images(
        manga_imgs,
        cookies=payload.cookies,
        user_agent=payload.user_agent,
        page_url=meta.url,
    )
    logger.info(f"  [v] Download xong {len(images_bytes)}/{len(manga_imgs)} anh")

    if not images_bytes:
        return JSONResponse(content={
            "status":  "download_failed",
            "message": "Khong download duoc anh nao (co the bi chan hotlink)",
        })

    logger.info(f"  [>] Bat dau dich {len(images_bytes)} trang...")
    logger.info("=" * 60)

    result = await translate_chapter(
        images_bytes,
        manga_id=manga_id,
        manga_name=clean_title,
        chapter_number=clean_chapter,
    )

    logger.info("=" * 60)
    logger.info(f"  [v] Dich xong -> {result['translated_pages']}/{result['total_pages']} trang")
    logger.info("=" * 60)

    return JSONResponse(content={
        "status":           "ok",
        "manga_id":         manga_id,
        "title":            canonical_name,

        "chapter":          clean_chapter,
        "total_pages":      result["total_pages"],
        "translated_pages": result["translated_pages"],
        "output_dir":       result["output_dir"],
        "page_results":     result["page_results"],
        "hint": f"Xem anh tai: http://localhost:8000/library/{manga_id}/{clean_chapter}/page_001.png",
    })


# ---------------------------------------------------------------------------
# /scrape/submit Async: tra job_id + reader_url ngay, dich nen
# ---------------------------------------------------------------------------

@router.post("/submit")
async def scrape_submit(payload: ScrapePayload) -> JSONResponse:
    """
    Submit chapter de dich trong background.
    Tra ve ngay: { job_id, reader_url } -- extension mo reader_url trong tab moi.
    Reader page tu poll /scrape/job/{job_id} de hien tung trang khi xong.
    """
    from app.services.db_service import AsyncSessionLocal, MangaCRUD
    from app.services.anilist_service import resolve_manga_id

    meta     = payload.metadata
    all_imgs  = payload.image_urls
    is_novel  = payload.mode == "novel"

    clean_title, clean_chapter = parse_title_and_chapter(meta.title, meta.chapter)
    async with AsyncSessionLocal() as session:
        manga_id, canonical_name = await resolve_manga_id(
            clean_title, session, is_novel=is_novel
        )
    novel_display_name = canonical_name

    if is_novel:
        # Novel mode: validate text_paragraphs thay vi anh
        if not payload.text_paragraphs:
            raise HTTPException(status_code=400, detail="Novel mode: text_paragraphs trong.")
        novel_paragraphs = payload.text_paragraphs
        manga_imgs       = []          # khong co anh
        total_items      = len(novel_paragraphs)
    else:
        # Manga mode: validate images
        manga_imgs = [img for img in all_imgs if is_manga_page(img)]
        if not manga_imgs:
            raise HTTPException(status_code=400, detail="Khong tim thay anh manga sau khi filter.")
        novel_paragraphs = []
        total_items      = len(manga_imgs)

    job_id = uuid.uuid4().hex
    job    = _make_job(
        job_id, manga_id, novel_display_name, clean_chapter,
        total_items, content_type="novel" if is_novel else "manga",
    )
    _jobs[job_id] = job

    logger.info(
        f"[Job {job_id[:8]}] Created -- {novel_display_name!r} "
        f"ch={clean_chapter}, type={job['content_type']}, "
        f"{'paragraphs' if is_novel else 'pages'}={total_items}"
    )

    async def _run():
        from app.services.translation_strategies import get_strategy, JobContext
        try:
            ctx = JobContext(
                payload=payload,
                job=job,
                manga_id=manga_id,
                display_name=novel_display_name,
                chapter_number=clean_chapter,
                manga_imgs=manga_imgs,
                paragraphs=novel_paragraphs,
            )
            result = await get_strategy(payload.mode).execute(ctx)

            try:
                from app.celery_tasks.background_task import process_chapter_background_after_VLM_response
                process_chapter_background_after_VLM_response.delay(
                    manga_id=manga_id,
                    chapter_number=clean_chapter,
                    manga_name=novel_display_name,
                )
            except Exception as bg_err:
                logger.warning(f"[Job {job_id[:8]}] BG task dispatch skip: {bg_err}")

            job["status"] = "done"
            logger.info(f"[Job {job_id[:8]}] Done -- {job['translated_pages']}/{job['total_pages']}p")

        except Exception as exc:
            job["status"] = "error"
            job["error"]  = str(exc)
            logger.error(f"[Job {job_id[:8]}] Error: {exc}", exc_info=True)

    asyncio.create_task(_run())

    return JSONResponse(content={
        "status":      "accepted",
        "job_id":      job_id,
        "reader_url":  job["reader_url"],
        "manga_id":    manga_id,
        "title":       novel_display_name,
        "chapter":     clean_chapter,
        "total_pages": total_items,
        "content_type": job["content_type"],
    })


# ---------------------------------------------------------------------------
# /scrape/job/{job_id}  -- Polling status endpoint cho reader
# ---------------------------------------------------------------------------

@router.get("/job/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """Tra ve trang thai hien tai cua job -- reader poll moi 2s."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    return JSONResponse(content={
        "job_id":           job["job_id"],
        "content_type":     job.get("content_type", "manga"),
        "status":           job["status"],
        "manga_name":       job["manga_name"],
        "chapter_number":   job["chapter_number"],
        "total_pages":      job["total_pages"],
        "translated_pages": job["translated_pages"],
        "pages":            sorted(job["pages"], key=lambda p: p["page_index"]),
        "novel_text":       job.get("novel_text"),    # None cho manga, string cho novel
        "error":            job["error"],
    })


# ---------------------------------------------------------------------------
# /scrape/url  -- Server-side scraping for mobile (Scriptable / iOS)
# ---------------------------------------------------------------------------

_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Mobile/15E148 Safari/604.1"
)

_IMG_SRC_RE = re.compile(
    r'<img\b[^>]*?\s(?:data-original|data-src|data-lazy-src|data-lazy|src)'
    r'\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_IMG_SKIP_RE = re.compile(
    r'(logo|avatar|banner|favicon|spinner|loading|ad[_\-/]'
    r'|doubleclick|googlesyndication|adsystem|pixel\.gif|1x1|blank)',
    re.IGNORECASE,
)
_OG_TITLE_RE  = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE)
_TITLE_TAG_RE = re.compile(r'<title[^>]*>([^<]+)</title>', re.IGNORECASE)
_H1_TAG_RE    = re.compile(r'<h1[^>]*>([^<]+)</h1>', re.IGNORECASE)
_PARA_RE      = re.compile(r'<p[^>]*>(.*?)</p>', re.IGNORECASE | re.DOTALL)
_STRIP_TAG_RE = re.compile(r'<[^>]+>')
_CHAPTER_RE   = re.compile(
    r'(?:chapter|chap|ch|tap|episode)[/\-_\s\.#\(]*([0-9]+(?:\.[0-9]+)?)',
    re.IGNORECASE,
)


async def _fetch_page(url: str) -> tuple[str, str, str]:
    """Fetch HTML with mobile Safari UA. Returns (html_text, final_url, domain)."""
    headers = {
        "User-Agent":      _MOBILE_UA,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         url,
    }
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
    resp.raise_for_status()
    final_url = str(resp.url)
    domain    = urlparse(final_url).netloc.replace("www.", "")
    return resp.text, final_url, domain


def _parse_html_content(html: str, page_url: str) -> dict:
    """Parse raw HTML. Returns {title, chapter, chapter_raw, images, paragraphs}."""
    og  = _OG_TITLE_RE.search(html)
    h1  = _H1_TAG_RE.search(html)
    ttl = _TITLE_TAG_RE.search(html)
    raw_title = (
        (og  and og.group(1))  or
        (h1  and h1.group(1))  or
        (ttl and ttl.group(1)) or
        page_url
    )
    raw_title = _STRIP_TAG_RE.sub("", raw_title).strip()

    ch_match    = _CHAPTER_RE.search(page_url + " " + raw_title)
    chapter     = int(float(ch_match.group(1))) if ch_match else 0
    chapter_raw = ch_match.group(0) if ch_match else ""

    paragraphs: list[str] = []
    for m in _PARA_RE.finditer(html):
        text = _STRIP_TAG_RE.sub("", m.group(1)).strip()
        if len(text) > 20:
            paragraphs.append(text)

    images: list[ImageInfo] = []
    seen: set[str] = set()
    for m in _IMG_SRC_RE.finditer(html):
        src = m.group(1).strip()
        if not src.startswith("http"):
            continue
        if _IMG_SKIP_RE.search(src):
            continue
        if src in seen:
            continue
        seen.add(src)
        images.append(ImageInfo(url=src, index=len(images)))

    return {"title": raw_title, "chapter": chapter, "chapter_raw": chapter_raw,
            "images": images, "paragraphs": paragraphs}



async def _scrape_and_translate(url: str, body: UrlScrapePayload, job_id: str) -> None:
    """
    Background coroutine: Playwright scrape → DB → translate.
    Cập nhật placeholder job in-place để reader có thể poll tiến trình.
    """
    job = _jobs.get(job_id)
    if not job:
        return

    try:
        # ── Phase 1: Playwright ──────────────────────────────────────────────
        from app.services.browser_scraper import scrape_with_playwright
        parsed = await scrape_with_playwright(url)

        images = parsed["images"]
        paras  = parsed["paragraphs"]

        if not images and not paras:
            job["status"] = "error"
            job["error"]  = "Khong tim thay noi dung. Site co the yeu cau dang nhap."
            logger.warning(f"[Job {job_id[:8]}] No content found on {url!r}")
            return

        logger.info(
            f"[Job {job_id[:8]}] Scraped - title={parsed['title']!r:.50} "
            f"ch={parsed['chapter']} imgs={len(images)} paras={len(paras)}"
        )

        #Phase 2: DB + job setup 
        from app.services.db_service import AsyncSessionLocal, MangaCRUD

        clean_title, clean_chapter = parse_title_and_chapter(parsed["title"], parsed["chapter"])
        is_novel     = (body.mode == "novel")

        from app.services.anilist_service import resolve_manga_id
        async with AsyncSessionLocal() as session:
            manga_id, display_name = await resolve_manga_id(
                clean_title, session, is_novel=is_novel
            )
        # display_name: romaji từ AniList hoặc clean_title fallback (+ "[Novel]" nếu là novel)

        manga_imgs       = [img for img in images if is_manga_page(img)] if not is_novel else []
        novel_paragraphs = paras if is_novel else []
        total_items      = len(novel_paragraphs if is_novel else manga_imgs)

        if not is_novel and not manga_imgs:
            job["status"] = "error"
            job["error"]  = "Không tìm thấy ảnh manga sau khi filter."
            return

        # Cập nhật job
        job["manga_id"]       = manga_id
        job["manga_name"]     = display_name
        job["chapter_number"] = clean_chapter
        job["total_pages"]    = total_items
        job["content_type"]   = "novel" if is_novel else "manga"
        job["status"]         = "queued"
        job["reader_url"]     = (
            f"http://localhost:8000/reader"
            f"?job_id={job_id}&manga_id={manga_id}&chapter={clean_chapter}"
            f"&title={display_name.replace(' ', '+')}&type={job['content_type']}"
        )

        logger.info(
            f"[Job {job_id[:8]}] Created {display_name!r} ch={clean_chapter}, "
            f"type={job['content_type']}, pages={total_items}"
        )
        #Phase 3: Translate
        from app.services.translation_strategies import get_strategy, JobContext

        payload = ScrapePayload(
            metadata=MangaMetadata(
                title=clean_title, chapter=clean_chapter,
                chapter_raw=parsed["chapter_raw"],
                url=parsed["final_url"], domain=parsed["domain"],
                extracted_at=datetime.now(timezone.utc).isoformat(),
            ),
            image_urls      = manga_imgs if not is_novel else [],
            text_paragraphs = novel_paragraphs,
            total_pages     = total_items,
            cookies=None, user_agent=_MOBILE_UA,
            mode=body.mode, model=body.model, genre_list=body.genre_list,
        )

        ctx = JobContext(
            payload=payload,
            job=job,
            manga_id=manga_id,
            display_name=display_name,
            chapter_number=clean_chapter,
            manga_imgs=manga_imgs,
            paragraphs=novel_paragraphs,
        )

        result = await get_strategy(body.mode).execute(ctx)

        try:
            from app.celery_tasks.background_task import process_chapter_background_after_VLM_response
            process_chapter_background_after_VLM_response.delay(
                manga_id=manga_id,
                chapter_number=clean_chapter,
                manga_name=display_name,
            )
        except Exception as bg_err:
            logger.warning(f"[Job {job_id[:8]}] BG task dispatch skip: {bg_err}")

        job["status"] = "done"
        logger.info(f"[Job {job_id[:8]}] Done - {job['translated_pages']}/{job['total_pages']}p")

    except Exception as exc:
        if job := _jobs.get(job_id):
            job["status"] = "error"
            job["error"]  = str(exc)
        logger.error(f"[Job {job_id[:8]}] Error: {exc}", exc_info=True)


@router.post("/url")
async def scrape_from_url(body: UrlScrapePayload) -> JSONResponse:
    """
    Server-side scraping for mobile (Scriptable / iOS Share Sheet).

    Tra reader_url NGAY LAP TUC (<1s), toan bo Playwright + dich chay background.
    Reader page poll /scrape/job/{job_id} de xem tien do.
    """
    # Normalize URL: strip duplicate .html.html
    url = re.sub(r"(\.html){2,}$", ".html", body.url, flags=re.IGNORECASE)
    if url != body.url:
        logger.info(f"[MobileScrape] URL normalized: {body.url!r} -> {url!r}")

    # Tạo placeholder job ngay để trả reader về cho iphone load ngay
    job_id = uuid.uuid4().hex
    placeholder = {
        "job_id":           job_id,
        "content_type":     "manga",
        "status":           "scraping",      #PLaywright đang load
        "manga_id":         "",
        "manga_name":       "Đang scrape...",
        "chapter_number":   0,
        "total_pages":      0,
        "translated_pages": 0,
        "pages":            [],
        "novel_text":       None,
        "error":            None,
        "reader_url":       f"http://localhost:8000/reader?job_id={job_id}",
    }
    _jobs[job_id] = placeholder

    logger.info(f"[MobileScrape] job={job_id[:8]} URL: {url!r} -> Playwright (background)")

    # Chay toan bo pipeline trong background - khong block HTTP response
    asyncio.create_task(_scrape_and_translate(url, body, job_id))

    return JSONResponse(content={
        "status":       "accepted",
        "job_id":       job_id,
        "reader_url":   placeholder["reader_url"],
        "title":        "Đang xử lí...",
        "chapter":      0,
        "total_pages":  0,
        "content_type": "manga",
    })
