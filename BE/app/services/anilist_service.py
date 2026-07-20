import difflib
import json
import logging
import os
import re
import uuid as _uuid

import httpx
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession
# pyrefly: ignore [missing-import]
from sqlalchemy import select

from app.schemas import Manga

logger = logging.getLogger(__name__)

_CACHE_PATH = "anilist_cache.json"
_cache: dict | None = None

_ANILIST_URL = "https://graphql.anilist.co"
_ANILIST_QUERY = 'query ($search: String) {\n  Media(search: $search, type: MANGA) {\n    id\n    title { romaji english }\n  }\n}'

_SLUG_CHAPTER_RE = re.compile(
    "(?:\u7b2c\\s*|(?:chapter|chap|ch\\.?|tap|t\u1eadp|ep(?:isode)?)[\\s.\\-#]*)"
    "\\d+(?:\\s*[\u8a71\u7ae0\u5dfb])?",
    re.IGNORECASE,
)


def _to_slug(title: str) -> str:
    s = _SLUG_CHAPTER_RE.sub("", title)
    s = re.sub(r"[-_]+$", "", s.strip())
    return s.strip().lower()


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            logger.debug(f"[AniList] Cache loaded: {len(_cache)} entries")
        except Exception as e:
            logger.warning(f"[AniList] Cache load failed: {e}")
            _cache = {}
    else:
        _cache = {}
    return _cache


def _write_cache(key: str, manga_id: str, name: str) -> None:
    global _cache
    if _cache is None:
        _load_cache()
    _cache[key] = {"manga_id": manga_id, "name": name}
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[AniList] Cache write failed: {e}")


async def _query_anilist(title: str) -> tuple[str, str] | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                _ANILIST_URL,
                json={"query": _ANILIST_QUERY, "variables": {"search": title}},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        if resp.status_code != 200:
            logger.warning(f"[AniList] HTTP {resp.status_code} for title={title!r}")
            return None
        data = resp.json()
        media = (data.get("data") or {}).get("Media")
        if not media:
            logger.info(f"[AniList] Not found: {title!r}")
            return None
        anilist_id = media["id"]
        t = media.get("title") or {}
        canonical = t.get("romaji") or t.get("english") or title
        return f"al_{anilist_id}", canonical
    except Exception as e:
        logger.warning(f"[AniList] API error for {title!r}: {e}")
        return None


async def _fuzzy_db_match(slug: str, session: AsyncSession) -> tuple[str, str] | None:
    result = await session.execute(select(Manga.manga_id, Manga.name, Manga.name_slug))
    rows = result.all()
    if not rows:
        return None
    best_score = 0.0
    best_pair: tuple[str, str] | None = None
    for manga_id, name, name_slug in rows:
        cmp = name_slug or _to_slug(name or "")
        score = difflib.SequenceMatcher(None, slug, cmp).ratio()
        if score > best_score:
            best_score = score
            best_pair = (manga_id, name)
    if best_score >= 0.85 and best_pair:
        logger.info(
            f"[AniList] DB fuzzy: {slug!r} -> {best_pair[0]!r} score={best_score:.2f}"
        )
        return best_pair
    return None


async def resolve_manga_id(
    clean_title: str,
    session: AsyncSession,
    *,
    is_novel: bool = False,
) -> tuple[str, str]:
    slug = _to_slug(clean_title)
    cache_key = f"{slug}__novel" if is_novel else slug

    cache = _load_cache()
    if cache_key in cache:
        entry = cache[cache_key]
        logger.info(f"[AniList] Cache hit: {cache_key!r} -> {entry['manga_id']!r}")
        return entry["manga_id"], entry["name"]

    api_result = await _query_anilist(clean_title)
    if api_result:
        manga_id, canonical_name = api_result
        if is_novel:
            canonical_name = canonical_name + " [Novel]"
        _write_cache(cache_key, manga_id, canonical_name)
        logger.info(
            f"[AniList] API resolved: {clean_title!r} -> {manga_id!r} ({canonical_name!r})"
        )
        return manga_id, canonical_name

    fuzzy = await _fuzzy_db_match(slug, session)
    if fuzzy:
        existing_id, existing_name = fuzzy
        display = (existing_name + " [Novel]") if is_novel else existing_name
        _write_cache(cache_key, existing_id, display)
        return existing_id, display

    fallback_id = _uuid.uuid4().hex
    display = (clean_title + " [Novel]") if is_novel else clean_title
    _write_cache(cache_key, fallback_id, display)
    logger.warning(f"[AniList] UUID fallback for {clean_title!r}: {fallback_id}")
    return fallback_id, display
