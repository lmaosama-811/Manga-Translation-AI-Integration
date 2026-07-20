"""
Module: app.services.translation_strategies
Description: Strategy Pattern cho các pipeline dịch.

Mỗi strategy đại diện cho 1 chế độ (mode):
  NovelStrategy      — text-only, 1 API call, web novel
  AsyncMangaStrategy — ảnh, song song, Flash Lite (default)
  SyncMangaStrategy  — ảnh, tuần tự, chọn model
  RichMangaStrategy  — ảnh, tier-1 key, chọn model

Để thêm mode mới:
  1. Viết class kế thừa TranslationStrategy (hoặc _BaseMangaStrategy)
  2. Đăng ký vào STRATEGY_MAP
  => KHÔNG cần chỉnh sửa scrape.py hay bất kỳ caller nào khác.

Lưu ý circular import:
  - scrape.py  import get_strategy, JobContext từ module này (top-level OK)
  - Module này import từ scrape.py và translate.py LAZILY (trong method body)
    → an toàn vì các module đó đã load xong trước khi method được gọi.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


# ===========================================================================
# JobContext — túi dữ liệu truyền từ scrape_submit vào strategy
# ===========================================================================

@dataclass
class JobContext:
    """
    Tất cả dữ liệu cần thiết để chạy 1 translation job.

    Dùng `object` thay vì `ScrapePayload` để tránh circular import;
    các field bên trong payload được truy cập qua attribute access bình thường.
    """
    payload:        object  # ScrapePayload (type erased để tránh circular)
    job:            dict    # mutable job dict — strategy mutate in-place
    manga_id:       str
    display_name:   str
    chapter_number: int
    manga_imgs:     list = field(default_factory=list)  # manga: pre-filtered ImageInfo list
    paragraphs:     list = field(default_factory=list)  # novel: text paragraph list


# ===========================================================================
# Abstract Base
# ===========================================================================

class TranslationStrategy(ABC):
    """
    Interface chung cho mọi pipeline dịch.

    Contract:
      - Nhận JobContext, chạy toàn bộ pipeline
      - Mutate ctx.job in-place (status, pages/novel_text, translated_pages, v.v.)
      - Return dict kết quả raw (để Celery background task xử lý tiếp nếu cần)
      - Raise Exception nếu có lỗi nghiêm trọng → scrape.py sẽ set job error
    """

    @abstractmethod
    async def execute(self, ctx: JobContext) -> dict:
        ...


# ===========================================================================
# Novel Strategy
# ===========================================================================

class NovelStrategy(TranslationStrategy):
    """
    Text-only pipeline — web novel / light novel.

    Không có phase download ảnh, không có render/inpainting.
    Toàn bộ chapter được dịch trong 1 API call (tối đa 2 nếu RPM).
    """

    async def execute(self, ctx: JobContext) -> dict:
        from app.api.routes.translate import translate_novel

        job     = ctx.job
        payload = ctx.payload

        job["status"] = "translating"
        result = await translate_novel(
            ctx.paragraphs,
            manga_id=ctx.manga_id,
            manga_name=ctx.display_name,
            chapter_number=ctx.chapter_number,
            model=payload.model,
            genre_list=payload.genre_list,
        )

        job["novel_text"]       = result["translated_text"]
        job["chapter_summary"]  = result["chapter_summary"]
        job["total_pages"]      = 1   # novel = 1 đơn vị logic
        job["translated_pages"] = 1
        return result


# ===========================================================================
# Manga Strategies — shared download phase
# ===========================================================================

class _BaseMangaStrategy(TranslationStrategy):
    """
    Base chung cho mọi manga strategy.

    Phase 1 (download) giống nhau cho tất cả manga modes.
    Subclass chỉ cần implement _translate() với translate function tương ứng.
    """

    async def execute(self, ctx: JobContext) -> dict:
        # Lazy import để tránh circular (scrape.py được load trước module này)
        from app.api.routes.scrape import download_all_images

        job     = ctx.job
        payload = ctx.payload

        # ── Phase 1: Download ──
        job["status"] = "downloading"
        images_bytes = await download_all_images(
            ctx.manga_imgs,
            cookies=payload.cookies,
            user_agent=payload.user_agent,
            page_url=payload.metadata.url,   # Referer = chapter page URL (fix CDN 403)
        )
        if not images_bytes:
            raise RuntimeError("Không download được ảnh (hotlink blocked hoặc URL hết hạn?)")

        # ── Phase 2: Translate ──
        job["status"]      = "translating"
        job["total_pages"] = len(images_bytes)

        def _on_page_ready(page_index: int, preview_url: str) -> None:
            """Callback được gọi khi mỗi trang dịch xong — cập nhật job real-time."""
            job["pages"].append({"page_index": page_index, "preview_url": preview_url})
            job["translated_pages"] = len(job["pages"])

        return await self._translate(ctx, images_bytes, _on_page_ready)

    @abstractmethod
    async def _translate(
        self,
        ctx: JobContext,
        images_bytes: list[bytes],
        on_page_ready: Callable[[int, str], None],
    ) -> dict:
        """Override với translate function của từng mode."""
        ...


class AsyncMangaStrategy(_BaseMangaStrategy):
    """mode='async' — song song, Flash Lite, tự retry RPM."""

    async def _translate(self, ctx, images_bytes, on_page_ready):
        from app.api.routes.translate import translate_chapter
        return await translate_chapter(
            images_bytes,
            manga_id=ctx.manga_id,
            manga_name=ctx.display_name,
            chapter_number=ctx.chapter_number,
            genre_list=ctx.payload.genre_list,
            on_page_ready=on_page_ready,
        )


class SyncMangaStrategy(_BaseMangaStrategy):
    """mode='sync' — tuần tự, chọn model."""

    async def _translate(self, ctx, images_bytes, on_page_ready):
        from app.api.routes.translate import translate_sync
        return await translate_sync(
            images_bytes,
            manga_id=ctx.manga_id,
            manga_name=ctx.display_name,
            chapter_number=ctx.chapter_number,
            model=ctx.payload.model,
            genre_list=ctx.payload.genre_list,
            on_page_ready=on_page_ready,
        )


class RichMangaStrategy(_BaseMangaStrategy):
    """mode='rich' — tier-1 key, 20 workers, chọn model."""

    async def _translate(self, ctx, images_bytes, on_page_ready):
        from app.api.routes.translate import translate_rich
        return await translate_rich(
            images_bytes,
            manga_id=ctx.manga_id,
            manga_name=ctx.display_name,
            chapter_number=ctx.chapter_number,
            model=ctx.payload.model,
            genre_list=ctx.payload.genre_list,
            on_page_ready=on_page_ready,
        )


# ===========================================================================
# Registry — đăng ký mode → strategy instance
# ===========================================================================

STRATEGY_MAP: dict[str, TranslationStrategy] = {
    "novel":  NovelStrategy(),
    "async":  AsyncMangaStrategy(),
    "sync":   SyncMangaStrategy(),
    "rich":   RichMangaStrategy(),
}


def get_strategy(mode: str) -> TranslationStrategy:
    """
    Tra cứu strategy theo mode string.
    Fallback về AsyncMangaStrategy nếu mode không tìm thấy.
    """
    strategy = STRATEGY_MAP.get(mode)
    if strategy is None:
        logger.warning(f"[Strategy] Mode không nhận ra: {mode!r} — fallback về 'async'")
        strategy = STRATEGY_MAP["async"]
    return strategy
