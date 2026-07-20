"""
Module: app.celery_tasks.background_task
Description: Định nghĩa các Celery background task chạy sau mỗi lần dịch trang truyện.

Workflow sau khi nhận kết quả từ VLM:
  1. process_page_intelligence(): Task điều phối chính — nhận page_summary,
     character_updates, pronoun_shift rồi phân luồng sang 3 sub-task bên dưới.
  2. synthesize_chapter_summary(): Tổng hợp các page_summary thành chapter_summary
     bằng model local, lưu vào database.
  3. update_character_graph(): Cập nhật character_graph với character_updates mới.
  4. apply_pronoun_shifts(): Sửa mối quan hệ trong character_graph theo pronoun_shift.

TODO: Implement từng sub-task sau khi DB schema và model local được xác định.
"""

import logging
import asyncio
from app.celery_tasks.celery_config import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task điều phối chính
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.celery_tasks.background_task.process_chapter_background_after_VLM_response",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def process_chapter_background_after_VLM_response(
    self,
    manga_id: str,
    chapter_number: int,
    page_summaries: list[str],
    character_updates: list,
    pronoun_shifts: list,
    manga_name: str = "",           # tên truyện — để lưu đúng vào chapter_summaries.name
):
    """
    Task điều phối nhận kết quả Page Intelligence đã gộp toàn chapter.
    Chạy 1 lần per-chapter (thay vì per-page trước đây).

    Args:
        manga_id:          ID bộ truyện.
        chapter_number:    Số thứ tự chapter.
        page_summaries:    List tóm tắt từng trang → tổng hợp thành chapter_summary.
        character_updates: List nhân vật xuất hiện trong chapter → update graph.
        pronoun_shifts:    List thay đổi xưng hô → apply vào graph.
        manga_name:        Tên truyện — lưu vào chapter_summaries.name.
    """
    logger.info(
        f"📬 [BG Task] process_chapter — manga={manga_id!r}, ch={chapter_number}, "
        f"{len(page_summaries)} trang summary, "
        f"{len(character_updates)} char updates, {len(pronoun_shifts)} shifts"
    )
    try:
        # Sub-task 1: Tổng hợp chapter summary từ tất cả page summaries
        if page_summaries:
            synthesize_chapter_summary.delay(
                manga_id, chapter_number, " ".join(page_summaries),
                manga_id=manga_id,
                chapter_number=chapter_number,
                chapter_name=manga_name or None,   # ← lưu tên truyện vào .name
            )

        # Sub-task 2: Merge tất cả character_updates của chapter vào graph
        if character_updates:
            update_character_graph.delay(manga_id, character_updates)

        # Sub-task 3: Apply tất cả pronoun_shifts của chapter vào graph
        if pronoun_shifts:
            apply_pronoun_shifts.delay(manga_id, pronoun_shifts)

    except Exception as exc:
        logger.error(f"❌ [BG Task] Lỗi process_chapter: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# (Legacy alias) — giữ lại để không bị break nếu có task cũ trong queue Redis
# ---------------------------------------------------------------------------
process_page_background_after_VLM_response = process_chapter_background_after_VLM_response


# ---------------------------------------------------------------------------
# Sub-task 1: Tổng hợp chapter summary
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.celery_tasks.background_task.synthesize_chapter_summary",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def synthesize_chapter_summary(
    self,
    chapter_id: str,
    page_index: int,
    page_summary: str,
    manga_id: str = "",
    chapter_number: int = 0,
    chapter_name: str | None = None,
):
    """
    Gọi Ollama qwen2.5:3b để tổng hợp page_summary thành chapter_summary,
    sau đó lưu kết quả vào PostgreSQL.

    Celery task chạy sync nên dùng asyncio.run() để wrap coroutine.
    """
    import asyncio
    from app.ai.ollama import summarize_chapter

    logger.info(
        f"[BG Task] synthesize_chapter_summary — chapter={chapter_id}, page={page_index}"
    )
    try:
        from app.ai.ollama import summarize_chapter
        from app.services.db_service import make_task_session

        # Celery worker là sync context → dùng asyncio.run
        asyncio.run(
            summarize_chapter(
                page_summaries=[page_summary],
                manga_id=manga_id or chapter_id,
                chapter_number=chapter_number or page_index,
                chapter_name=chapter_name,
                session_factory=make_task_session(),  # fresh engine cho thread này
            )
        )
    except Exception as exc:
        logger.error(f"[BG Task] synthesize_chapter_summary thất bại: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Sub-task 2: Cập nhật character graph
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.celery_tasks.background_task.update_character_graph",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def update_character_graph(self, chapter_id: str, character_updates: list):
    """
    Merge character_updates từ VLM vào character graph của manga.
    Flow: Load graph từ DB → merge updates → lưu lại DB.
    """
    logger.info(
        f"👥 [BG Task] update_character_graph — chapter={chapter_id}, "
        f"{len(character_updates)} update(s)"
    )
    try:
        from app.services.graph_service import GraphService
        from app.services.db_service import make_task_session

        async def _run():
            sf = make_task_session()  # engine mới gắn với event loop của thread này
            G = await GraphService.load(chapter_id, use_cache=False, session_factory=sf)
            count = GraphService.merge_character_updates(G, character_updates)
            await GraphService.save(chapter_id, G, session_factory=sf)
            GraphService.invalidate_cache(chapter_id)
            return count

        count = asyncio.run(_run())
        logger.info(f"[BG Task] update_character_graph done — {count} node(s) upserted.")

    except Exception as exc:
        logger.error(f"[BG Task] update_character_graph thất bại: {exc}", exc_info=True)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Sub-task 3: Áp dụng pronoun shifts
# ---------------------------------------------------------------------------

@celery_app.task(
    name="app.celery_tasks.background_task.apply_pronoun_shifts",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def apply_pronoun_shifts(self, chapter_id: str, pronoun_shift: list):
    """
    Áp dụng pronoun_shift lên character graph của manga.
    Flow: Load graph từ DB → apply shifts → lưu lại DB.
    """
    logger.info(
        f"🔄 [BG Task] apply_pronoun_shifts — chapter={chapter_id}, "
        f"{len(pronoun_shift)} shift(s)"
    )
    try:
        from app.services.graph_service import GraphService
        from app.services.db_service import make_task_session

        async def _run():
            sf = make_task_session()  # engine mới gắn với event loop của thread này
            G = await GraphService.load(chapter_id, use_cache=False, session_factory=sf)
            applied = GraphService.merge_pronoun_shifts(G, pronoun_shift)
            if applied > 0:
                await GraphService.save(chapter_id, G, session_factory=sf)
                GraphService.invalidate_cache(chapter_id)
            return applied

        applied = asyncio.run(_run())
        logger.info(f"[BG Task] apply_pronoun_shifts done — {applied} shift(s) applied.")

    except Exception as exc:
        logger.error(f"[BG Task] apply_pronoun_shifts thất bại: {exc}", exc_info=True)
        raise self.retry(exc=exc)
