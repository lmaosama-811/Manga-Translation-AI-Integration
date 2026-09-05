"""
Module: app.celery_tasks.background_task
Description: Định nghĩa các Celery background task chạy sau mỗi lần dịch trang/chapter truyện.
"""

import logging
from app.celery_tasks.celery_config import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task điều phối chính (dùng cho mở rộng background task sau này)
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
    page_summaries: list[str] | None = None,
    character_updates: list | None = None,
    pronoun_shifts: list | None = None,
    manga_name: str = "",
):
    """
    Task điều phối chạy nền sau khi hoàn tất dịch 1 chapter.
    Giữ lại làm entrypoint cho các xử lý background bổ sung về sau.
    """
    logger.info(
        f"📬 [BG Task] process_chapter — manga={manga_id!r}, ch={chapter_number}"
    )
    # Placeholder cho các tác vụ xử lý nền tương lai


# Legacy alias
process_page_background_after_VLM_response = process_chapter_background_after_VLM_response

