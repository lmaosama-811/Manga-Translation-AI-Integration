"""
Module: app.celery_tasks.periodic_task
Description: Định nghĩa Celery periodic task chạy mỗi ngày vào giờ reset quota của Gemini.

Task này gọi KeyManager.reset_daily_tracking() để:
  - Xóa toàn bộ exhausted_models trên mọi Key
  - Xóa reset_day trên mọi Key
  - Lưu trạng thái mới xuống keys_state.json
"""

import logging
from app.celery_tasks.celery_config import celery_app
from app.ai.key_manager import KeyManager

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.celery_tasks.periodic_task.reset_gemini_daily_quota",
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # Retry sau 60 giây nếu thất bại
)
def reset_gemini_daily_quota(self):
    """
    Reset toàn bộ quota tracking cho tất cả API key Gemini.
    Được gọi tự động bởi Celery Beat vào 14:00 VN (tháng 4–10)
    hoặc 15:00 VN (tháng 11–3).
    """
    logger.info("⏰ [Celery Beat] Bắt đầu reset Gemini daily quota tracking...")
    try:
        key_manager = KeyManager()
        key_manager.reset_daily_tracking()
        logger.info(
            f"✅ [Celery Beat] Reset thành công {len(key_manager.keys)} API key(s). "
            "Tất cả key và model đã được khôi phục quota."
        )
    except Exception as exc:
        logger.error(f"❌ [Celery Beat] Lỗi khi reset quota: {exc}", exc_info=True)
        raise self.retry(exc=exc)
