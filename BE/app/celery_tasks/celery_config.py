"""
Module: app.celery_tasks.celery_config
Description: Khai báo Celery app instance và cấu hình Beat schedule.

Gemini daily quota reset theo giờ Việt Nam (UTC+7):
  - Tháng 4–10 (PDT):  14:00 VN = 07:00 UTC
  - Tháng 11–3 (PST):  15:00 VN = 08:00 UTC

Beat schedule dùng crontab UTC để đảm bảo tính chính xác.
Tháng được chia làm 2 nhóm, mỗi nhóm một entry riêng.
"""

# pyrefly: ignore [missing-import]
from celery import Celery
# pyrefly: ignore [missing-import]
from celery.schedules import crontab
from app.core.config import settings

# ---------------------------------------------------------------------------
# Celery App
# ---------------------------------------------------------------------------

celery_app = Celery(
    "manga_translator",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.celery_tasks.periodic_task",
        "app.celery_tasks.background_task",
    ],
)

# ---------------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Timezone
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Task behavior
    task_acks_late=True,           # Chỉ ack sau khi task hoàn thành (tránh mất task khi worker crash)
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # Không prefetch nhiều task (tránh bottleneck)

    # Result TTL
    result_expires=3600,           # Kết quả task hết hạn sau 1h
)

# ---------------------------------------------------------------------------
# Beat Schedule (Periodic Tasks)
# ---------------------------------------------------------------------------
# Gemini quota reset vào:
#   - 07:00 UTC (= 14:00 VN) cho tháng 4–10 (UTC+7 = PDT offset)
#   - 08:00 UTC (= 15:00 VN) cho tháng 11–3
#
# Dùng 2 entry crontab riêng để xử lý 2 mùa trong năm.
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    # Tháng 4–10: reset lúc 14:00 VN (= 07:00 UTC)
    "reset-gemini-keys-summer": {
        "task": "app.celery_tasks.periodic_task.reset_gemini_daily_quota",
        "schedule": crontab(hour=23, minute=3, month_of_year="4-10"),
        "options": {"expires": 300},  # Task tự hủy nếu không được thực thi trong 5 phút
    },
    # Tháng 11–3: reset lúc 15:00 VN (= 08:00 UTC)
    "reset-gemini-keys-winter": {
        "task": "app.celery_tasks.periodic_task.reset_gemini_daily_quota",
        "schedule": crontab(hour=15, minute=0, month_of_year="11,12,1,2,3"),
        "options": {"expires": 300},
    },
}
