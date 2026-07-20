"""
Module: app.ai.ollama
Description: Implements the local/remote Ollama VLM Model subclass of BaseModel.
             Also provides summarize_chapter() for chapter-level summary generation.
"""

import logging
import httpx
# pyrefly: ignore [missing-import]
from fastapi import HTTPException
from app.ai.base_model import BaseModel
from app.core.config import settings

logger = logging.getLogger(__name__)

AVAI_OLLAMA_MODELS = []
SUMMARIZER_MODEL = "qwen2.5:3b"   # Model local dùng để tổng hợp chapter summary

class OllamaModel(BaseModel):
    async def translate(self, image_base64: str, prompt: str) -> str:
        """
        Implementation of the translate method for local/remote Ollama API.
        """
        ollama_url = settings.OLLAMA_API_URL
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": "Please perform OCR, translate this manga page to Vietnamese, and classify each speech bubble background as 'clr' (1 for white, 2 for complex/screentone, 3 for black). Return strictly a JSON list complying with the system instructions.",
                    "images": [image_base64]
                }
            ],
            "stream": False,
            "format": "json"
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(ollama_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Lỗi khi kết nối với Ollama: {response.text}"
                )
                
            result_data = response.json()
            return result_data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Chapter Summary Generator
# ---------------------------------------------------------------------------

async def summarize_chapter(
    page_summaries: list[str],
    manga_id: str,
    chapter_number: int,
    chapter_name: str | None = None,
    session_factory=None,  # None → AsyncSessionLocal (FastAPI); make_task_session() cho Celery
) -> str:
    """
    Gọi model local qwen2.5:3b để tổng hợp danh sách page_summary thành
    1 chapter_summary hoàn chỉnh (tối đa 8 dòng), sau đó lưu vào PostgreSQL.

    Args:
        page_summaries:  Danh sách tóm tắt từng trang trong chapter.
        manga_id:        Slug/ID của bộ truyện (khóa tra cứu trong DB).
        chapter_number:  Số thứ tự chapter.
        chapter_name:    Tên chapter (optional, vd: "Chương 42: ...").

    Returns:
        Chuỗi chapter summary đã được lưu vào DB.

    Raises:
        RuntimeError nếu không gọi được Ollama hoặc không lưu được DB.
    """
    if not page_summaries:
        raise ValueError("page_summaries không được rỗng")

    # --- Xây dựng prompt ---
    numbered_pages = "\n".join(
        f"[Trang {i + 1}] {s.strip()}"
        for i, s in enumerate(page_summaries)
        if s.strip()
    )

    system_prompt = (
        "Bạn là trợ lý tóm tắt truyện tranh tiếng Việt. "
        "Nhiệm vụ: tổng hợp các tóm tắt trang thành 1 đoạn tóm tắt chapter mạch lạc, "
        "ngắn gọn, tối đa 8 dòng. "
        "Chỉ trả về đoạn tóm tắt, không có tiêu đề hay giải thích thêm."
    )

    user_prompt = (
        f"Dưới đây là tóm tắt từng trang của chapter {chapter_number}:\n\n"
        f"{numbered_pages}\n\n"
        "Hãy tổng hợp thành 1 đoạn tóm tắt chapter hoàn chỉnh (tối đa 8 dòng)."
    )

    # --- Gọi Ollama (text-only, không có ảnh) ---
    # Dùng base URL từ OLLAMA_API_URL, thay /api/chat nếu cần
    base_url = settings.OLLAMA_API_URL.rstrip("/")
    if not base_url.endswith("/api/chat"):
        ollama_url = base_url + "/api/chat"
    else:
        ollama_url = base_url

    payload = {
        "model": SUMMARIZER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
    }

    logger.info(
        f"[Summarizer] Gọi {SUMMARIZER_MODEL} — manga={manga_id!r}, ch={chapter_number}, "
        f"{len(page_summaries)} trang"
    )

    # Background task — không cần lo trễ FE/BE, tăng timeout để mộ hình local không bị cắt giữa chừng
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(ollama_url, json=payload, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(
            f"[Summarizer] Ollama trả lỗi {response.status_code}: {response.text}"
        )

    result_data = response.json()
    chapter_summary = result_data.get("message", {}).get("content", "").strip()

    if not chapter_summary:
        raise RuntimeError("[Summarizer] Model trả về chapter summary rỗng.")

    logger.info(
        f"[Summarizer] Tổng hợp xong — {len(chapter_summary)} ký tự. Đang lưu vào DB..."
    )

    # --- Lưu vào PostgreSQL ---
    from app.services.db_service import AsyncSessionLocal, ChapterSummaryCRUD
    factory = session_factory or AsyncSessionLocal
    async with factory() as session:
        try:
            await ChapterSummaryCRUD.upsert(
                session,
                manga_id=manga_id,
                chapter_number=chapter_number,
                summary=chapter_summary,
                name=chapter_name,
            )
            await session.commit()
            logger.info(
                f"[Summarizer] Đã lưu chapter summary vào DB: manga={manga_id!r}, ch={chapter_number}"
            )
        except Exception as db_err:
            await session.rollback()
            logger.error(f"[Summarizer] Lỗi khi lưu DB: {db_err}", exc_info=True)
            raise RuntimeError(f"[Summarizer] Lưu DB thất bại: {db_err}") from db_err

    return chapter_summary
