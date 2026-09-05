"""
Module: app.models.base_model
Description: Defines the abstract BaseModel class with a generic prompt renderer
             that accepts any template string.
"""

import logging
from abc import ABC, abstractmethod
from app.prompts.image_main_prompt import SYSTEM_PROMPT
from app.prompts.SKILLS import load_skill

logger = logging.getLogger(__name__)

class BaseModel(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @classmethod
    def build_system_prompt(
        cls,
        template: str = SYSTEM_PROMPT,   # Mặc định = manga prompt; novel/future truyền riêng
        genre: list[str] = [],
        manga_name: str = "",
        chapter_number: int = 0,
    ) -> str:
        """
        Generic prompt renderer — nhận bất kỳ template nào và inject context.

        Template phải chứa các placeholder:
          {{MANGA_TITLE}}              — tên bộ truyện
          {{CHAPTER_NUMBER}}           — số chapter hiện tại
          {{GENRE_SKILL}}              — skill text theo thể loại
        """
        genre_skill = "\n".join(load_skill(g) for g in genre) if genre else ""

        prompt = template
        prompt = prompt.replace("{{MANGA_TITLE}}",    manga_name or "Unknown")
        prompt = prompt.replace("{{CHAPTER_NUMBER}}", str(chapter_number) if chapter_number else "?")
        prompt = prompt.replace(
            "{{GENRE_SKILL}}",
            genre_skill or "Không có skill thể loại.",
        )
        return prompt

    @abstractmethod
    async def translate(self, image_base64: str, prompt: str) -> str:
        """
        Abstract method to call the underlying VLM API. Must be implemented by subclasses.
        """
        pass
