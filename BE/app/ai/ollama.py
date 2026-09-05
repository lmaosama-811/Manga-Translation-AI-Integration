"""
Module: app.ai.ollama
Description: Implements the local/remote Ollama VLM Model subclass of BaseModel.
"""

import logging
import httpx
# pyrefly: ignore [missing-import]
from fastapi import HTTPException
from app.ai.base_model import BaseModel
from app.core.config import settings

logger = logging.getLogger(__name__)

AVAI_OLLAMA_MODELS = []


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

