"""
Module: app.ai.gemini
Description: Implements the Google Gemini VLM Model subclass of BaseModel.
"""

import httpx
# pyrefly: ignore [missing-import]
from fastapi import HTTPException
from app.ai.base_model import BaseModel

AVAI_GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
# ------------------------------------------------------------------
# Rate limit configuration (per-model cooldown)
# Parallel slots = tự động theo số key: len(KeyManager().keys)
#
# Công thức: mỗi key = 1 request duy nhất mỗi (avg_latency + cooldown)s
# Không còn rpm_retry → mỗi session chỉ gọi API 1 lần per model
#
#   Model               latency   cooldown   cycle    RPM/key   limit    margin
#   flash-lite          3–6s      10s        13–16s   3.8–4.6   15 RPM   70%  ✅
#   gemini-3.5-flash    4–7s      10s        14–17s   3.5–4.3    5 RPM   10%  ✅
#   gemini-2.5-flash    5–10s     10s        15–20s   3.0–4.0   10 RPM   60%  ✅
#
# Tại sao không để 4s (ngưỡng tối thiểu cho 15 RPM)?
#   Latency dao động: trang ít text có thể latency < 1s.
#   cycle=1+10=11s → 5.5 RPM → vẫn an toàn, nhưng 10s cho buffer tốt hơn.
# ------------------------------------------------------------------

GEMINI_REQUESTS_PARALLEL_LIMITS: dict[str, dict] = {
    "gemini-3.1-flash-lite": {"cooldown_seconds": 5},
    "gemini-3.5-flash":      {"cooldown_seconds": 10},
    "gemini-2.5-flash":      {"cooldown_seconds": 10},
}

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "bubble_id": {
                        "type": "INTEGER",
                        "description": (
                            "Số thứ tự màu đỏ in ở góc trên bên trái của bong bóng thoại "
                            "(bắt đầu từ 1). Đọc chính xác con số nhìn thấy trên ảnh."
                        )
                    },
                    "text_vi": {
                        "type": "STRING",
                        "description": "Bản dịch tiếng Việt tự nhiên của bong bóng thoại."
                    },
                    "clr": {
                        "type": "INTEGER",
                        "description": "Clean action: 1 (white bubble), 2 (screentone/complex), 3 (black bubble)."
                    }
                },
                "required": ["bubble_id", "text_vi", "clr"]
            }
        },

        "has_dialogue": {
            "type": "BOOLEAN",
            "description": "True if there is dialogue on the page, False otherwise."
        }
    },
    "required": [
        "translations",
        "has_dialogue"
    ]
}

class GeminiModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.api_key = api_key

    async def translate(self, image_base64: str, prompt: str) -> str:
        """
        Implementation of the translate method for Google Gemini API.
        """
        if not self.api_key:
            raise HTTPException(
                status_code=400, 
                detail="Chưa cấu hình biến môi trường GEMINI_API_KEY để sử dụng Google Gemini."
            )
            
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        gemini_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Please perform OCR, translate this manga page to Vietnamese, and classify each speech bubble background as 'clr' (1 for white, 2 for complex/screentone, 3 for black). Return strictly a JSON list complying with the system instructions."
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_RESPONSE_SCHEMA
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(gemini_url, json=gemini_payload, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Lỗi khi kết nối với Gemini API: {response.text}"
                )
                
            result_data = response.json()
            
            try:
                return result_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError) as key_err:
                raise HTTPException(
                    status_code=502,
                    detail=f"Cấu trúc phản hồi từ Gemini không đúng hoặc bị thiếu trường: {key_err}"
                )

    async def translate_text(self, text: str, prompt: str, response_schema: dict) -> str:
        """
        Text-only Gemini API call — dùng cho novel translation (không có ảnh).

        Cùng pattern với translate() nhưng phần 'parts' chỉ chứa text, không có inlineData.
        response_schema được truyền từ ngoài vào để linh hoạt (novel vs manga có schema khác nhau).

        Raises:
            HTTPException: khi API key thiếu hoặc Gemini trả về lỗi HTTP.
        """
        if not self.api_key:
            raise HTTPException(
                status_code=400,
                detail="Chưa cấu hình GEMINI_API_KEY.",
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}],
                }
            ],
            "systemInstruction": {"parts": [{"text": prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }

        async with httpx.AsyncClient(timeout=180.0) as client:  # novel chapters có thể dài hơn manga
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Gemini API lỗi (text-only): {response.text[:300]}",
                )

            result_data = response.json()

        try:
            return result_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini response format lỗi (text-only): {e}",
            )

