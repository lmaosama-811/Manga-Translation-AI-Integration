"""
Package: app.ai
Description: Provides factories to build concrete VLM instances and key/model pool helpers.
"""

from app.ai.base_model import BaseModel
from app.ai.gemini import GeminiModel, AVAI_GEMINI_MODELS
from app.ai.ollama import OllamaModel, AVAI_OLLAMA_MODELS


def get_vlm_model(model_name: str, api_key: str = "") -> BaseModel:
    """
    Factory: returns a concrete VLM instance based on the model name string.

    Routing rules (evaluated in order):
      "gemini-*"                    → GeminiModel (requires api_key)
      anything else                 → OllamaModel (local)
    """
    name_clean = model_name.strip()
    name_lower = name_clean.lower()

    if name_lower.startswith("gemini"):
        return GeminiModel(name_clean, api_key)
    else:
        return OllamaModel(name_clean)




def get_fallback_model_names(primary_model_name: str) -> list[str]:
    """
    Returns an ordered list of fallback Gemini model names, excluding the primary model.
    Used by the FallbackEngine to build the model cascade.
    """
    all_models = AVAI_GEMINI_MODELS + AVAI_OLLAMA_MODELS

    primary_lower = primary_model_name.strip().lower()
    return [m for m in all_models if m.lower() != primary_lower]
