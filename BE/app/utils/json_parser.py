"""
Module: app.utils.json_parser
Description: Handles extraction, sanitization, and structured parsing of VLM JSON responses.
             Supports the nested schema: {translations, has_dialogue, page_summary, character_updates, pronoun_shift}
"""

import re
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_block(text: str) -> str:
    """
    Extracts the first valid JSON object or array from the text by balancing brackets/braces.
    Handles string literals to avoid counting braces inside strings.
    """
    start_match = re.search(r'[\{\[]', text)
    if not start_match:
        return text
    
    start_idx = start_match.start()
    char_open = text[start_idx]
    char_close = '}' if char_open == '{' else ']'
    
    in_string = False
    escape = False
    stack = []
    
    for idx in range(start_idx, len(text)):
        char = text[idx]
        
        if escape:
            escape = False
            continue
        
        if char == '\\':
            escape = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == char_open:
                stack.append(char)
            elif char == char_close:
                if stack:
                    stack.pop()
                if not stack:
                    return text[start_idx:idx+1]
                    
    end_idx = text.rfind(char_close)
    if end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx+1]
        
    return text[start_idx:]


def parse_vlm_json(response_content: str) -> dict[str, Any]:
    """
    Parses raw VLM string response into a structured dict with fixed schema.
    
    Expected VLM output:
    {
        "translations": [...],
        "has_dialogue": true/false,
        "page_summary": "...",
        "character_updates": [...],
        "pronoun_shift": [...]
    }
    
    Returns a normalized dict. Missing fields are filled with safe defaults.
    If parsing fails entirely, returns a dict with empty translations and has_dialogue=False.
    """
    json_str = response_content.strip()
    
    # Strip markdown block formatting if present
    if "```" in json_str:
        matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
        if matches:
            json_str = matches[0].strip()
            
    # Extract clean JSON block by balancing braces/brackets (to handle extra trailing characters, etc.)
    json_str = extract_json_block(json_str)
            
    try:
        data = json.loads(json_str)
    except Exception as err:
        logger.error(f"Error parsing VLM response JSON: {err}")
        return _empty_response()
    
    # Handle the expected nested dict format
    if isinstance(data, dict):
        return _normalize_response(data)
    
    # Legacy fallback: flat array of translation boxes (old prompt format)
    if isinstance(data, list):
        logger.warning("VLM returned legacy flat array format. Wrapping into new schema.")
        return {
            "translations": data,
            "has_dialogue": len(data) > 0,
            "page_summary": "",
            "character_updates": [],
            "pronoun_shift": [],
        }
    
    logger.error(f"Unexpected VLM response type: {type(data)}")
    return _empty_response()


def _normalize_response(data: dict) -> dict[str, Any]:
    """
    Normalizes a parsed dict to ensure all expected fields exist with correct types.
    """
    translations = data.get("translations", [])
    if not isinstance(translations, list):
        translations = []
    
    has_dialogue = data.get("has_dialogue", len(translations) > 0)
    if not isinstance(has_dialogue, bool):
        has_dialogue = bool(has_dialogue)
    
    page_summary = data.get("page_summary", "")
    if not isinstance(page_summary, str):
        page_summary = str(page_summary) if page_summary else ""
    
    character_updates = data.get("character_updates", [])
    if not isinstance(character_updates, list):
        character_updates = []
    
    pronoun_shift = data.get("pronoun_shift", [])
    if not isinstance(pronoun_shift, list):
        pronoun_shift = []
    
    return {
        "translations": translations,
        "has_dialogue": has_dialogue,
        "page_summary": page_summary,
        "character_updates": character_updates,
        "pronoun_shift": pronoun_shift,
    }


def _empty_response() -> dict[str, Any]:
    """Returns a safe empty response when parsing fails."""
    return {
        "translations": [],
        "has_dialogue": False,
        "page_summary": "",
        "character_updates": [],
        "pronoun_shift": [],
    }
