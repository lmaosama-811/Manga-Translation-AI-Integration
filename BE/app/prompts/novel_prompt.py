"""
Module: app.prompts.novel_prompt
Description: System prompt và response schema cho dịch novel (web novel / light novel).
             Đây là text-only pipeline — không có ảnh, không có OCR, không có bounding box.

Điểm khác biệt so với manga prompt (image_main_prompt.py):
  - Input: plain text (đoạn văn) thay vì ảnh
  - Output: translated_text (string dài) thay vì translations[] (array bbox)
  - Vẫn extract character_updates + pronoun_shifts để background task xử lý như manga
  - Prompt rendering dùng chung BaseModel.build_system_prompt()
"""

# ---------------------------------------------------------------------------
# Response schema — Gemini sẽ trả đúng cấu trúc này nhờ responseSchema
# character_updates và pronoun_shifts giữ nguyên format của manga để
# background tasks (Celery) tái sử dụng mà không cần thay đổi.
# ---------------------------------------------------------------------------

NOVEL_RESPONSE_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "translated_text": {
            "type": "STRING",
            "description": (
                "Toàn bộ nội dung chapter đã được dịch sang tiếng Việt. "
                "Giữ nguyên cấu trúc đoạn văn (dùng \\n\\n để ngăn cách paragraphs). "
                "KHÔNG rút gọn, KHÔNG bỏ sót câu nào."
            ),
        },
        "chapter_summary": {
            "type": "STRING",
            "description": "Tóm tắt 2–4 câu những gì xảy ra trong chapter này (dành cho context chapter tiếp theo).",
        },
        "character_updates": {
            "type": "ARRAY",
            "description": "Danh sách nhân vật xuất hiện/được nhắc đến trong chapter này kèm thông tin mới.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name":       {"type": "STRING"},
                    "gender":     {"type": "STRING", "enum": ["male", "female", "unknown"]},
                    "age_range":  {"type": "STRING", "enum": ["child", "teen", "young_adult", "adult", "elder", "?"]},
                    "speaks_to": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "target":          {"type": "STRING"},
                                "caller_pronoun":  {"type": "STRING"},
                                "target_pronoun":  {"type": "STRING"},
                            },
                            "required": ["target", "caller_pronoun", "target_pronoun"],
                        },
                    },
                    "notes": {"type": "STRING"},
                },
                "required": ["name", "gender", "age_range", "speaks_to"],
            },
        },
        "pronoun_shifts": {
            "type": "ARRAY",
            "description": "Ghi lại khi nhân vật thay đổi cách xưng hô (do cảm xúc, sự kiện cốt truyện, v.v.).",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "character":        {"type": "STRING"},
                    "target":           {"type": "STRING"},
                    "previous_pronoun": {"type": "STRING"},
                    "new_pronoun":      {"type": "STRING"},
                    "reason":           {"type": "STRING"},
                },
                "required": ["character", "target", "previous_pronoun", "new_pronoun"],
            },
        },
    },
    "required": ["translated_text", "chapter_summary", "character_updates", "pronoun_shifts"],
}


# ---------------------------------------------------------------------------
# System prompt template
# Placeholder tokens: {{MANGA_TITLE}}, {{CHAPTER_NUMBER}},
#                     {{CHARACTER_GRAPH}}, {{PREVIOUS_CHAPTER_SUMMARY}}, {{GENRE_SKILL}}
# ---------------------------------------------------------------------------

_NOVEL_SYSTEM_PROMPT_TEMPLATE = """
You are an elite web novel / light novel translator specialized in translating Asian novels
(Chinese, Japanese, Korean) to Vietnamese with natural, culturally-appropriate output.

═══════════════════════════════════════════════
CONTEXT BLOCK (đọc trước — ảnh hưởng đến toàn bộ bản dịch)
═══════════════════════════════════════════════
<series_info>
Novel title : {{MANGA_TITLE}}
Chapter     : {{CHAPTER_NUMBER}}
</series_info>

<character_graph>
{{CHARACTER_GRAPH}}
</character_graph>

HOW TO USE character_graph:
  Format: {"CharName": {"g": gender, "a": age_range, "p": {target: [caller_pronoun, target_pronoun]}}}
  - Dùng để chọn đúng đại từ nhân xưng cho từng nhân vật (tôi/ta/mày/cậu/ngươi/...)
  - Nếu nhân vật chưa có trong graph → suy luận từ ngữ cảnh, ghi lại vào character_updates

<previous_chapter_summary>
{{PREVIOUS_CHAPTER_SUMMARY}}
</previous_chapter_summary>

<genre_skills>
{{GENRE_SKILL}}
</genre_skills>

═══════════════════════════════════════════════
NHIỆM VỤ
═══════════════════════════════════════════════
Dịch TOÀN BỘ nội dung novel được cung cấp sang tiếng Việt.

QUY TẮC DỊCH:
1. KHÔNG bỏ sót câu nào — dịch hết 100% nội dung
2. Giữ nguyên cấu trúc đoạn văn: 1 paragraph gốc = 1 paragraph dịch
3. Dùng đại từ nhân xưng từ character_graph; nếu không có → suy luận tự nhiên từ ngữ cảnh
4. Giữ tên riêng (nhân vật, địa danh, kỹ năng) theo nguyên tác — phiên âm nếu cần
5. Áp dụng phong cách từ genre_skills (ví dụ: văn phong hào hùng cho action, lãng mạn cho romance)
6. Hội thoại phải tự nhiên như người Việt Nam thực sự nói — không dịch cứng
7. Thuật ngữ tu luyện / hệ thống / kỹ năng: dịch hoặc giữ nguyên tùy ngữ cảnh, nhất quán

SAU KHI DỊCH, PHÂN TÍCH:
- chapter_summary: tóm tắt ngắn gọn để làm context cho chapter tiếp theo
- character_updates: chỉ ghi những nhân vật XUẤT HIỆN hoặc được NHẮC ĐẾN trong chapter này,
  và chỉ ghi thông tin MỚI hoặc CÓ THỂ XÁC NHẬN — không suy đoán
- pronoun_shifts: ghi lại nếu có sự thay đổi cách xưng hô đột ngột (do cảm xúc, sự kiện)

═══════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON
═══════════════════════════════════════════════
Output ONLY raw JSON. No markdown. No code fences. No explanation.

{
  "translated_text": "Toàn bộ nội dung đã dịch, paragraphs ngăn cách bằng \\n\\n",
  "chapter_summary": "2-4 câu tóm tắt chapter",
  "character_updates": [...],
  "pronoun_shifts": []
}
"""