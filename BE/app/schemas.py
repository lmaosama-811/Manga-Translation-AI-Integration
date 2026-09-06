"""
Module: app.schemas
Description: SQLModel table definitions cho dự án manga-image-translator.

Tables:
  - Manga           : Thông tin tổng quan của từng bộ truyện.
  - GlossaryTerm    : Thuật ngữ chuyên ngành / thế giới quan riêng của từng bộ truyện.
"""

from enum import Enum as PyEnum
from typing import Optional
# pyrefly: ignore [missing-import]
from sqlmodel import SQLModel, Field
# pyrefly: ignore [missing-import]
from sqlalchemy import UniqueConstraint


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MangaStatus(str, PyEnum):
    ONGOING   = "ONGOING"
    COMPLETED = "COMPLETED"
    DROP      = "DROP"


# ---------------------------------------------------------------------------
# Table: mangas
# ---------------------------------------------------------------------------

class Manga(SQLModel, table=True):
    __tablename__ = "mangas"

    id:              int | None     = Field(default=None, primary_key=True)
    name:            str
    name_slug:       str | None     = Field(default=None, index=True)   # chuẩn hóa để lookup
    manga_id:        str            = Field(unique=True, index=True)
    latest_chapter:  int | None     = None
    status:          MangaStatus    = MangaStatus.ONGOING


# ---------------------------------------------------------------------------
# Table: glossary_terms
# ---------------------------------------------------------------------------

class GlossaryTerm(SQLModel, table=True):
    """
    Thuật ngữ chuyên ngành / hệ thống sức mạnh / thế giới quan hư cấu
    riêng của từng bộ truyện, dùng để duy trì cách dịch nhất quán
    xuyên suốt các chapter.

    Ví dụ:
      manga_id="al_136", source_term="Nen", target_term="Niệm"
    """
    __tablename__ = "glossary_terms"
    __table_args__ = (
        UniqueConstraint("manga_id", "source_term", name="uq_glossary_manga_source"),
    )

    id:          int | None = Field(default=None, primary_key=True)
    manga_id:    str        = Field(index=True)          # Canonical ID từ anilist (ví dụ: al_136)
    source_term: str        = Field(index=True)           # Từ gốc (ví dụ: "Nen", "Domain Expansion")
    target_term: str                                      # Bản dịch (ví dụ: "Niệm", "Bành Trướng Lãnh Địa")

