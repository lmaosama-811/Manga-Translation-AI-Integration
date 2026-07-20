"""
Module: app.schemas
Description: SQLModel table definitions cho dự án manga-image-translator.

Tables:
  - Manga           : Thông tin tổng quan của từng bộ truyện.
  - ChapterSummary  : Tóm tắt từng chapter (tổng hợp từ page_summary).
"""

from enum import Enum as PyEnum
from typing import Optional
# pyrefly: ignore [missing-import]
from sqlmodel import SQLModel, Field


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
    overall_summary: str | None     = None
    latest_chapter:  int | None     = None
    status:          MangaStatus    = MangaStatus.ONGOING
    character_graph: str | None     = None  # JSON string (nx.node_link_data)


# ---------------------------------------------------------------------------
# Table: chapter_summaries
# ---------------------------------------------------------------------------

class ChapterSummary(SQLModel, table=True):
    __tablename__ = "chapter_summaries"

    id:             int | None  = Field(default=None, primary_key=True)
    manga_id:       str         = Field(foreign_key="mangas.manga_id", index=True)
    name:           str | None  = None
    chapter_number: int
    summary:        str
