"""
Module: app.schemas
Description: SQLModel table definitions cho dự án manga-image-translator.

Tables:
  - Manga           : Thông tin tổng quan của từng bộ truyện.
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
    latest_chapter:  int | None     = None
    status:          MangaStatus    = MangaStatus.ONGOING

