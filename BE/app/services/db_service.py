"""
Module: app.services.db_service
Description: Database service — quản lý kết nối PostgreSQL (async) và cung cấp
             các hàm CRUD cho bảng mangas và chapter_summaries.

Dùng SQLModel + SQLAlchemy 2.0 async engine.
DATABASE_URL được đọc từ settings.DATABASE_URL (.env).
"""

import logging
from typing import Optional

# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
# pyrefly: ignore [missing-import]
from sqlalchemy import select, update
# pyrefly: ignore [missing-import]
from sqlmodel import SQLModel

from app.core.config import settings
from app.schemas import Manga, MangaStatus, GlossaryTerm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine & Session Factory
# ---------------------------------------------------------------------------

_db_url = settings.DATABASE_URL or "sqlite+aiosqlite:///:memory:"
_engine_kwargs: dict = {
    "echo": False,
}
if "postgresql" in _db_url:
    _engine_kwargs.update({
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
        "connect_args": {"ssl": False},
    })

# Engine dùng cho FastAPI process (gắn với event loop của FastAPI)
_engine = create_async_engine(_db_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


def make_task_session():
    """
    Tạo session factory gắn với engine MỚI — dùng cho Celery tasks.

    WHY: Celery thread gọi asyncio.run() → tạo event loop riêng trong thread.
    Engine toàn cục (_engine) đã gắn với event loop FastAPI process → conflict
    → 'NoneType' object has no attribute 'send'.

    Pattern sử dụng trong Celery task:
        async def _run():
            async with make_task_session()() as session:
                await SomeCRUD.method(session, ...)
            # Engine tự dispose khi hàm trả về (nếu gọi dispose sau)
        asyncio.run(_run())
    """
    task_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,
        connect_args={"ssl": False},
    )
    return async_sessionmaker(
        bind=task_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def init_db() -> None:
    """
    Tạo tất cả bảng nếu chưa tồn tại.
    Gọi một lần khi khởi động ứng dụng (FastAPI lifespan).
    """
    async with _engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables initialized.")


# ---------------------------------------------------------------------------
# CRUD: Manga
# ---------------------------------------------------------------------------

class MangaCRUD:

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        name: str,
        manga_id: str | None = None,
        status: MangaStatus = MangaStatus.ONGOING,
        latest_chapter: Optional[int] = None,
    ) -> Manga:
        """
        Tạo bản ghi manga mới.
        manga_id  : Tự sinh uuid4().hex nếu không truyền.
        name_slug : Tự tính từ name (normalize lowercase, bỏ ký tự đặc biệt).
        """
        import uuid as _uuid
        # Import lazy để tránh circular dependency
        from app.api.routes.scrape import normalize_manga_title

        manga = Manga(
            name=name,
            name_slug=normalize_manga_title(name),
            manga_id=manga_id or _uuid.uuid4().hex,
            status=status,
            latest_chapter=latest_chapter,
        )
        session.add(manga)
        await session.flush()   # flush để có ID ngay, caller tự commit
        logger.info(f"[MangaCRUD] Created: {manga.manga_id!r} — name={name!r} slug={manga.name_slug!r}")
        return manga

    @staticmethod
    async def get_by_manga_id(session: AsyncSession, manga_id: str) -> Optional[Manga]:
        """Lấy manga theo manga_id (UUID hex)."""
        result = await session.execute(
            select(Manga).where(Manga.manga_id == manga_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_name_slug(session: AsyncSession, name_slug: str) -> Optional[Manga]:
        """
        Tìm manga theo name_slug (exact match trên cột đã index).
        Nhanh hơn ILIKE, dùng sau khi đã có cột name_slug trong DB.
        Fallback về ILIKE nếu name_slug chưa có trên các row cũ.
        """
        # 1. Exact match trên name_slug (indexed, nhanh)
        result = await session.execute(
            select(Manga).where(Manga.name_slug == name_slug).limit(1)
        )
        found = result.scalar_one_or_none()
        if found:
            return found

        # 2. Fallback ILIKE cho row cũ chưa có name_slug
        result = await session.execute(
            select(Manga).where(Manga.name.ilike(f"%{name_slug}%")).limit(1)
        )
        return result.scalar_one_or_none()

    # Alias cho code cũ
    get_by_name_fuzzy = get_by_name_slug

    @staticmethod
    async def get_all(session: AsyncSession) -> list[Manga]:
        """Lấy toàn bộ danh sách manga."""
        result = await session.execute(select(Manga).order_by(Manga.name))
        return list(result.scalars().all())

    @staticmethod
    async def update_latest_chapter(
        session: AsyncSession,
        manga_id: str,
        chapter_number: int,
    ) -> bool:
        """Cập nhật latest_chapter nếu chapter_number mới hơn hiện tại."""
        result = await session.execute(
            update(Manga)
            .where(Manga.manga_id == manga_id)
            .where(
                (Manga.latest_chapter == None) |  # noqa: E711
                (Manga.latest_chapter < chapter_number)
            )
            .values(latest_chapter=chapter_number)
        )
        await session.commit()
        return result.rowcount > 0

    @staticmethod
    async def update_status(
        session: AsyncSession,
        manga_id: str,
        status: MangaStatus,
    ) -> bool:
        """Cập nhật trạng thái phát hành."""
        result = await session.execute(
            update(Manga)
            .where(Manga.manga_id == manga_id)
            .values(status=status)
        )
        await session.commit()
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, manga_id: str) -> bool:
        """Xóa manga."""
        manga = await MangaCRUD.get_by_manga_id(session, manga_id)
        if not manga:
            return False
        await session.delete(manga)
        await session.commit()
        logger.info(f"[MangaCRUD] Deleted: {manga_id!r}")
        return True


# ---------------------------------------------------------------------------
# CRUD: Glossary
# ---------------------------------------------------------------------------

class GlossaryCRUD:

    @staticmethod
    async def get_by_manga_id(
        session: AsyncSession,
        manga_id: str,
    ) -> list[GlossaryTerm]:
        """
        Lấy toàn bộ thuật ngữ glossary của một bộ truyện.
        Dùng để nạp vào System Prompt trước khi gọi VLM.
        """
        result = await session.execute(
            select(GlossaryTerm)
            .where(GlossaryTerm.manga_id == manga_id)
            .order_by(GlossaryTerm.source_term)
        )
        return list(result.scalars().all())

    @staticmethod
    async def save_chapter_terms(
        session: AsyncSession,
        manga_id: str,
        new_terms: dict[str, str],
    ) -> int:
        """
        Lưu các thuật ngữ mới phát hiện sau khi dịch xong 1 chapter.

        new_terms: dict mapping {source_term: target_term}
          - source_term đã được chuẩn hóa (strip/lower) bởi Aggregator.
          - target_term có thể là chuỗi ghép (ví dụ "Niệm, ý niệm")
            nếu Gemini đề xuất nhiều phương án dịch.

        Nếu source_term đã tồn tại trong DB cho manga_id này → bỏ qua.
        (Tuân thủ nguyên tắc: Database là Chân lý tuyệt đối).

        Returns: Số thuật ngữ mới thực sự được thêm.
        """
        added = 0
        for source, target in new_terms.items():
            # Kiểm tra xem từ đã tồn tại chưa
            existing = await session.execute(
                select(GlossaryTerm).where(
                    GlossaryTerm.manga_id == manga_id,
                    GlossaryTerm.source_term == source,
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue  # Đã có trong DB → bỏ qua, không ghi đè

            term = GlossaryTerm(
                manga_id=manga_id,
                source_term=source,
                target_term=target,
            )
            session.add(term)
            added += 1

        if added > 0:
            await session.commit()
            logger.info(
                f"[GlossaryCRUD] Saved {added} new term(s) for manga={manga_id!r}"
            )
        return added


