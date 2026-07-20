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
from app.schemas import Manga, ChapterSummary, MangaStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine & Session Factory
# ---------------------------------------------------------------------------

# Engine dùng cho FastAPI process (gắn với event loop của FastAPI)
_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"ssl": False},
)

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
        overall_summary: Optional[str] = None,
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
            overall_summary=overall_summary,
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
    async def update_summary(
        session: AsyncSession,
        manga_id: str,
        overall_summary: str,
    ) -> bool:
        """Cập nhật overall_summary."""
        result = await session.execute(
            update(Manga)
            .where(Manga.manga_id == manga_id)
            .values(overall_summary=overall_summary)
        )
        await session.commit()
        return result.rowcount > 0

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
        """Xóa manga (cascade xóa cả chapter_summaries liên quan)."""
        manga = await MangaCRUD.get_by_manga_id(session, manga_id)
        if not manga:
            return False
        await session.delete(manga)
        await session.commit()
        logger.info(f"[MangaCRUD] Deleted: {manga_id!r}")
        return True

    @staticmethod
    async def update_character_graph(
        session: AsyncSession,
        manga_id: str,
        graph_json: str,
    ) -> bool:
        """Ghi đè character_graph (JSON string) vào bảng mangas."""
        result = await session.execute(
            update(Manga)
            .where(Manga.manga_id == manga_id)
            .values(character_graph=graph_json)
        )
        await session.commit()
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# CRUD: ChapterSummary
# ---------------------------------------------------------------------------

class ChapterSummaryCRUD:

    @staticmethod
    async def upsert(
        session: AsyncSession,
        *,
        manga_id: str,
        chapter_number: int,
        summary: str,
        name: Optional[str] = None,
        manga_name: Optional[str] = None,
    ) -> ChapterSummary:
        """
        Tạo mới hoặc cập nhật chapter summary.
        Tự động tạo manga record nếu chưa tồn tại (tránh ForeignKeyViolationError).
        Tự động cập nhật latest_chapter trên bảng manga sau khi lưu.
        """
        # effective_name: ưu tiên tên truyện, fallback về chapter_name, cuối cùng mới "Chapter N"
        # Trước đây chỉ có `name or f"Chapter {chapter_number}"` → lưu sai "Chapter 300"
        effective_name = name or manga_name or f"Chapter {chapter_number}"

        # Đảm bảo manga tồn tại trước khi insert chapter_summary (FK constraint)
        manga = await MangaCRUD.get_by_manga_id(session, manga_id)
        if not manga:
            manga = Manga(
                manga_id=manga_id,
                name=manga_name or manga_id,  # Dùng manga_id làm tên tạm nếu chưa có
                status=MangaStatus.ONGOING,
            )
            session.add(manga)
            await session.flush()  # Flush để có row trong DB trước khi insert FK
            logger.info(f"[ChapterCRUD] Auto-created manga record: {manga_id!r}")

        existing = await ChapterSummaryCRUD.get(session, manga_id, chapter_number)

        if existing:
            existing.summary = summary
            existing.name = effective_name  # luôn cập nhật với effective_name
            session.add(existing)
            await session.commit()
            await session.refresh(existing)
            logger.info(f"[ChapterCRUD] Updated ch={chapter_number} manga={manga_id!r}")
            return existing
        else:
            chapter = ChapterSummary(
                manga_id=manga_id,
                chapter_number=chapter_number,
                summary=summary,
                name=effective_name,  # không bao giờ NULL
            )
            session.add(chapter)
            await session.commit()
            await session.refresh(chapter)
            logger.info(f"[ChapterCRUD] Created ch={chapter_number} manga={manga_id!r}")

            # Cập nhật latest_chapter trên bảng manga
            await MangaCRUD.update_latest_chapter(session, manga_id, chapter_number)
            return chapter

    @staticmethod
    async def get(
        session: AsyncSession,
        manga_id: str,
        chapter_number: int,
    ) -> Optional[ChapterSummary]:
        """Lấy summary của 1 chapter cụ thể."""
        result = await session.execute(
            select(ChapterSummary).where(
                ChapterSummary.manga_id == manga_id,
                ChapterSummary.chapter_number == chapter_number,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_for_manga(
        session: AsyncSession,
        manga_id: str,
    ) -> list[ChapterSummary]:
        """Lấy tất cả chapter summary của một manga, theo thứ tự chapter."""
        result = await session.execute(
            select(ChapterSummary)
            .where(ChapterSummary.manga_id == manga_id)
            .order_by(ChapterSummary.chapter_number)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_recent(
        session: AsyncSession,
        manga_id: str,
        limit: int = 5,
    ) -> list[ChapterSummary]:
        """Lấy N chapter summary mới nhất (dùng cho context window)."""
        result = await session.execute(
            select(ChapterSummary)
            .where(ChapterSummary.manga_id == manga_id)
            .order_by(ChapterSummary.chapter_number.desc())
            .limit(limit)
        )
        # Trả về theo thứ tự cũ → mới
        return list(reversed(result.scalars().all()))

    @staticmethod
    async def delete(
        session: AsyncSession,
        manga_id: str,
        chapter_number: int,
    ) -> bool:
        """Xóa summary của 1 chapter."""
        chapter = await ChapterSummaryCRUD.get(session, manga_id, chapter_number)
        if not chapter:
            return False
        await session.delete(chapter)
        await session.commit()
        logger.info(f"[ChapterCRUD] Deleted ch={chapter_number} manga={manga_id!r}")
        return True
