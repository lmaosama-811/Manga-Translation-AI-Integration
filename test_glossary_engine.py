"""
Test Suite for Glossary Engine:
1. Prompt rendering (Manga & Novel)
2. Database Schema & GlossaryCRUD (async)
3. Aggregator logic & deduplication
"""

import asyncio
import os
import sys
import unittest

# Ensure BE is in PYTHONPATH
sys.path.insert(0, os.path.abspath("BE"))

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.schemas import GlossaryTerm
from app.services.db_service import GlossaryCRUD
from app.ai.base_model import BaseModel
from app.prompts.image_main_prompt import SYSTEM_PROMPT
from app.prompts.novel_prompt import _NOVEL_SYSTEM_PROMPT_TEMPLATE
from app.services.glossary_service import aggregate_and_save_glossary, load_glossary_for_manga


class TestGlossaryPromptRendering(unittest.TestCase):
    def test_manga_prompt_without_glossary(self):
        prompt = BaseModel.build_system_prompt(
            SYSTEM_PROMPT,
            genre=["Action"],
            manga_name="Hunter x Hunter",
            chapter_number=1,
            glossary_terms=None,
        )
        self.assertNotIn("{{GLOSSARY_BLOCK}}", prompt)
        self.assertNotIn("<glossary_terms>", prompt)
        self.assertIn("Hunter x Hunter", prompt)

    def test_manga_prompt_with_empty_glossary(self):
        prompt = BaseModel.build_system_prompt(
            SYSTEM_PROMPT,
            genre=["Action"],
            manga_name="Hunter x Hunter",
            chapter_number=1,
            glossary_terms=[],
        )
        self.assertNotIn("{{GLOSSARY_BLOCK}}", prompt)
        self.assertNotIn("<glossary_terms>", prompt)

    def test_manga_prompt_with_terms(self):
        terms = [
            ("Nen", "Niệm"),
            ("Domain Expansion", "Bành Trướng Lãnh Địa"),
            ("Bankai", "Vạn Khai, Giải phóng kiếm"),
        ]
        prompt = BaseModel.build_system_prompt(
            SYSTEM_PROMPT,
            genre=["Action", "Fantasy"],
            manga_name="Hunter x Hunter",
            chapter_number=350,
            glossary_terms=terms,
        )
        self.assertNotIn("{{GLOSSARY_BLOCK}}", prompt)
        self.assertIn("SERIES LORE & POWER-SYSTEM GLOSSARY", prompt)
        self.assertIn('- "Nen" → "Niệm"', prompt)
        self.assertIn('- "Domain Expansion" → "Bành Trướng Lãnh Địa"', prompt)
        self.assertIn('- "Bankai" → "Vạn Khai, Giải phóng kiếm"', prompt)
        self.assertIn("choose ONLY the single most contextually appropriate one", prompt)

    def test_novel_prompt_with_terms(self):
        terms = [("Qi Condensation", "Ngưng Khí Kỳ")]
        prompt = BaseModel.build_system_prompt(
            _NOVEL_SYSTEM_PROMPT_TEMPLATE,
            genre=["Cultivation"],
            manga_name="A Will Eternal",
            chapter_number=10,
            glossary_terms=terms,
        )
        self.assertNotIn("{{GLOSSARY_BLOCK}}", prompt)
        self.assertIn("SERIES LORE & POWER-SYSTEM GLOSSARY", prompt)
        self.assertIn('- "Qi Condensation" → "Ngưng Khí Kỳ"', prompt)


class TestGlossaryCRUDAndAggregator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create an isolated in-memory SQLite database
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        self.async_session = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_crud_save_and_get(self):
        async with self.async_session() as session:
            manga_id = "test_manga_1"

            # Initially empty
            terms = await GlossaryCRUD.get_by_manga_id(session, manga_id)
            self.assertEqual(len(terms), 0)

            # Save first batch
            new_terms = {
                "Nen": "Niệm",
                "Ten": "Triền",
            }
            added = await GlossaryCRUD.save_chapter_terms(session, manga_id, new_terms)
            self.assertEqual(added, 2)

            # Read back
            terms = await GlossaryCRUD.get_by_manga_id(session, manga_id)
            self.assertEqual(len(terms), 2)
            term_dict = {t.source_term: t.target_term for t in terms}
            self.assertEqual(term_dict["Nen"], "Niệm")
            self.assertEqual(term_dict["Ten"], "Triền")

            # Try saving duplicate terms — should NOT overwrite or duplicate
            dup_terms = {
                "Nen": "Ý niệm khác",  # already exists
                "Ren": "Luyện",        # new
            }
            added2 = await GlossaryCRUD.save_chapter_terms(session, manga_id, dup_terms)
            self.assertEqual(added2, 1)

            # Check final DB state
            terms_after = await GlossaryCRUD.get_by_manga_id(session, manga_id)
            self.assertEqual(len(terms_after), 3)
            term_dict_after = {t.source_term: t.target_term for t in terms_after}
            self.assertEqual(term_dict_after["Nen"], "Niệm")  # Unchanged!
            self.assertEqual(term_dict_after["Ren"], "Luyện")


class TestAggregatorLogic(unittest.IsolatedAsyncioTestCase):
    async def test_aggregation_and_deduplication(self):
        from unittest.mock import patch

        # Mock DB session in db_service
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        test_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        manga_id = "test_manga_agg"
        existing_glossary = [("Zetsu", "Tuyệt")]

        # Seed existing term in DB
        async with test_session() as session:
            await GlossaryCRUD.save_chapter_terms(session, manga_id, {"Zetsu": "Tuyệt"})

        # Simulated VLM results from 3 pages
        vlm_results = [
            # Page 0: Nen -> Niệm, and existing term Zetsu
            {
                "vlm_response": {
                    "translations": [],
                    "new_terms_discovered": [
                        {"source_term": "Nen", "target_term": "Niệm"},
                        {"source_term": "Zetsu", "target_term": "Tuyệt"},  # Should be skipped!
                    ],
                }
            },
            # Page 1: case variation "nen" with alternate target "ý niệm" + new term "Gyo"
            {
                "vlm_response": {
                    "translations": [],
                    "new_terms_discovered": [
                        {"source_term": "nen", "target_term": "ý niệm"},
                        {"source_term": "Gyo", "target_term": "Ngưng"},
                    ],
                }
            },
            # Page 2: Error page or no new terms
            {
                "error": "Failed page",
            },
            # Page 3: Duplicate of Gyo
            {
                "vlm_response": {
                    "translations": [],
                    "new_terms_discovered": [
                        {"source_term": "Gyo", "target_term": "Ngưng"},
                    ],
                }
            },
        ]

        with patch("app.services.glossary_service.AsyncSessionLocal", test_session):
            added = await aggregate_and_save_glossary(manga_id, vlm_results, existing_glossary)
            self.assertEqual(added, 2)  # Nen and Gyo (Zetsu was skipped)

            # Verify in DB
            async with test_session() as session:
                terms = await GlossaryCRUD.get_by_manga_id(session, manga_id)
                term_dict = {t.source_term: t.target_term for t in terms}

                # "Nen" should combine "Niệm" and "ý niệm"
                self.assertIn("Nen", term_dict)
                self.assertIn("Niệm", term_dict["Nen"])
                self.assertIn("ý niệm", term_dict["Nen"])

                # "Gyo" should be "Ngưng" without duplication
                self.assertEqual(term_dict["Gyo"], "Ngưng")

                # "Zetsu" stays "Tuyệt"
                self.assertEqual(term_dict["Zetsu"], "Tuyệt")

        # Test load_glossary_for_manga
        with patch("app.services.glossary_service.AsyncSessionLocal", test_session):
            loaded = await load_glossary_for_manga(manga_id)
            self.assertEqual(len(loaded), 3)
            loaded_dict = dict(loaded)
            self.assertIn("Nen", loaded_dict)
            self.assertIn("Gyo", loaded_dict)
            self.assertIn("Zetsu", loaded_dict)

        # Test empty VLM results
        with patch("app.services.glossary_service.AsyncSessionLocal", test_session):
            added_empty = await aggregate_and_save_glossary(manga_id, [], loaded)
            self.assertEqual(added_empty, 0)

            added_no_new = await aggregate_and_save_glossary(
                manga_id,
                [{"vlm_response": {"translations": []}}],
                loaded,
            )
            self.assertEqual(added_no_new, 0)

        await engine.dispose()


if __name__ == "__main__":
    unittest.main()
