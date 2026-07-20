"""
Module: app.services.graph_service
Description: Quản lý character relationship graph bằng NetworkX DiGraph.

Architecture:
  - Storage   : PostgreSQL (cột character_graph TEXT trong bảng mangas)
  - In-memory : Cache dict đơn giản cho API process (đọc nhanh khi inject prompt)
  - Serialize : nx.node_link_data() ↔ JSON string
  - Celery    : Luôn read→modify→write trực tiếp từ DB (không dùng cache)

Node attributes:
  - name   : str  — tên nhân vật
  - gender : str  — "M" | "F" | "?"
  - age    : str  — "child" | "teen" | "young_adult" | "adult" | "elder" | "?"

Edge attributes (A → B: A gọi B như thế nào):
  - caller  : str  — đại từ A dùng khi nói về/với B  (vd: "tao", "tôi")
  - target  : str  — đại từ A dùng để gọi B           (vd: "mày", "cậu", "cô")
  - notes   : str  — ghi chú thêm về mối quan hệ
"""

import json
import logging
from typing import Any

# pyrefly: ignore [missing-import]
import networkx as nx

from app.services.db_service import AsyncSessionLocal, MangaCRUD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache (chỉ dùng cho FastAPI process để inject prompt)
# ---------------------------------------------------------------------------
_graph_cache: dict[str, nx.DiGraph] = {}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize(G: nx.DiGraph) -> str:
    """NetworkX DiGraph → JSON string (lưu DB)."""
    return json.dumps(nx.node_link_data(G), ensure_ascii=False)


def _deserialize(json_str: str) -> nx.DiGraph:
    """JSON string → NetworkX DiGraph."""
    data = json.loads(json_str)
    return nx.node_link_graph(data, directed=True, multigraph=False)


def _empty_graph() -> nx.DiGraph:
    return nx.DiGraph()


# ---------------------------------------------------------------------------
# GraphService
# ---------------------------------------------------------------------------

class GraphService:
    """
    Stateless utility class — tất cả method đều là classmethod,
    không giữ state instance để an toàn khi dùng từ nhiều process.
    """

    # ── Load / Save ────────────────────────────────────────────────────────

    @classmethod
    async def load(
        cls,
        manga_id: str,
        *,
        use_cache: bool = False,
        session_factory=None,
    ) -> nx.DiGraph:
        """
        Đọc character graph từ DB.

        Args:
            manga_id        : ID của bộ truyện.
            use_cache       : Trả từ cache nếu có (dùng cho FastAPI).
                              Celery task LUÔN để False.
            session_factory : None → dùng AsyncSessionLocal (FastAPI).
                              Truyền make_task_session()() để dùng trong Celery.
        """
        if use_cache and manga_id in _graph_cache:
            return _graph_cache[manga_id]

        factory = session_factory or AsyncSessionLocal
        async with factory() as session:
            manga = await MangaCRUD.get_by_manga_id(session, manga_id)

        if manga and manga.character_graph:
            G = _deserialize(manga.character_graph)
        else:
            G = _empty_graph()
            if use_cache:
                # FastAPI: graph chưa có trong DB → dùng graph rỗng tạm cho việc inject prompt.
                # Graph thực sự sẽ được build sau bởi Celery task update_character_graph.
                logger.debug(
                    f"[GraphService] manga={manga_id!r} chưa có graph trong DB — "
                    f"dùng graph rỗng tạm cho VLM context."
                )
            else:
                # Celery: khởi tạo graph mới thực sự (lần đầu tiên build).
                logger.info(
                    f"[GraphService] manga={manga_id!r} — khởi tạo graph mới "
                    f"(lần đầu tiên hoặc chưa có dữ liệu)."
                )

        if use_cache:
            _graph_cache[manga_id] = G

        return G

    @classmethod
    async def save(
        cls,
        manga_id: str,
        G: nx.DiGraph,
        *,
        session_factory=None,
    ) -> None:
        """
        Lưu graph vào DB và cập nhật cache nếu có.
        """
        graph_json = _serialize(G)
        factory = session_factory or AsyncSessionLocal
        async with factory() as session:
            await MangaCRUD.update_character_graph(session, manga_id, graph_json)

        if manga_id in _graph_cache:
            _graph_cache[manga_id] = G

        logger.info(
            f"[GraphService] Saved graph — manga={manga_id!r} "
            f"| nodes={G.number_of_nodes()} edges={G.number_of_edges()}"
        )

    @classmethod
    def invalidate_cache(cls, manga_id: str) -> None:
        """Xóa cache entry (gọi sau khi Celery task cập nhật xong)."""
        _graph_cache.pop(manga_id, None)

    # ── Graph Mutation ─────────────────────────────────────────────────────

    @classmethod
    def add_node(
        cls,
        G: nx.DiGraph,
        name: str,
        *,
        gender: str = "?",
        age: str = "?",
    ) -> None:
        """
        Thêm hoặc cập nhật node nhân vật.
        Nếu node đã tồn tại → merge (không ghi đè các field đang có giá trị).
        """
        if G.has_node(name):
            existing = G.nodes[name]
            if gender != "?" and existing.get("gender", "?") == "?":
                G.nodes[name]["gender"] = gender
            if age != "?" and existing.get("age", "?") == "?":
                G.nodes[name]["age"] = age
        else:
            G.add_node(name, gender=gender, age=age)
            logger.debug(f"[GraphService] Node added: {name!r} ({gender}, {age})")

    @classmethod
    def add_edge(
        cls,
        G: nx.DiGraph,
        source: str,
        target: str,
        *,
        caller: str = "?",
        target_pronoun: str = "?",
        notes: str = "",
    ) -> None:
        """
        Thêm hoặc cập nhật edge xưng hô (source → target).
        Tự động tạo node nếu chưa có.
        """
        # Đảm bảo cả 2 node tồn tại
        cls.add_node(G, source)
        cls.add_node(G, target)

        if G.has_edge(source, target):
            # Chỉ update field có giá trị mới thực sự
            if caller != "?":
                G[source][target]["caller"] = caller
            if target_pronoun != "?":
                G[source][target]["target"] = target_pronoun
            if notes:
                G[source][target]["notes"] = notes
        else:
            G.add_edge(source, target, caller=caller, target=target_pronoun, notes=notes)
            logger.debug(f"[GraphService] Edge added: {source!r} → {target!r}")

    @classmethod
    def apply_pronoun_shift(
        cls,
        G: nx.DiGraph,
        source: str,
        target: str,
        *,
        new_caller: str | None = None,
        new_target: str | None = None,
        reason: str = "",
    ) -> bool:
        """
        Áp dụng pronoun shift lên 1 cặp (source → target).

        Returns:
            True nếu edge tồn tại và được cập nhật.
            False nếu edge chưa tồn tại (không tạo mới).
        """
        if not G.has_edge(source, target):
            logger.warning(
                f"[GraphService] Pronoun shift ignored — edge {source!r}→{target!r} chưa tồn tại."
            )
            return False

        if new_caller:
            G[source][target]["caller"] = new_caller
        if new_target:
            G[source][target]["target"] = new_target
        if reason:
            old_notes = G[source][target].get("notes", "")
            G[source][target]["notes"] = f"{old_notes} | Shift: {reason}".strip(" |")

        logger.info(
            f"[GraphService] Pronoun shift applied: {source!r}→{target!r} "
            f"caller={new_caller!r} target={new_target!r}"
        )
        return True

    # ── Merge từ VLM output ────────────────────────────────────────────────

    @classmethod
    def merge_character_updates(cls, G: nx.DiGraph, character_updates: list[dict]) -> int:
        """
        Merge danh sách character_updates từ VLM vào graph.

        Format mỗi entry:
          {
            "name": "Killua",
            "gender": "male",       # hoặc "female" / "?"
            "age_range": "teen",
            "speaks_to": [
              {"target": "Gon", "caller_pronoun": "tao", "target_pronoun": "mày"}
            ],
            "notes": "..."
          }

        Returns:
            Số lượng node được thêm/cập nhật.
        """
        count = 0
        for entry in character_updates:
            name = entry.get("name", "").strip()
            if not name:
                continue

            gender = _normalize_gender(entry.get("gender", "?"))
            age    = entry.get("age_range", "?")
            cls.add_node(G, name, gender=gender, age=age)
            count += 1

            for rel in entry.get("speaks_to", []):
                t = rel.get("target", "").strip()
                if not t:
                    continue
                cls.add_edge(
                    G, name, t,
                    caller=rel.get("caller_pronoun", "?"),
                    target_pronoun=rel.get("target_pronoun", "?"),
                    notes=entry.get("notes", ""),
                )

        return count

    @classmethod
    def merge_pronoun_shifts(cls, G: nx.DiGraph, pronoun_shifts: list[dict]) -> int:
        """
        Áp dụng danh sách pronoun_shift từ VLM lên graph.

        Format mỗi entry:
          {
            "character": "Killua",
            "target": "Gon",
            "new_caller_pronoun": "tôi",   # optional
            "new_target_pronoun": "bạn",   # optional
            "reason": "..."                # optional
          }

        Returns:
            Số lượng shift được áp dụng thành công.
        """
        applied = 0
        for shift in pronoun_shifts:
            source  = shift.get("character", "").strip()
            target  = shift.get("target", "").strip()
            if not source or not target:
                continue

            ok = cls.apply_pronoun_shift(
                G, source, target,
                new_caller=shift.get("new_caller_pronoun"),
                new_target=shift.get("new_target_pronoun"),
                reason=shift.get("reason", ""),
            )
            if ok:
                applied += 1

        return applied

    # ── Prompt serialization ───────────────────────────────────────────────

    @classmethod
    def to_prompt(cls, G: nx.DiGraph) -> str:
        """
        Serialize graph thành compact JSON string để inject vào system prompt.
        Format tiết kiệm token (key viết tắt):
          g=gender, a=age, p={target:[caller, target_pronoun]}

        Ví dụ:
          {"Killua":{"g":"M","a":"teen","p":{"Gon":["tao","mày"]}}}
        """
        if G.number_of_nodes() == 0:
            return ""

        result: dict[str, Any] = {}
        for node, attrs in G.nodes(data=True):
            entry: dict[str, Any] = {
                "g": attrs.get("gender", "?"),
                "a": attrs.get("age", "?"),
            }
            pronouns: dict[str, list[str]] = {}
            for _, tgt, edata in G.out_edges(node, data=True):
                pronouns[tgt] = [
                    edata.get("caller", "?"),
                    edata.get("target", "?"),
                ]
            if pronouns:
                entry["p"] = pronouns
            result[node] = entry

        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_gender(raw: str) -> str:
    """Chuẩn hóa gender string về M / F / ?"""
    r = raw.lower()
    if r in ("male", "m", "nam"):
        return "M"
    if r in ("female", "f", "nữ", "nu"):
        return "F"
    return "?"
