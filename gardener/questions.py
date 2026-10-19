"""Unanswered questions from the answer bot's query_log, or a JSON export.

A row is a FAQ candidate unless the answer cited a wiki chunk. FAQ chunk ids
are negative. Retrieved chunks in window_ids do not count; only cited_ids do.
Admin asks and very short texts are dropped. The same question text is kept
once, newest first.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True)
class Question:
    ts: int
    text: str
    user_id: int | None = None


def normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def is_gap(cited_ids: list[int]) -> bool:
    """True unless the model answer cited a FAQ chunk."""
    return all(int(i) > 0 for i in cited_ids)


def _id_list(raw) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [int(i) for i in raw]
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("id list must be a list")
        return [int(i) for i in parsed]
    raise TypeError(f"id list must be a list or JSON string, not {type(raw).__name__}")


def filter_rows(
    rows: list[dict],
    *,
    since_ts: int,
    admin_ids: set[int] | frozenset[int],
    min_words: int,
) -> list[Question]:
    """Apply the gap, admin, length, and time filters. Newest unique text wins."""
    best: dict[str, Question] = {}
    for row in rows:
        text = str(row.get("question") or row.get("text") or "").strip()
        if not text:
            continue
        ts = int(row.get("ts") or 0)
        if ts < since_ts:
            continue
        user_raw = row.get("user_id")
        user_id = int(user_raw) if user_raw is not None else None
        if user_id is not None and user_id in admin_ids:
            continue
        if len(text.split()) < min_words:
            continue
        try:
            cited = _id_list(row.get("cited_ids"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not is_gap(cited):
            continue
        key = normalize(text)
        current = best.get(key)
        if current is None or ts > current.ts:
            best[key] = Question(ts=ts, text=text, user_id=user_id)
    return sorted(best.values(), key=lambda q: q.ts, reverse=True)


def _ro_uri(path: Path) -> str:
    return "file:" + quote(str(path.resolve()), safe="/:") + "?mode=ro"


def load_sqlite(
    path: Path,
    *,
    since_ts: int,
    admin_ids: set[int] | frozenset[int],
    min_words: int,
) -> list[Question]:
    """Read query_log without writing the answer bot's database."""
    if not path.is_file():
        raise FileNotFoundError(f"answer bot database not found: {path}")
    conn = sqlite3.connect(_ro_uri(path), uri=True)
    try:
        try:
            rows = conn.execute(
                """SELECT ts, question, user_id, cited_ids
                   FROM query_log WHERE ts >= ?""",
                (since_ts,),
            ).fetchall()
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                f"{path} has no readable query_log (ts, question, user_id, cited_ids)"
            ) from e
    finally:
        conn.close()
    payload = [
        {"ts": ts, "question": question, "user_id": user_id, "cited_ids": cited_ids}
        for ts, question, user_id, cited_ids in rows
    ]
    return filter_rows(payload, since_ts=since_ts, admin_ids=admin_ids, min_words=min_words)


def load_export(
    path: Path,
    *,
    since_ts: int,
    admin_ids: set[int] | frozenset[int],
    min_words: int,
) -> list[Question]:
    """JSON list of query_log-shaped objects. Missing cited_ids counts as a gap."""
    if not path.is_file():
        raise FileNotFoundError(f"questions export not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if "cited_ids" not in row:
            row["cited_ids"] = []
        rows.append(row)
    return filter_rows(rows, since_ts=since_ts, admin_ids=admin_ids, min_words=min_words)
