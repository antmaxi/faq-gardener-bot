"""Remember cluster fingerprints and question decisions already sent to admins."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .questions import normalize


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sent (
             fingerprint TEXT PRIMARY KEY,
             sent_at INTEGER NOT NULL,
             heading TEXT NOT NULL,
             questions TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reported (
             key TEXT PRIMARY KEY,
             sent_at INTEGER NOT NULL,
             question TEXT NOT NULL,
             label TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS faq_suggestions (
             key TEXT PRIMARY KEY,
             saved_at INTEGER NOT NULL,
             card TEXT NOT NULL,
             answer TEXT NOT NULL
           )"""
    )
    conn.commit()
    return conn


def save_suggestion(conn: sqlite3.Connection, card: dict) -> None:
    with conn:
        conn.execute(
            """INSERT INTO faq_suggestions (key, saved_at, card, answer)
               VALUES (?, ?, ?, ?) ON CONFLICT(key) DO NOTHING""",
            (card["key"], int(time.time()), json.dumps(card, ensure_ascii=False), card["suggestion"]),
        )


def get_suggestion(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT card, answer FROM faq_suggestions WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    card = json.loads(row[0])
    card["suggestion"] = row[1]
    return card


def list_suggestions(conn: sqlite3.Connection, *, limit: int, offset: int) -> list[dict]:
    return [
        json.loads(row[0]) for row in conn.execute(
            "SELECT card FROM faq_suggestions ORDER BY saved_at DESC, key LIMIT ? OFFSET ?",
            (limit, offset),
        )
    ]


def edit_suggestion(conn: sqlite3.Connection, key: str, answer: str) -> bool:
    with conn:
        return conn.execute(
            "UPDATE faq_suggestions SET answer = ? WHERE key = ?", (answer, key)
        ).rowcount > 0


def seen(conn: sqlite3.Connection, fingerprint: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sent WHERE fingerprint=?", (fingerprint,)
    ).fetchone()
    return row is not None


def mark(
    conn: sqlite3.Connection,
    fingerprint: str,
    heading: str,
    questions: list[str],
) -> None:
    conn.execute(
        """INSERT INTO sent (fingerprint, sent_at, heading, questions)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(fingerprint) DO NOTHING""",
        (fingerprint, int(time.time()), heading, json.dumps(questions, ensure_ascii=False)),
    )
    conn.commit()


def decision_key(question: str, label: str) -> str:
    """Same question and decision is one report. A new label is a new report."""
    body = normalize(question) + "\n" + label.strip().lower()
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def reported(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute("SELECT 1 FROM reported WHERE key=?", (key,)).fetchone()
    return row is not None


def mark_reported(conn: sqlite3.Connection, question: str, label: str) -> None:
    conn.execute(
        """INSERT INTO reported (key, sent_at, question, label)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(key) DO NOTHING""",
        (decision_key(question, label), int(time.time()), question, label),
    )
    conn.commit()
