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
    conn.commit()
    return conn


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
