"""Remember cluster fingerprints that were already sent to admins."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


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
