import json
import sqlite3

from gardener.questions import filter_rows, load_export, load_sqlite

from tests.conftest import FIXTURES


def test_export_filters_gaps_admins_and_short_text():
    questions = load_export(
        FIXTURES / "questions.json",
        since_ts=100,
        admin_ids={1},
        min_words=3,
    )
    texts = [q.text for q in questions]
    assert texts == [
        "как оформить семейную страховку в Цюрихе",
        "как оформить семейную страховку для детей",
        "сколько стоила поездка на лыжах в феврале",
        "сколько стоила поездка на лыжах в прошлом году",
    ]


def test_missing_window_ids_count_as_a_gap():
    kept = filter_rows(
        [{"ts": 5, "question": "как устроить ребёнка в сад", "user_id": 9}],
        since_ts=0,
        admin_ids=set(),
        min_words=3,
    )
    assert len(kept) == 1


def test_duplicate_keeps_the_newer_wording():
    kept = filter_rows(
        [
            {"ts": 1, "question": "Как оформить страховку семьи", "window_ids": []},
            {"ts": 2, "question": "как  оформить страховку семьи", "window_ids": []},
        ],
        since_ts=0,
        admin_ids=set(),
        min_words=3,
    )
    assert len(kept) == 1
    assert kept[0].text == "как  оформить страховку семьи"
    assert kept[0].ts == 2


def test_sqlite_is_read_only(tmp_path):
    db = tmp_path / "answerbot.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE query_log (
             id INTEGER PRIMARY KEY,
             ts INTEGER NOT NULL,
             question TEXT NOT NULL,
             window_ids TEXT NOT NULL,
             cited_ids TEXT NOT NULL,
             user_id INTEGER
           )"""
    )
    rows = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))
    conn.executemany(
        """INSERT INTO query_log (ts, question, window_ids, cited_ids, user_id)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (
                row["ts"],
                row["question"],
                json.dumps(row["window_ids"]),
                json.dumps(row["cited_ids"]),
                row["user_id"],
            )
            for row in rows
        ],
    )
    conn.commit()
    conn.close()

    questions = load_sqlite(db, since_ts=100, admin_ids={1}, min_words=3)
    assert len(questions) == 4
    assert not (tmp_path / "answerbot.db-journal").exists()


def test_missing_database(tmp_path):
    try:
        load_sqlite(tmp_path / "nope.db", since_ts=0, admin_ids=set(), min_words=3)
    except FileNotFoundError as e:
        assert "nope.db" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
