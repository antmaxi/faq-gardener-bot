import json
import urllib.error

from gardener.cli import pause_for_admin, reply_dry_run
from gardener.store import connect, seen
from gardener.telegram import (
    admin_private_chats,
    advance_offset,
    format_message,
    get_updates,
    notify,
    split_message,
)
from tests.conftest import settings

DRAFT = {
    "fingerprint": "abc",
    "label": "missing",
    "heading": "Семейная страховка",
    "nearest_url": "https://ru-ch.github.io/faq/docs/Страховка.html",
    "questions": ["как оформить семейную страховку в Цюрихе"],
    "outline": "- TODO: какие кассы",
}


def test_format_has_outline_and_no_user_id():
    text = format_message(DRAFT)
    assert "Семейная страховка" in text
    assert "https://ru-ch.github.io/faq/docs/Страховка.html" in text
    assert "TODO" in text
    assert "user_id" not in text
    assert "999001" not in text


def test_split_stays_under_telegram_limit():
    parts = split_message("строка\n" * 2000)
    assert len(parts) > 1
    assert all(len(part) <= 4096 for part in parts)
    assert parts[0].startswith("(1/")


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"ok": true}'


def test_one_admin_failure_still_records(tmp_path):
    conn = connect(tmp_path / "s.db")
    calls = []

    def opener(req, timeout=None):
        body = json.loads(req.data.decode())
        calls.append(body["chat_id"])
        if body["chat_id"] == 1:
            raise urllib.error.URLError("down")
        return _Resp()

    delivered = notify("token", [1, 2], [DRAFT], dry_run=False, conn=conn, opener=opener)
    assert delivered == 1
    assert calls == [1, 2]
    assert seen(conn, "abc")
    conn.close()


def test_total_failure_is_not_recorded(tmp_path):
    conn = connect(tmp_path / "s.db")

    def opener(req, timeout=None):
        raise urllib.error.URLError("down")

    delivered = notify("token", [1], [DRAFT], dry_run=False, conn=conn, opener=opener)
    assert delivered == 0
    assert not seen(conn, "abc")
    conn.close()


def test_dry_run_prints_and_does_not_record(tmp_path, capsys):
    conn = connect(tmp_path / "s.db")

    def opener(req, timeout=None):
        raise AssertionError("dry run must not send")

    delivered = notify(None, [], [DRAFT], dry_run=True, conn=conn, opener=opener)
    assert delivered == 1
    assert not seen(conn, "abc")
    assert "Семейная страховка" in capsys.readouterr().out
    conn.close()


def test_admin_messages_ignore_groups_and_other_users():
    updates = [
        {
            "update_id": 2,
            "message": {
                "text": "/start",
                "from": {"id": 1},
                "chat": {"id": -100, "type": "supergroup"},
            },
        },
        {
            "update_id": 3,
            "message": {
                "text": "hi",
                "from": {"id": 9},
                "chat": {"id": 9, "type": "private"},
            },
        },
        {
            "update_id": 4,
            "message": {
                "text": "/start",
                "from": {"id": 1},
                "chat": {"id": 1, "type": "private"},
            },
        },
        {
            "update_id": 5,
            "message": {
                "text": "again",
                "from": {"id": 1},
                "chat": {"id": 1, "type": "private"},
            },
        },
    ]
    assert admin_private_chats(updates, {1}) == [1, 1]
    assert advance_offset(updates, None) == 6


def test_get_updates_posts_offset():
    def opener(req, timeout=None):
        assert req.full_url.endswith("/getUpdates")
        body = json.loads(req.data.decode())
        assert body["timeout"] == 5
        assert body["offset"] == 4
        assert body["allowed_updates"] == ["message"]
        return _Json(b'{"ok": true, "result": [{"update_id": 4}]}')

    assert get_updates("token", offset=4, timeout=5, opener=opener) == [{"update_id": 4}]


class _Json:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_pause_previews_each_admin_once(tmp_path):
    import threading

    stop = threading.Event()
    seen = []
    batches = [
        [
            {
                "update_id": 4,
                "message": {
                    "text": "/start",
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "private"},
                },
            },
            {
                "update_id": 5,
                "message": {
                    "text": "again",
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "private"},
                },
            },
        ]
    ]

    def fetch(token, *, offset, timeout):
        assert token == "test-token"
        if batches:
            return batches.pop(0)
        stop.set()
        return []

    offset = pause_for_admin(
        settings(tmp_path),
        stop,
        30,
        offset=None,
        run_preview=seen.append,
        fetch=fetch,
    )
    assert seen == [1]
    assert offset == 6


def test_reply_dry_run_sends_outline(tmp_path, monkeypatch):
    texts = []

    def opener(req, timeout=None):
        texts.append(json.loads(req.data.decode())["text"])
        return _Resp()

    def fake_execute(settings, **kwargs):
        assert kwargs["dry_run"] is True
        assert kwargs["notify_admins"] is False
        return {"questions": 2, "drafts": [DRAFT], "delivered": 0}

    monkeypatch.setattr("gardener.cli.execute", fake_execute)
    reply_dry_run(settings(tmp_path), 42, opener=opener)
    assert texts[0] == "Checking for FAQ gaps…"
    assert "Семейная страховка" in texts[1]
    assert "TODO" in texts[1]


def test_reply_dry_run_reports_empty(tmp_path, monkeypatch):
    texts = []

    def opener(req, timeout=None):
        texts.append(json.loads(req.data.decode())["text"])
        return _Resp()

    monkeypatch.setattr(
        "gardener.cli.execute",
        lambda settings, **kwargs: {"questions": 0, "drafts": [], "delivered": 0},
    )
    reply_dry_run(settings(tmp_path), 42, opener=opener)
    assert texts == ["Checking for FAQ gaps…", "Dry run: questions=0 drafts=0"]
