import json
import urllib.error

from gardener.store import connect, seen
from gardener.telegram import format_message, notify, split_message

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
