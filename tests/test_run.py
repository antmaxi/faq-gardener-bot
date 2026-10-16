import json

import pytest

from gardener.cli import execute, main, watch
from gardener.config import load_settings
from gardener.store import connect, seen
from tests.conftest import FIXTURES, settings
from tests.keyword_embed import KeywordEmbedding
from tests.test_graph import ScriptLLM


class _Stop:
    def __init__(self):
        self.flag = False
        self.waits = 0

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True

    def wait(self, _interval: float) -> bool:
        self.waits += 1
        return self.flag


def test_debug_includes_admin_questions(tmp_path):
    cfg = settings(tmp_path, debug=True)
    pages = {
        "docs/Страховка.md": (FIXTURES / "faq" / "docs" / "Страховка.md").read_text(encoding="utf-8"),
    }
    summary = execute(
        cfg,
        dry_run=True,
        questions_path=FIXTURES / "questions.json",
        since_days=1,
        faq_pages=pages,
        llm=ScriptLLM(),
        embed_model=KeywordEmbedding(),
        now=86500,
    )
    assert summary["questions"] == 5
    by_text = {row["text"]: row for row in summary["decisions"]}
    assert by_text["как получить визу"]["label"] == "too_small"
    assert "at least 2" in by_text["как получить визу"]["reason"]
    assert by_text["как оформить семейную страховку в Цюрихе"]["label"] == "missing"
    assert by_text["как оформить семейную страховку в Цюрихе"]["reason"]


def test_dry_run_then_send_then_skip(tmp_path, capsys):
    cfg = settings(tmp_path)
    pages = {
        "docs/Страховка.md": (FIXTURES / "faq" / "docs" / "Страховка.md").read_text(encoding="utf-8"),
        "inbox/note.md": (FIXTURES / "faq" / "inbox" / "note.md").read_text(encoding="utf-8"),
    }
    common = dict(
        questions_path=FIXTURES / "questions.json",
        since_days=1,
        faq_pages=pages,
        llm=ScriptLLM(),
        embed_model=KeywordEmbedding(),
        now=86500,
    )
    preview = execute(cfg, dry_run=True, **common)
    assert preview["questions"] == 4
    assert len(preview["drafts"]) == 1
    assert "Семейная страховка" in capsys.readouterr().out
    conn = connect(cfg.gardener_db)
    assert not seen(conn, preview["drafts"][0]["fingerprint"])
    conn.close()

    sent = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def opener(req, timeout=None):
        sent.append(json.loads(req.data.decode())["chat_id"])
        return Resp()

    delivered = execute(cfg, dry_run=False, opener=opener, **common)
    assert delivered["delivered"] == 1
    assert sent == [1]
    conn = connect(cfg.gardener_db)
    assert seen(conn, delivered["drafts"][0]["fingerprint"])
    conn.close()

    again = execute(cfg, dry_run=False, opener=opener, **common)
    assert again["drafts"] == []
    assert again["delivered"] == 0
    assert sent == [1]


def test_watch_retries_then_stops(tmp_path, monkeypatch):
    stop = _Stop()
    calls = []

    def fake(settings, **kwargs):
        calls.append(kwargs["embed_model"])
        if len(calls) == 1:
            raise RuntimeError("down")
        stop.set()
        return {"questions": 0, "drafts": [], "delivered": 0}

    monkeypatch.setattr("gardener.cli.execute", fake)
    model = object()
    code = watch(
        settings(tmp_path),
        interval=3600,
        stop=stop,
        embed_model=model,
        llm=object(),
    )
    assert code == 0
    assert calls == [model, model]
    assert stop.waits == 1


def test_watch_loads_models_once(tmp_path, monkeypatch):
    stop = _Stop()
    loads = []

    def embedder(*_args):
        loads.append("embed")
        return "model"

    def llm(_settings):
        loads.append("llm")
        return "llm"

    monkeypatch.setattr("gardener.cli.huggingface_embedder", embedder)
    monkeypatch.setattr("gardener.cli.get_llm", llm)
    passes = {"n": 0}

    def counting(_settings, **_kwargs):
        passes["n"] += 1
        if passes["n"] == 2:
            stop.set()
        return {"questions": 0, "drafts": [], "delivered": 0}

    monkeypatch.setattr("gardener.cli.execute", counting)
    assert watch(settings(tmp_path), interval=3600, stop=stop) == 0
    assert loads == ["embed", "llm"]
    assert passes["n"] == 2


def test_watch_seconds_blank_is_one_shot(monkeypatch):
    monkeypatch.setenv("WATCH_SECONDS", "")
    assert load_settings().watch_seconds == 0


def test_watch_seconds_must_not_be_negative(monkeypatch):
    monkeypatch.setenv("WATCH_SECONDS", "-5")
    with pytest.raises(ValueError):
        load_settings()


def test_main_dry_run_refuses_watch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gardener.cli.load_settings", lambda: settings(tmp_path, watch_seconds=30)
    )
    assert main(["run", "--dry-run"]) == 2


def test_watch_seconds_zero_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gardener.cli.load_settings", lambda: settings(tmp_path, watch_seconds=30)
    )
    calls = []
    monkeypatch.setattr(
        "gardener.cli.execute",
        lambda settings, **kwargs: calls.append(1) or {"questions": 0, "drafts": [], "delivered": 0},
    )
    assert main(["run", "--watch-seconds", "0"]) == 0
    assert calls == [1]
