import json

from gardener.cli import execute
from gardener.store import connect, seen
from tests.conftest import FIXTURES, settings
from tests.keyword_embed import KeywordEmbedding
from tests.test_graph import ScriptLLM


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
