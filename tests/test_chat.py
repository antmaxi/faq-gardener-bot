import json
import sqlite3
from contextlib import closing

from gardener.cli import execute
from gardener.telegram import format_message
from tests.conftest import settings
from tests.keyword_embed import KeywordEmbedding


def test_selected_chat_evidence_reaches_admin_preview(tmp_path):
    cfg = settings(tmp_path, faq_chat_ids={-100123})
    with closing(sqlite3.connect(cfg.answerbot_db)) as conn:
        conn.executescript("""
            CREATE TABLE query_log(ts, question, user_id, cited_ids);
            INSERT INTO query_log VALUES (120, 'WHERE do I renew?', 7, '[]');
            CREATE TABLE messages(chat_id, msg_id, ts, sender_id, text, reply_to);
        """)
        rows = [
            (-100123, 1, 100, 7, "Where do I renew?", None),
            (-100123, 2, 101, 7, "I need help renewing", None),
            (-100123, 3, 102, 8, "At the desk", 1),
            (-100123, 4, 86500, 9, "Renew online now", None),
            (-100123, 5, 104, 9, "Different thread", 99),
            (-100123, 7, 105, 7, "Renew where?", None),
            (-100123, 6, 86501, 9, "Too late to infer", None),
            (-100456, 4, 86500, 8, "Unselected group", None),
        ]
        conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
        conn.commit()

    class Model:
        def complete(self, system, user):
            assert "Unselected group" not in user
            if user.startswith("["):
                return json.dumps({"questions": [1, 2, 7], "answers": [[1, 4], [1, 5], [1, 6]]})
            if user.startswith("Task: grade"):
                assert "At the desk" not in user
                return '{"label":"missing"}'
            assert "leave the disputed detail open" in system
            assert "At the desk" in user and "Renew online now" in user
            return '{"heading":"Renewal", "outline":"Verify whether renewal moved online."}'

    result = execute(cfg, dry_run=True, llm=Model(), embed_model=KeywordEmbedding(),
                     faq_pages={"docs/other.md": "# Other\nOther FAQ topic."}, now=86501)
    assert result["questions"] == 2
    draft, = result["drafts"]
    assert {(s["message_id"], s["kind"]) for s in draft["sources"]} == {
        (1, "question"), (2, "question"), (3, "direct reply"), (4, "inferred match")
    }
    preview = format_message(draft)
    assert "question · https://t.me/c/123/2" in preview
    assert "direct reply · https://t.me/c/123/3" in preview
    assert "inferred match · https://t.me/c/123/4" in preview
