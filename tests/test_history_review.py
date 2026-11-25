import json
import sqlite3
from contextlib import closing
from threading import Event
from types import SimpleNamespace

from gardener.cli import pause_for_admin
from gardener.review import HistoryReview
from gardener.store import connect, mark
from tests.conftest import settings


class ReviewLLM:
    def complete(self, system, user):
        assert "Excluded group" not in user
        if user.startswith("Task: grade"):
            return '{"label":"covered", "reason":"The wiki covers membership."}'
        conversation = json.loads(user)
        if isinstance(conversation, list):
            return json.dumps({
                "questions": [message["msg_id"] for message in conversation if "?" in message["text"]],
                "answers": [],
            })
        messages = conversation["messages"]
        question = conversation["question"]
        if question == "Renew membership?":
            assert [m["text"] for m in messages] == ["Renew at the desk", "Renew online"]
            return '{"status":"conflicting", "sources":[1,2], "suggestion":"Verify the renewal location."}'
        if question == "Where is the office?":
            assert [m["text"] for m in messages] == ["The office is on the second floor."]
            return '{"status":"answered", "sources":[1], "suggestion":"Visit the second floor."}'
        if question == "Is there a refund form?":
            answer = next(m for m in messages if m["text"] == "Fill out form 17.")
            return json.dumps({
                "status": "answered", "sources": [answer["id"]], "suggestion": "Fill out form 17.",
            })
        assert [m["text"] for m in messages] == ["Thanks for asking"]
        # Even if the model supplies text with a "none" result, the card must omit it.
        return '{"status":"none", "sources":[], "suggestion":"Unsupported text"}'


class ReviewIndex:
    def retrieve(self, query, *, k):
        return [SimpleNamespace(
            heading="Membership", text="Membership help.", url="https://faq.test/membership",
        )]


def seed_archive(path):
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
            CREATE TABLE query_log(id INTEGER PRIMARY KEY, ts, question, answer, cited_ids);
            INSERT INTO query_log VALUES (1, 12, 'Where do I register?', 'Visit the reception desk.', '[]');
            INSERT INTO query_log VALUES (2, 86400, 'Outside the date range?', NULL, '[]');
            INSERT INTO query_log VALUES (3, 22, 'Where is the office?', 'Possibly upstairs.', '[1,2]');
            INSERT INTO query_log VALUES (4, 23, 'Is there a refund form?', NULL, '[]');
            CREATE TABLE messages(chat_id, msg_id, ts, sender_id, text, reply_to);
            INSERT INTO messages VALUES (-100123, 1, 11, 2, 'Renew membership?', NULL);
            INSERT INTO messages VALUES (-100123, 2, 12, 3, 'Renew at the desk', 1);
            INSERT INTO messages VALUES (-100123, 3, 13, 4, 'Renew online', NULL);
            INSERT INTO messages VALUES (-100123, 4, 86399, 2, 'Can I transfer it?', NULL);
            INSERT INTO messages VALUES (-100123, 5, 86400, 3, 'Thanks for asking', 4);
            INSERT INTO messages VALUES (-100123, 6, -1000, 3, 'The office is on the second floor.', NULL);
            INSERT INTO messages VALUES (-100123, 7, 100000, 2, 'How do I request a refund?', NULL);
            INSERT INTO messages VALUES (-100123, 8, 100001, 3, 'Fill out form 17.', 7);
            INSERT INTO messages VALUES (-100456, 1, 11, 2, 'Excluded group question?', NULL);
            CREATE TABLE windows(id INTEGER PRIMARY KEY, chat_id, first_msg, last_msg);
            INSERT INTO windows VALUES (1, -100123, 6, 6);
            INSERT INTO windows VALUES (2, -100456, 1, 1);
        """)


def update(number, text, *, admin=1, chat_type="private"):
    return {
        "update_id": number,
        "message": {"from": {"id": admin}, "chat": {"id": admin, "type": chat_type}, "text": text},
    }


def run_admin_updates(cfg, review, updates):
    stop = Event()
    sent = []
    previews = []

    def fetch(*args, **kwargs):
        if updates:
            batch = updates[:]
            updates.clear()
            return batch
        stop.set()
        return []

    def handle(chat_id, text):
        return review.handle(chat_id, text, sent.append)

    pause_for_admin(cfg, stop, 30, offset=None, run_preview=previews.append, run_review=handle, fetch=fetch)
    return sent, previews


def test_admin_reviews_and_edits_archived_questions(tmp_path, monkeypatch):
    def unexpected_error(*args, **kwargs):
        raise AssertionError("The admin command failed.")

    monkeypatch.setattr("gardener.cli.send_text", unexpected_error)
    cfg = settings(tmp_path, faq_chat_ids=frozenset({-100123}), min_question_words=2)
    seed_archive(cfg.answerbot_db)
    original = cfg.answerbot_db.read_bytes()
    with closing(connect(cfg.gardener_db)) as conn:
        mark(conn, "old", "Membership", ["Renew membership?"])

    review = HistoryReview(cfg, ReviewLLM, ReviewIndex)
    sent, previews = run_admin_updates(cfg, review, [
        update(1, "/review 1970-01-01 1970-01-01", admin=99),
        update(2, "/review_open q:1", chat_type="group"),
        update(3, "/review 1970-01-01 1970-01-01"),
        update(4, "/review_search 1970-01-01 1970-01-01 1 RENEW"),
        update(5, "/review_open g:-100123:1"),
        update(6, "/review_save g:-100123:1"),
        update(7, "/review_open q:1"),
        update(8, "/review_save q:1"),
        update(9, "/review_open q:3"),
        update(10, "/review_open g:-100123:4"),
        update(11, "Check for FAQ gaps"),
        update(19, "/review_open q:4"),
    ])
    assert len(sent) == 9
    assert previews == [1]
    assert all(name in sent[0] for name in ("Renew membership?", "Can I transfer it?", "Where do I register?"))
    assert "Outside the date range" not in sent[0]
    assert "Excluded group" not in sent[0]
    assert sent[1].endswith("/review_open g:-100123:1")
    assert "Where do I register?" not in sent[1]
    assert "Conflicting answers found" in sent[2] and "FAQ · covered" in sent[2]
    assert "Direct reply · https://t.me/c/123/2" in sent[2]
    assert "Inferred match · https://t.me/c/123/3" in sent[2]
    assert "No chat answer found" in sent[4] and "Visit the reception desk." in sent[4]
    assert "Suggested answer" not in sent[4]
    assert "https://t.me/c/123/6" in sent[6] and "Visit the second floor." in sent[6]
    assert "No chat answer found" in sent[7] and "Unsupported text" not in sent[7]
    assert "Suggested answer\nFill out form 17." in sent[8]
    assert "Inferred match · https://t.me/c/123/8" in sent[8]

    restarted = HistoryReview(cfg, ReviewLLM, ReviewIndex)
    replies, _ = run_admin_updates(cfg, restarted, [
        update(12, "/review_saved"),
        update(13, "/review_edit g:-100123:1 Renewal is online."),
        update(14, "/review_save g:-100123:1"),
        update(15, "/review_draft g:-100123:1"),
        update(16, "/review_draft q:1"),
        update(17, "/review 1970-02-01 1970-01-01"),
        update(18, "/review 1970-01-01 1970-01-01 2"),
    ])
    assert replies[0].count("/review_draft") == 2
    assert "Updated." in replies[1]
    assert "Already saved." in replies[2]
    assert "Saved answer\nRenewal is online." in replies[3]
    assert "No chat answer found" in replies[4] and "Saved answer" not in replies[4]
    assert "Unable to complete" in replies[5]
    assert "No questions on this page" in replies[6]
    assert cfg.answerbot_db.read_bytes() == original


def test_browse_limits_classification_and_search_filters_first(tmp_path):
    cfg = settings(tmp_path, faq_chat_ids=frozenset({-100123}))
    with closing(sqlite3.connect(cfg.answerbot_db)) as conn:
        conn.executescript("""
            CREATE TABLE query_log(id INTEGER PRIMARY KEY, ts, question);
            CREATE TABLE messages(chat_id, msg_id, ts, sender_id, text, reply_to);
            INSERT INTO messages VALUES (-100123, 1, 1, 2, 'Where is the refund form?', NULL);
        """)
        conn.executemany(
            "INSERT INTO messages VALUES (-100123, ?, ?, 2, 'Chatter', NULL)",
            [(n, n) for n in range(2, 26)],
        )
        conn.commit()

    class Classifier:
        def __init__(self):
            self.batches = []

        def complete(self, system, user):
            messages = json.loads(user)
            self.batches.append(messages)
            return json.dumps({
                "questions": [row["msg_id"] for row in messages if row["text"].endswith("?")],
                "answers": [],
            })

    model = Classifier()
    review = HistoryReview(cfg, lambda: model, ReviewIndex)
    replies, _ = run_admin_updates(cfg, review, [
        update(1, "/review 1970-01-01 1970-01-01"),
        update(2, "/review 1970-01-01 1970-01-01 2"),
        update(3, "/review_search 1970-01-01 1970-01-01 1 refund"),
    ])
    assert "No questions on this page." in replies[0]
    assert "Next page · /review 1970-01-01 1970-01-01 2" in replies[0]
    assert "Where is the refund form?" in replies[1] and "Next page" not in replies[1]
    assert "Where is the refund form?" in replies[2]
    assert len(model.batches) == 4
    assert [m["text"] for m in model.batches[-1]] == ["Where is the refund form?"]
