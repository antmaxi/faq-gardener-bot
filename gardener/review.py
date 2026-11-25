"""Admin-only Telegram commands for reviewing the archive."""

from __future__ import annotations

import json
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone

from . import store
from .history import assess, load_history_page, load_item
from .questions import normalize
from .telegram import message_reference, split_message

_PAGE_SIZE = 10
_ARCHIVE_PAGE_SIZE = 24

_HELP = """Review past questions (dates are inclusive, UTC)
/review START END [PAGE]
/review_search START END PAGE WORDS
/review_open ID
/review_save ID
/review_saved [PAGE]
/review_draft ID
/review_edit ID ANSWER

Use YYYY-MM-DD for dates. For search, use 1 for the first page.
Open a question before saving its suggestion."""


def _page(raw: str = "1") -> int:
    page = int(raw)
    if page < 1:
        raise ValueError("Page numbers start at 1.")
    return page


def _range(start: str, end: str) -> tuple[int, int]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first.isoformat() != start or last.isoformat() != end or first > last:
        raise ValueError("Enter an ordered range using YYYY-MM-DD dates.")
    return (
        int(datetime.combine(first, time.min, timezone.utc).timestamp()),
        int(datetime.combine(last + timedelta(days=1), time.min, timezone.utc).timestamp()),
    )


def _listing(rows: list[str], page: int, next_command: str) -> str:
    start = (page - 1) * _PAGE_SIZE
    selected = rows[start:start + _PAGE_SIZE]
    if not selected:
        return "No questions on this page."
    if len(rows) > start + _PAGE_SIZE:
        selected.append(f"Next page · {next_command}")
    return "\n\n".join(selected)


def _format_card(card: dict, *, saved: bool = False) -> str:
    chat_status = {
        "none": "No chat answer found",
        "answered": "Answer found",
        "partial": "Partial answer found",
        "conflicting": "Conflicting answers found",
    }
    lines = [
        f"{card['date']} · {card['key']}", card["question"], card["origin"],
        "", f"Chat · {chat_status[card['chat_status']]}",
        f"FAQ · {card['faq_label']} · {card['faq_reason']}",
    ]
    if saved:
        lines.insert(0, "Saved FAQ suggestion")
    if card["faq_url"]:
        lines.append(card["faq_url"])
    if card["bot_response"]:
        lines.extend(["", "Answer bot response (context)", card["bot_response"]])
    for source in card["sources"]:
        link = message_reference(source["chat_id"], source["msg_id"])
        lines.extend(["", f"{source['kind']} · {link}", source["text"]])
    if card["suggestion"]:
        lines.extend(["", "Saved answer" if saved else "Suggested answer", card["suggestion"]])
    if saved:
        lines.extend(["", f"Edit · /review_edit {card['key']} ANSWER"])
    else:
        lines.extend(["", f"Save for FAQ · /review_save {card['key']}"])
    return "\n".join(lines)


class HistoryReview:
    def __init__(self, settings, get_llm, get_index):
        self.settings = settings
        self.get_llm = get_llm
        self.get_index = get_index
        self.open_cards = {}

    def handle(self, chat_id: int, text: str, send) -> bool:
        command, _, args = text.partition(" ")
        command = command.split("@", 1)[0]
        if command != "/review" and not command.startswith("/review_"):
            return False
        try:
            response = self._command(chat_id, command, args.strip())
        except (ValueError, OverflowError):
            response = "Unable to complete that request. Check the dates, page, or question ID and try again.\n\n" + _HELP
        for part in split_message(response):
            send(part)
        return True

    def _command(self, chat_id: int, command: str, args: str) -> str:
        if command in ("/review", "/review_search") and args:
            return self._browse(args, search=command == "/review_search")
        with closing(store.connect(self.settings.gardener_db)) as conn:
            if command == "/review_draft" and args:
                saved = store.get_suggestion(conn, args)
                return _format_card(saved, saved=True) if saved is not None else "Saved suggestion not found."
            if command == "/review_open" and args:
                item, sources = load_item(self.settings, args)
                card = assess(item, sources, self.get_llm(), self.get_index(), self.settings.top_k)
                self.open_cards[chat_id] = card
                return _format_card(card)
            if command == "/review_save" and args:
                if store.get_suggestion(conn, args) is not None:
                    return f"Already saved. Open with /review_draft {args}"
                card = self.open_cards.get(chat_id)
                if card is None or card["key"] != args:
                    return f"Open this question first with /review_open {args}"
                store.save_suggestion(conn, card)
                return f"Saved for FAQ. Open with /review_draft {args}"
            if command == "/review_saved":
                page = _page(args or "1")
                rows = [
                    f"{_preview(card['question'])}\n/review_draft {card['key']}"
                    for card in store.list_suggestions(conn, limit=_PAGE_SIZE + 1, offset=(page - 1) * _PAGE_SIZE)
                ]
                return _listing(rows, 1, f"/review_saved {page + 1}")
            if command == "/review_edit":
                key, _, answer = args.partition(" ")
                if answer.strip() and store.edit_suggestion(conn, key, answer.strip()):
                    return f"Updated. Open with /review_draft {key}"
        return _HELP

    def _browse(self, args: str, *, search: bool) -> str:
        fields = args.split(maxsplit=3)
        if (search and len(fields) != 4) or (not search and len(fields) not in (2, 3)):
            return _HELP
        start, end = fields[:2]
        page = _page(fields[2] if len(fields) > 2 else "1")
        since, until = _range(start, end)
        terms = normalize(fields[3]).split() if search else []
        items, more = load_history_page(
            self.settings, since, until, self.get_llm(),
            terms=terms, page=page, page_size=_ARCHIVE_PAGE_SIZE,
        )
        rows = [
            f"{item.date} · {_preview(item.text)}\n/review_open {item.key}"
            for item in items
        ]
        next_command = (
            f"/review_search {start} {end} {page + 1} {fields[3]}" if search
            else f"/review {start} {end} {page + 1}"
        )
        if not rows:
            rows.append("No questions on this page.")
        if more:
            rows.append(f"Next page · {next_command}")
        return "\n\n".join(rows)


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text[:157] + "…" if len(text) > 158 else text
