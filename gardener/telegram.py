"""DM each FAQ gap to admins. One gap per message, split under Telegram's cap.

The watch process also long-polls getUpdates. A private message from an admin
runs a dry run and the reply is sent back to that chat.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request

from . import store

log = logging.getLogger("gardener")

# Telegram's hard cap is 4096. The part prefix "(12/12)\n" still fits.
_LIMIT = 4096
_BODY = 4080


def format_message(draft: dict) -> str:
    """Plain text an admin can paste from. No user ids."""
    lines = [
        f"FAQ gap ({draft['label']})",
        draft["heading"],
        f"Nearest: {draft.get('nearest_url') or 'none'}",
        "",
        "Questions:",
    ]
    for question in draft.get("questions") or []:
        lines.append(f"- {question}")
    lines.append("")
    lines.append(draft.get("outline") or "")
    return "\n".join(lines).strip()


def split_message(text: str) -> list[str]:
    """Split on newlines so each part, plus a page prefix, stays within 4096."""
    if len(text) <= _LIMIT:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        if len(rest) <= _BODY:
            parts.append(rest)
            break
        cut = rest.rfind("\n", 0, _BODY)
        if cut < _BODY // 2:
            cut = _BODY
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if len(parts) == 1:
        return parts
    numbered = [f"({i}/{len(parts)})\n{part}" for i, part in enumerate(parts, 1)]
    for part in numbered:
        if len(part) > _LIMIT:
            raise RuntimeError("Telegram message part exceeds 4096 characters")
    return numbered


def send_text(
    token: str,
    chat_id: int,
    text: str,
    *,
    timeout: float = 30,
    opener=urllib.request.urlopen,
) -> None:
    payload = json.dumps({"chat_id": chat_id, "text": text}, ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Telegram send to {chat_id} failed: {e.code} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Telegram send to {chat_id} failed: {e.reason}") from e
    if not isinstance(body, dict) or not body.get("ok"):
        raise RuntimeError(f"Telegram send to {chat_id} was not accepted")


def get_updates(
    token: str,
    *,
    offset: int | None,
    timeout: int,
    opener=urllib.request.urlopen,
) -> list[dict]:
    """Long-poll private messages. `timeout` is Telegram's wait, in seconds."""
    payload: dict = {"timeout": timeout, "allowed_updates": ["message"]}
    if offset is not None:
        payload["offset"] = offset
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/getUpdates",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener(req, timeout=timeout + 10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Telegram getUpdates failed: {e.code} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Telegram getUpdates failed: {e.reason}") from e
    if not isinstance(body, dict) or not body.get("ok") or not isinstance(body.get("result"), list):
        raise RuntimeError("Telegram getUpdates was not accepted")
    return body["result"]


def advance_offset(updates: list[dict], offset: int | None) -> int | None:
    if not updates:
        return offset
    return max(int(item["update_id"]) for item in updates) + 1


def admin_private_chats(updates: list[dict], admin_ids: set[int] | frozenset[int]) -> list[int]:
    """Chat ids for private texts from admins, in arrival order."""
    chats: list[int] = []
    for update in updates:
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        if chat.get("type") != "private":
            continue
        user = (message.get("from") or {}).get("id")
        text = str(message.get("text") or "").strip()
        if not text or user is None or int(user) not in admin_ids:
            continue
        chats.append(int(chat["id"]))
    return chats


def format_decision(question: str, label: str, reason: str) -> str:
    """One question, the decision, and the reason when there is one."""
    lines = [f"Question: {question}", f"Decision: {label}"]
    if reason.strip():
        lines.append(f"Why: {reason.strip()}")
    return "\n".join(lines)


def notify_decisions(
    token: str | None,
    admin_ids: list[int],
    decisions: list[dict],
    *,
    dry_run: bool,
    conn: sqlite3.Connection | None = None,
    opener=urllib.request.urlopen,
) -> None:
    """DM each question's decision. Skip a question whose label was already delivered.

    A dry run prints every decision and does not record it. A label is recorded
    only after at least one admin DM succeeds, so a failed run retries. The same
    question with a new label is sent again.
    """
    for item in decisions:
        text = str(item.get("text") or "")
        label = str(item.get("label") or "")
        key = store.decision_key(text, label)
        if conn is not None and not dry_run and store.reported(conn, key):
            log.info("skip already reported %s %s", label, text[:40])
            continue
        parts = split_message(
            format_decision(text, label, str(item.get("reason") or ""))
        )
        if dry_run:
            for part in parts:
                print(part)
                print("---")
            continue
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        if not admin_ids:
            raise RuntimeError("ADMIN_USER_IDS is empty")
        ok = False
        for admin_id in admin_ids:
            try:
                for part in parts:
                    send_text(token, admin_id, part, opener=opener)
                ok = True
            except RuntimeError:
                log.warning("telegram decision to %s failed", admin_id, exc_info=True)
        if ok and conn is not None:
            store.mark_reported(conn, text, label)


def notify(
    token: str | None,
    admin_ids: list[int],
    drafts: list[dict],
    *,
    dry_run: bool,
    conn: sqlite3.Connection | None,
    opener=urllib.request.urlopen,
) -> int:
    """Send each draft. Return how many were delivered to at least one admin.

    A dry run prints the messages and does not record them. A fingerprint is
    recorded only after at least one admin DM succeeds, so a failed run retries.
    """
    delivered = 0
    for draft in drafts:
        text = format_message(draft)
        parts = split_message(text)
        if dry_run:
            for part in parts:
                print(part)
                print("---")
            delivered += 1
            continue
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        if not admin_ids:
            raise RuntimeError("ADMIN_USER_IDS is empty")
        ok = False
        for admin_id in admin_ids:
            try:
                for part in parts:
                    send_text(token, admin_id, part, opener=opener)
                ok = True
            except RuntimeError:
                log.warning("telegram DM to %s failed", admin_id, exc_info=True)
        if ok:
            if conn is not None:
                store.mark(
                    conn,
                    draft["fingerprint"],
                    draft["heading"],
                    list(draft.get("questions") or []),
                )
            delivered += 1
    return delivered
