"""Load questions, grade them against the FAQ, and DM admins.

One pass by default. WATCH_SECONDS keeps the process up and repeats the pass,
so new clusters are sent without a cron trigger. Clusters already delivered
are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

from . import store
from .chat import load_chat
from .config import Settings, load_settings
from .faq_index import huggingface_embedder, index_pages, index_repo, pages_from_dir
from .graph import compile_graph, question_decisions
from .llm import LLM, get_llm
from .questions import load_export, load_sqlite, normalize
from .review import HistoryReview
from .telegram import (
    admin_private_messages,
    advance_offset,
    format_message,
    get_updates,
    notify,
    notify_decisions,
    send_text,
    split_message,
)

log = logging.getLogger("gardener")


def _faq_index(settings, model, faq_dir, faq_pages):
    if faq_pages is None and faq_dir is not None:
        faq_pages = pages_from_dir(faq_dir)
    options = dict(
        embed_model=model, site=settings.faq_site, cache_dir=settings.faq_cache, top_k=settings.top_k,
    )
    if faq_pages is not None:
        return index_pages(faq_pages, **options)
    return index_repo(settings.faq_repo, **options)


def execute(
    settings: Settings,
    *,
    dry_run: bool = False,
    questions_path: Path | None = None,
    since_days: int | None = None,
    faq_dir: Path | None = None,
    faq_pages: dict[str, str] | None = None,
    llm: LLM | None = None,
    embed_model=None,
    opener=None,
    now: float | None = None,
    notify_admins: bool = True,
) -> dict:
    """Run one pass. Returns counts and the drafts that were produced."""
    since = settings.since_days if since_days is None else since_days
    since_ts = int((now if now is not None else time.time()) - since * 86400)
    # DEBUG keeps admin asks in the candidate set. Those ids are still the DM recipients.
    gap_admin_ids: frozenset[int] = frozenset() if settings.debug else settings.admin_ids
    if settings.debug:
        log.info("debug mode: including questions asked by admins")
    if questions_path is not None:
        questions = load_export(
            questions_path,
            since_ts=since_ts,
            admin_ids=gap_admin_ids,
            min_words=settings.min_question_words,
        )
    else:
        questions = load_sqlite(
            settings.answerbot_db,
            since_ts=since_ts,
            admin_ids=gap_admin_ids,
            min_words=settings.min_question_words,
        )
    evidence = {}
    if questions_path is None and settings.faq_chat_ids:
        llm = llm or get_llm(settings)
        archived, evidence = load_chat(
            settings.answerbot_db, settings.faq_chat_ids, since_ts, gap_admin_ids, llm,
            min_words=settings.min_question_words,
        )
        unique = {}
        for question in sorted(questions + archived, key=lambda q: q.ts, reverse=True):
            unique.setdefault(normalize(question.text), question)
        questions = list(unique.values())
    if not questions:
        log.info("no FAQ candidate questions")
        return {"questions": 0, "drafts": [], "delivered": 0, "decisions": []}

    if not dry_run:
        if not settings.telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        if not settings.admin_ids:
            raise RuntimeError("ADMIN_USER_IDS is empty")

    model = embed_model or huggingface_embedder(
        settings.embed_model, settings.hf_token, settings.embed_threads
    )
    index = _faq_index(settings, model, faq_dir, faq_pages)

    completer = llm or get_llm(settings)
    conn = store.connect(settings.gardener_db)
    try:
        graph = compile_graph(
            index=index,
            llm=completer,
            conn=conn,
            min_cluster_size=settings.min_cluster_size,
            cluster_cosine=settings.cluster_cosine,
            top_k=settings.top_k,
        )
        result = graph.invoke(
            {"questions": [{"ts": q.ts, "text": q.text, "sources": evidence.get(normalize(q.text), [])} for q in questions]}
        )
        drafts = list(result.get("drafts") or [])
        decisions = question_decisions(list(result.get("clusters") or []))
        delivered = 0
        if settings.debug and notify_admins:
            notify_decisions(
                settings.telegram_token,
                sorted(settings.admin_ids),
                decisions,
                dry_run=dry_run,
                conn=conn,
                **({} if opener is None else {"opener": opener}),
            )
        if notify_admins:
            delivered = notify(
                settings.telegram_token,
                sorted(settings.admin_ids),
                drafts,
                dry_run=dry_run,
                conn=conn,
                **({} if opener is None else {"opener": opener}),
            )
    finally:
        conn.close()
    return {
        "questions": len(questions),
        "drafts": drafts,
        "delivered": delivered,
        "decisions": decisions,
    }


def _report(summary: dict, *, dry_run: bool) -> int:
    print(
        f"questions={summary['questions']} drafts={len(summary['drafts'])} "
        f"delivered={summary['delivered']}"
    )
    if not dry_run and summary["drafts"] and summary["delivered"] == 0:
        return 1
    return 0


_POLL_MAX = 25


def reply_dry_run(
    settings: Settings,
    chat_id: int,
    *,
    opener=None,
    **execute_kwargs,
) -> dict:
    """Run one pass and reply in `chat_id`. Delivered gaps and decisions are recorded."""
    if not settings.telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    send_kwargs = {} if opener is None else {"opener": opener}
    send_text(settings.telegram_token, chat_id, "Checking for FAQ gaps…", **send_kwargs)
    summary = execute(
        settings,
        dry_run=True,
        notify_admins=False,
        opener=opener,
        **execute_kwargs,
    )
    conn = store.connect(settings.gardener_db)
    try:
        if settings.debug:
            notify_decisions(
                settings.telegram_token,
                [chat_id],
                list(summary.get("decisions") or []),
                dry_run=False,
                conn=conn,
                **send_kwargs,
            )
        drafts = list(summary.get("drafts") or [])
        if not drafts:
            send_text(
                settings.telegram_token,
                chat_id,
                f"Dry run: questions={summary['questions']} drafts=0",
                **send_kwargs,
            )
            return summary
        for draft in drafts:
            try:
                for part in split_message(format_message(draft)):
                    send_text(settings.telegram_token, chat_id, part, **send_kwargs)
            except RuntimeError:
                log.warning("dry-run outline to %s failed", chat_id, exc_info=True)
                continue
            store.mark(
                conn,
                draft["fingerprint"],
                draft["heading"],
                list(draft.get("questions") or []),
            )
    finally:
        conn.close()
    return summary


def pause_for_admin(
    settings: Settings,
    stop: threading.Event,
    seconds: float,
    *,
    offset: int | None,
    run_preview,
    run_review=None,
    fetch=get_updates,
) -> int | None:
    """Wait out `seconds`, and run a dry-run reply when an admin writes."""
    deadline = time.monotonic() + seconds
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return offset
        timeout = max(1, min(_POLL_MAX, int(remaining)))
        try:
            updates = fetch(settings.telegram_token, offset=offset, timeout=timeout)
        except Exception:
            log.exception("telegram poll failed")
            if stop.wait(min(5.0, remaining)):
                return offset
            continue
        offset = advance_offset(updates, offset)
        previewed = set()
        for chat_id, text in admin_private_messages(updates, settings.admin_ids):
            try:
                if run_review is not None and run_review(chat_id, text):
                    continue
                if chat_id not in previewed:
                    previewed.add(chat_id)
                    run_preview(chat_id)
            except Exception:
                log.exception("admin request for %s failed", chat_id)
                try:
                    send_text(settings.telegram_token, chat_id, "Admin request failed.")
                except Exception:
                    log.warning("could not report the dry-run failure to %s", chat_id, exc_info=True)
    return offset


def watch(
    settings: Settings,
    *,
    interval: int,
    stop: threading.Event,
    dry_run: bool = False,
    questions_path: Path | None = None,
    since_days: int | None = None,
    faq_dir: Path | None = None,
    faq_pages: dict[str, str] | None = None,
    llm: LLM | None = None,
    embed_model=None,
    opener=None,
    now: float | None = None,
    listen: bool = False,
) -> int:
    """Repeat passes until `stop` is set. A failed pass is logged and retried.

    With `listen`, a private message from an admin during the pause runs a dry
    run and replies in that chat. The preview is not recorded.
    """
    if interval <= 0:
        raise ValueError("watch interval must be positive")
    held_embed = embed_model
    held_llm = llm
    offset: int | None = None
    hear = listen and bool(settings.telegram_token) and bool(settings.admin_ids)
    review = HistoryReview(
        settings, lambda: held_llm, lambda: _faq_index(settings, held_embed, faq_dir, faq_pages),
    )
    log.info("watching every %s seconds", interval)
    if hear:
        log.info("private messages from admins run a dry run and are not saved")
    while not stop.is_set():
        try:
            if held_embed is None:
                held_embed = huggingface_embedder(
                    settings.embed_model, settings.hf_token, settings.embed_threads
                )
            if held_llm is None:
                held_llm = get_llm(settings)
            summary = execute(
                settings,
                dry_run=dry_run,
                questions_path=questions_path,
                since_days=since_days,
                faq_dir=faq_dir,
                faq_pages=faq_pages,
                llm=held_llm,
                embed_model=held_embed,
                opener=opener,
                now=now,
            )
            if _report(summary, dry_run=dry_run) != 0:
                log.error("watch pass produced drafts but delivered none")
        except Exception:
            log.exception("watch pass failed")
        if hear:
            def run_review(chat_id: int, text: str) -> bool:
                send_kwargs = {} if opener is None else {"opener": opener}
                return review.handle(
                    chat_id, text,
                    lambda message: send_text(settings.telegram_token, chat_id, message, **send_kwargs),
                )

            def run_preview(chat_id: int) -> None:
                nonlocal held_embed, held_llm
                if held_embed is None:
                    held_embed = huggingface_embedder(
                        settings.embed_model, settings.hf_token, settings.embed_threads
                    )
                if held_llm is None:
                    held_llm = get_llm(settings)
                reply_dry_run(
                    settings,
                    chat_id,
                    opener=opener,
                    questions_path=questions_path,
                    since_days=since_days,
                    faq_dir=faq_dir,
                    faq_pages=faq_pages,
                    llm=held_llm,
                    embed_model=held_embed,
                )

            offset = pause_for_admin(
                settings,
                stop,
                interval,
                offset=offset,
                run_preview=run_preview,
                run_review=run_review,
            )
            if stop.is_set():
                break
        elif stop.is_set() or stop.wait(interval):
            break
    log.info("watch stopped")
    return 0


def _install_stop(stop: threading.Event) -> None:
    def _handle(signum, _frame) -> None:
        log.info("received signal %s; stopping after this pass", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="gardener", description="Draft FAQ outlines from unanswered questions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="cluster recent gaps and DM admins")
    run.add_argument("--dry-run", action="store_true", help="print drafts, do not send or record them")
    run.add_argument("--questions", type=Path, help="JSON export instead of the live answer-bot database")
    run.add_argument("--since-days", type=int, default=None)
    run.add_argument("--faq-dir", type=Path, help="local docs/ and inbox/ markdown instead of GitHub")
    run.add_argument(
        "--watch-seconds",
        type=int,
        default=None,
        help="seconds between passes; 0 runs once (default: WATCH_SECONDS)",
    )
    args = parser.parse_args(argv)
    if args.watch_seconds is not None and args.watch_seconds < 0:
        log.error("--watch-seconds must be >= 0")
        return 2
    settings = load_settings()
    interval = settings.watch_seconds if args.watch_seconds is None else args.watch_seconds
    if interval > 0 and args.dry_run:
        log.error("dry-run is a single pass; leave WATCH_SECONDS unset")
        return 2
    if interval > 0:
        stop = threading.Event()
        _install_stop(stop)
        return watch(
            settings,
            interval=interval,
            stop=stop,
            dry_run=args.dry_run,
            questions_path=args.questions,
            since_days=args.since_days,
            faq_dir=args.faq_dir,
            listen=True,
        )
    summary = execute(
        settings,
        dry_run=args.dry_run,
        questions_path=args.questions,
        since_days=args.since_days,
        faq_dir=args.faq_dir,
    )
    return _report(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
