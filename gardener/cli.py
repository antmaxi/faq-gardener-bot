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
from .config import Settings, load_settings
from .faq_index import huggingface_embedder, index_pages, index_repo, pages_from_dir
from .graph import compile_graph
from .llm import LLM, get_llm
from .questions import load_export, load_sqlite
from .telegram import (
    admin_private_chats,
    advance_offset,
    format_message,
    get_updates,
    notify,
    send_text,
    split_message,
)

log = logging.getLogger("gardener")


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
    if not questions:
        log.info("no FAQ candidate questions")
        return {"questions": 0, "drafts": [], "delivered": 0}

    if not dry_run:
        if not settings.telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
        if not settings.admin_ids:
            raise RuntimeError("ADMIN_USER_IDS is empty")

    model = embed_model or huggingface_embedder(
        settings.embed_model, settings.hf_token, settings.embed_threads
    )
    if faq_pages is None and faq_dir is not None:
        faq_pages = pages_from_dir(faq_dir)
    if faq_pages is not None:
        index = index_pages(
            faq_pages,
            embed_model=model,
            site=settings.faq_site,
            cache_dir=settings.faq_cache,
            top_k=settings.top_k,
        )
    else:
        index = index_repo(
            settings.faq_repo,
            embed_model=model,
            site=settings.faq_site,
            cache_dir=settings.faq_cache,
            top_k=settings.top_k,
        )

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
            {"questions": [{"ts": q.ts, "text": q.text} for q in questions]}
        )
        drafts = list(result.get("drafts") or [])
        delivered = 0
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
    return {"questions": len(questions), "drafts": drafts, "delivered": delivered}


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
    """Run one dry pass and send the outlines to `chat_id`. Nothing is recorded."""
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
        for part in split_message(format_message(draft)):
            send_text(settings.telegram_token, chat_id, part, **send_kwargs)
    return summary


def pause_for_admin(
    settings: Settings,
    stop: threading.Event,
    seconds: float,
    *,
    offset: int | None,
    run_preview,
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
        for chat_id in dict.fromkeys(admin_private_chats(updates, settings.admin_ids)):
            try:
                run_preview(chat_id)
            except Exception:
                log.exception("dry run for %s failed", chat_id)
                try:
                    send_text(settings.telegram_token, chat_id, "Dry run failed.")
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
