"""One-shot run: load questions, grade them against the FAQ, DM admins."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from . import store
from .config import Settings, load_settings
from .faq_index import huggingface_embedder, index_pages, index_repo, pages_from_dir
from .graph import compile_graph
from .llm import LLM, get_llm
from .questions import load_export, load_sqlite
from .telegram import notify

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
) -> dict:
    """Run one pass. Returns counts and the drafts that were produced."""
    since = settings.since_days if since_days is None else since_days
    since_ts = int((now if now is not None else time.time()) - since * 86400)
    if questions_path is not None:
        questions = load_export(
            questions_path,
            since_ts=since_ts,
            admin_ids=settings.admin_ids,
            min_words=settings.min_question_words,
        )
    else:
        questions = load_sqlite(
            settings.answerbot_db,
            since_ts=since_ts,
            admin_ids=settings.admin_ids,
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
    args = parser.parse_args(argv)
    summary = execute(
        load_settings(),
        dry_run=args.dry_run,
        questions_path=args.questions,
        since_days=args.since_days,
        faq_dir=args.faq_dir,
    )
    print(
        f"questions={summary['questions']} drafts={len(summary['drafts'])} "
        f"delivered={summary['delivered']}"
    )
    if not args.dry_run and summary["drafts"] and summary["delivered"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
