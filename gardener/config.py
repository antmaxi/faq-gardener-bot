"""Settings read from the environment or a .env file in the working directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "cursor": "composer-2.5",
}


def _admins(raw: str | None) -> frozenset[int]:
    if not raw:
        return frozenset()
    return frozenset(int(part) for part in raw.replace(",", " ").split() if part.strip())


def _optional(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _flag(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _seconds(raw: str | None) -> int:
    """Empty or unset is a single pass. A positive value is the pause between passes."""
    if raw is None or not raw.strip():
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError("WATCH_SECONDS must be >= 0")
    return value


@dataclass(frozen=True)
class Settings:
    answerbot_db: Path
    gardener_db: Path
    faq_repo: str
    faq_site: str
    embed_model: str
    since_days: int
    min_cluster_size: int
    min_question_words: int
    cluster_cosine: float
    admin_ids: frozenset[int]
    debug: bool
    telegram_token: str | None
    llm_provider: str
    anthropic_api_key: str | None
    openai_api_key: str | None
    openai_base_url: str
    cursor_api_key: str | None
    answer_model: str | None
    llm_timeout: float
    faq_cache: Path
    hf_token: str | None
    embed_threads: int
    top_k: int
    watch_seconds: int

    def model_name(self) -> str:
        if self.answer_model:
            return self.answer_model
        try:
            return DEFAULT_MODELS[self.llm_provider]
        except KeyError as e:
            raise ValueError(f"unknown LLM_PROVIDER: {self.llm_provider!r}") from e


def load_settings() -> Settings:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    return Settings(
        answerbot_db=Path(
            os.getenv("ANSWERBOT_DB", "../answer-chat-history-bot/data/answerbot.db")
        ),
        gardener_db=Path(os.getenv("GARDENER_DB", "data/gardener.db")),
        faq_repo=os.getenv("FAQ_REPO", "ru-ch/faq").strip(),
        faq_site=os.getenv("FAQ_SITE", "https://ru-ch.github.io/faq").rstrip("/"),
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small").strip(),
        since_days=int(os.getenv("SINCE_DAYS", "14")),
        min_cluster_size=int(os.getenv("MIN_CLUSTER_SIZE", "2")),
        min_question_words=int(os.getenv("MIN_QUESTION_WORDS", "3")),
        cluster_cosine=float(os.getenv("CLUSTER_COSINE", "0.82")),
        admin_ids=_admins(os.getenv("ADMIN_USER_IDS")),
        debug=_flag(os.getenv("DEBUG")),
        telegram_token=_optional(os.getenv("TELEGRAM_BOT_TOKEN")),
        llm_provider=provider,
        anthropic_api_key=_optional(os.getenv("ANTHROPIC_API_KEY")),
        openai_api_key=_optional(os.getenv("OPENAI_API_KEY")),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        cursor_api_key=_optional(os.getenv("CURSOR_API_KEY")),
        answer_model=_optional(os.getenv("ANSWER_MODEL")),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        faq_cache=Path(os.getenv("FAQ_CACHE", "data/faq_index")),
        hf_token=_optional(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")),
        embed_threads=int(os.getenv("EMBED_THREADS", "1")),
        top_k=int(os.getenv("FAQ_TOP_K", "4")),
        watch_seconds=_seconds(os.getenv("WATCH_SECONDS")),
    )
