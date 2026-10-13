from pathlib import Path

from gardener.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        answerbot_db=tmp_path / "missing.db",
        gardener_db=tmp_path / "gardener.db",
        faq_repo="ru-ch/faq",
        faq_site="https://ru-ch.github.io/faq",
        embed_model="intfloat/multilingual-e5-small",
        since_days=14,
        min_cluster_size=2,
        min_question_words=3,
        cluster_cosine=0.82,
        admin_ids=frozenset({1}),
        telegram_token="test-token",
        llm_provider="anthropic",
        anthropic_api_key=None,
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        cursor_api_key=None,
        answer_model=None,
        llm_timeout=30,
        faq_cache=tmp_path / "faq_index",
        hf_token=None,
        embed_threads=1,
        top_k=4,
    )
    base.update(overrides)
    return Settings(**base)
