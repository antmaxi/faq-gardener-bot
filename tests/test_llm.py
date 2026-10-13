import pytest

from gardener.llm import get_llm

from tests.conftest import settings


def test_cursor_reports_missing_sdk(tmp_path):
    cfg = settings(tmp_path, llm_provider="cursor", cursor_api_key="test-key")
    llm = get_llm(cfg)
    with pytest.raises(RuntimeError, match="cursor-sdk"):
        llm.complete("system", "user")


def test_unknown_provider(tmp_path):
    cfg = settings(tmp_path, llm_provider="nope", answer_model="whatever")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_llm(cfg)


def test_anthropic_requires_a_key(tmp_path):
    cfg = settings(tmp_path, llm_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(RuntimeError, match="Anthropic"):
        get_llm(cfg)


def test_cursor_default_model(tmp_path):
    cfg = settings(tmp_path, llm_provider="cursor", cursor_api_key="k", answer_model=None)
    assert cfg.model_name() == "composer-2.5"
