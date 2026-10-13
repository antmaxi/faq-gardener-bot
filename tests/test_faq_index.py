import pytest

from gardener.faq_index import (
    chunk_markdown,
    chunks_from_pages,
    faq_rel,
    huggingface_kwargs,
    index_pages,
    pages_from_dir,
)
from tests.conftest import FIXTURES
from tests.keyword_embed import KeywordEmbedding


def test_heading_trail_keeps_the_parent():
    chunks = chunk_markdown(
        "# Быт\n\nintro\n\n## Zürich\n\nмусор\n",
        title="Быт",
    )
    headings = [heading for heading, _ in chunks]
    assert "Быт · Zürich" in headings


def test_faq_rel_is_one_directory_deep():
    assert faq_rel("faq-master/docs/Быт.md") == "docs/Быт.md"
    assert faq_rel("faq-master/inbox/note.md") == "inbox/note.md"
    assert faq_rel("faq-master/docs/nested/skip.md") is None
    assert faq_rel("faq-master/README.md") is None


def test_pages_from_dir_skips_nested_markdown():
    pages = pages_from_dir(FIXTURES / "faq")
    assert set(pages) == {"docs/Страховка.md", "inbox/note.md"}


def test_retrieve_prefers_the_matching_page(tmp_path):
    pages = pages_from_dir(FIXTURES / "faq")
    index = index_pages(
        pages,
        embed_model=KeywordEmbedding(),
        site="https://ru-ch.github.io/faq",
        cache_dir=tmp_path / "cache",
        top_k=2,
    )
    hits = index.retrieve("как оформить семейную страховку")
    assert hits
    assert hits[0].url == "https://ru-ch.github.io/faq/docs/Страховка.html"
    assert "страхов" in hits[0].text.lower()


def test_cache_is_reused(tmp_path, monkeypatch):
    pages = {"docs/Страховка.md": "# Страховка\n\nсемейную страховку"}
    cache = tmp_path / "cache"
    embed = KeywordEmbedding()
    index_pages(
        pages,
        embed_model=embed,
        site="https://ru-ch.github.io/faq",
        cache_dir=cache,
        top_k=2,
    )
    import gardener.faq_index as faq_index

    def boom(*_args, **_kwargs):
        raise AssertionError("index was rebuilt")

    monkeypatch.setattr(faq_index, "build_index", boom)
    again = faq_index.index_pages(
        pages,
        embed_model=embed,
        site="https://ru-ch.github.io/faq",
        cache_dir=cache,
        top_k=2,
    )
    hits = again.retrieve("семейную страховку")
    assert hits[0].url.endswith("/docs/Страховка.html")


def test_e5_prefixes_are_explicit():
    kwargs = huggingface_kwargs("intfloat/multilingual-e5-small", "hf_token")
    assert kwargs["query_instruction"] == "query: "
    assert kwargs["text_instruction"] == "passage: "
    assert kwargs["token"] == "hf_token"
    assert kwargs["device"] == "cpu"
    other = huggingface_kwargs("BAAI/bge-small-en-v1.5", None)
    assert "query_instruction" not in other
    assert "token" not in other


def test_empty_pages_are_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        index_pages(
            {},
            embed_model=KeywordEmbedding(),
            site="https://ru-ch.github.io/faq",
            cache_dir=tmp_path,
            top_k=1,
        )
    assert chunks_from_pages({}, site="https://example.test") == []
