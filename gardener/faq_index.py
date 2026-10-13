"""Fetch ru-ch/faq, chunk it by heading, and retrieve with LlamaIndex.

Markdown one directory under docs/ and inbox/ is the same slice the answer bot
indexes. This module does not import that package. Embeddings for the live
index are multilingual-e5-small; callers can pass another embed model in tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("gardener")

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_KEEP_DIRS = ("docs", "inbox")
_USER_AGENT = "faq-gardener-bot"
_HTTP_TIMEOUT = 60.0
_MAX_CHARS = 1500


@dataclass(frozen=True)
class Chunk:
    path: str
    heading: str
    text: str
    url: str


@dataclass(frozen=True)
class FaqHit:
    heading: str
    text: str
    url: str
    score: float


def page_url(path: str, site: str) -> str:
    """docs/Быт.md → https://ru-ch.github.io/faq/docs/Быт.html"""
    base = site.rstrip("/")
    rel = path[:-3] + ".html" if path.endswith(".md") else path
    return f"{base}/{rel}"


def page_title(path: str) -> str:
    return Path(path).stem


def chunk_markdown(markdown: str, *, title: str, max_chars: int = _MAX_CHARS) -> list[tuple[str, str]]:
    """Split on # / ## / ###. Heading trail is prepended so a subsection still
    carries its parent titles (e.g. recycling → Zürich). Returns (heading, text)."""
    levels = {1: "", 2: "", 3: ""}
    buf: list[str] = []
    chunks: list[tuple[str, str]] = []

    def trail() -> str:
        return " · ".join(levels[i] for i in (1, 2, 3) if levels[i])

    def speakers(heading: str) -> str:
        if not heading:
            return title
        if heading == title or heading.startswith(f"{title} · "):
            return heading
        return f"{title} · {heading}"

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        heading = trail()
        if not body and not heading:
            return
        label = speakers(heading)
        text = f"{label}\n\n{body}".strip() if body else label
        if not text.strip():
            return
        for piece in _split_chars(text, max_chars):
            chunks.append((label, piece))

    for line in markdown.replace("\r\n", "\n").split("\n"):
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            levels[level] = match.group(2).strip()
            for lv in range(level + 1, 4):
                levels[lv] = ""
            continue
        buf.append(line)
    flush()
    return chunks


def _split_chars(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    parts = re.split(r"\n\s*\n", text)
    out: list[str] = []
    cur = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if not cur:
            cur = piece
            continue
        if len(cur) + 2 + len(piece) <= max_chars:
            cur = f"{cur}\n\n{piece}"
            continue
        out.append(cur)
        cur = piece
    if cur:
        out.append(cur)
    hard: list[str] = []
    for block in out:
        if len(block) <= max_chars:
            hard.append(block)
            continue
        for i in range(0, len(block), max_chars):
            hard.append(block[i : i + max_chars])
    return hard


def faq_rel(name: str) -> str | None:
    """faq-master/docs/Быт.md → docs/Быт.md. One directory deep only."""
    parts = Path(name).parts
    if len(parts) < 3:
        return None
    rel_parts = parts[1:]
    if rel_parts[0] not in _KEEP_DIRS:
        return None
    if len(rel_parts) != 2 or not rel_parts[1].endswith(".md"):
        return None
    return "/".join(rel_parts)


def chunks_from_pages(pages: dict[str, str], *, site: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(pages):
        title = page_title(path)
        url = page_url(path, site)
        for heading, text in chunk_markdown(pages[path], title=title):
            chunks.append(Chunk(path=path, heading=heading, text=text, url=url))
    return chunks


def content_sha(pages: dict[str, str]) -> str:
    payload = "\n".join(f"{path}\n{pages[path]}" for path in sorted(pages))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def pages_from_dir(root: Path) -> dict[str, str]:
    """Local stand-in for the GitHub tree: docs/*.md and inbox/*.md only."""
    pages: dict[str, str] = {}
    for folder in _KEEP_DIRS:
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            if path.parent != directory:
                continue
            pages[f"{folder}/{path.name}"] = path.read_text(encoding="utf-8")
    return pages


def _http_get(url: str, timeout: float = _HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/vnd.github+json, application/octet-stream, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def remote_sha(repo: str) -> str | None:
    url = f"https://api.github.com/repos/{repo}/commits/master"
    try:
        payload = json.loads(_http_get(url, timeout=20).decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        log.warning("faq: could not read remote SHA for %s", repo, exc_info=True)
        return None
    sha = payload.get("sha")
    return str(sha) if sha else None


def _from_tarball(repo: str) -> dict[str, str]:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/master"
    blob = _http_get(url)
    pages: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel = faq_rel(member.name)
            if rel is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            pages[rel] = extracted.read().decode("utf-8", errors="replace")
    return pages


def _from_raw_tree(repo: str) -> dict[str, str]:
    url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
    payload = json.loads(_http_get(url).decode("utf-8"))
    pages: dict[str, str] = {}
    for entry in payload.get("tree") or []:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path") or ""
        parts = path.split("/")
        if len(parts) != 2 or parts[0] not in _KEEP_DIRS or not parts[1].endswith(".md"):
            continue
        raw = f"https://raw.githubusercontent.com/{repo}/master/{path}"
        pages[path] = _http_get(raw).decode("utf-8", errors="replace")
    return pages


def fetch_repo(repo: str) -> tuple[str, dict[str, str]]:
    """Return (commit sha, {path: markdown}). Falls back from the tarball to raw files."""
    sha = remote_sha(repo)
    try:
        pages = _from_tarball(repo)
    except (urllib.error.URLError, TimeoutError, tarfile.TarError, OSError):
        log.warning("faq: tarball failed, falling back to raw files", exc_info=True)
        pages = _from_raw_tree(repo)
    if not sha:
        sha = content_sha(pages)
    return sha, pages


class FaqIndex:
    """LlamaIndex vector index plus query embeddings for clustering."""

    def __init__(self, index, embed_model, top_k: int):
        self.index = index
        self.embed_model = embed_model
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[FaqHit]:
        if self.index is None or not query.strip():
            return []
        nodes = self.index.as_retriever(similarity_top_k=k or self.top_k).retrieve(query)
        hits: list[FaqHit] = []
        for node in nodes:
            meta = node.node.metadata or {}
            hits.append(
                FaqHit(
                    heading=str(meta.get("heading") or ""),
                    text=node.node.get_content(),
                    url=str(meta.get("url") or ""),
                    score=float(node.score or 0.0),
                )
            )
        return hits

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = [self.embed_model.get_query_embedding(text) for text in texts]
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        return arr / norms


def build_index(chunks: list[Chunk], embed_model, top_k: int) -> FaqIndex:
    if not chunks:
        return FaqIndex(None, embed_model, top_k)
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import TextNode

    nodes = [
        TextNode(
            text=chunk.text,
            metadata={"path": chunk.path, "heading": chunk.heading, "url": chunk.url},
            excluded_embed_metadata_keys=["path", "url"],
            excluded_llm_metadata_keys=["path", "url"],
        )
        for chunk in chunks
    ]
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=False)
    return FaqIndex(index, embed_model, top_k)


def _load_persisted(persist_dir: Path, embed_model, top_k: int) -> FaqIndex:
    from llama_index.core import StorageContext, load_index_from_storage

    ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(ctx, embed_model=embed_model)
    return FaqIndex(index, embed_model, top_k)


def cached_index(
    chunks: list[Chunk],
    sha: str,
    embed_model,
    cache_dir: Path | None,
    top_k: int,
) -> FaqIndex:
    """Rebuild when `sha` changes. `cache_dir` None keeps the index in memory."""
    sha_path = cache_dir / "sha" if cache_dir is not None else None
    persist = cache_dir / "storage" if cache_dir is not None else None
    if (
        sha_path is not None
        and persist is not None
        and sha_path.is_file()
        and persist.is_dir()
        and sha_path.read_text(encoding="utf-8").strip() == sha
    ):
        try:
            return _load_persisted(persist, embed_model, top_k)
        except Exception:
            log.warning("faq cache load failed; rebuilding", exc_info=True)
    index = build_index(chunks, embed_model, top_k)
    if persist is not None and index.index is not None:
        if persist.exists():
            shutil.rmtree(persist)
        persist.parent.mkdir(parents=True, exist_ok=True)
        index.index.storage_context.persist(persist_dir=str(persist))
        sha_path.write_text(sha, encoding="utf-8")
    return index


def index_pages(
    pages: dict[str, str],
    *,
    embed_model,
    site: str,
    cache_dir: Path | None,
    top_k: int,
) -> FaqIndex:
    if not pages:
        raise RuntimeError("FAQ pages are empty")
    chunks = chunks_from_pages(pages, site=site)
    return cached_index(chunks, content_sha(pages), embed_model, cache_dir, top_k)


def index_repo(
    repo: str,
    *,
    embed_model,
    site: str,
    cache_dir: Path | None,
    top_k: int,
) -> FaqIndex:
    """Fetch the wiki. On network failure, use the last cached index if there is one."""
    persist = cache_dir / "storage" if cache_dir is not None else None
    sha_path = cache_dir / "sha" if cache_dir is not None else None
    try:
        sha, pages = fetch_repo(repo)
    except Exception:
        log.warning("faq fetch failed", exc_info=True)
        if persist is not None and sha_path is not None and persist.is_dir() and sha_path.is_file():
            return _load_persisted(persist, embed_model, top_k)
        raise
    if not pages:
        raise RuntimeError(f"no markdown pages in {repo}")
    chunks = chunks_from_pages(pages, site=site)
    return cached_index(chunks, sha, embed_model, cache_dir, top_k)


def huggingface_kwargs(model_name: str, token: str | None) -> dict:
    """Arguments for LlamaIndex HuggingFaceEmbedding.

    e5 models need asymmetric prefixes. llama-index-embeddings-huggingface 0.8
    does not add them from the model name; SentenceTransformer applies the
    prompt strings as prefixes. `token` is forwarded to SentenceTransformer.
    """
    kwargs: dict = {
        "model_name": model_name,
        "normalize": True,
        "embed_batch_size": 64,
        "device": "cpu",
    }
    if "e5" in model_name.lower():
        kwargs["query_instruction"] = "query: "
        kwargs["text_instruction"] = "passage: "
    if token:
        kwargs["token"] = token
    return kwargs


def huggingface_embedder(model_name: str, token: str | None, threads: int):
    """Local embedder. Imported lazily so tests can skip the Hugging Face stack."""
    nthreads = max(1, threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(nthreads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(**huggingface_kwargs(model_name, token))
