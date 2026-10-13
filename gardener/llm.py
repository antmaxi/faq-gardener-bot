"""One completion helper for Anthropic, OpenAI-compatible endpoints, and Cursor.

Cursor matches the answer bot: a local agent with tools disabled, billed to the
Cursor subscription. The SDK is imported only when that provider is used.
"""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.request
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def _openai_message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts).strip()
    return ""


class AnthropicLLM:
    def __init__(self, api_key: str | None, model: str, timeout: float):
        import anthropic

        if not api_key:
            raise RuntimeError("Anthropic API key is not set")
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise RuntimeError("Anthropic returned no text")
        return text


class OpenAICompatLLM:
    """Chat Completions client. Point OPENAI_BASE_URL at Groq, OpenRouter, or Ollama."""

    def __init__(self, api_key: str | None, model: str, base_url: str, timeout: float):
        if not api_key:
            raise RuntimeError("OpenAI API key is not set")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "faq-gardener-bot/0.1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"OpenAI-compatible request failed: {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI-compatible request failed: {e.reason}") from e
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("OpenAI-compatible response had no text") from e
        text = _openai_message_text(content)
        if not text:
            raise RuntimeError("OpenAI-compatible response had no text")
        return text


def _cursor_sdk():
    """Lazy import so other providers do not need cursor-sdk installed."""
    try:
        from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
    except ImportError as e:
        raise RuntimeError(
            "Cursor provider requires the cursor-sdk package. "
            "Install with: pip install 'faq-gardener[cursor]'"
        ) from e
    return Agent, AgentOptions, CursorAgentError, LocalAgentOptions


class CursorLLM:
    """Cursor SDK agent billed to the Cursor subscription, not a chat-completions API.

    Concatenates system + user into one prompt (the SDK has no system role) and
    offers no tools so the local agent can only return text. There is no HTTP
    timeout cap; the local agent run ends when Cursor does.
    """

    def __init__(self, api_key: str | None, model: str, cwd: str | None = None):
        self.api_key = api_key
        self.model = model
        self.cwd = cwd

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("Cursor API key is not set")
        Agent, AgentOptions, CursorAgentError, LocalAgentOptions = _cursor_sdk()
        prompt = f"{system.strip()}\n\n{user.strip()}"

        def run(cwd: str):
            try:
                return Agent.prompt(
                    prompt,
                    AgentOptions(
                        api_key=self.api_key,
                        model=self.model,
                        tools=[],
                        local=LocalAgentOptions(cwd=cwd),
                    ),
                )
            except CursorAgentError as e:
                raise RuntimeError(f"Cursor: {e.message}") from e

        if self.cwd:
            result = run(self.cwd)
        else:
            with tempfile.TemporaryDirectory(prefix="faq-gardener-cursor-") as cwd:
                result = run(cwd)

        if result.status != "finished":
            run_id = getattr(result, "id", "") or ""
            suffix = f" ({run_id})" if run_id else ""
            raise RuntimeError(f"Cursor run {result.status}{suffix}")
        text = (result.result or "").strip()
        if not text:
            raise RuntimeError("Cursor returned no text")
        return text


def get_llm(settings: Settings) -> LLM:
    model = settings.model_name()
    provider = settings.llm_provider
    if provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, model, settings.llm_timeout)
    if provider == "openai":
        return OpenAICompatLLM(
            settings.openai_api_key, model, settings.openai_base_url, settings.llm_timeout
        )
    if provider == "cursor":
        return CursorLLM(settings.cursor_api_key, model)
    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")
