# faq-gardener-bot

Reads questions the [answer-chat-history-bot](https://github.com/antmaxi/answer-chat-history-bot) could not ground in the [ru-ch/faq](https://github.com/ru-ch/faq) wiki, clusters the ones that repeat, and DMs admins a paste-ready outline.

The outline is a heading, the nearest existing page, up to three example questions, and a TODO list. It does not invent procedures, prices, or legal steps. Nothing is written to GitHub.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
# optional, to grade and draft with Cursor instead of Anthropic:
.venv/bin/pip install -e '.[cursor]'
cp .env.example .env
```

`ANSWERBOT_DB` is opened read-only (`mode=ro`), so this can run while the answer bot is up. A JSON export of the same `query_log` fields (`ts`, `question`, `user_id`, `window_ids`, `cited_ids`) replaces the live file:

```bash
.venv/bin/python -m gardener run --questions export.json --dry-run
```

`--dry-run` prints drafts and does not send or remember them.

## What gets drafted

A question is a candidate when `window_ids` is empty, or every id is positive (chat hits only; FAQ chunk ids are stored negative). Admins, very short texts, and clusters already sent are skipped. Default `MIN_CLUSTER_SIZE=2`, over the last `SINCE_DAYS` (14).

LlamaIndex embeds the wiki (`docs/` and `inbox/` markdown, chunked by heading) with `intfloat/multilingual-e5-small` and retrieves the nearest sections. LangGraph grades each cluster as covered, partial, missing, or not a FAQ topic. Only partial and missing clusters are drafted.

## Telegram

Use a separate bot token from the answer bot. Each id in `ADMIN_USER_IDS` must open a chat with this bot once (`/start`). One gap is one message, split under 4096 characters. A fingerprint is stored in `data/gardener.db` only after at least one DM succeeds.

## Models

`LLM_PROVIDER`:

- `anthropic` (default) — `ANTHROPIC_API_KEY`
- `openai` — `OPENAI_API_KEY` and `OPENAI_BASE_URL` (Groq, OpenRouter, or Ollama)
- `cursor` — `CURSOR_API_KEY`, default model `composer-2.5` (`ANSWER_MODEL` overrides). Requires `pip install 'faq-gardener[cursor]'`. The local agent runs with no tools.

## Schedule

```cron
0 8 * * * cd /path/to/faq-gardener-bot && .venv/bin/python -m gardener run
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Tests use a fake embedder and a scripted model. They do not download the embedding model or call the network.
