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

## Docker

Same one-shot job, with CPU torch, the default embed model, and the Cursor extra already in the image. Copy `.env.example` to `.env` and fill in the keys you need first. Set `HF_TOKEN` there before `docker compose build` if Hub rate-limits anonymous downloads or the embed model is gated.

The answer-bot database is mounted read-only from `ANSWERBOT_DATA_DIR` (default `../answer-chat-history-bot/data`, the directory that contains `answerbot.db` and its `-wal` / `-shm` files). This repo's `./data` holds `gardener.db` and the FAQ cache. Compose overrides `ANSWERBOT_DB`, `GARDENER_DB`, and `FAQ_CACHE` to those container paths.

```bash
mkdir -p data
docker compose build
docker compose run --rm gardener
```

Dry run, or a JSON export dropped into `./data`:

```bash
docker compose run --rm gardener python -m gardener run --dry-run
docker compose run --rm gardener python -m gardener run --questions /data/questions.json --dry-run
```

`docker compose run --rm gardener` is one pass and exits. To keep checking without cron, leave the watch service up (default pause 3600 seconds, or `WATCH_SECONDS` from `.env`):

```bash
docker compose --profile watch up -d
```

`docker compose stop watch` ends it. Ollama on the host: `OPENAI_BASE_URL=http://host.docker.internal:11434/v1`.

## What gets drafted

A question is a candidate unless `cited_ids` contains a FAQ chunk (those ids are negative). A chunk that was only retrieved stays a candidate; that list is `window_ids`. Admins, very short texts, and clusters already sent are skipped. Set `DEBUG=1` to include questions asked by `ADMIN_USER_IDS` and to DM each processed question with its decision and a short reason. Default `MIN_CLUSTER_SIZE=2`, over the last `SINCE_DAYS` (14).

LlamaIndex embeds the wiki (`docs/` and `inbox/` markdown, chunked by heading) with `intfloat/multilingual-e5-small` and retrieves the nearest sections. LangGraph grades each cluster as covered, partial, missing, or not a FAQ topic. Only partial and missing clusters are drafted.

## Telegram

Use a separate bot token from the answer bot. Each id in `ADMIN_USER_IDS` must open a chat with this bot once (`/start`). One gap is one message, split under 4096 characters. A fingerprint is stored in `data/gardener.db` only after at least one DM succeeds.

While the watch process is running, any private message from an admin runs a pass and replies in that chat. A gap or a question decision already delivered is not sent again. The same question is sent again when its decision changes. `docker compose run --rm gardener` exits, so it does not hear messages. A command-line `--dry-run` still only prints and does not remember what it printed.

## Models

`LLM_PROVIDER`:

- `anthropic` (default) — `ANTHROPIC_API_KEY`
- `openai` — `OPENAI_API_KEY` and `OPENAI_BASE_URL` (Groq, OpenRouter, or Ollama)
- `cursor` — `CURSOR_API_KEY`, default model `composer-2.5` (`ANSWER_MODEL` overrides). Requires `pip install 'faq-gardener[cursor]'`. The local agent runs with no tools.

## Schedule

Leave `WATCH_SECONDS` empty to run once. Set it to keep a process polling; a cluster is sent once, and later passes only send new ones.

```bash
WATCH_SECONDS=3600 .venv/bin/python -m gardener run
```

Or cron for a single daily pass:

```cron
0 8 * * * cd /path/to/faq-gardener-bot && .venv/bin/python -m gardener run
```

With Docker, the same daily pass is `docker compose run --rm gardener`. The background service is `docker compose --profile watch up -d`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Tests use a fake embedder and a scripted model. They do not download the embedding model or call the network.
