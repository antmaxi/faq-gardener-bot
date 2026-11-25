# faq-gardener-bot

Gardener finds gaps in the [ru-ch/faq](https://github.com/ru-ch/faq) wiki using questions from the [answer-chat-history-bot](https://github.com/antmaxi/answer-chat-history-bot) query log and selected archived Telegram groups. It clusters similar questions and sends drafts to admins. In watch mode, admins can also browse and search past questions, check chat answers and FAQ coverage, and save suggestions for later editing.

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

Set `FAQ_CHAT_IDS` to a comma-separated list of group IDs archived by the answer bot. An empty list disables group archive scanning. An explicit `--questions` export supplies the entire input for that pass.

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

- **Answer bot query log.** A question is a candidate unless the bot cited a FAQ chunk (a negative ID in `cited_ids`). FAQ chunks that the bot only retrieved appear in `window_ids` and leave the question eligible.
- **Selected group archives.** The configured model identifies explicit questions and clear requests for help, including those phrased as statements. A question can qualify whether or not someone answered it.

Each pass checks the last `SINCE_DAYS` (14 by default). The default `MIN_CLUSTER_SIZE` is two distinct question texts on a topic. Repeated wording counts once after normalizing case and whitespace. The scheduled pass filters short questions and questions from admins. Set `DEBUG=1` to include questions from `ADMIN_USER_IDS` and DM each processed question’s decision and reason.

LlamaIndex embeds the wiki (`docs/` and `inbox/` Markdown, chunked by heading) with `intfloat/multilingual-e5-small` and retrieves the nearest sections. LangGraph checks FAQ coverage. Gardener drafts only partial or missing topics and skips clusters already delivered.

A draft includes a heading, the nearest FAQ page when available, and up to three example questions. It can suggest answer text grounded in archived replies or give the admin a list of details to verify. Gardener shows source references for archived questions and replies, labels inferred answers, and asks the admin to resolve conflicting claims. For a group question, it checks direct replies and can infer answers from the next six messages in that group within 24 hours.

## Telegram

Use a separate bot token from the answer bot. Each id in `ADMIN_USER_IDS` must open a chat with this bot once (`/start`). One gap is one message, split under 4096 characters. A fingerprint is stored in `data/gardener.db` only after at least one DM succeeds.

While the watch process is running, the review commands below open historical review. Other private messages from admins run a pass and return the results in that chat. A gap or a question decision already delivered is not sent again. The same question is sent again when its decision changes. `docker compose run --rm gardener` exits, so it does not hear messages. A command-line `--dry-run` still only prints and does not remember what it printed.

### Review past questions

In watch mode, send these commands to Gardener in a private admin chat. Dates use `YYYY-MM-DD` and include both endpoints in UTC.

| Command | What it does |
| --- | --- |
| `/review START END [PAGE]` | Browse past questions in the selected groups and answer bot query log, newest first. |
| `/review_search START END PAGE WORDS` | Find questions containing all the search words. Start with page 1. |
| `/review_open ID` | Check a question against the current FAQ and look for archived chat answers. |
| `/review_save ID` | Save the suggestion from the review card you just opened. |
| `/review_saved [PAGE]` | List saved suggestions. |
| `/review_draft ID` | Reopen a saved suggestion and its review context. |
| `/review_edit ID ANSWER` | Replace the answer in a saved suggestion. |

Each page checks a batch of messages. A page can have no questions and still offer a next page. Search filters by text before checking messages.

Manual review includes questions asked once and questions already shown in an automatic report. The card shows chat answer status and current FAQ coverage separately, with source references where available. A recorded bot response appears as context. For questions from the query log, Gardener also searches the selected groups for supporting chat messages.

Admins can save a chat-backed suggestion or leave the answer open for later editing. Saved suggestions and edits live in `GARDENER_DB`. Admins publish FAQ changes separately.

## Improvement ideas

1. **Collect messages live from selected groups outside the archive.** Add a group handler with its own allowlist. Persist messages and edits before advancing a durable update offset, deduplicate by chat and message ID, and run grading separately from collection. Preserve topic IDs where available.
2. **Allow time for replies before sending a draft.** Add a configurable delay so community answers have time to arrive before Gardener notifies admins.
3. **Rank recurring demand.** Track occurrences and distinct askers separately from unique wording. Rank gaps by frequency, distinct askers, and recency. Use a stable topic ID to avoid repeat proposals when the wording changes.
4. **Add review actions and a digest.** Let admins dismiss, snooze, link, or resolve topics. Offer a daily digest of the highest-priority gaps and recheck resolved topics after relevant FAQ edits.
5. **Track scan health and quality.** Report collection totals, rejected candidates, pending clusters, and the last successful pass. Retry malformed model responses and flag repeated failures for review. Evaluate question detection and coverage grading against labeled examples.
6. **Offer opt-in replies in groups.** Use a confidence threshold and per-topic cooldown for automatic public FAQ replies.

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
