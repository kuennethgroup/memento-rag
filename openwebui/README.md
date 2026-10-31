# Open WebUI integration

Open WebUI 0.11.4 is Memento's interface: accounts, groups, chats and the model menu. Everything else (parsing,
search, answers, Thorough check) runs in Memento's own code, reached through the functions in `functions/`. Open
WebUI's own document handling is switched off.

**The code that runs is the copy in Open WebUI's database, not the file.** After editing a file, push it with the
[setup script](#setup-script). Settings are *Valves*: Admin Panel → Functions → gear icon.

## Functions

| File | Function id | Users see | Who |
|---|---|---|---|
| `functions/query_pipe.py` | `memento_query` | **Library Chat** (one entry per answer model) and **Document Chat** | groups `contributors` and the member group (`MEMENTO_MEMBER_GROUP`) |
| `functions/ingestion_pipe.py` | `memento_ingestion` | **Library Upload** | admins and the same groups |
| `functions/thorough_filter.py` | `memento_thorough` | the **Thorough check** switch in the chat box | everyone with Library or Document Chat |
| `functions/qdrant_pipe.py` | `qdrant_manager` | **Qdrant Manager** | admins |

A second library runs the same two files as `memento_query_polymer` and `memento_ingestion_polymer`, with its own
Valves and the group `User study` (`setup_openwebui.py study`).

## Library Chat

What happens with a question:

1. **History:** the last 3 exchanges. For the router, earlier citation numbers are replaced by the document and page
   they cited; the answer model reads earlier answers without their citations and cites only with `[N]`.
2. **Router** (one model call): rewrites a follow-up into a full question and picks the path: normal question,
   question across documents, list, summary of a document, rework of the last answer, library contents, figures,
   a section verbatim, research gaps, research proposal, greeting, or a general question.
3. **Shortcuts** without search: library contents, "do we have document X?", a person's documents ("does the library
   contain documents from X?": their own, supervised and possible documents), a section verbatim, a rework (in the
   request's language unless it asks for another), greetings.
4. **Search:** dense + BM25 (80 candidates), then the reranker keeps the best 20 (at most 6 per document). A named
   document, or a person's documents with a topic, narrow the search. If even the best passage scores below 0.1, the
   answer says the library has nothing on it.
5. **Lists** ("which dissertations are about X"): the topic is searched in the documents of the kind asked for, two
   passages each from up to 60 documents; up to 50 are listed; lists over 20 fold.
6. **Answer:** the model reads the 20 passages and up to 6 figures as images, cites every statement and answers in
   the question's language. A summary reads the whole document if it fits `SUMMARY_MAX_TOKENS`, otherwise its main
   parts. Ordinary answers do not think.
7. **Thorough check** (the switch): the model may think first (at most 2,000 tokens), then every claim is checked
   against its cited passages and failing statements are rewritten (up to 2 rounds). The answer ends with a note;
   statements that still fail carry ⚠.
8. **Citations:** each `[N]` opens the cited page in a side panel (files server, port 8024). Below the answer: the
   cited sources and other documents with related passages.

Time limits: 120 s per answer, 270 s with Thorough check; any model call that writes nothing for 90 s is stopped with
a note. The Stop button stops the model at once.

### Valves of Memento Query

| Valve | After setup | Meaning |
|---|---|---|
| `DOMAIN_NAME`, `COLLECTION` | from `trial` | the library this function serves |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY_ENV` | `REASONING_URL` + `/v1`, `REASONING_MODEL_NAME`, `REASONING_API_KEY` | the answer model of `config/.env`; `LLM_API_KEY_ENV` names the variable that holds its key |
| `VARIANTS_FILE` | `query_models.yaml` | the [answer models](#answer-models) of the menu |
| `FILES_URL` | `http://localhost:8443/files` | the citation viewer as the user's browser reaches it (through the proxy) |
| `RETRIEVE_K`, `RERANK_TOP`, `PER_PAPER_CAP` | 80, 20, 6 | candidates, passages read, passages per document |
| `HYBRID_ENABLED`, `RERANK_ENABLED` | on, on | BM25 next to dense search; reranker |
| `RERANK_MIN_SCORE`, `RELATED_LOW_SCORE` | 0.03, 0.1 | drop weak passages; "nothing found" below this |
| `MAX_IMAGES_TO_LLM`, `MAX_FIGURES_SHOWN` | 6, 6 | figures sent to the model; figures shown under the answer |
| `LIST_SCAN_DOCS`, `LIST_SCAN_MAX` | 60, 50 | documents a list question reads; documents it lists |
| `SUMMARY_MAX_TOKENS`, `GAP_BUDGET_TOKENS` | 100000, 20000 | longest whole-document summary; research-gap passages (both must fit the model's context) |
| `LIVE_THINKING` | off | thinking for ordinary answers |
| `THINKING_BUDGET`, `THOROUGH_THINKING_BUDGET`, `GAP_THINKING_BUDGET` | 12288, 2000, 2000 | thinking caps: stored answers, Thorough check, research gaps |
| `VALIDATION_ENABLED`, `VALIDATION_MAX_ROUNDS` | on, 2 | Thorough check available; revision rounds |
| `ANSWER_BUDGET_S`, `THOROUGH_BUDGET_S` | 120, 270 | time per answer |
| `TEMPERATURE`, `MAX_TOKENS` | 0.15, 16384 | generation |
| `HISTORY_TURNS` | 3 | exchanges a follow-up sees |
| `DOC_CHAT_MAX_FILES`, `DOC_CHAT_MAX_PAGES` | 5, 0 | Document Chat: files per chat (also in `loader.js`), pages per file (0: no limit) |
| `METRICS_URL`, `WARMUP_MAX_RUNNING` | `127.0.0.1:8090/metrics`, 2 | when the background writer may use the model ([stored answers](#start-screen-and-stored-answers)) |
| `GRAPH_RETRIEVAL` | off | research knowledge graph |

### Answer models

`config/query_models.yaml` defines the Library Chat entries of the menu:

- exactly one `provider: local` entry, which uses the Valves above;
- any number of `provider: cloud` entries (model name, and the variables holding address and key: `base_url_env`,
  `key_env`); `enabled: false` hides one; `accent` / `accent_dark` set its colour.

Only the answer uses the chosen model; the router, chat titles and Thorough check always use the local one. Each
answer ends with "_Answered by <model>._" (plus ", checked by <local model>" after a Thorough check). A cloud model that
is out of quota or unreachable is replaced by the local model, with a note. Under each cloud model the menu shows the
calls left this hour (only through the proxy, port 8443). The menu is rebuilt from the file at every Open WebUI start.

## Document Chat

Questions about files attached to the chat: up to 5 files per chat (`DOC_CHAT_MAX_FILES`; with more in the message box,
sending is blocked), with no page limit (`DOC_CHAT_MAX_PAGES` 0; a file's reading ends after MinerU's 30 minutes). The
files are parsed like uploads into a separate collection, searched only within that chat, answered by the local model,
and never added to the library. While they are read, the chat shows the same progress bar as Library Upload. A
start-screen suggestion clicked while a file still uploads starts once the upload is done. A summary reads the whole
file; Thorough check works as in Library Chat. Files are removed about an hour after their chat is
deleted.

## Library Upload

Attach files and send. Memento processes one upload at a time, skips unchanged files, hides documents from answers
until they are fully stored, and reports every file with a reason when it is skipped. A question typed here gets a
link to Library Chat. Admins can also type a folder or file path on the server.

| Valve | After setup | Meaning |
|---|---|---|
| `DOMAIN_NAME`, `COLLECTION` | from `trial` | the library uploads go to |
| `MINERU_URL`, `EMBEDDER_URL`, `QDRANT_URL` | localhost ports | services |
| `BACKEND` | `vlm-engine` | MinerU backend |
| `PARSE_WORKERS`, `EMBED_BATCH_SIZE` | 8, 2 | documents parsed at once; inputs per embedding request |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY_ENV` | | model for figure descriptions and gap tags |
| `RESEARCH_GAP_TAG` | from `config/.env` | tag research-gap passages |
| `CONTRIBUTOR_GROUP` | `contributors` and the member group | groups allowed to upload, besides admins |
| `SKIP_PARSE`, `FORCE_REPARSE`, `FORCE_REEMBED` | off | maintenance switches |

## Start screen and stored answers

- **Buttons** under the chat box: *What's in the library?*, *Summarize a document* and *Limitations & future work*
  (both with a searchable document picker), and *Draft a proposal*. The picker's list is
  `OUTPUT_ROOT/_runtime/library.json`, served at `/files/library`.
- **Stored answers:** summaries and research gaps asked through the start-screen buttons are stored per document and
  replayed after that; a regenerated answer replaces them. After an upload, a background writer creates them while the answer model has room
  (at most `WARMUP_MAX_RUNNING` requests running, none waiting, no upload in progress).
- **Research gaps:** *Limitations & future work* reports what a document's authors state as open. The gap ledger keeps
  the open questions of the whole library, checked against the other documents and the literature; the
  `gap-updater` process keeps it current. *Draft a proposal* writes a research proposal from it.
- **Tables** in Open WebUI's Postgres: `memento_documents` (document list), `memento_answer_cache` (stored answers),
  `memento_warmup_queue`, `memento_gap*` (gap ledger), `memento_doc_people` (authors and supervisors).

## Qdrant Manager (admin)

```
list | info <name> | count <name> | papers <name> [page] | chunks <name> <source> [page] | chunk <id>
create <name> | delete <name> [confirm] | delete paper <name> <source> [confirm] | delete chunk <id> | delete all confirm | help
```

## Setup script

`scripts/setup_openwebui.py` signs in with `OPENWEBUI_API_KEY` from `config/.env`. Add `--dry-run` to see changes
first.

| Command | Does |
|---|---|
| `trial --domain <d> [--accounts people.csv]` | first setup of a library: all functions, groups, model access, sign-up on, interface settings; **empties the document list and stored answers** |
| `--only <id> keep-valves` | push one function's code, keep its saved Valves (the normal way to deploy a change) |
| `check` | compare code and Valves on disk with the database |
| `overwrite` | push code and reset Valves to the `config/.env` defaults |
| `sync-models` | rebuild the model menu from `config/query_models.yaml` (runs at every Open WebUI start) |
| `trial-ui` | only the interface settings (permissions, banner, default model) |
| `study`, `remove-study` | set up or remove a second library in the same Open WebUI (`SECOND_LIBRARY_*` in `config/.env`) |
| `flash-next` | point the default library at the answer model of `config/.env` (`REASONING_URL`, `REASONING_MODEL_NAME`, `REASONING_API_KEY`) |

Open WebUI rewrites the words `from config`, `from utils`, `from apps` and `from main` anywhere in a function's text when
it saves it; the functions contain none of them, so the stored code stays equal to the file.

## Look and feel

`loader.js`, `custom.css` and `static/memento/` (copied into Open WebUI at every start by `scripts/services.sh`) add:

- start screens with a drawing per model, guide clips (`static/memento/guides/`) and the start-screen buttons;
- the model menu with Local / Cloud labels, model colours and the cloud quota;
- the Thorough check switch, citation pills "[N] p. X", the side panel in the sidebar's colours, a light/dark button;
- the upload progress bar (Library Upload and Document Chat) and, in Document Chat, sending blocked with more files in
  the message box than `DOC_CHAT_MAX_FILES`;
- the Künneth group logo and the footer "© 2026 Kuenneth Group · Built with Open WebUI";
- hidden Open WebUI items Memento does not use; the interface switches each browser to English once.

These additions find Open WebUI's elements by structure and are written for 0.11.4. After an upgrade: open both start
screens, a chat with citations and an upload in light and dark mode, and fix the selectors in `custom.css` /
`loader.js` where needed.

## Tuning for other hardware

| Setting | Default | Adjust when |
|---|---|---|
| `SUMMARY_MAX_TOKENS`, `GAP_BUDGET_TOKENS` | 100000, 20000 | the model's context is smaller: both plus `MAX_TOKENS` must fit |
| `WARMUP_MAX_RUNNING` | 2 | keep it below the model server's parallel requests, so one place stays free for questions |
| `METRICS_URL` | `http://127.0.0.1:8090/metrics` | your vLLM server's metrics address (without it, only the upload rule applies) |
| `THOROUGH_THINKING_BUDGET`, `GAP_THINKING_BUDGET` | 2000, 2000 | your model thinks much faster or slower |
| prefix caching on the model server | on | keep `--enable-prefix-caching`: repeated long prompts are read from the cache |

## Licence and branding

Open WebUI's licence (`open_webui-0.11.4.dist-info/licenses/LICENSE`, clause 4) allows changing its branding only for
deployments with at most 50 users in any 30 days, with written permission, or with an enterprise licence. This
installation stays below 50 users and replaces the logo (`WEBUI_LOGO`) and drops the " (Open WebUI)" name suffix;
the footer credits Open WebUI. **Above 50 users:** set `OPENWEBUI_NAME_SUFFIX=on`, unset `WEBUI_LOGO`, reinstall the
package (`.venv/bin/pip install --force-reinstall --no-deps open-webui==0.11.4`) and restart Open WebUI. The serif font
Fraunces is served locally under the SIL Open Font License (`static/memento/fonts/OFL-Fraunces.txt`).

## Developer notes

- In `functions/`, never write `from config`, `from utils`, `from apps` or `from main`, comments included (Open WebUI
  rewrites them); `tests/query/test_query_pipe_domain.py` checks every function through the rewrite.
- Long work must not block Open WebUI's event loop: run it in a thread.
- Status messages stay visible as a log: send only real progress, in plain words; counts belong in the server log.
- Separate consecutive `<details>` blocks with blank lines; unknown HTML, comments included, is shown as text.
- Toggle filters switched on by the user reach the function in `__metadata__["filter_ids"]`.
- Upgrading Open WebUI changes its database irreversibly: back up Postgres and `~/.open-webui` first, then run
  `check` and retest streaming, citations and the sources block.
