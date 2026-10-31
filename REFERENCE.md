# Memento reference

Settings, commands and how the repository is laid out. To install Memento, see [`INSTALL.md`](INSTALL.md).

## Contents

1. [Configuration](#configuration)
2. [Command line](#command-line)
3. [Day-to-day operation](#day-to-day-operation)
4. [Deployment options](#deployment-options)
5. [Optional and research components](#optional-and-research-components)
6. [Tests](#tests)
7. [Project structure](#project-structure)

---

## Configuration

### Domains

A domain profile in `config/domains/` sets the corpus folder, the collection, the instruction the embedder and
reranker use, the answer instructions, the section vocabulary and the model-free claim checks of Thorough check
(`claim_rules`: `numbers`, `units`, `phases`; the last two are written for materials science).

| Profile | Corpus folder | Collection |
|---|---|---|
| `generic.yaml` | `Generic` | `memento-generic-rag` |
| `ceramics.yaml` | `Ceramics` | `memento-ceramics-rag` |
| `polymer-engineering.yaml` | `PolymerEngineering` | `memento-polymer-rag` |
| `computational-materials.yaml` | `ComputationalMaterials` | `memento-comp-mat-rag` |

For a new field: copy `generic.yaml`, adapt it, then set `DOMAIN_NAME` and run `setup_openwebui.py trial --domain
<name>`. Changing the retrieval instruction later means embedding the documents again.

### Answer models in the model menu

`config/query_models.yaml` lists the Library Chat entries: exactly one `provider: local` entry (it uses the Memento
Query Valves of step 9; its `name` and `label` are what users see) and any number of `provider: cloud` entries, which
read their address and key from the variables they name (`base_url_env`, `key_env`). Set `enabled: false` to hide
one. The menu is rebuilt from this file
every time Open WebUI starts (`bash scripts/services.sh start openwebui`).

The cloud entries are optional examples: any OpenAI-compatible chat-completions API can serve them. A cloud entry is
offered only while its key variable is set, so an installation that configured no cloud service shows none; removing
the key (or `enabled: false`) takes the model out of the menu at the next start. Set
`model` to a name your API serves, and say in `label` and `host` which provider it is, so users know where their
questions go.

### Important settings (Valves)

Settings live in each function's Valves in Open WebUI (Admin Panel → Functions → gear icon). Their defaults come from
`config/.env` when a function is installed; a saved value always wins afterwards.

| Valve (Memento Query) | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY_ENV` | `REASONING_URL` + `/v1`, `REASONING_MODEL_NAME`, `REASONING_API_KEY` | the answer model ([step 9](INSTALL.md#step-9-check-your-answer-model)) |
| `DOMAIN_NAME`, `COLLECTION` | set by `trial` | which library this function serves |
| `FILES_URL` | from `config/.env` (`http://localhost:8443/files`) | the citation viewer's address as the user's browser sees it |
| `SUMMARY_MAX_TOKENS` | 100,000 | longest document summarised as a whole (needs a large context) |
| `LIVE_THINKING` | off | ordinary answers without thinking (faster); Thorough check still thinks |
| `THOROUGH_THINKING_BUDGET` | 2,000 | thinking tokens of a Thorough answer and its revisions |
| `ANSWER_BUDGET_S`, `THOROUGH_BUDGET_S` | 120, 270 | time budget of an answer, without and with Thorough check |
| `RERANK_TOP` | 20 | passages the answer model reads |

The main Valves are described in [`openwebui/README.md`](openwebui/README.md).

### Environment variables (`config/.env`)

"Default" is the value used when the variable is missing from `config/.env`; `.env.example` sets some of them
differently.

| Variable | Default | Meaning |
|---|---|---|
| `MEMENTO_DIR`, `INPUT_ROOT`, `OUTPUT_ROOT` | none | repository, documents and parsed output |
| `DOMAIN_NAME` | `polymer-engineering` | domain profile for the command-line tools and new installations |
| `MODEL_PATH`, `RERANKER_MODEL_PATH` | none | weights of embedder and reranker |
| `MINERU_GPU`, `EMBEDDER_GPU`, `RERANKER_GPU` | 0, 1, 2 | GPU per service |
| `EMBEDDER_GPU_MEM`, `RERANKER_GPU_MEM` | 0.5 | share of the GPU memory vLLM may use |
| `MINERU_BACKEND` | `vlm-engine` | parser mode (`hybrid-engine`, or `pipeline` for classic OCR) |
| `PARSE_WORKERS`, `EMBED_BATCH_SIZE` | 8, 2 | documents parsed at once; inputs per embedding request |
| `HYBRID_ENABLED`, `HYBRID_PREFETCH_K` | true, 80 | dense + BM25 search; candidates before reranking |
| `RESEARCH_GAP_TAG` | false | mark research-gap passages during ingestion |
| `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | none | PostgreSQL for Open WebUI and Memento |
| `WEBUI_SECRET_KEY` | none (required) | signs login sessions |
| `OPENWEBUI_API_KEY` | none | admin key for `setup_openwebui.py` |
| `OPENAI_BASE_URL`, `CLOUD_API_KEY` | `https://api.openai.com/v1`, none | hosted answer models |
| `REASONING_URL`, `REASONING_MODEL_NAME`, `REASONING_API_KEY` | none | the answer model: its server without `/v1`, its model name and its key; `trial`, `flash-next` and `study` write it into the Valves |
| `MEMENTO_MEMBER_GROUP` | `members` | the group new accounts join (may upload and ask) |
| `MEMENTO_BUILT_BY` | none | who built Memento, as the general answers state it ("built at …") |
| `FILES_URL`, `PROXY_PORT` | `http://localhost:8443/files`, 8443 | citation viewer address (through the proxy), proxy port |
| `WEBUI_NAME`, `WEBUI_LOGO`, `OPENWEBUI_PORT` | Memento, none, 8086 | name, logo and port of the interface |
| `S2_API_KEY`, `OPENALEX_API_KEY` | none | literature search of the gap ledger |
| `GRAPH_BUILD`, `GRAPH_RETRIEVAL`, `NEO4J_*` | false | research knowledge graph (not used in production) |

---

## Command line

**Ask a question** with the same code path and saved settings as the chat:

```bash
bash run_query.sh "What sintering temperature was used for the alumina samples?"
bash run_query.sh "question" --model qwen35              # an id of config/query_models.yaml (default: local)
bash run_query.sh "question" --thorough --json out.json  # with Thorough check; save answer and sources
bash run_query.sh --file questions.txt                   # one question per line
```

**Add documents:** `bash run_ingestion.sh --domain <name>` ([step 10](INSTALL.md#step-10-add-documents)). Further options:
`--pdf <file>` for one document, `--force` to embed again, `--skip-parse` to reuse parsed output, `--staggered` to
parse everything first and embed afterwards (frees the GPU when parser and embedder share one), `--ocr` for the OCR
baseline collection. Details: [`memento/ingestion/README.md`](memento/ingestion/README.md).

---

## Day-to-day operation

```bash
bash scripts/services.sh status                  # every service, GPU and the interface
bash scripts/services.sh start openwebui         # (re)start the interface, e.g. after editing config/.env
bash scripts/services.sh stop                    # stop everything
tmux attach -t openwebui                         # the interface's console (detach: Ctrl+B, then D)
```

- **Logs:** `logs/openwebui-<date>.log` (questions, answers, errors), `logs/gap_updater.log` (research-gap ledger),
  `logs/ingestion_<time>.log` (command-line ingestion).
- **After changing Memento's code:** push it into Open WebUI without touching the saved settings:
  `.venv/bin/python scripts/setup_openwebui.py --only memento_query keep-valves` (likewise `memento_ingestion`).
  Changes to `memento/` or `config/domains/` also need an Open WebUI restart.
- **Stored answers:** document summaries and limitations are stored when first asked and replayed afterwards; after
  an upload they are written in the background.

---

## Deployment options

| What | Where | Notes |
|---|---|---|
| An answer model server for every library | `REASONING_URL`, `REASONING_MODEL_NAME`, `REASONING_API_KEY`; `setup_openwebui.py flash-next` | `flash-next` points both functions of the default library at it and keeps every other setting |
| The name users see for the local model | `config/query_models.yaml`, the `local` entry's `name` and `label` | rename them after your model, then restart Open WebUI |
| Where a cloud model runs | `config/query_models.yaml`, a cloud entry's `host` | one line on the start screen; default "Runs on a cloud service." |
| The member group | `MEMENTO_MEMBER_GROUP` | if you rename it in Open WebUI, change Memento Ingestion's `CONTRIBUTOR_GROUP` Valve too (uploads are allowed by group name) |
| A second library in the same Open WebUI | `setup_openwebui.py study` / `remove-study`, `SECOND_LIBRARY_*` in `config/.env`, `config/query_models_study.yaml` | its own functions, models, group and collection; not needed for one library. The two libraries are not separated at file level: a logged-in user who knows a file's path can open it, so groups that must not see each other's documents need separate installations |
| An offline literature index for the research-gap ledger | `ALEXANDRIA_DB`, `scripts/build_alexandria_index.py`, `memento/research_gaps/literature.py` | optional; skipped when the index is missing |
| Full texts in the literature check | `LITERATURE_FULLTEXT=1`, `ALEXANDRIA_DB`, `scripts/build_fulltext_paths.py`, `memento/research_gaps/fulltext.py` | optional, off by default; the judge also reads the best-matching passages of papers held in full; the database is only read. Turning it on sends those passages to the answer model (`REASONING_URL`), the reranker (`RERANKER_URL`) and the verifier (`VERIFIER_URL`, if set), so point these only at servers allowed to hold that text. Cloud models never receive them: a report or proposal that draws on full texts, and the chat that holds it, stay on the local model. Readers see the abstract and the judge's one-sentence reason, not the passages |
| Look-alike characters against the PDF's text layer | `TEXT_LAYER_CHECK`, `memento/ingestion/text_layer.py` | on by default for PDFs, `0` switches it off; applies when a document is parsed; each change is listed in `text_layer_changes.json` next to the parse |

---

## Optional and research components

Disabled or optional features retained for the thesis evaluations. Keep their source modules: current code also imports shared graph utilities.

- **Knowledge graph** (`memento/graph/build/build_graph.py`, `memento/graph/query/graph_retrieval.py`, the `neo4j*` services):
  off by default (`GRAPH_BUILD`, `GRAPH_RETRIEVAL`).
- **Fact sheets** (`memento/graph/build/fact_sheets.py`): per-document records for list questions.
- **Local answer model container** (`reasoning` in `docker-compose.yml`): build with `bash scripts/build_images.sh
  reasoning`, start with `bash scripts/services.sh start reasoning`, then point the Valves at
  `http://localhost:8023/v1`.

---

## Tests

```bash
.venv/bin/pip install -r requirements/dev.txt
# One-time tokenizer download before running without network access:
.venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
bash scripts/test_offline.sh
```

The suite covers ingestion, retrieval, the Open WebUI functions (with Open WebUI's own import rewrite), roles, Document
Chat isolation, the verification loop and the setup script. The contract tests (`tests/contracts/test_query_adapter_contract.py`,
`tests/contracts/test_ingestion_adapter_contract.py`) pin what Library Chat and Library Upload send and write, against
fakes of the outside services; after an intended change, `MEMENTO_UPDATE_CONTRACTS=1` rewrites their transcripts in
`tests/fixtures/contracts/`. The offline runner uses temporary database/static/output directories and unreachable service endpoints. Install test
dependencies with `pip install -r requirements/dev.txt`; Node.js is needed for the loader tests.

Tests that need a real database (the locks between libraries on PostgreSQL, the knowledge graph on Neo4j, fact sheets
on Qdrant) run only against throwaway servers named by `MEMENTO_TEST_PG_URL`, `NEO4J_TEST_URI` and
`MEMENTO_TEST_QDRANT_URL`, and are skipped without them. They create and drop tables, graphs and collections: never
point them at the deployment's servers.

```bash
podman run -d --rm --name memento-test-pg -e POSTGRES_PASSWORD=test -p 127.0.0.1:55432:5432 postgres:16-alpine
podman run -d --rm --name memento-test-neo4j -e NEO4J_AUTH=none -p 127.0.0.1:7690:7687 neo4j:5-community
podman run -d --rm --name memento-test-qdrant -p 127.0.0.1:16333:6333 qdrant/qdrant
MEMENTO_TEST_PG_URL=postgresql://postgres:test@127.0.0.1:55432/postgres NEO4J_TEST_URI=bolt://127.0.0.1:7690 \
    MEMENTO_TEST_QDRANT_URL=http://127.0.0.1:16333 bash scripts/test_offline.sh
podman stop memento-test-pg memento-test-neo4j memento-test-qdrant
```

---

## Project structure

| Folder | What is there |
|---|---|
| `config/` | the files you edit or a container mounts: `.env` (from `.env.example`), `domains/*.yaml`, `query_models*.yaml`, `templates/`, `config.sh` |
| `openwebui/` | what Open WebUI stores and serves: `functions/` (the four functions, short entry points into `memento/`), `loader.js`, `custom.css`, `static/` |
| `memento/config/` | settings, the corpus folder rule, domain profiles, `--domain` for command lines |
| `memento/query/` | answering one chat message: `service.py` (the ordered steps), `request.py`, `router.py`, `flows/` (one module per kind of question), `scope/` (which documents a message is about), the search step, model calls, stored answers, user texts, the warm-up worker |
| `memento/document_chat/` | Document Chat: reading a chat's own files, their store, removing chats that are gone |
| `memento/ingestion/` | storing a document: convert, parse, enrich, embed; `run_pipeline.py` is the command line, `upload/` is Library Upload |
| `memento/research_gaps/` | research-gap tags, the ledger and its jobs, the literature search, the open-questions report and proposals |
| `memento/retrieval/` | the search core (`search.py`), the answer messages, Library Chat from the command line (`pipe_cli.py`, `generate.py`), a standalone search command (`retrieve.py`) |
| `memento/library/` | a library's records in Postgres: documents and titles, front matter, people, stored answers, locks, copy state, warm-up queue |
| `memento/graph/` | the knowledge graph: store, entity registries, `build/`, `query/` |
| `memento/agentic_validation/` | the Thorough check: claims, verifiers, scientific checks, strategies |
| `memento/common/` | small helpers several packages use: model-server requests, cloud quota, atomic files, text language, QR codes |
| `scripts/` | `services.sh`, `setup_openwebui.py`, image builds, model downloads, the gap-ledger commands |
| `services/` | the container programs: the file server (citation viewer), the optional local answer-model API |
| `docker/`, `docker-compose.yml` | the container images and the services |
| `evaluation/` | RQ1 to RQ4: building the ground truth and running the evaluations |
| `tests/` | the test suite, one folder per package it tests |
| `run_query.sh`, `run_ingestion.sh`, `one_click_setup.sh` | ask a question, add documents, install everything in one go |
| `data/`, `logs/`, `backups/` | created at run time, not in git |
