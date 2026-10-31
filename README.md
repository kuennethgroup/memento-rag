# Memento

**A multimodal research assistant for a group's own documents.**

Memento turns a research group's archive (papers, theses, reports, slides, spreadsheets) into a library you can ask
questions about in a chat. Answers include source citations, and a citation opens the original page with
the passage marked. Figures, charts, tables and equations are indexed with what the authors write about them, so they
can be found and read too. Built at the University of Bayreuth for a Master's thesis ([Thesis](#thesis)).

What users get:

- **Library Chat:** questions about the library in English or German, with follow-up questions, cited answers and an
  optional **Thorough check** that checks each statement against its sources before the answer is shown.
- **Document Chat:** questions about files attached to one chat; they never enter the shared library.
- **Library Upload:** adding documents to the shared library from the chat.
- **Start-screen actions:** what's in the library, a summary of a document, a document's limitations and future work,
  and a draft research proposal built from the open questions of the library.

<!-- evaluation -->
Running the thesis evaluations on your own documents: [evaluation/README.md](evaluation/README.md).
<!-- /evaluation -->

License, data and what has been tested: [Publication status](#publication-status).

## Contents

1. [How it works](#how-it-works)
2. [What you need](#what-you-need)
3. [Install and run](#install-and-run) (step by step)
4. [Access from other computers](#access-from-other-computers)
5. [Using Memento](#using-memento)
6. [Configuration](#configuration)
7. [Command line](#command-line)
8. [Day-to-day operation](#day-to-day-operation)
9. [Deployment options](#deployment-options)
10. [Optional and research components](#optional-and-research-components)
11. [Tests](#tests), [Project structure](#project-structure), [Further documentation](#further-documentation),
    [Publication status](#publication-status), [Thesis](#thesis)

---

## How it works

```
Documents: PDF, Word, PowerPoint, Excel, images (.doc/.ppt/.xls/.odt/.odp/.ods/.rtf are converted first)
   │
   ├─ 1  Parsing      MinerU 3.4.0 (VLM backend): text, titles, tables, figures, equations with page and position
   ├─ 2  Enrichment   joins split paragraphs, merges multi-panel figures, links figures, tables and equations
   │                  to the sentences that discuss them, assigns sections, drops page furniture
   ├─ 3  Embedding    Qwen3-VL-Embedding-8B: text and figure images in one vector space, plus BM25 keywords
   ├─ 4  Storage      Qdrant (one point per passage, figure, table or equation)
   └─ 5  Gap tags     optional: the answer model marks passages that state limitations or future work
                                   ▼
Question in the chat
   ├─ 6  Router       rewrites follow-ups into full questions and picks the path (search, summary, list, ...)
   ├─ 7  Search       dense + BM25 together, 80 candidates
   ├─ 8  Reranking    Qwen3-VL-Reranker-8B reads question and candidates (figures as images), keeps 20
   ├─ 9  Answer       an OpenAI-compatible model with vision writes the answer and cites every statement
   ├─ 10 Thorough     on request: each claim checked against its cited passages, failing statements revised
   └─ 11 Interface    Open WebUI 0.11.4 with Memento's functions; citations open the page in a side panel
```

Each part runs as its own service:

| Service | What runs it | Port (localhost) |
|---|---|---|
| Parser | MinerU container, GPU | 8010 |
| Embedder | vLLM 0.30.0 container, GPU | 8021 |
| Reranker | vLLM 0.30.0 container, GPU | 8022 |
| Vector database | Qdrant container | 6333 |
| Database of Open WebUI and Memento's tables | PostgreSQL container | 5433 |
| Citation viewer (files server) | small container | 8024 |
| Interface | Open WebUI, a host process in tmux | 8086 |
| Remote access (optional) | Caddy container | 8443 |
| Answer model | **not included:** any OpenAI-compatible server or API ([below](#the-answer-model)) | yours |

Containers run rootless under Podman, defined in `docker-compose.yml` and managed by `scripts/services.sh`. Model
weights stay on the host and are mounted read-only.

---

## What you need

### Hardware

| | Minimum | Tested on |
|---|---|---|
| GPU | three NVIDIA GPUs with ≥ 40 GB: one each for the parser, the embedder and the reranker | 4× A100 40 GB |
| RAM | 32 GB | 64 GB+ |
| Disk | 200 GB free (model weights about 32 GB, container images about 75 GB, plus documents and their parsed output) | 400 GB+ |
| OS, CUDA | Ubuntu 22.04 or later, CUDA 12.4 or later | Ubuntu 24.04, driver 595 (CUDA 13.2) |

The embedder and the reranker each use half of their GPU (`EMBEDDER_GPU_MEM`, `RERANKER_GPU_MEM`), which leaves room
for other jobs on the same card. On one 40 GB card the two do not fit together at these settings.

### Software

Podman with podman-compose, the NVIDIA Container Toolkit, Python 3.12, tmux and LibreOffice (for old Office formats).
Installation commands are in [step 1](#step-1-system-packages); the Hugging Face CLI comes with the Python
environment (step 2).

### The answer model

Memento does not ship an answer model. It calls any server with the OpenAI chat-completions API
(`/v1/chat/completions`), for example a vLLM server of your own or a hosted API. The model should:

- **accept images**, so that figures and charts reach it as pictures, not only as captions;
- **have a context of about 128,000 tokens** for whole-document summaries (default up to 100,000 tokens of document);
  with a smaller model, lower the `SUMMARY_MAX_TOKENS` setting ([Configuration](#important-settings-valves));
- ideally support Qwen's thinking switch and vLLM's `thinking_token_budget`, which Thorough check and the
  research-gap answers use to cap their thinking.

Tested with Qwen3.8 models served by vLLM and with Qwen3.5-397B and Qwen3.6-35B from GWDG's Academic Cloud.

Memento uses **one** answer model for everything that writes or checks text: Library Chat, Document Chat, Library
Upload's figure descriptions and gap tags, and command-line uploads. You name it once, in `config/.env`:
`REASONING_URL` (its server, without `/v1`), `REASONING_MODEL_NAME` (the model name it serves) and
`REASONING_API_KEY` (its key). The key stays in your `config/.env` and is never committed. `REASONING_PROVIDER=cloud`
sends figure descriptions, contextual retrieval and command-line answers to the cloud API instead.

---

## Install and run

For a fresh Ubuntu server. `one_click_setup.sh` runs steps 2 to 5 in one go; if you use it, create `.venv` with
Python 3.12 first and fill in `config/.env` as in step 3.

### Step 1: System packages

```bash
sudo apt install podman podman-compose tmux libreoffice git python3.12-venv
```

The NVIDIA Container Toolkit comes from NVIDIA's package repository: add it as in NVIDIA's
[installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then `sudo apt install nvidia-container-toolkit`.

Memento's model containers get their GPU through `NVIDIA_VISIBLE_DEVICES`, which Podman passes on with NVIDIA's
runtime. Set it once for your account (if `~/.config/containers/containers.conf` exists already, add these lines to
its `[engine]` section instead), then check that a container sees a GPU:

```bash
mkdir -p ~/.config/containers
cat > ~/.config/containers/containers.conf <<'EOF'
[engine]
runtime = "nvidia-container-runtime"
[engine.runtimes]
nvidia-container-runtime = ["/usr/bin/nvidia-container-runtime"]
EOF
podman run --rm -e NVIDIA_VISIBLE_DEVICES=0 docker.io/nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

### Step 2: Clone and create the Python environment

Memento needs Python 3.12 (`python3.12 --version`; on Ubuntu 22.04 it comes from the deadsnakes PPA).

```bash
git clone https://github.com/kuennethgroup/memento-rag.git memento-rag
cd memento-rag
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

This environment also holds Open WebUI (version 0.11.4) and its PostgreSQL driver. The model servers run in containers.

### Step 3: Configure `config/.env`

```bash
cp config/.env.example config/.env
python3 -c "import secrets; print(secrets.token_hex(32))"   # a value for WEBUI_SECRET_KEY
```

Open `config/.env` and set at least these (all paths absolute). Never commit this file.

| Variable | Set it to |
|---|---|
| `MEMENTO_DIR` | the folder you cloned into |
| `INPUT_ROOT`, `OUTPUT_ROOT` | where documents and parsed output go, e.g. `<MEMENTO_DIR>/data/input` and `.../data/output` |
| `DOMAIN_NAME` | **add this line**: the domain profile to use, e.g. `generic` ([Domains](#domains)) |
| `MODEL_PATH`, `RERANKER_MODEL_PATH` | where step 4 puts the weights: `$HOME/models/Qwen3-VL-Embedding-8B` and `$HOME/models/Qwen3-VL-Reranker-8B` |
| `MINERU_GPU`, `EMBEDDER_GPU`, `RERANKER_GPU` | GPU numbers (as in `nvidia-smi`) for parser, embedder and reranker |
| `POSTGRES_PASSWORD` and the password inside `DATABASE_URL` | the same new password, twice |
| `WEBUI_SECRET_KEY` | the random value printed above |
| `WEBUI_LOGO` | a PNG logo, or leave it empty for Open WebUI's own |
| `RESEARCH_GAP_TAG` | **add** `RESEARCH_GAP_TAG=true` if you want the research-gap features (tagging uses the answer model during uploads) |
| `REASONING_URL`, `REASONING_MODEL_NAME`, `REASONING_API_KEY` | **the answer model** ([The answer model](#the-answer-model)): its server **without** `/v1`, the model name it serves, and its key (empty for a server without one). Library Chat, Document Chat, Library Upload and command-line uploads all use these three; `trial` (step 8) writes them into both functions |

Optional now, needed later:

- `OPENWEBUI_API_KEY`: created in step 7.
- Optional, for the cloud models in `config/query_models.yaml`: `OPENAI_BASE_URL` (any OpenAI-compatible API) and **add**
  `CLOUD_API_KEY=<your key>`; without the key they are not offered ([Answer models](#answer-models-in-the-model-menu)).
- `S2_API_KEY`, `OPENALEX_API_KEY`: literature search of the research-gap ledger (it works without them, but slower).
- `FILES_URL`, `PROXY_PORT`: change them only for [access from other computers](#access-from-other-computers).

### Step 4: Download the model weights

```bash
bash scripts/download_models.sh      # Qwen3-VL-Embedding-8B and Qwen3-VL-Reranker-8B into ~/models, about 32 GB
```

Another folder: `MODELS_DIR=/data/models bash scripts/download_models.sh` (then adjust `MODEL_PATH` and
`RERANKER_MODEL_PATH`). Re-running it resumes an interrupted download.

### Step 5: Build the container images

```bash
bash scripts/build_images.sh        # MinerU (about 20 min, includes its models), files server; pulls the vLLM image
```

### Step 6: Start the services

```bash
mkdir -p data/input data/output logs
bash scripts/services.sh start postgres qdrant mineru embedder reranker files openwebui proxy
bash scripts/services.sh status
```

The first start of the embedder and reranker takes a few minutes. The proxy serves Open WebUI and the citation viewer
under one address (port 8443); Memento's start screen reads the library's models through it. Name the services as
above: a bare `services.sh start` also starts the research graph database (Neo4j), which `status` otherwise lists as
not reachable (it is not needed). On this first start Open WebUI also reports `model sync failed`: expected, since the
API key it needs is created in step 7; the restart in step 9 syncs the models.

### Step 7: Create the admin account and an API key

1. Open **http://localhost:8443**. On a remote server, tunnel first: `ssh -L 8443:localhost:8443 you@server`.
   Keep the local port at 8443: citations use `FILES_URL` (`http://localhost:8443/files`). With another local port,
   set that port in `FILES_URL` and in Memento Query's `FILES_URL` Valve.
2. Register. The first account becomes the administrator.
3. In **Admin Panel → Settings → Authentication**, switch on **API Keys** and save. Then, in **Settings → Account →
   API keys**, click **Show** next to *Secrets* and choose **Create new secret key**.
4. Add it to `config/.env` as `OPENWEBUI_API_KEY=...`.

### Step 8: Install Memento into Open WebUI

```bash
.venv/bin/python scripts/setup_openwebui.py trial --domain generic --dry-run   # shows what it would change
.venv/bin/python scripts/setup_openwebui.py trial --domain generic
```

Use the same domain as `DOMAIN_NAME`. This installs Library Chat, Document Chat, Library Upload, the Thorough check
switch and the admin's Qdrant Manager, creates the groups `contributors` and the member group (`MEMENTO_MEMBER_GROUP`,
default `members`), and turns sign-up on: new accounts join the member group and wait until an admin approves them
(**Admin Panel → Users**). For a moderated session where people should start at once, add `--open-signup`.

> `trial` also empties the library's document list and stored answers. Run it on a new installation or when you
> switch to another domain, not on a library people are using.

### Step 9: Check your answer model

`trial` has written the answer model of `config/.env` (`REASONING_URL`, `REASONING_MODEL_NAME`, key in
`REASONING_API_KEY`) into both functions. To check it, open **Admin Panel → Functions** and the gear icon (**Valves**)
of **Memento Query** (Library Chat and Document Chat) and **Memento Ingestion** (Library Upload: figure descriptions
and gap tags): `LLM_BASE_URL` is your server with `/v1`, `LLM_MODEL` its model name, and `LLM_API_KEY_ENV` reads
`REASONING_API_KEY`, the **name** of the variable that holds the key, never the key itself.

If your model's context is smaller than about 128,000 tokens, also lower `SUMMARY_MAX_TOKENS` in Memento Query (for
example to 50,000 for a 64,000-token model). Then restart Open WebUI so it reads `config/.env` again:

```bash
bash scripts/services.sh start openwebui
```

### Step 10: Add documents

**From the chat:** pick **Library Upload** in the model menu, attach files and send. The reply reports each document.

**In bulk, from the command line:** copy the files into `INPUT_ROOT/<corpus_dir>/` (any folder depth; `corpus_dir`
comes from the domain profile, e.g. `Generic`), then:

```bash
mkdir -p data/input/Generic && cp -r /path/to/archive/* data/input/Generic/
bash run_ingestion.sh --domain generic
```

Documents already in the library are skipped.

### Step 11: Ask a first question

Pick **Library Chat** in the model menu and ask, for example, *"What's in the library?"* or a question about one of
your documents. If something does not work, `bash scripts/services.sh status` shows each service, and the log is in
`logs/openwebui-<date>.log`.

---

## Access from other computers

Open WebUI and every service listen on `127.0.0.1` only. The proxy started in step 6 serves Open WebUI at `/` and the
citation viewer at `/files` (only to logged-in users) on port 8443, by default only on the server itself. For one person, the SSH tunnel of
step 7 is enough. For users on other computers:

1. In `config/.env`: `PROXY_BIND=0.0.0.0` (the proxy otherwise answers only on the server itself; for IPv6 as well,
   `PROXY_BIND="0.0.0.0 ::"`) and
   `FILES_URL=http://<server-name>:8443/files` (and `PROXY_PORT` if you want another port).
2. Set the same address in Memento Query's `FILES_URL` Valve (a saved Valve wins over `config/.env`).
3. `bash scripts/services.sh start proxy openwebui`.

Anyone who reaches that address can register. With `trial`'s default, a new account can do nothing until an admin
approves it; without a certificate the connection is plain HTTP, so users should not reuse a personal password.

Users then open `http://<server-name>:8443`. The proxy speaks plain HTTP; to add a certificate, see the note in
`docker/Caddyfile`.

---

## Using Memento

| In the model menu | Who sees it | What it does |
|---|---|---|
| **Library Chat** (one entry per answer model) | `contributors`, the member group, admins | questions about the shared library |
| **Document Chat** | `contributors`, the member group | questions about files attached to this chat only |
| **Library Upload** | `contributors`, the member group, admins | adds documents to the shared library |
| **Qdrant Manager** | admins | inspect and delete indexed documents |

- **Answers:** each statement carries a citation marker; clicking it opens the cited page with the passage marked.
  Below the answer, the cited sources and other documents with related passages are listed. Questions and answers
  work in English and German, whatever the documents' language.
- **Thorough check:** a switch under the chat box. With it on, the model thinks the question through, then each
  claim of the answer is checked against the passages it cites and failing statements are rewritten (up to two
  rounds). Answers take longer (about a minute); statements that still fail are marked with ⚠.
- **List questions** ("which dissertations are about ...") scan up to 60 candidate documents and list up to 50;
  long lists fold.
- **Research gaps:** "Limitations & future work" lists what a document's authors state as open; the gap ledger
  collects the open questions of the whole library, checks them against the other documents and the published
  literature, and can draft a research proposal (needs `RESEARCH_GAP_TAG=true`).
- **Document Chat** files are stored in their own collection, answered only by the local answer model, and removed
  within about an hour after the chat is deleted. Their pages, figures and downloads open only for the account that
  attached them. The one exception: a download link that a page shows works without a login for four hours (or until
  the file server restarts), so whoever is given it can download that file.

Accounts: `trial` turns sign-up on. To create accounts yourself instead, pass a CSV of `name,email` lines:
`setup_openwebui.py trial --domain generic --accounts people.csv` (the generated passwords are written to
`data/rq4/trial_accounts.csv`, never printed).

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
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY_ENV` | `REASONING_URL` + `/v1`, `REASONING_MODEL_NAME`, `REASONING_API_KEY` | the answer model ([step 9](#step-9-check-your-answer-model)) |
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

**Add documents:** `bash run_ingestion.sh --domain <name>` ([step 10](#step-10-add-documents)). Further options:
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

<!-- evaluation -->
## Reproducing the evaluations

[`evaluation/README.md`](evaluation/README.md) lists, for RQ1 to RQ4, the commands that build the ground truth and run
each evaluation on your own documents; no corpora, ground truth or results are included. The thesis' results come
from an earlier revision of this code, so a new run can give other numbers.

---
<!-- /evaluation -->

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
| `evaluation/` | RQ1 to RQ4: building the ground truth and running the evaluations | <!-- evaluation -->
| `tests/` | the test suite, one folder per package it tests |
| `run_query.sh`, `run_ingestion.sh`, `one_click_setup.sh` | ask a question, add documents, install everything in one go |
| `data/`, `logs/`, `backups/` | created at run time, not in git |

---

## Further documentation

| Topic | File |
|---|---|
| The Open WebUI functions and their Valves | [`openwebui/README.md`](openwebui/README.md) |
| Ingestion in detail | [`memento/ingestion/README.md`](memento/ingestion/README.md) |
| Running the evaluations, and which code the thesis' results come from | [`evaluation/README.md`](evaluation/README.md) | <!-- evaluation -->

---

## Publication status

- **License:** none yet; this repository has no `LICENSE` file. The same holds for the branding (the theme keeps the
  chair's logo and footer), bundled assets and third-party material.
- **Data:** no corpora, ground truth, results or study records are included.
- **Installation:** tested from a fresh clone, under a new account, on one server. The tests run offline, so they do
  not show that model downloads, GPU and container combinations or external APIs work on another machine.

---

## Thesis

**Persistent Intelligence: Mitigating Knowledge Loss in Ephemeral Working Environments with Multimodal Agentic RAG**

Ranganath Pujari, University of Bayreuth. Supervisors: Prof. Dr. Christopher Künneth, Prof. Dr. Daniel Buschek.
