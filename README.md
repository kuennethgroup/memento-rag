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
3. [Install](#install)
4. [Using Memento](#using-memento)
5. [Further documentation](#further-documentation)
6. [Publication status](#publication-status), [Thesis](#thesis)

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
Installation commands are in [step 1](INSTALL.md#step-1-system-packages); the Hugging Face CLI comes with the Python
environment (step 2).

### The answer model

Memento does not ship an answer model. It calls any server with the OpenAI chat-completions API
(`/v1/chat/completions`), for example a vLLM server of your own or a hosted API. The model should:

- **accept images**, so that figures and charts reach it as pictures, not only as captions;
- **have a context of about 128,000 tokens** for whole-document summaries (default up to 100,000 tokens of document);
  with a smaller model, lower the `SUMMARY_MAX_TOKENS` setting ([Configuration](REFERENCE.md#important-settings-valves));
- ideally support Qwen's thinking switch and vLLM's `thinking_token_budget`, which Thorough check and the
  research-gap answers use to cap their thinking.

Tested with Qwen3.8 models served by vLLM and with Qwen3.5-397B and Qwen3.6-35B from GWDG's Academic Cloud.

Memento uses **one** answer model for everything that writes or checks text: Library Chat, Document Chat, Library
Upload's figure descriptions and gap tags, and command-line uploads. You name it once, in `config/.env`:
`REASONING_URL` (its server, without `/v1`), `REASONING_MODEL_NAME` (the model name it serves) and
`REASONING_API_KEY` (its key). The key stays in your `config/.env` and is never committed. `REASONING_PROVIDER=cloud`
sends figure descriptions, contextual retrieval and command-line answers to the cloud API instead.

---

## Install

[`INSTALL.md`](INSTALL.md) takes you from a fresh Ubuntu server to a first question in eleven steps, and covers access
from other computers. [`REFERENCE.md`](REFERENCE.md) has the settings, commands and day-to-day operation.

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

<!-- evaluation -->
## Reproducing the evaluations

[`evaluation/README.md`](evaluation/README.md) lists, for RQ1 to RQ4, the commands that build the ground truth and run
each evaluation on your own documents; no corpora, ground truth or results are included. The thesis' results come
from an earlier revision of this code, so a new run can give other numbers.

---
<!-- /evaluation -->

## Further documentation

| Topic | File |
|---|---|
| Installing Memento, step by step | [`INSTALL.md`](INSTALL.md) |
| Settings, command line, operation, tests, project structure | [`REFERENCE.md`](REFERENCE.md) |
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
