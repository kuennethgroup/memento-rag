# Install Memento

These steps are for a fresh Ubuntu server; check [what you need](README.md#what-you-need) first. `one_click_setup.sh`
runs steps 2 to 5 in one go; if you use it, create `.venv` with Python 3.12 first and fill in `config/.env` as in step 3.

## Step 1: System packages

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

## Step 2: Clone and create the Python environment

Memento needs Python 3.12 (`python3.12 --version`; on Ubuntu 22.04 it comes from the deadsnakes PPA).

```bash
git clone https://github.com/kuennethgroup/memento-rag.git memento-rag
cd memento-rag
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

This environment also holds Open WebUI (version 0.11.4) and its PostgreSQL driver. The model servers run in containers.

## Step 3: Configure `config/.env`

```bash
cp config/.env.example config/.env
python3 -c "import secrets; print(secrets.token_hex(32))"   # a value for WEBUI_SECRET_KEY
```

Open `config/.env` and set at least these (all paths absolute). Never commit this file.

| Variable | Set it to |
|---|---|
| `MEMENTO_DIR` | the folder you cloned into |
| `INPUT_ROOT`, `OUTPUT_ROOT` | where documents and parsed output go, e.g. `<MEMENTO_DIR>/data/input` and `.../data/output` |
| `DOMAIN_NAME` | **add this line**: the domain profile to use, e.g. `generic` ([Domains](REFERENCE.md#domains)) |
| `MODEL_PATH`, `RERANKER_MODEL_PATH` | where step 4 puts the weights: `$HOME/models/Qwen3-VL-Embedding-8B` and `$HOME/models/Qwen3-VL-Reranker-8B` |
| `MINERU_GPU`, `EMBEDDER_GPU`, `RERANKER_GPU` | GPU numbers (as in `nvidia-smi`) for parser, embedder and reranker |
| `POSTGRES_PASSWORD` and the password inside `DATABASE_URL` | the same new password, twice |
| `WEBUI_SECRET_KEY` | the random value printed above |
| `WEBUI_LOGO` | a PNG logo, or leave it empty for Open WebUI's own |
| `RESEARCH_GAP_TAG` | **add** `RESEARCH_GAP_TAG=true` if you want the research-gap features (tagging uses the answer model during uploads) |
| `REASONING_URL`, `REASONING_MODEL_NAME`, `REASONING_API_KEY` | **the answer model** ([The answer model](README.md#the-answer-model)): its server **without** `/v1`, the model name it serves, and its key (empty for a server without one). Library Chat, Document Chat, Library Upload and command-line uploads all use these three; `trial` (step 8) writes them into both functions |

Optional now, needed later:

- `OPENWEBUI_API_KEY`: created in step 7.
- Optional, for the cloud models in `config/query_models.yaml`: `OPENAI_BASE_URL` (any OpenAI-compatible API) and **add**
  `CLOUD_API_KEY=<your key>`; without the key they are not offered ([Answer models](REFERENCE.md#answer-models-in-the-model-menu)).
- `S2_API_KEY`, `OPENALEX_API_KEY`: literature search of the research-gap ledger (it works without them, but slower).
- `FILES_URL`, `PROXY_PORT`: change them only for [access from other computers](#access-from-other-computers).

## Step 4: Download the model weights

```bash
bash scripts/download_models.sh      # Qwen3-VL-Embedding-8B and Qwen3-VL-Reranker-8B into ~/models, about 32 GB
```

Another folder: `MODELS_DIR=/data/models bash scripts/download_models.sh` (then adjust `MODEL_PATH` and
`RERANKER_MODEL_PATH`). Re-running it resumes an interrupted download.

## Step 5: Build the container images

```bash
bash scripts/build_images.sh        # MinerU (about 20 min, includes its models), files server; pulls the vLLM image
```

## Step 6: Start the services

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

## Step 7: Create the admin account and an API key

1. Open **http://localhost:8443**. On a remote server, tunnel first: `ssh -L 8443:localhost:8443 you@server`.
   Keep the local port at 8443: citations use `FILES_URL` (`http://localhost:8443/files`). With another local port,
   set that port in `FILES_URL` and in Memento Query's `FILES_URL` Valve.
2. Register. The first account becomes the administrator.
3. In **Admin Panel → Settings → Authentication**, switch on **API Keys** and save. Then, in **Settings → Account →
   API keys**, click **Show** next to *Secrets* and choose **Create new secret key**.
4. Add it to `config/.env` as `OPENWEBUI_API_KEY=...`.

## Step 8: Install Memento into Open WebUI

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

## Step 9: Check your answer model

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

## Step 10: Add documents

**From the chat:** pick **Library Upload** in the model menu, attach files and send. The reply reports each document.

**In bulk, from the command line:** copy the files into `INPUT_ROOT/<corpus_dir>/` (any folder depth; `corpus_dir`
comes from the domain profile, e.g. `Generic`), then:

```bash
mkdir -p data/input/Generic && cp -r /path/to/archive/* data/input/Generic/
bash run_ingestion.sh --domain generic
```

Documents already in the library are skipped.

## Step 11: Ask a first question

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

Settings, commands and day-to-day operation: [`REFERENCE.md`](REFERENCE.md).
