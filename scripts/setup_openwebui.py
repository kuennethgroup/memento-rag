#!/usr/bin/env python3
"""Open WebUI setup for Memento through its API.

Full setup creates the pipe functions (ingestion, query, Qdrant manager), the workspace models, the OpenAI-compatible
connection (OPENAI_API_KEY and OPENAI_BASE_URL from config/.env) and turns on Bypass Embedding and Retrieval.

Usage:
# full first-time setup
python3 scripts/setup_openwebui.py --email admin@example.com --password yourpassword

# one function: check (diff code and Valves, change nothing), overwrite (push code, Valves back to the .env
# defaults: after a domain switch or new Valve fields), keep-valves (push code, keep saved Valves: the default)
python3 scripts/setup_openwebui.py --email ... --password ... --only memento_ingestion check
python3 scripts/setup_openwebui.py --email ... --password ... --only memento_ingestion overwrite
python3 scripts/setup_openwebui.py --email ... --password ... --only memento_ingestion keep-valves
python3 scripts/setup_openwebui.py --email ... --password ... check          # every function

# deployment (without --email/--password: OPENWEBUI_API_KEY from config/.env): pipe code (Valves kept) plus the
# deployment's Valves, the groups "contributors" and MEMENTO_MEMBER_GROUP, wrapper models readable by them (raw
# pipes admin-only), sign-up on (new accounts join MEMENTO_MEMBER_GROUP), the document list saved to data/rq4/
# and emptied; --accounts people.csv (name,email per line) creates accounts and writes their generated passwords
# to data/rq4/trial_accounts.csv (mode 600, never printed)
python3 scripts/setup_openwebui.py trial --domain generic --dry-run
python3 scripts/setup_openwebui.py trial --domain generic [--accounts people.csv]
# what members see (also part of trial): permissions, global switches, default model, banner, follow-up
# suggestions off, model descriptions; read, change, write
python3 scripts/setup_openwebui.py trial-ui [--dry-run]"""
import argparse
import csv
import difflib
import json
import os
import secrets
import sys
from pathlib import Path

import requests

PROJECT_ROOT  = Path(__file__).parent.parent
PIPELINES_DIR = PROJECT_ROOT / "openwebui" / "functions"
ENV_FILE      = PROJECT_ROOT / "config" / ".env"


def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _env_value(key: str, default: str = "") -> str:
    """A deployment value: the process environment, else config/.env, else `default`."""
    return os.getenv(key) or load_env().get(key) or default


sys.path.insert(0, str(PROJECT_ROOT))
from memento.config.domain_loader import default_collection, load_domain  # noqa: E402
from memento.query.models import (CONFIG as VARIANTS_CONFIG, load_variants, unoffered_cloud_variants,  # noqa: E402
                                  workspace_id)

VARIANTS        = load_variants()                   # Memento Query models users can pick (config/query_models.yaml)
QUERY_MODEL_IDS = [workspace_id(v) for v in VARIANTS]
DOC_CHAT_ID     = "memento-document-chat"           # Memento Document Chat
# picker order: everything local first (Query on the local model, Document Chat, Ingestion), then the cloud Query models
# in config order
MODEL_ORDER     = ([workspace_id(v) for v in VARIANTS if not v.is_cloud] + [DOC_CHAT_ID, "memento-ingestion-prod"]
                   + [workspace_id(v) for v in VARIANTS if v.is_cloud])
CONTRIBUTORS  = "contributors"      # RQ4 roles: admin / contributor (upload + ask) / reader (ask only, later)
TRIAL_GROUP   = _env_value("MEMENTO_MEMBER_GROUP", "members")   # new accounts join it; may upload and ask
UPLOAD_GROUPS = [CONTRIBUTORS, TRIAL_GROUP]
# who sees Document Chat: the groups that may upload
DOC_CHAT_GROUPS: list = UPLOAD_GROUPS
ACCOUNTS_OUT  = PROJECT_ROOT / "data" / "rq4" / "trial_accounts.csv"

FUNCTIONS = [
    {
        "id":   "memento_ingestion",
        "name": "Memento Ingestion",
        "file": "ingestion_pipe.py",
        "meta": {"description": "Multimodal ingestion pipe: parse and embed. Admin only."},
    },
    {
        "id":   "memento_query",
        "name": "Memento Query",
        "file": "query_pipe.py",
        "meta": {"description": "Multimodal RAG query pipe: retrieve, rerank and generate with citations."},
    },
    {
        "id":   "memento_thorough",
        "name": "Thorough check",
        "file": "thorough_filter.py",
        # hover text of the button; same as thorough_filter.py's frontmatter (live answers think only with Thorough
        # check on, so it names both steps)
        "meta": {"description": "Takes time to think the question through, then checks each statement against the "
                                "sources. Answers take longer."},
    },
    {
        "id":   "qdrant_manager",
        "name": "Qdrant Manager",
        "file": "qdrant_pipe.py",
        "meta": {"description": "Admin utility for managing Qdrant collections. Admin only."},
    },
]

def _start_suggestions(domain: str) -> list[dict]:
    """A library's start-screen suggestions (default model only): its domain profile's `start_suggestions`, else the
    generic profile's."""
    return load_domain(domain).get("start_suggestions") or load_domain("generic").get("start_suggestions") or []


QUERY_SUGGESTIONS = _start_suggestions(_env_value("DOMAIN_NAME", "polymer-engineering"))

# Memento Document Chat's start-screen suggestions (without them Open WebUI shows its defaults)
DOC_CHAT_SUGGESTIONS = [
    {"title": ["Summarize the file", "in a few paragraphs"], "content": "Summarize the attached file."},
    {"title": ["Main findings", "and the evidence for them"],
     "content": "What are the main findings of the attached file, and what evidence supports them?"},
    {"title": ["Methods and data", "what was done and how"],
     "content": "Which methods, materials and data does the attached file use?"},
]

MODELS = [
    {
        "id":            "memento-ingestion-prod",
        "name":          "Library Upload",
        "base_model_id": "memento_ingestion",
        "meta": {
            "description": "Add documents to the shared library: PDF, Word, PowerPoint or Excel.",
            "capabilities": {"vision": False, "file_context": False},   # no images in the chat box
        },
        "params": {},
        "groups": UPLOAD_GROUPS,
    },
    # one workspace model per Memento Query variant; the local one keeps the id memento-query, so the default model and
    # existing chats keep working
    *({
        "id":            workspace_id(v),
        "name":          v.name,
        "base_model_id": f"memento_query.{v.id}",
        "meta": {
            "description": v.full_description,
            # no images in the chat box; no attachments at all (Library Chat cannot read them, and one re-sent to
            # Library Upload later in the same chat would be stored twice)
            "capabilities": {"vision": False, "file_context": False, "file_upload": False},
            "filterIds": ["memento_thorough"],   # "Thorough check" toggle button in the chat input bar (RQ3)
            **({} if v.is_cloud else {"suggestion_prompts": QUERY_SUGGESTIONS}),
        },
        "params": {},
        "groups": UPLOAD_GROUPS,
    } for v in VARIANTS),
    # Memento Document Chat: attach files and ask about them, privately (the Query pipe's manifold entry
    # "document-chat"); the description is the privacy message
    {
        "id":            DOC_CHAT_ID,
        "name":          "Document Chat",
        "base_model_id": "memento_query.document-chat",
        "meta": {
            "description": ("Attach a document and ask about it. This file is stored only by Memento on the chair's "
                            "server and used only in this chat. It is not added to the library and is deleted when "
                            "you delete the chat. Document Chat uses only the local model for reasoning and "
                            "answering."),
            "capabilities": {"vision": False, "file_context": False},   # no images in the chat box
            "filterIds": ["memento_thorough"],
            "suggestion_prompts": DOC_CHAT_SUGGESTIONS,
        },
        "params": {},
        "groups": DOC_CHAT_GROUPS,
    },
    {
        "id":            "qdrant-manager",
        "name":          "Qdrant Manager",
        "base_model_id": "qdrant_manager",
        "meta": {
            "description": "Manage Qdrant collections. Type 'help' for commands. Qdrant must be running first.",
            "capabilities": {"file_context": False},
        },
        "params": {},
    },
    # The raw pipes the two Memento models wrap. Since Open WebUI 0.11 a user needs read access to every model in the
    # base_model_id chain; a pipe with no model entry is admin-only ("Model not found" for everyone else). These entries
    # grant the groups access and stay hidden from the model picker.
    *({
        "id":            fn_id,
        "name":          f"{name} (pipe)",
        "base_model_id": None,
        "meta": {"hidden": True, "capabilities": {"file_context": False}},
        "params": {},
        "groups": DOC_CHAT_GROUPS if fn_id == "memento_query.document-chat" else UPLOAD_GROUPS,
    } for fn_id, name in ((("memento_ingestion", "Memento Ingestion"),)
                          + tuple((f"memento_query.{v.id}", f"Memento Query {v.id}") for v in VARIANTS)
                          + (("memento_query.document-chat", "Memento Document Chat"),))),
]


def signin_or_signup(base_url: str, email: str, password: str) -> str:
    r = requests.post(f"{base_url}/api/v1/auths/signin",
                      json={"email": email, "password": password}, timeout=10)
    if r.status_code == 200:
        token = r.json().get("token")
        if token:
            return token
    print("  Signin failed — attempting to create admin account (first-time setup) ...")
    r2 = requests.post(f"{base_url}/api/v1/auths/signup",
                       json={"name": "Admin", "email": email, "password": password},
                       timeout=10)
    if r2.status_code == 200:
        print("  ✓ Admin account created")
        r3 = requests.post(f"{base_url}/api/v1/auths/signin",
                           json={"email": email, "password": password}, timeout=10)
        if r3.status_code == 200:
            token = r3.json().get("token")
            if token:
                return token
    print(f"  ERROR: could not sign in or create account ({r.status_code}): {r.text[:200]}")
    sys.exit(1)


# ── Valve helpers ────────────────────────────────────────────────────────────

def get_valves(base_url: str, headers: dict, fn_id: str) -> dict | None:
    r = requests.get(f"{base_url}/api/v1/functions/id/{fn_id}/valves",
                     headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json()
    return None


def set_valves(base_url: str, headers: dict, fn_id: str, valves: dict) -> bool:
    r = requests.post(f"{base_url}/api/v1/functions/id/{fn_id}/valves/update",
                      json=valves, headers=headers, timeout=10)
    if r.status_code == 200:
        print(f"  ✓ Valves set for '{fn_id}'")
        return True
    print(f"  ⚠ Valve update failed for '{fn_id}' ({r.status_code}): {r.text[:200]}")
    return False


# ── Function code helpers ────────────────────────────────────────────────────

def get_function_code(base_url: str, headers: dict, fn_id: str) -> str | None:
    r = requests.get(f"{base_url}/api/v1/functions/id/{fn_id}",
                     headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json().get("content")
    return None


def push_function(base_url: str, headers: dict, fn: dict, activate: bool = True) -> bool:
    """Create the function or update its code (an update keeps its active state). `activate`: switch a newly created one
    on (the default copy); the second copy is created off and switched on as setup's last step."""
    content = (PIPELINES_DIR / fn["file"]).read_text()
    form_data = {"id": fn["id"], "name": fn["name"], "content": content,
                 "meta": fn["meta"]}
    create_payload = {**form_data, "is_active": activate}
    r = requests.post(f"{base_url}/api/v1/functions/create",
                      json=create_payload, headers=headers, timeout=30)
    if r.status_code == 200 and r.json():
        print(f"  ✓ Function '{fn['name']}' created" + ("" if activate else " (inactive)"))
        if activate and not r.json().get("is_active"):   # OpenWebUI creates filters inactive; a toggle needs it active
            requests.post(f"{base_url}/api/v1/functions/id/{fn['id']}/toggle", headers=headers, timeout=10)
        return True
    r2 = requests.post(f"{base_url}/api/v1/functions/id/{fn['id']}/update",
                       json=form_data, headers=headers, timeout=30)
    if r2.status_code == 200:
        print(f"  ↻ Function '{fn['name']}' updated")
        return True
    print(f"  ERROR: function '{fn['name']}' update failed ({r2.status_code}): {r2.text[:200]}")
    return False


# ── The three sync modes ─────────────────────────────────────────────────────

def cmd_check(base_url: str, headers: dict, functions: list[dict]):
    """Compare disk vs Postgres — code diff + valve drift. Changes nothing."""
    all_synced = True
    for fn in functions:
        print(f"\n{'─'*60}")
        print(f"  {fn['name']}  ({fn['id']})")
        print(f"{'─'*60}")

        disk_code = (PIPELINES_DIR / fn["file"]).read_text()
        db_code = get_function_code(base_url, headers, fn["id"])

        if db_code is None:
            print("  Code: NOT IN POSTGRES (needs initial push)")
            all_synced = False
        elif disk_code.strip() == db_code.strip():
            print("  Code: IN SYNC ✓")
        else:
            all_synced = False
            disk_lines = disk_code.splitlines(keepends=True)
            db_lines = db_code.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                db_lines, disk_lines,
                fromfile=f"postgres:{fn['id']}",
                tofile=f"disk:{fn['file']}",
                n=1,
            ))
            added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
            removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
            print(f"  Code: OUT OF SYNC — {added} lines added, {removed} lines removed on disk")

        valves = get_valves(base_url, headers, fn["id"])
        if valves:
            print(f"  Valves ({len(valves)} keys):")
            for k, v in sorted(valves.items()):
                display = v if not isinstance(v, str) or len(str(v)) < 60 else str(v)[:57] + "..."
                print(f"    {k:25s} = {display}")
        else:
            print("  Valves: none saved (will use code defaults)")

    print()
    if all_synced:
        print("All functions: code in sync ✓")
    else:
        print("Some functions are out of sync. Use 'overwrite' or 'keep-valves' to push.")
    print()


def compute_valve_defaults(fn: dict) -> dict | None:
    """Compute Valve defaults by loading .env into os.environ and then
    importing the function module to instantiate its Valves class.
    This runs locally — independent of the OpenWebUI process's env."""

    env = load_env()
    old_env = {}
    for k, v in env.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        # Ensure the project root is on sys.path so `memento.config.settings` etc. resolve
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        source = (PIPELINES_DIR / fn["file"]).read_text()

        # Load the module the same way OpenWebUI does — exec into a fresh namespace
        import types
        module = types.ModuleType(f"_valve_extract_{fn['id']}")
        module.__dict__["__file__"] = str(PIPELINES_DIR / fn["file"])
        exec(source, module.__dict__)

        # Find the Pipe class and its Valves
        Pipe = getattr(module, "Pipe", None)
        if Pipe is None:
            return None
        Valves = getattr(Pipe, "Valves", None)
        if Valves is None:
            return None

        instance = Valves()
        return instance.model_dump()
    except Exception as e:
        print(f"  ⚠ Could not compute valve defaults: {e}")
        return None
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def cmd_overwrite(base_url: str, headers: dict, functions: list[dict]):
    """Push code from disk, reset valves to .env defaults."""
    for fn in functions:
        print(f"\n  Pushing {fn['name']}...")
        if push_function(base_url, headers, fn):
            defaults = compute_valve_defaults(fn)
            if defaults:
                if set_valves(base_url, headers, fn["id"], defaults):
                    for k, v in sorted(defaults.items()):
                        print(f"    {k:25s} = {v}")
            else:
                print(f"  ⚠ Could not compute valve defaults — valves unchanged")
    print("\nDone. Valves overwritten with .env defaults.\n")


def cmd_keep_valves(base_url: str, headers: dict, functions: list[dict]):
    """Push code from disk, preserve existing Postgres valve values."""
    for fn in functions:
        print(f"\n  Pushing {fn['name']}...")
        saved_valves = get_valves(base_url, headers, fn["id"])
        if push_function(base_url, headers, fn):
            if saved_valves:
                set_valves(base_url, headers, fn["id"], saved_valves)
            else:
                print(f"  (no saved valves to restore — using .env defaults)")
    print("\nDone. Existing valve values preserved.\n")


# ── Models / OpenAI / bypass (full setup only) ──────────────────────────────

def create_model(base_url: str, headers: dict, m: dict, group_ids: dict | None = None) -> bool:
    meta = {**m["meta"]}
    meta.setdefault("profile_image_url", None)
    grants = [{"principal_type": "group", "principal_id": group_ids[g], "permission": "read"}
              for g in m.get("groups", []) if group_ids and g in group_ids]
    payload = {
        "id": m["id"], "name": m["name"], "base_model_id": m["base_model_id"],
        "meta": meta, "params": m["params"], "access_grants": grants, "is_active": True,
    }
    r = requests.post(f"{base_url}/api/v1/models/create",
                      json=payload, headers=headers, timeout=10)
    if r.status_code == 200 and r.json():
        print(f"  ✓ Model '{m['name']}' created")
        return True
    r2 = requests.post(f"{base_url}/api/v1/models/model/update?id={m['id']}",
                       json=payload, headers=headers, timeout=10)
    if r2.status_code == 200 and r2.json():
        print(f"  ↻ Model '{m['name']}' updated")
        return True
    print(f"  ERROR: model '{m['name']}' failed ({r2.status_code}): {r2.text[:200]}")
    return False


def configure_openai(base_url: str, headers: dict, api_key: str, api_base_url: str) -> bool:
    payload = {
        "ENABLE_OPENAI_API":    True,
        "OPENAI_API_BASE_URLS": [api_base_url],
        "OPENAI_API_KEYS":      [api_key],
        "OPENAI_API_CONFIGS":   {},
    }
    r = requests.post(f"{base_url}/openai/config/update",
                      json=payload, headers=headers, timeout=10)
    if r.status_code == 200:
        print(f"  ✓ OpenAI connection set ({api_base_url})")
        return True
    print(f"  ERROR: OpenAI config failed ({r.status_code}): {r.text[:200]}")
    return False


def configure_bypass(base_url: str, headers: dict) -> bool:
    payload = {"BYPASS_EMBEDDING_AND_RETRIEVAL": True}
    r = requests.post(f"{base_url}/api/v1/retrieval/config/update",
                      json=payload, headers=headers, timeout=10)
    if r.status_code == 200:
        print("  ✓ Bypass Embedding and Retrieval = ON")
        return True
    print(f"  ERROR: retrieval config failed ({r.status_code}): {r.text[:200]}")
    return False


# ── RQ4 trial ────────────────────────────────────────────────────────────────

def trial_valves(fn_id: str, domain: str, env: dict) -> dict:
    """Valve values the trial needs, on top of what is saved (everything else — URLs, models — is kept)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import yaml
    d = yaml.safe_load((PROJECT_ROOT / "config" / "domains" / f"{domain}.yaml").read_text())
    common = {"DOMAIN_NAME": domain, "COLLECTION": f"{d['collection_prefix']}-rag"}
    if fn_id == "memento_query":
        return {**common, "GRAPH_RETRIEVAL": False,                  # RQ2: graphs not adopted
                "THINKING_AUTO": True,                               # router skips thinking for simple questions
                "VALIDATION_ENABLED": True, "VALIDATION_STRATEGY": "A",   # RQ3's best configuration: verify → revise
                "VALIDATION_MAX_ROUNDS": 2,
                # the answer model of config/.env, and the load its warm-up reads
                **(FLASH_NEXT_QUERY if FLASH_NEXT_URL else {})}
    if fn_id == "memento_ingestion":
        return {**common, "SKIP_PARSE": False, "SKIP_EMBED": False, "FORCE_REEMBED": False,
                "FORCE_REPARSE": False, "GRAPH_BUILD": False,
                "BACKEND": env.get("MINERU_BACKEND", "vlm-engine"), "CONTRIBUTOR_GROUP": ", ".join(UPLOAD_GROUPS),
                # figure descriptions and gap tags on the answer model of config/.env
                **(FLASH_NEXT if FLASH_NEXT_URL else {})}
    return {}


# ── What members see ──
# Default user permissions (the groups set none, so these apply to every non-admin): no chat controls (system prompt,
# parameters, valves), no continue / read aloud / voice mode / side-by-side models, no web-page attach, no code
# interpreter, memories, notes or calendar; no dictation (on plain HTTP browsers refuse the microphone). Kept: uploads,
# rating, copy, edit, regenerate, share links, export/import, folders, search.
TRIAL_PERMISSIONS = {
    "chat": {"controls": False, "valves": False, "system_prompt": False, "params": False, "continue_response": False,
             "tts": False, "call": False, "stt": False, "multiple_models": False, "web_upload": False},
    "features": {"code_interpreter": False, "memories": False, "notes": False, "calendar": False},
}
# global switches: no publishing chats to openwebui.com, no presence/status, notes/calendar/memories off
TRIAL_GLOBAL = {"ENABLE_COMMUNITY_SHARING": False, "ENABLE_USER_STATUS": False, "ENABLE_NOTES": False,
                "ENABLE_CALENDAR": False, "ENABLE_MEMORIES": False}
TRIAL_BANNER = {"id": "memento-trial", "type": "info", "title": "", "dismissible": True,
                "content": "Memento is a research prototype for a Master's thesis. Usage is logged for evaluation. "
                           "Please don't upload confidential documents."}


def apply_trial_ui(base_url: str, headers: dict, dry: bool) -> None:
    """Read → change → write for each setting, printing what changes; everything else is kept."""
    import time

    def get(path):
        r = requests.get(f"{base_url}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def post(path, body, what):
        if dry:
            return
        r = requests.post(f"{base_url}{path}", headers=headers, json=body, timeout=10)
        print(f"  ✓ {what}" if r.status_code == 200 else f"  ERROR {what} ({r.status_code}): {r.text[:200]}")

    perms = get("/api/v1/users/default/permissions")
    changes = [f"{sec}.{k}" for sec, vals in TRIAL_PERMISSIONS.items() for k, v in vals.items()
               if perms.get(sec, {}).get(k) != v]
    for sec, vals in TRIAL_PERMISSIONS.items():
        perms.setdefault(sec, {}).update(vals)
    print(f"  user permissions off: {', '.join(changes) or 'none to change'}")
    if changes:
        post("/api/v1/users/default/permissions", perms, "default user permissions")

    # permissions add up across a user's groups (anything one group allows is allowed), so a group with its own
    # permission set (the default "Users" group from the first setup allows everything) gets the same switches off
    for g in get("/api/v1/groups/"):
        full = get(f"/api/v1/groups/id/{g['id']}")
        gp = full.get("permissions")
        if not gp:
            continue
        changes = [f"{sec}.{k}" for sec, vals in TRIAL_PERMISSIONS.items() for k, v in vals.items()
                   if gp.get(sec, {}).get(k) is not False]
        if not changes:
            continue
        for sec, vals in TRIAL_PERMISSIONS.items():
            gp.setdefault(sec, {}).update(vals)
        print(f"  group '{full['name']}' permissions off: {', '.join(changes)}")
        post(f"/api/v1/groups/id/{g['id']}/update", {"name": full["name"], "description": full.get("description") or "",
                                                      "permissions": gp, "data": full.get("data")},
             f"group '{full['name']}' permissions")

    cfg = get("/api/v1/auths/admin/config")
    changes = [k for k, v in TRIAL_GLOBAL.items() if cfg.get(k) != v]
    cfg.update(TRIAL_GLOBAL)
    print(f"  global switches off: {', '.join(changes) or 'none to change'}")
    if changes:
        post("/api/v1/auths/admin/config", cfg, "global switches")

    sync_model_order(base_url, headers, dry)

    # follow-up suggestions and chat tags off (tags: one extra model call per chat, seen only in the sidebar search);
    # chat titles stay on
    tasks = get("/api/v1/tasks/config")
    off = [k for k in ("ENABLE_FOLLOW_UP_GENERATION", "ENABLE_TAGS_GENERATION") if tasks.get(k) is not False]
    if off:
        print(f"  {', '.join(off)}: on → off (titles unchanged)")
        tasks.update(dict.fromkeys(off, False))
        post("/api/v1/tasks/config/update", tasks, "follow-ups and tags off")

    current = get("/api/v1/configs/banners")
    # unchanged banner → no re-post (a new timestamp would show it again to everyone who dismissed it)
    if not any(b.get("id") == TRIAL_BANNER["id"] and b.get("content") == TRIAL_BANNER["content"] for b in current):
        print("  banner:", TRIAL_BANNER["content"])
        others = [b for b in current if b.get("id") != TRIAL_BANNER["id"]]
        post("/api/v1/configs/banners", {"banners": others + [{**TRIAL_BANNER, "timestamp": int(time.time())}]}, "banner")

    for m in MODELS:
        if m["id"] not in MODEL_ORDER:
            continue
        r = requests.get(f"{base_url}/api/v1/models/model?id={m['id']}", headers=headers, timeout=10)
        if r.status_code == 404:          # not created yet (a dry run before sync-models): created with its description
            print(f"  {m['name']}: new, created with its description")
            continue
        r.raise_for_status()
        live = r.json()
        if (live.get("meta") or {}).get("description") == m["meta"]["description"]:
            continue
        print(f"  {m['name']} description → {m['meta']['description']!r}")
        body = {k: live[k] for k in ("id", "name", "base_model_id", "meta", "params", "is_active") if k in live}
        body["meta"] = {**live["meta"], "description": m["meta"]["description"]}
        body["access_grants"] = [{k: g[k] for k in ("principal_type", "principal_id", "permission")}
                                 for g in live.get("access_grants") or []]
        post(f"/api/v1/models/model/update?id={m['id']}", body, f"{m['name']} description")


def ensure_group(base_url: str, headers: dict, name: str, dry: bool) -> str | None:
    r = requests.get(f"{base_url}/api/v1/groups/", headers=headers, timeout=10)
    r.raise_for_status()
    for g in r.json():
        if g["name"] == name:
            print(f"  = group '{name}' exists")
            return g["id"]
    if dry:
        print(f"  + would create group '{name}'")
        return "<new>"
    r = requests.post(f"{base_url}/api/v1/groups/create", headers=headers, timeout=10,
                      json={"name": name, "description": "Memento trial: may upload documents and ask questions"})
    r.raise_for_status()
    print(f"  ✓ group '{name}' created")
    return r.json()["id"]


def add_accounts(base_url: str, headers: dict, path: Path, group_id: str, dry: bool) -> None:
    """name,email per line → user accounts (role user) in the contributors group; new passwords are written
    to ACCOUNTS_OUT (mode 600) and never printed. Existing accounts are only added to the group."""
    people = [(row[0].strip(), row[1].strip().lower()) for row in csv.reader(path.open())
              if len(row) >= 2 and "@" in row[1]]
    r = requests.get(f"{base_url}/api/v1/users/all", headers=headers, timeout=10)
    r.raise_for_status()
    existing = {u["email"].lower(): u["id"] for u in r.json()["users"]}
    new_creds, ids = [], []
    for name, email in people:
        if email in existing:
            print(f"  = {email} exists — will be in '{CONTRIBUTORS}'")
            ids.append(existing[email])
            continue
        if dry:
            print(f"  + would create {email}")
            continue
        pw = secrets.token_urlsafe(12)
        r = requests.post(f"{base_url}/api/v1/auths/add", headers=headers, timeout=10,
                          json={"name": name, "email": email, "password": pw, "role": "user"})
        if r.status_code != 200:
            print(f"  ERROR: {email} ({r.status_code}): {r.text[:150]}")
            continue
        ids.append(r.json()["id"])
        new_creds.append((name, email, pw))
        print(f"  ✓ {email} created")
    if dry or not ids:
        return
    r = requests.post(f"{base_url}/api/v1/groups/id/{group_id}/users/add", headers=headers, timeout=10,
                      json={"user_ids": ids})
    r.raise_for_status()
    print(f"  ✓ {len(ids)} account(s) in '{CONTRIBUTORS}'")
    if new_creds:
        ACCOUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        new_file = not ACCOUNTS_OUT.exists()
        with ACCOUNTS_OUT.open("a") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["name", "email", "password"])
            w.writerows(new_creds)
        os.chmod(ACCOUNTS_OUT, 0o600)
        print(f"  → passwords for {len(new_creds)} new account(s) in {ACCOUNTS_OUT.relative_to(PROJECT_ROOT)}")


def reset_document_list(env: dict, dry: bool, collection: str, engine=None) -> None:
    """The served library's document list (memento_documents, memento/library/documents.py; per collection, so switching
    libraries needs no reset): its rows are saved as CSV, then the list and the library's stored answers are emptied;
    the query pipe rebuilds the list from the collection on the next question."""
    url = env.get("DATABASE_URL", "")
    if engine is None and not url:
        print("  ⚠ DATABASE_URL not set in config/.env — document list not reset")
        return
    from datetime import datetime
    sys.path.insert(0, str(PROJECT_ROOT))
    from memento.library import documents as ldb
    if engine is None:
        from sqlalchemy import create_engine
        engine = create_engine(url)
    ldb.ensure_tables(engine)
    rows = ldb.list_documents(engine, collection)
    if dry:
        print(f"  (dry run) would save + empty the document list of {collection} ({len(rows)} entries)")
        return
    if rows:
        out = PROJECT_ROOT / "data" / "rq4" / f"memento_documents_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["collection", "source", "title", "content_hash", "uploaded_by", "added_at"])
            w.writerows((collection, r["source"], r["title"], r["content_hash"], r["uploaded_by"],
                         r["added_at"].isoformat() if r["added_at"] else "") for r in rows)
        print(f"  ✓ {len(rows)} entries saved to {out.relative_to(PROJECT_ROOT)}")
    ldb.reset_library(engine, collection=collection, output_root=env.get("OUTPUT_ROOT") or PROJECT_ROOT / "data" / "output")
    print(f"  ✓ document list of {collection} emptied (rebuilt from the collection on the next question)")


def sync_model_order(base_url: str, headers: dict, dry: bool, study: bool = False) -> bool:
    """Default model memento-query, picker order = MODEL_ORDER (local entries first, then the cloud Query models). With
    the second copy enabled: defaults are both copies' Library Chats, today's first (the frontend picks the first one the
    user may see, so each group lands on its own Library Chat) and the second copy's models after the default ones in the
    order; without it, its ids are gone from both."""
    r = requests.get(f"{base_url}/api/v1/configs/models", headers=headers, timeout=10)
    r.raise_for_status()
    cfg = r.json()
    default = "memento-query" + (f",{STUDY_IDS['query']}" if study else "")
    order = MODEL_ORDER + (STUDY_MODEL_ORDER if study else [])
    if cfg.get("DEFAULT_MODELS") == default and cfg.get("MODEL_ORDER_LIST") == order:
        print("  = default model and order unchanged")
        return True
    print(f"  default model '{default}', order {order}")
    if dry:
        return True
    cfg.update({"DEFAULT_MODELS": default, "MODEL_ORDER_LIST": order})
    r = requests.post(f"{base_url}/api/v1/configs/models", headers=headers, json=cfg, timeout=10)
    print("  ✓ default model and order" if r.status_code == 200 else f"  ERROR default model ({r.status_code}): {r.text[:200]}")
    return r.status_code == 200


def remove_unoffered(base_url: str, headers: dict, models: dict[str, str], dry: bool) -> list[str]:
    """Delete the workspace models of cloud variants the config no longer offers (disabled, or the key removed): Open
    WebUI would keep listing them although their answer function is gone. `models`: workspace id → the base it must
    have (this library's Library Chat function and the variant); a record with another base is kept and reported.
    Returns the ids that could not be removed."""
    failed = []
    for mid, base in models.items():
        r = requests.get(f"{base_url}/api/v1/models/model?id={mid}", headers=headers, timeout=10)
        if _absent(r):
            continue
        if r.status_code != 200:
            failed.append(mid)
        elif (r.json() or {}).get("base_model_id") != base:
            print(f"  ⚠ model {mid} kept: not offered any more, but it belongs to "
                  f"{(r.json() or {}).get('base_model_id')!r}, not {base!r}")
        elif dry:
            print(f"  (dry run) model {mid}: not offered any more, would be removed")
        elif _delete_model(base_url, headers, mid):
            print(f"  ✓ model {mid} removed: not offered any more")
        else:
            failed.append(mid)
    return failed


def shared_model_ids(default_config, study_config) -> set[str]:
    """Workspace model ids both libraries would use (every entry of their configs, offered or not): such a clash would
    let one library's models overwrite or remove the other's."""
    def ids(path, base, fixed):
        return set(fixed) | {workspace_id(v, base) for v in load_variants(path) + unoffered_cloud_variants(path)}
    return (ids(default_config, "memento-query", [m["id"] for m in MODELS] + [DOC_CHAT_ID])
            & ids(study_config, STUDY_IDS["query"], STUDY_IDS.values()))


def require_unique_model_ids() -> None:
    """Stop before any change, dry run included, if both libraries would use the same workspace model id."""
    clash = shared_model_ids(VARIANTS_CONFIG, PROJECT_ROOT / "config" / STUDY_VARIANTS_FILE)
    if clash:
        print(f"  ERROR: model ids used by both libraries: {', '.join(sorted(clash))}; rename the entry in "
              "config/query_models.yaml. Nothing changed.")
        raise SystemExit(1)


def cmd_sync_models(base_url: str, headers: dict, dry: bool) -> None:
    """The workspace models (names, descriptions, group access; one per Memento Query model in config/query_models.yaml)
    and the default model and order. Run by scripts/services.sh at every Open WebUI start, so editing the config and
    restarting is enough. Nothing else is touched: pipes, valves, permissions, banner, sign-up and the document list
    stay as they are. Models edited by hand in Admin → Models are reset to the config."""
    print("\nModels:")
    require_unique_model_ids()
    failed = []                 # reported at the end with a nonzero exit (services.sh logs a warning)
    group_ids = {g: ensure_group(base_url, headers, g, dry) for g in UPLOAD_GROUPS}
    for m in MODELS:
        if dry:
            who = ", ".join(m.get("groups", [])) or "admins only"
            print(f"  (dry run) model '{m['name']}' ({m['id']}, base {m['base_model_id']}) → {who}")
        elif not create_model(base_url, headers, m, group_ids):
            failed.append(m["id"])
    failed += remove_unoffered(base_url, headers, {workspace_id(v): f"memento_query.{v.id}"
                                                   for v in unoffered_cloud_variants()}, dry)
    # the second copy: its models only while it is enabled; never the group or a function, so a removed or
    # half-installed copy is never brought back by a restart
    state = study_state(base_url, headers)
    print(f"  {STUDY_NAME} copy: {'enabled' if state else 'not enabled' if state is False else 'state unknown'}")
    if state:
        gid = _group_id(base_url, headers, STUDY_GROUP)
        for m in STUDY_MODELS:
            if dry:
                print(f"  (dry run) model '{m['name']}' ({m['id']}) → {STUDY_GROUP}")
            elif not gid or not create_model(base_url, headers, m, {STUDY_GROUP: gid}):
                failed.append(m["id"])
        stale = unoffered_cloud_variants(PROJECT_ROOT / "config" / STUDY_VARIANTS_FILE)
        failed += remove_unoffered(base_url, headers, {workspace_id(v, STUDY_IDS["query"]): f"{STUDY_QUERY_FN}.{v.id}"
                                                   for v in stale}, dry)
    if state is None:
        print("  ⚠ the study copy's state could not be read: default model and order left as they are")
        raise SystemExit(1)
    if not sync_model_order(base_url, headers, dry, study=bool(state)):
        failed.append("default model and order")
    if failed:
        print(f"  ⚠ not synced: {', '.join(failed)}")
        raise SystemExit(1)


def signup_settings(group_id: str, open_signup: bool) -> dict:
    """Open WebUI's sign-up settings for trial: people register themselves into the member group. By default a new
    account waits for an admin's approval (role pending), so a stranger who reaches the address cannot use Memento;
    open_signup (a moderated study) makes new accounts active at once."""
    return {"ENABLE_SIGNUP": True, "DEFAULT_USER_ROLE": "user" if open_signup else "pending", "DEFAULT_GROUP_ID": group_id}


def cmd_trial(base_url: str, headers: dict, domain: str, accounts: str | None, dry: bool, open_signup: bool = False):
    # this mode sets up the default copy only, and it empties the library's stored answers and document list: never
    # while the second copy is in use
    print("⚠ trial mode empties the library's stored answers and document list; do not run it while the library is in use")
    require_unique_model_ids()
    env = load_env()
    print(f"\nPipes (code from disk, saved valves kept, trial valves for domain '{domain}'):")
    for fn in FUNCTIONS:
        saved = get_valves(base_url, headers, fn["id"]) or {}
        want = trial_valves(fn["id"], domain, env)
        changes = {k: (saved.get(k), v) for k, v in want.items() if saved.get(k) != v}
        for k, (old, new) in changes.items():
            print(f"  {fn['id']}.{k}: {old!r} → {new!r}")
        if dry:
            print(f"  (dry run) {fn['id']}: code push + {len(changes)} valve change(s)")
            continue
        if push_function(base_url, headers, fn):
            set_valves(base_url, headers, fn["id"], {**saved, **want})

    print("\nGroups and models:")
    group_ids = {g: ensure_group(base_url, headers, g, dry) for g in UPLOAD_GROUPS}
    gid = group_ids[CONTRIBUTORS]
    for m in MODELS:
        who = ", ".join(m.get("groups", [])) or "admins only"
        if dry:
            print(f"  (dry run) model '{m['name']}' → {who}")
        else:
            create_model(base_url, headers, m, group_ids)

    print("\nSign-up:")
    r = requests.get(f"{base_url}/api/v1/auths/admin/config", headers=headers, timeout=10)
    r.raise_for_status()
    cfg = r.json()
    want = signup_settings(group_ids[TRIAL_GROUP], open_signup)
    for k, v in want.items():
        print(f"  {k}: {cfg.get(k)!r} → {v!r}")
    if not dry:
        cfg.update(want)
        r = requests.post(f"{base_url}/api/v1/auths/admin/config", headers=headers, json=cfg, timeout=10)
        print(f"  ✓ sign-up on, new accounts in '{TRIAL_GROUP}'"
              + ("" if open_signup else ", waiting for an admin's approval (Admin Panel → Users)") if r.status_code == 200
              else f"  ERROR ({r.status_code}): {r.text[:200]}")
    if not dry:
        configure_bypass(base_url, headers)

    print("\nTrial UI:")
    apply_trial_ui(base_url, headers, dry)

    print("\nDocument list (memento_documents):")
    reset_document_list(env, dry, trial_valves("memento_query", domain, env)["COLLECTION"])

    # the command line (run_pipeline.py, run_ingestion.sh) reads DOMAIN_NAME from config/.env: a different domain there
    # would ingest into another collection, with another domain's embedding instruction
    if env.get("DOMAIN_NAME") != domain:
        print(f"\n⚠ config/.env has DOMAIN_NAME={env.get('DOMAIN_NAME')!r}, not {domain!r}: set it to {domain!r} "
              "before pre-loading documents from the command line.")

    if accounts:
        print("\nAccounts:")
        add_accounts(base_url, headers, Path(accounts), gid, dry)
    print("\nDone." if not dry else "\nDry run — nothing changed.")


# ── Two libraries in one Open WebUI: the second copy ──
# The default copy (memento_query / memento_ingestion, its library and groups) stays as it is. The second copy runs the
# same code with its own settings (SECOND_LIBRARY_* in config/.env): its domain's library, the local model server, its
# own models and group (accounts made by hand). `study` mode sets it up (idempotent; created off, switched on last);
# sync-models (every Open WebUI start) keeps its models only while it is enabled; `remove-study` takes it down and
# archives its Document Chat rows (memento/document_chat/store.archive).
STUDY_ID = _env_value("SECOND_LIBRARY_ID", "polymer")                # its functions' and models' id suffix
STUDY_NAME = _env_value("SECOND_LIBRARY_NAME", "user study")         # its name in Open WebUI's admin lists
STUDY_GROUP = _env_value("SECOND_LIBRARY_GROUP", "User study")
STUDY_QUERY_FN, STUDY_INGEST_FN = f"memento_query_{STUDY_ID}", f"memento_ingestion_{STUDY_ID}"
STUDY_IDS = {"query": f"memento-query-{STUDY_ID}", "upload": f"memento-ingestion-{STUDY_ID}",
             "doc": f"memento-document-chat-{STUDY_ID}"}
STUDY_DOMAIN = _env_value("SECOND_LIBRARY_DOMAIN", "polymer-engineering")
STUDY_COLLECTION = default_collection(STUDY_DOMAIN)
# the answer model both libraries use (config/.env): REASONING_URL (the server, without /v1), REASONING_MODEL_NAME and
# the key in REASONING_API_KEY
ANSWER_SERVER = _env_value("REASONING_URL").rstrip("/").removesuffix("/v1")
FLASH_NEXT_URL = f"{ANSWER_SERVER}/v1" if ANSWER_SERVER else ""
FLASH_NEXT = {"LLM_BASE_URL": FLASH_NEXT_URL, "LLM_MODEL": _env_value("REASONING_MODEL_NAME", "qwen-flash-next"),
              "LLM_API_KEY_ENV": "REASONING_API_KEY"}
FLASH_NEXT_METRICS = f"{ANSWER_SERVER}/metrics" if ANSWER_SERVER else ""


def _require_flash_next() -> None:
    if not FLASH_NEXT_URL:
        raise SystemExit("ERROR: set REASONING_URL in config/.env (the answer model's server, without /v1)")


# the default library on the same server; its warm-up reads that server's load, as the second copy's does
FLASH_NEXT_QUERY = {**FLASH_NEXT, "METRICS_URL": FLASH_NEXT_METRICS,
                    "METRICS_API_KEY_ENV": FLASH_NEXT["LLM_API_KEY_ENV"]}
STUDY_VARIANTS_FILE = "query_models_study.yaml"
STUDY_VARIANTS = load_variants(PROJECT_ROOT / "config" / STUDY_VARIANTS_FILE)
STUDY_FUNCTIONS = [
    {"id": STUDY_INGEST_FN, "name": f"Memento Ingestion ({STUDY_NAME})", "file": "ingestion_pipe.py",
     "meta": {"description": f"Library Upload of the {STUDY_NAME}'s library ({STUDY_ID}). Same code as Memento "
                             "Ingestion."}},
    {"id": STUDY_QUERY_FN, "name": f"Memento Query ({STUDY_NAME})", "file": "query_pipe.py",
     "meta": {"description": f"Library Chat and Document Chat of the {STUDY_NAME}'s library ({STUDY_ID}). Same code "
                             "as Memento Query."}},
]
STUDY_QUERY_SUGGESTIONS = _start_suggestions(STUDY_DOMAIN)


def _study_models() -> list[dict]:
    """The study's workspace models, as today's MODELS with the study's ids and group (names as in the trial)."""
    today = {m["id"]: m for m in MODELS}
    out = [{**today["memento-ingestion-prod"], "id": STUDY_IDS["upload"], "base_model_id": STUDY_INGEST_FN,
            "groups": [STUDY_GROUP]}]
    for v in STUDY_VARIANTS:
        out.append({
            "id": workspace_id(v, STUDY_IDS["query"]), "name": v.name, "base_model_id": f"{STUDY_QUERY_FN}.{v.id}",
            "meta": {"description": v.full_description,
                     "capabilities": {"vision": False, "file_context": False, "file_upload": False},
                     "filterIds": ["memento_thorough"],
                     **({} if v.is_cloud else {"suggestion_prompts": STUDY_QUERY_SUGGESTIONS})},
            "params": {}, "groups": [STUDY_GROUP]})
    out.append({**today[DOC_CHAT_ID], "id": STUDY_IDS["doc"], "base_model_id": f"{STUDY_QUERY_FN}.document-chat",
                "groups": [STUDY_GROUP]})
    # the raw pipes the wrappers stand on (read access to the whole base_model_id chain, hidden)
    for fn_id, name in ((STUDY_INGEST_FN, f"Memento Ingestion ({STUDY_NAME})"),
                        *((f"{STUDY_QUERY_FN}.{v.id}", f"Memento Query {v.id} ({STUDY_NAME})") for v in STUDY_VARIANTS),
                        (f"{STUDY_QUERY_FN}.document-chat", f"Memento Document Chat ({STUDY_NAME})")):
        out.append({"id": fn_id, "name": f"{name} (pipe)", "base_model_id": None,
                    "meta": {"hidden": True, "capabilities": {"file_context": False}}, "params": {},
                    "groups": [STUDY_GROUP]})
    return out


STUDY_MODELS = _study_models()
STUDY_MODEL_ORDER = ([workspace_id(v, STUDY_IDS["query"]) for v in STUDY_VARIANTS if not v.is_cloud]
                     + [STUDY_IDS["doc"], STUDY_IDS["upload"]]
                     + [workspace_id(v, STUDY_IDS["query"]) for v in STUDY_VARIANTS if v.is_cloud])


def _pipe_module(name: str):
    """openwebui/functions/<name>.py as a plain module (never `function_…`: no worker starts)."""
    import importlib.util
    os.environ.setdefault("MEMENTO_DIR", str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(f"setup_{name}", PIPELINES_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_flash_next(base_url: str, headers: dict, dry: bool) -> None:
    """The default copy (memento_query, memento_ingestion) on the local model server: its model, its key and the load
    its warm-up reads; every other saved setting kept. Both are read before anything is written, so a missing function
    changes nothing. Its stored answers were written by the old model (the model is part of their key), so they are
    written again."""
    _require_flash_next()
    print("\nToday's copy on Qwen3.8 Flash-Next" + (" (dry run)" if dry else "") + ":")
    plan = [("memento_query", FLASH_NEXT_QUERY), ("memento_ingestion", FLASH_NEXT)]
    saved = {fid: _saved_valves(base_url, headers, fid) for fid, _ in plan}
    missing = [fid for fid, v in saved.items() if v is None]
    if missing:
        raise SystemExit(f"ERROR: no saved settings for {', '.join(missing)}; nothing changed")
    for fid, want in plan:
        changes = {k: v for k, v in want.items() if saved[fid].get(k) != v}
        print(f"  {fid}: {len(changes)} setting(s) to save")
        for k, v in sorted(changes.items()):
            print(f"    {k}: {saved[fid].get(k)!r} → {v!r}")
        if dry or not changes:
            continue
        if not set_valves(base_url, headers, fid, {**saved[fid], **want}):
            raise SystemExit(f"ERROR: {fid}'s settings could not be saved")
        back = _saved_valves(base_url, headers, fid) or {}
        if any(back.get(k) != v for k, v in want.items()):
            raise SystemExit(f"ERROR: {fid}'s saved settings differ from what was written")


def study_valves(comp_query: dict | None, comp_ingest: dict | None) -> tuple[dict, dict]:
    """(query, ingestion) Valves of the study's copy, every field explicit (OpenWebUI saves only explicit fields, and
    an unsaved one would take config/.env's comp-mat value): the pipe's defaults, today's copy's saved values, then the
    study's own library, model, key, ids and trial settings."""
    _require_flash_next()
    qmod, imod = _pipe_module("query_pipe"), _pipe_module("ingestion_pipe")
    q = {**qmod.Pipe.Valves().model_dump(), **{k: v for k, v in (comp_query or {}).items() if v is not None}}
    q.update(DOMAIN_NAME=STUDY_DOMAIN, COLLECTION=STUDY_COLLECTION, **FLASH_NEXT, METRICS_URL=FLASH_NEXT_METRICS,
             METRICS_API_KEY_ENV=FLASH_NEXT["LLM_API_KEY_ENV"], WARMUP_MAX_RUNNING=2,
             QUERY_MODEL_ID=STUDY_IDS["query"], UPLOAD_MODEL_ID=STUDY_IDS["upload"],
             DOC_CHAT_MODEL_ID=STUDY_IDS["doc"], INGESTION_FUNCTION_ID=STUDY_INGEST_FN,
             VARIANTS_FILE=STUDY_VARIANTS_FILE, GRAPH_RETRIEVAL=False, THINKING_AUTO=True,
             VALIDATION_ENABLED=True, VALIDATION_STRATEGY="A", VALIDATION_MAX_ROUNDS=2)
    # no page limit in the second copy's Document Chat, whatever the default library saved; a read still ends at
    # MinerU's 30 minutes (DOC_CHAT_STALE_S, memento/document_chat/chat.py)
    q["DOC_CHAT_MAX_PAGES"] = 0
    i = {**imod.Pipe.Valves().model_dump(), **{k: v for k, v in (comp_ingest or {}).items() if v is not None}}
    i.update(DOMAIN_NAME=STUDY_DOMAIN, COLLECTION=STUDY_COLLECTION, **FLASH_NEXT,
             QUERY_MODEL_ID=STUDY_IDS["query"], UPLOAD_MODEL_ID=STUDY_IDS["upload"],
             DOC_CHAT_MODEL_ID=STUDY_IDS["doc"], CONTRIBUTOR_GROUP=STUDY_GROUP, GRAPH_BUILD=False,
             SKIP_PARSE=False, SKIP_EMBED=False, FORCE_REEMBED=False, FORCE_REPARSE=False)
    q = {k: v for k, v in q.items() if k in qmod.Pipe.Valves.model_fields}
    i = {k: v for k, v in i.items() if k in imod.Pipe.Valves.model_fields}
    return q, i


# OpenWebUI 0.11.4 answers a missing function, model or group with 401 and this text (constants.ERROR_MESSAGES)
NOT_FOUND_DETAIL = "We could not find what you're looking for :/"


def _absent(r) -> bool:
    """The item does not exist: 404, or 401 with Open WebUI's "not found" text. Any other answer that is not 200
    (another 401, a wrong key; a 5xx) is a failed request, never taken as absence."""
    if r.status_code == 404:
        return True
    if r.status_code != 401:
        return False
    try:
        return (r.json() or {}).get("detail") == NOT_FOUND_DETAIL
    except Exception:
        return False


def _function(base_url: str, headers: dict, fn_id: str) -> dict | None:
    """The function, None when it does not exist; a failed request: RuntimeError."""
    r = requests.get(f"{base_url}/api/v1/functions/id/{fn_id}", headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json() or None
    if _absent(r):
        return None
    raise RuntimeError(f"OpenWebUI answered {r.status_code} for function {fn_id}")


def _saved_valves(base_url: str, headers: dict, fn_id: str) -> dict | None:
    """A function's saved Valves, None when it does not exist (or nothing is saved); a failed request: RuntimeError
    (unlike get_valves, which reads every failure as "none")."""
    r = requests.get(f"{base_url}/api/v1/functions/id/{fn_id}/valves", headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json()
    if _absent(r):
        return None
    raise RuntimeError(f"OpenWebUI answered {r.status_code} for the settings of {fn_id}")


def study_state(base_url: str, headers: dict):
    """True: the study copy is enabled (both functions installed, active, every required setting saved); False: not;
    None: its state cannot be read (then nothing that depends on it is changed)."""
    from memento.library import copies as cs
    try:
        for fid, role in ((STUDY_QUERY_FN, "query"), (STUDY_INGEST_FN, "ingestion")):
            f = _function(base_url, headers, fid)
            if not f or not f.get("is_active"):
                return False
            v = _saved_valves(base_url, headers, fid)          # a failed read: unknown (None), not "not set up"
            if not isinstance(v, dict) or cs.missing(v, role):
                return False
        return True
    except Exception as e:
        print(f"  ⚠ study copy state not readable: {e!r}")
        return None


def activate(base_url: str, headers: dict, fn_id: str, active: bool) -> None:
    """Switch a function on or off: read its state, toggle only when needed, read it again. A failed call or a wrong
    final state stops the setup (SystemExit), never reported as success."""
    f = _function(base_url, headers, fn_id)
    if not f:
        raise SystemExit(f"ERROR: {fn_id} is not installed")
    if bool(f.get("is_active")) == active:
        return
    r = requests.post(f"{base_url}/api/v1/functions/id/{fn_id}/toggle", headers=headers, timeout=10)
    if r.status_code != 200:
        raise SystemExit(f"ERROR: switching {fn_id} {'on' if active else 'off'} failed ({r.status_code})")
    f = _function(base_url, headers, fn_id)
    if not f or bool(f.get("is_active")) != active:
        raise SystemExit(f"ERROR: {fn_id} is still {'off' if active else 'on'} after switching")
    print(f"  ✓ {fn_id} switched {'on' if active else 'off'}")


def _group_id(base_url: str, headers: dict, name: str) -> str | None:
    r = requests.get(f"{base_url}/api/v1/groups/", headers=headers, timeout=10)
    r.raise_for_status()
    return next((g["id"] for g in r.json() if g["name"] == name), None)


def stop_study_updater(dry: bool) -> None:
    """The second library's gap updater (services.sh session gap-updater-<SECOND_LIBRARY_ID>), if it runs."""
    import subprocess
    if subprocess.run(["tmux", "has-session", "-t", f"gap-updater-{STUDY_ID}"], capture_output=True).returncode != 0:
        return
    print(f"  stop gap-updater-{STUDY_ID}" + (" (dry run)" if dry else ""))
    if not dry:
        subprocess.run(["tmux", "kill-session", "-t", f"gap-updater-{STUDY_ID}"], check=False)


def _temp_collection() -> str:
    from memento.document_chat.intake import temp_collection
    return temp_collection(STUDY_COLLECTION)          # computed from the library, never typed (N12)


def _transfer_or_stop(fn, *args, **kw):
    from memento.document_chat import store as td
    try:
        return fn(*args, **kw)
    except td.TransferConflict as e:
        for moved, there in e.conflicts:
            print(f"  CONFLICT {moved['chat_id']}/{moved['file_id']}:\n    moving: {moved}\n    there:  {there}")
        raise SystemExit(f"ERROR: {e}")
    except td.LockTimeout as e:
        raise SystemExit(f"ERROR: {e}")


def cmd_study(base_url: str, headers: dict, dry: bool, engine, output_root) -> None:
    """Set up (or update) the second copy, in this order: groups checked, group, functions (new ones off), their
    complete Valves, archived Document Chat rows restored, runtime files published, models, functions on, default model
    and order. Idempotent; a re-run never switches the copy off."""
    from memento.document_chat import chat, store as td
    print(f"\nSecond library ({STUDY_NAME}):" + (" (dry run)" if dry else ""))
    require_unique_model_ids()
    unknown = sorted({g for m in STUDY_MODELS for g in m.get("groups", [])} - {STUDY_GROUP})
    if unknown:
        raise SystemExit(f"ERROR: unknown group(s) in the study models: {', '.join(unknown)}")
    gid = ensure_group(base_url, headers, STUDY_GROUP, dry)
    # today's saved settings (a failed read stops here: the study copy must never get defaults by mistake)
    q, i = study_valves(_saved_valves(base_url, headers, "memento_query"),
                        _saved_valves(base_url, headers, "memento_ingestion"))
    for fn, want in ((STUDY_FUNCTIONS[0], i), (STUDY_FUNCTIONS[1], q)):
        saved = get_valves(base_url, headers, fn["id"]) if _function(base_url, headers, fn["id"]) else None
        changes = {k: v for k, v in want.items() if (saved or {}).get(k) != v}
        print(f"  {fn['id']}: {'update code' if saved is not None else 'install (off)'}, "
              f"{len(changes)} setting(s) to save")
        for k, v in sorted(changes.items()):
            print(f"    {k} = {v!r}")
        if dry:
            continue
        if not push_function(base_url, headers, fn, activate=False):
            raise SystemExit(f"ERROR: {fn['id']} could not be installed")
        if not set_valves(base_url, headers, fn["id"], {**(saved or {}), **want}):
            raise SystemExit(f"ERROR: {fn['id']}'s settings could not be saved")
    temp = _temp_collection()
    if dry:
        with engine.connect() as c:
            from sqlalchemy import func, select
            td.ensure_tables(engine)
            n = c.execute(select(func.count()).select_from(td.ARCHIVE)
                          .where(td.ARCHIVE.c.collection == temp)).scalar()
        print(f"  archived Document Chat rows to restore: {n}")
    else:
        qp = _pipe_module("query_pipe")
        from memento.query import workers
        r = _transfer_or_stop(td.restore, engine, temp, timeout_s=chat.DOC_CHAT_STALE_S)
        print(f"  Document Chat rows restored: {r['moved']} (already there: {r['identical']})")
        old_root = os.environ.get("OUTPUT_ROOT")
        os.environ["OUTPUT_ROOT"] = str(output_root)
        try:
            workers.publish_runtime(workers.valves_from(qp.Pipe.Valves, q), function_id=STUDY_QUERY_FN, engine=engine)
        finally:
            if old_root is None:
                os.environ.pop("OUTPUT_ROOT", None)
            else:
                os.environ["OUTPUT_ROOT"] = old_root
        # the pipe only logs a failed model-map write (its worker must go on): read it back here
        try:
            mapped = STUDY_QUERY_FN in (json.loads((Path(output_root) / "_runtime" / "query_models.json")
                                                   .read_text()).get("copies") or {})
        except (OSError, ValueError):
            mapped = False
        if not mapped:
            raise SystemExit("ERROR: the model map could not be written; the study copy stays off")
        print("  ✓ model map and library list published")
    failed = []
    for m in STUDY_MODELS:
        if dry:
            print(f"  (dry run) model '{m['name']}' ({m['id']}) → {STUDY_GROUP}")
        elif not create_model(base_url, headers, m, {STUDY_GROUP: gid}):
            failed.append(m["id"])
    if failed:
        raise SystemExit(f"ERROR: model(s) not registered: {', '.join(failed)}; the study copy stays off (run study "
                         "again)")
    if not dry:
        activate(base_url, headers, STUDY_INGEST_FN, True)
        activate(base_url, headers, STUDY_QUERY_FN, True)
    if not sync_model_order(base_url, headers, dry, study=True):
        raise SystemExit("ERROR: the default model and order could not be saved (run study again)")
    print("\nDone." if not dry else "\nDry run — nothing changed.")


def _delete_model(base_url: str, headers: dict, model_id: str) -> bool:
    """Deleted, or confirmed gone (a model already deleted counts as deleted; any other failure: False)."""
    r = requests.post(f"{base_url}/api/v1/models/model/delete", json={"id": model_id}, headers=headers, timeout=10)
    if r.status_code == 200 and r.json() is True:
        return True
    if _absent(r):
        return True
    return _absent(requests.get(f"{base_url}/api/v1/models/model?id={model_id}", headers=headers, timeout=10))


def cmd_remove_study(base_url: str, headers: dict, dry: bool, engine, backup_root, restore_from=None) -> None:
    """Take the second copy down, each step only after the one before succeeded: back up its settings and Document Chat
    rows; switch it off; stop its gap updater; wait for running file reads and cleaners (exclusive lifecycle lock), back
    the rows up again and archive them (transfer rule); delete its models, functions and group; drop it from the model
    map, default model and order; optionally put the default functions back from a deployment backup (`restore_from`:
    its functions.json). Vectors and files stay; accounts and chats stay."""
    from datetime import datetime
    from sqlalchemy import select
    from memento.document_chat import chat, store as td
    print(f"\nRemove the second library ({STUDY_NAME}):" + (" (dry run)" if dry else ""))
    temp = _temp_collection()
    td.ensure_tables(engine)
    with engine.connect() as c:
        rows = [dict(r) for r in c.execute(select(td.TEMP_DOCS).where(td.TEMP_DOCS.c.collection == temp)).mappings()]
    installed = [fid for fid in (STUDY_QUERY_FN, STUDY_INGEST_FN) if _function(base_url, headers, fid)]
    print(f"  installed: {', '.join(installed) or 'nothing'}; Document Chat rows to archive: {len(rows)}")
    if dry:
        print("\nDry run — nothing changed.")
        return
    bdir = Path(backup_root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_remove_study"
    bdir.mkdir(parents=True, exist_ok=False)
    dump = lambda p, data: p.write_text(json.dumps(data, indent=1, default=str, ensure_ascii=False))  # noqa: E731
    # a failed read stops here, before anything is switched off (never a backup with "none" in it)
    dump(bdir / "study_valves.json", {fid: _saved_valves(base_url, headers, fid) for fid in installed})
    dump(bdir / "study_rows_before_drain.json", rows)
    for fid in installed:
        activate(base_url, headers, fid, False)
    stop_study_updater(dry)
    r = _transfer_or_stop(td.archive, engine, temp, timeout_s=chat.DOC_CHAT_STALE_S,
                          before_move=lambda moving: dump(bdir / "study_rows_after_drain.json", moving))
    print(f"  Document Chat rows archived: {r['moved']} (already there: {r['identical']}); backup {bdir}")
    for m in STUDY_MODELS:
        if not _delete_model(base_url, headers, m["id"]):
            raise SystemExit(f"ERROR: model {m['id']} could not be deleted; its functions and group stay (switched "
                             "off): run remove-study again")
    for fid in installed:
        d = requests.delete(f"{base_url}/api/v1/functions/id/{fid}/delete", headers=headers, timeout=10)
        if d.status_code != 200:
            raise SystemExit(f"ERROR: {fid} could not be deleted ({d.status_code})")
    gid = _group_id(base_url, headers, STUDY_GROUP)
    if gid:
        d = requests.delete(f"{base_url}/api/v1/groups/id/{gid}/delete", headers=headers, timeout=10)
        if not (d.status_code == 200 or _absent(d)) and _group_id(base_url, headers, STUDY_GROUP):
            raise SystemExit(f"ERROR: the group '{STUDY_GROUP}' could not be deleted ({d.status_code}): run "
                             "remove-study again")
    root = os.environ.get("OUTPUT_ROOT")
    if root:
        from memento.query.models import merge_model_map
        merge_model_map(Path(root) / "_runtime" / "query_models.json", STUDY_QUERY_FN, None)
    if not sync_model_order(base_url, headers, dry, study=False):
        raise SystemExit("ERROR: the default model and order could not be saved: run remove-study again")
    if restore_from:
        saved = json.loads((Path(restore_from) / "functions.json").read_text())
        for fid in ("memento_query", "memento_ingestion"):
            f = saved.get(fid)
            if not f:
                continue
            cur = _function(base_url, headers, fid) or {}
            r2 = requests.post(f"{base_url}/api/v1/functions/id/{fid}/update", headers=headers, timeout=30,
                               json={"id": fid, "name": cur.get("name") or fid, "content": f["content"],
                                     "meta": cur.get("meta") or {}})
            if r2.status_code != 200 or not set_valves(base_url, headers, fid, f["valves"] or {}):
                raise SystemExit(f"ERROR: {fid} could not be restored from {restore_from}")
            activate(base_url, headers, fid, True)
            print(f"  ✓ {fid} restored from {restore_from}")
    print("\nDone: the study copy is removed (its Document Chat vectors and files are kept).")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manage Memento pipe functions in OpenWebUI.",
        epilog="Modes: check | overwrite | keep-valves (default)",
    )
    parser.add_argument("--url",      default="http://localhost:8086")
    parser.add_argument("--email",    help="admin login (else OPENWEBUI_API_KEY from config/.env)")
    parser.add_argument("--password")
    parser.add_argument("--domain",   default="generic", help="trial: domain YAML under config/domains/")
    parser.add_argument("--accounts", help="trial: CSV with name,email per line")
    parser.add_argument("--dry-run",  action="store_true", help="trial: show what would change")
    parser.add_argument("--open-signup", action="store_true",
                        help="trial: new accounts are active at once instead of waiting for an admin's approval")
    parser.add_argument("--only", metavar="FUNCTION_ID",
                        help=f"Target one function. Choices: {', '.join(f['id'] for f in FUNCTIONS)}")
    parser.add_argument("--restore-from", metavar="BACKUP_DIR",
                        help="remove-study: put today's functions back from this deployment backup (functions.json)")
    parser.add_argument("--backup-root", default=str(PROJECT_ROOT / "backups"),
                        help="remove-study: where its backups go")
    parser.add_argument("mode", nargs="?", default=None,
                        choices=["check", "overwrite", "keep-valves", "trial", "trial-ui", "sync-models", "study",
                                 "remove-study", "flash-next"],
                        help="check: diff disk vs Postgres. "
                             "overwrite: push code + reset valves to .env defaults. "
                             "keep-valves: push code + preserve existing valve values (default).")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\nConnecting to OpenWebUI at {base_url} ...")
    if args.email and args.password:
        token = signin_or_signup(base_url, args.email, args.password)
    else:
        token = load_env().get("OPENWEBUI_API_KEY", "")
        if not token:
            print("ERROR: give --email/--password or set OPENWEBUI_API_KEY in config/.env")
            sys.exit(1)
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{base_url}/api/v1/auths/", headers=headers, timeout=10)
    if r.status_code != 200 or r.json().get("role") != "admin":
        print(f"ERROR: not authenticated as an admin ({r.status_code})")
        sys.exit(1)
    print("  ✓ Signed in as admin")

    # Determine which functions to target
    if args.only:
        fn = next((f for f in FUNCTIONS if f["id"] == args.only), None)
        if fn is None:
            ids = ", ".join(f["id"] for f in FUNCTIONS)
            print(f"ERROR: unknown function id '{args.only}' — choices: {ids}")
            sys.exit(1)
        targets = [fn]
    else:
        targets = FUNCTIONS

    # A mode was given — run that mode and exit
    if args.mode:
        if args.mode == "check":
            cmd_check(base_url, headers, targets)
        elif args.mode == "overwrite":
            cmd_overwrite(base_url, headers, targets)
        elif args.mode == "keep-valves":
            cmd_keep_valves(base_url, headers, targets)
        elif args.mode == "trial-ui":
            print("\nTrial UI:" + (" (dry run)" if args.dry_run else ""))
            apply_trial_ui(base_url, headers, args.dry_run)
        elif args.mode == "trial":
            cmd_trial(base_url, headers, args.domain, args.accounts, args.dry_run, args.open_signup)
        elif args.mode == "sync-models":
            cmd_sync_models(base_url, headers, args.dry_run)
        elif args.mode == "flash-next":                       # the default library on the local model server
            cmd_flash_next(base_url, headers, args.dry_run)
        elif args.mode in ("study", "remove-study"):         # the second copy
            from sqlalchemy import create_engine
            env = load_env()
            url = os.environ.get("DATABASE_URL") or env.get("DATABASE_URL", "")
            root = os.environ.get("OUTPUT_ROOT") or env.get("OUTPUT_ROOT", "")
            if not url or not root:
                print("ERROR: DATABASE_URL and OUTPUT_ROOT are needed (source config/.env first)")
                sys.exit(1)
            os.environ.setdefault("OUTPUT_ROOT", root)
            if args.mode == "study":
                cmd_study(base_url, headers, args.dry_run, create_engine(url), root)
            else:
                cmd_remove_study(base_url, headers, args.dry_run, create_engine(url), args.backup_root,
                                 restore_from=args.restore_from)
        return

    # No mode given — two cases:
    #   --only without mode: default to keep-valves (routine code push)
    #   neither --only nor mode: full first-time setup
    if args.only:
        cmd_keep_valves(base_url, headers, targets)
        return

    # Full setup (no --only, no mode)
    require_unique_model_ids()
    env      = load_env()
    api_key  = env.get("OPENAI_API_KEY", "")
    api_base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key or api_key == "your-api-key-here":
        print("WARNING: OPENAI_API_KEY not set in config/.env — skipping OpenAI config.")
        api_key = None

    print("\nCreating functions (pipes):")
    for fn in FUNCTIONS:
        saved_valves = get_valves(base_url, headers, fn["id"])
        if push_function(base_url, headers, fn):
            if saved_valves:
                set_valves(base_url, headers, fn["id"], saved_valves)

    print("\nCreating workspace models:")
    for m in MODELS:
        create_model(base_url, headers, m)

    print("\nConfiguring settings:")
    if api_key:
        configure_openai(base_url, headers, api_key, api_base)
    configure_bypass(base_url, headers)

    print("\nDone. All Memento pipes, models, and settings configured.\n")


if __name__ == "__main__":
    main()
