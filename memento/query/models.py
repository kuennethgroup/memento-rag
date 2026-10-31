# memento/query/models.py: the Library Chat models users can pick
"""Variants of the Memento Query model, from config/query_models.yaml. The chosen variant only writes the answer; the
router, chat titles and Thorough check stay on the local model."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from memento.config.paths import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "config" / "query_models.yaml"
# first sentence of every model's description: what Memento does (the start screen shows the selected model's text); a
# cloud model adds a second line, its short speed hint and where it runs
LIBRARY_DESCRIPTION = "Ask about the shared library in English or German; every answer cites its sources."
CLOUD_DESCRIPTION = "Runs on a cloud service."   # a cloud entry's `host` line replaces it


@dataclass(frozen=True)
class Variant:
    id: str
    name: str
    label: str
    provider: str                  # local | cloud
    model: str
    description: str
    base_url_env: str | None = None
    key_env: str | None = None
    vision: bool = True            # figures go to the model as images
    thinking: bool = True          # the model has Qwen's thinking switch
    checker: str = "local"         # who runs Thorough check (only "local" is used so far)
    enabled: bool = True
    accent: str | None = None      # the model's colour in the chat box and the dropdown, light mode ("#rrggbb");
    accent_dark: str | None = None # dark mode (default: accent). None = ours: the theme green
    host: str | None = None        # a cloud model's line on where it runs (markdown); default CLOUD_DESCRIPTION

    @property
    def is_cloud(self) -> bool:
        return self.provider == "cloud"

    @property
    def full_description(self) -> str:
        if self.is_cloud:
            # one newline: Open WebUI shows each as a <br> (a blank line would give two)
            return f"{LIBRARY_DESCRIPTION}\n{self.description} {self.host or CLOUD_DESCRIPTION}".replace("  ", " ")
        return f"{LIBRARY_DESCRIPTION} {self.description}".strip()


def load_variants(path: Path | None = None) -> list[Variant]:
    """The enabled variants in config order; a cloud variant only while its key variable (key_env) is set, so an
    installation that configured no cloud service offers none. Raises ValueError on an unknown provider, a cloud variant
    without its env var names, duplicate ids, or not exactly one enabled local variant."""
    path = Path(path or CONFIG)
    variants = _read(path)
    for v in variants:
        if v.provider not in ("local", "cloud"):
            raise ValueError(f"variant {v.id}: provider must be local or cloud, not {v.provider!r}")
        if v.is_cloud and not (v.base_url_env and v.key_env):
            raise ValueError(f"variant {v.id}: a cloud variant needs base_url_env and key_env")
        for field in ("accent", "accent_dark"):
            value = getattr(v, field)
            if value is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", str(value)):
                raise ValueError(f"variant {v.id}: {field} must be a colour like \"#0369a1\", not {value!r}")
    ids = [v.id for v in variants]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate variant ids")
    enabled = [v for v in variants if v.enabled]
    if sum(not v.is_cloud for v in enabled) != 1:
        raise ValueError(f"{path}: exactly one enabled local variant is required")
    return [v for v in enabled if not v.is_cloud or os.getenv(v.key_env or "")]


def _read(path: Path) -> list[Variant]:
    fields = set(Variant.__dataclass_fields__)
    return [Variant(**{k: v for k, v in e.items() if k in fields}) for e in yaml.safe_load(path.read_text()) or []]


def unoffered_cloud_variants(path: Path | None = None) -> list[Variant]:
    """The cloud variants the config names but load_variants leaves out (disabled, or their key variable unset): their
    workspace models are removed at the next start."""
    path = Path(path or CONFIG)
    offered = {v.id for v in load_variants(path)}
    return [v for v in _read(path) if v.is_cloud and v.id not in offered]


def workspace_id(v: Variant, base: str = "memento-query") -> str:
    """The Open WebUI workspace model wrapping the variant: the copy's Library Chat id (`base`, its QUERY_MODEL_ID) for
    the local one, `<base>-<variant id>` for a cloud one ("memento-query", "memento-query-qwen35")."""
    return base if not v.is_cloud else f"{base}-{v.id}"


# the fields the browser script reads from the top-level "models" list
_TODAY_FIELDS = ("id", "name", "cloud", "accent", "accent_dark", "hint")


def map_entries(variants: list[Variant], *, base: str = "memento-query", collection: str | None = None,
                siblings: dict | None = None) -> list[dict]:
    """One copy's entries of the browser's model map: each Library Chat model (kind "query": workspace id, name,
    cloud, colours, the cloud dropdown hint) and the copy's Document Chat and Library Upload (kinds "doc", "upload"),
    each with the copy's collection and its three model ids (`siblings`: query, upload, doc) for the links."""
    siblings = dict(siblings or {"query": base, "upload": "memento-ingestion-prod", "doc": "memento-document-chat"})
    out = []
    for v in variants:
        light = v.accent.lower() if v.accent else None
        # the dropdown's info tooltip of a cloud model: its second description line as plain text
        hint = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", v.full_description.split("\n", 1)[1]) if v.is_cloud else None
        out.append({"id": workspace_id(v, base), "name": v.name, "cloud": v.is_cloud, "accent": light,
                    "accent_dark": v.accent_dark.lower() if v.accent_dark else light, "hint": hint,
                    "kind": "query", "collection": collection, "siblings": siblings})
    for kind, name in (("doc", "Document Chat"), ("upload", "Library Upload")):
        out.append({"id": siblings[kind], "name": name, "cloud": False, "accent": None, "accent_dark": None,
                    "hint": None, "kind": kind, "collection": collection, "siblings": siblings})
    return out


def merge_model_map(path: Path, function_id: str, entries: list[dict] | None) -> None:
    """OUTPUT_ROOT/_runtime/query_models.json for loader.js (served as /files/models): `copies` holds each copy's
    entries (this copy's replaced as a whole; entries None removes the copy), and the top-level `models` list every
    copy's Library Chat models with the shared fields only (the default copy first). Under a file lock (copies may write
    it at once) and written atomically."""
    import fcntl
    from memento.common.atomic_file import write_text_atomic
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        copies = data.get("copies") if isinstance(data.get("copies"), dict) else {}
        if entries is None:                      # a removed copy (setup_openwebui.py remove-study)
            copies.pop(function_id, None)
        else:
            copies[function_id] = entries
        order = sorted(copies, key=lambda fid: (fid != "memento_query", fid))
        models = [{k: e.get(k) for k in _TODAY_FIELDS} for fid in order for e in copies[fid]
                  if e.get("kind", "query") == "query"]
        write_text_atomic(path, json.dumps({"models": models, "copies": {fid: copies[fid] for fid in order}}))


def write_model_colours(path: Path, variants: list[Variant]) -> None:
    """The default copy's entries of the model map (see merge_model_map)."""
    merge_model_map(path, "memento_query", map_entries(variants))


def resolve(model_id: str | None, variants: list[Variant], base: str = "memento-query") -> Variant:
    """"memento_query.qwen35", "qwen35" or the workspace id ("memento-query-qwen35", with the copy's `base`) → that
    variant; anything unknown or disabled → the copy's local variant (a chat opened on a model that was later removed
    still gets an answer)."""
    mid = model_id or ""
    key = mid.split(".", 1)[-1]
    for v in variants:
        if key == v.id or mid == workspace_id(v, base):
            return v
    return next(v for v in variants if not v.is_cloud)
