"""Make the corpus domain an explicit, required command-line argument for ingestion and evaluation scripts.
memento/config/settings.py resolves DOMAIN (instruction, collection prefix, prompts) from DOMAIN_NAME at import time, and
domain_loader falls back to generic.yaml for an unknown name, so an unset or mistyped variable would change the
embedding instruction without any sign. Scripts call apply_domain_arg() before importing config, and print
domain_banner() after."""
import os
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

_DOMAINS_DIR = PROJECT_ROOT / "config" / "domains"


def apply_domain_arg(required: bool = True) -> str | None:
    """Read --domain from argv and set DOMAIN_NAME before config is imported.
    The script's argparse must also declare --domain (so it is accepted)."""
    argv = sys.argv[1:]
    name = None
    for i, a in enumerate(argv):
        if a == "--domain" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--domain="):
            name = a.split("=", 1)[1]
    if name is None:
        if required:
            known = sorted(p.stem for p in _DOMAINS_DIR.glob("*.yaml"))
            sys.exit(f"ERROR: --domain is required (one of: {', '.join(known)})")
        return None
    if not (_DOMAINS_DIR / f"{name}.yaml").exists():
        known = sorted(p.stem for p in _DOMAINS_DIR.glob("*.yaml"))
        sys.exit(f"ERROR: unknown --domain '{name}' (one of: {', '.join(known)})")
    env = os.environ.get("DOMAIN_NAME")
    if env and env != name:
        print(f"[domain] WARNING: DOMAIN_NAME={env} from the environment is overridden by --domain {name}",
              file=sys.stderr)
    os.environ["DOMAIN_NAME"] = name
    return name


def domain_banner() -> dict:
    """Print and return the resolved domain settings (for logs and provenance)."""
    from memento.config.settings import DOMAIN, INPUT_DIR, OUTPUT_DIR
    from memento.config.paths import describe
    info = {
        "domain": os.environ.get("DOMAIN_NAME"),
        "collection_prefix": DOMAIN["collection_prefix"],
        "instruction": DOMAIN["instruction"],
        "rerank_instruction": DOMAIN["rerank_instruction"],
        "embed_instruction_override": bool(os.environ.get("EMBED_INSTRUCTION")),
        "collection_prefix_override": bool(os.environ.get("COLLECTION_PREFIX")),
    }
    info |= {"input_dir": str(INPUT_DIR), "output_dir": str(OUTPUT_DIR)}
    print(f"[domain] {info['domain']}  collection prefix: {info['collection_prefix']}", flush=True)
    print(f"[domain] {describe(DOMAIN['name'], INPUT_DIR, OUTPUT_DIR)}", flush=True)
    print(f"[domain] instruction: {info['instruction']}", flush=True)
    if info["embed_instruction_override"]:
        print("[domain] WARNING: EMBED_INSTRUCTION override is active", flush=True)
    if info["collection_prefix_override"]:
        print("[domain] WARNING: COLLECTION_PREFIX override is active", flush=True)
    return info
