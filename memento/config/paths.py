"""One rule for where a corpus lives: documents in INPUT_ROOT/<corpus_dir>, parsed output in OUTPUT_ROOT/<corpus_dir>.
INPUT_ROOT / OUTPUT_ROOT come from config/.env (or an Open WebUI Valve); corpus_dir comes from the domain YAML
(config/domains/<name>.yaml). Everything that needs a corpus folder (config.py, the Open WebUI functions,
build_graph.py, scripts) calls corpus_dirs(); nothing else builds these paths.

Standalone on purpose (stdlib + yaml only): Open WebUI loads its functions without config/.env or memento.config.settings and
imports this module directly."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_DIR = PROJECT_ROOT / "config" / "domains"


def roots(input_root: str | os.PathLike | None = None,
          output_root: str | os.PathLike | None = None) -> tuple[Path, Path]:
    """(INPUT_ROOT, OUTPUT_ROOT): explicit arguments, else the environment, else <repo>/data/{input,output}."""
    return (Path(input_root or os.getenv("INPUT_ROOT") or PROJECT_ROOT / "data" / "input"),
            Path(output_root or os.getenv("OUTPUT_ROOT") or PROJECT_ROOT / "data" / "output"))


def corpus_dir_name(domain: str | dict) -> str:
    """The domain's corpus folder name, from a DOMAIN dict or a domain name (its YAML)."""
    if isinstance(domain, dict):
        name, where = domain.get("corpus_dir"), f"domain '{domain.get('name', '?')}'"
    else:
        path = DOMAINS_DIR / f"{domain}.yaml"
        if not path.exists():
            raise ValueError(f"unknown domain '{domain}': {path} does not exist")
        name, where = (yaml.safe_load(path.read_text()) or {}).get("corpus_dir"), str(path)
    if not name:
        raise ValueError(f"{where} has no corpus_dir — add e.g. 'corpus_dir: MyCorpus' to its domain YAML")
    return str(name)


def corpus_dirs(domain: str | dict, input_root=None, output_root=None) -> tuple[Path, Path]:
    """(input folder, output folder) of one corpus."""
    in_root, out_root = roots(input_root, output_root)
    name = corpus_dir_name(domain)
    return in_root / name, out_root / name


def describe(domain_name: str, input_dir: Path, output_dir: Path) -> str:
    """One startup line so users see exactly which folders are read and written."""
    return f"Corpus {domain_name}: documents {input_dir} | parsed output {output_dir}"
