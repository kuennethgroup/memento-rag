"""Graph-blind term statistics from raw chunk text."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

from memento.graph.registry_base import fold_latex

TERM_PATTERNS = [
    re.compile(r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+\b"),                       # formulas: Al2O3, ZrB2, SiO2
    re.compile(r"\b[A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]{1,8})*\b"),                    # acronyms/codes: TGMDA, PA66-GF30
    re.compile(r"\bpoly[a-zäöüß]{3,}\b", re.I),                                   # polysulfone
    re.compile(r"\b[a-zäöüß]+(?:amide|imide|ester|urethane|ethylene|propylene|styrene)\b", re.I),
    re.compile(r"\b[A-Za-zÄÖÜäöüß]{3,}-[A-Za-z0-9ÄÖÜäöüß]{2,}(?:-[A-Za-z0-9ÄÖÜäöüß]{2,})?\b"),  # carbon-fibre
]
_STOP = {"DOI", "PDF", "ISBN", "ISSN", "URL", "HTTP", "HTTPS", "FIG", "TAB", "EQ", "ET", "AL", "II", "III", "IV",
         "VI", "VII", "VIII", "IX", "XI", "XII", "USA", "GER", "ISO", "DIN", "EN", "ASTM", "AG", "GMBH"}


def candidate_terms(text: str) -> set[str]:
    text = fold_latex(text or "")
    out = set()
    for rx in TERM_PATTERNS:
        for m in rx.finditer(text):
            t = m.group(0).strip("-")
            if len(t) < 3 or t.upper() in _STOP or t.replace("-", "").isdigit():
                continue
            out.add(t)
    return out


def term_doc_freq(chunks: list[dict]) -> dict[str, set[str]]:
    df: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        for t in candidate_terms(c.get("text", "")):
            df[t].add(c["source"])
    return dict(df)


def term_key(term: str) -> str:
    s = unicodedata.normalize("NFKC", term).lower()
    s = re.sub(r"[-_\s]+", " ", s).strip()
    return re.sub(r"(?<=[a-zäöüß]{3})s\b", "", s)


def hub_cap(n_docs: int, share: float = 0.02) -> int:
    return max(2, math.floor(share * n_docs))


GATE_SYSTEM = (
    "You filter candidate terms found in scientific documents. Keep a term only if it names a specific scientific "
    "entity: a material, compound, phase or element; a process or synthesis method; a material property or measured "
    "quantity; a processing or measurement condition; or a measurement or characterisation technique. Drop "
    "organisations, standards bodies, companies, people, places, software, units, generic words, adjectives and "
    "phrases that are not entities. Output ONLY JSON {\"keep\": [the kept terms, exactly as given]}.")


def gate_terms(client, model, terms, *, batch=40):
    """Domain-agnostic, graph-blind term quality gate: which candidate
    bridge/seed terms name a scientific entity. Returns (kept terms, number of terms whose batch
    could not be parsed; those are dropped). Only terms that were asked about can be kept."""
    from evaluation.RQ2.questions.common import llm_json
    kept, unparsed = set(), 0
    terms = list(dict.fromkeys(terms))
    for i in range(0, len(terms), batch):
        part = terms[i:i + batch]
        r = llm_json(client, model, GATE_SYSTEM, "Terms:\n" + "\n".join(f"- {t}" for t in part))
        if not isinstance(r, dict) or not isinstance(r.get("keep"), list):
            unparsed += len(part)
            continue
        kept |= set(part) & {k for k in r["keep"] if isinstance(k, str)}
    return kept, unparsed
