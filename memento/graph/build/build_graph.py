#!/usr/bin/env python3
"""Materialize the dual-graph knowledge graph into Neo4j. For one document's enriched content list
(enrich_parsed_content_list.py) it builds: one Paper node; Section nodes (one per canonical section present),
(Paper)-[:HAS_SECTION]->(Section); Chunk nodes for every item ingestion embeds, (Chunk)-[:IN_SECTION]->(Section); Entity
nodes and (Chunk)-[:MENTIONS]->(Entity) edges from memento/graph/build/extract_entities.py (registry + LLM); (Entity)-[:RELATION
{rel_type, paper_id, chunk_id}]->(Entity) edges from the LLM pass; (Chunk)-[:REFERENCES]->(Chunk) edges from the
figure/equation reference links the enrichment computes (_reference_edges).

Chunk node ids are the Qdrant point ids: sha256(f"{paper_id}_{page_idx}_{item_type}_{done}") % 2**63, `done` being the
0-based index over embeddable items in document order (embed.enumerate_embeddable is the single source of chunk
identity).

Ablation D (--no-llm): the dictionary pass only (MENTIONS edges, the REFERENCES and section structure), no RELATION
edges. --all skips documents whose graph exists (Neo4jStore.paper_exists) unless --force; naming one paper_id always
rebuilds it.

--domain is required; --output-dir defaults to the domain's OUTPUT_ROOT/<corpus_dir>, the collection to its VLM
collection:
python memento/graph/build/build_graph.py --domain ceramics --output-dir data/output/Ceramics <paper_id>
python memento/graph/build/build_graph.py --domain ceramics --output-dir data/output/Ceramics --all [--force] [--resolve]
python memento/graph/build/build_graph.py ... --papers-file ids.txt (a slice of the collection)
python memento/graph/build/build_graph.py ... <paper_id> --no-llm (ablation D) | --no-clear (do not wipe first)
python memento/graph/build/build_graph.py ... --resolve (re-resolve citations only)"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Optional
from memento.config.paths import PROJECT_ROOT

if __name__ == "__main__":
    # --domain must be applied before memento.config.settings is imported (instruction,
    # collection and Neo4j instance are all resolved from it at import time)
    from memento.config.cli_domain import apply_domain_arg
    apply_domain_arg()
from memento.ingestion.embed import chunk_point_id
from concurrent.futures import ThreadPoolExecutor, as_completed
from memento.config.settings import (  # noqa: E402
    COLLECTION, EXTRACT_WORKERS, GRAPH_ENRICH_COMBINED, MINERU_OUT_SUBDIR, NEO4J_URI, OUTPUT_DIR,
)
from memento.graph import get_by_id, normalize  # noqa: E402
from memento.graph.store import GraphEdge, GraphNode, GraphStore, get_store  # noqa: E402
from memento.ingestion.enrich_parsed_content_list import (  # noqa: E402
    enrich_content_list,
    find_all_equation_reference_indices,
    find_all_label_reference_indices,
)
from memento.graph.build.extract_entities import extract_entities  # noqa: E402
from memento.graph.build.llm_enrich import llm_enrich_items  # noqa: E402
from memento.ingestion.extract_metadata import (  # noqa: E402
    extract_paper_metadata,
    link_intext_citations,
    parse_bibliography,
)

# ── Embedding-eligibility filter ───────────────────────────────────────────
# Imported, not duplicated: enumerate_embeddable is the single source of which
# items are embedded and under which Qdrant point id.
from memento.ingestion.embed import IMAGE_TYPES, TABLE_TYPES, TEXT_TYPES, enumerate_embeddable  # noqa: E402,F401

# Chunk types whose text is prose — the LLM relation-extraction pass
# (memento/graph/build/extract_entities.py) only runs on these; table/figure chunks
# still get the (cheap, no-LLM) dictionary-match pass for MENTIONS edges.
PROSE_TYPES = {"text", "list", "equation"}

# Ignore tiny enriched_context lines when matching REFERENCES edges — avoids
# spurious matches on short fragments.
MIN_REF_EXCERPT_CHARS = 20


# ── Corpus-metadata fallback from snowball ledger ─────────────────────────
_LEDGER_PATH = PROJECT_ROOT / "data" / "input_snowball" / "snowball_ledger.json"
_ledger_cache: dict[str, dict] | None = None


def _load_ledger() -> dict[str, dict]:
    global _ledger_cache
    if _ledger_cache is None:
        try:
            led = json.loads(_LEDGER_PATH.read_text())
            _ledger_cache = {}
            for wid, v in led.items():
                entry: dict = {}
                y = v.get("year")
                if y and 1900 <= int(y) <= 2099:
                    entry["year"] = int(y)
                t = (v.get("title") or "").strip()
                if t:
                    entry["title"] = t
                if entry:
                    _ledger_cache[f"openalex_{wid}"] = entry
        except Exception:
            _ledger_cache = {}
    return _ledger_cache


def _ledger_year(paper_id: str) -> int | None:
    return _load_ledger().get(paper_id, {}).get("year")


def _ledger_title(paper_id: str) -> str | None:
    return _load_ledger().get(paper_id, {}).get("title")


# ── Chunk id (aligned with Qdrant point ids) ───────────────────────────────
def _chunk_id(paper_id: str, page_idx: int, item_type: str, done: int) -> str:
    """The shared Qdrant point id, as a decimal string."""
    return str(chunk_point_id(paper_id, page_idx, item_type, done))


# ── Embeddable-chunk enumeration ───────────────────────────────────────────
def _embeddable_chunks(items: list[dict], paper_dir: Path, paper_id: str) -> list[dict]:
    """Adapter over embed.enumerate_embeddable (single source of chunk
    identity). Never generates captions: cached LLM captions are reused.

    Returns a list of dicts (one per embeddable item, in document order):
        chunk_id, item_idx (index into `items`), item, type,
        disp_text (citation/display text), extract_text (text fed to
        extract_entities), section, page, page_end
    """
    out = []
    for r in enumerate_embeddable(items, paper_dir, paper_id, generate_captions=False):
        item, payload = r["item"], r["payload"]
        disp = payload.get("text") or ""
        # Entities come from the chunk's own text: not the embed text (it may
        # carry the paper-level situating context, which would make every chunk
        # MENTION the paper's headline entities) and not MinerU's `content`
        # reconstruction (excluded from embedding as hallucination-prone).
        # Equations add their surrounding sentences, as the embed text does.
        extract = "\n".join(filter(None, [payload.get("equation_context"), disp]))
        out.append({
            "chunk_id": r["chunk_id"], "item_idx": r["item_idx"], "item": item,
            "type": payload.get("type"), "disp_text": disp, "extract_text": extract,
            "section": payload.get("section") or "Unknown",
            "page": payload.get("page", 0), "page_end": item.get("page_end"),
        })
    return out


# ── REFERENCES edges (figure/table/equation -> citing chunk) ───────────────
def _reference_edges(items: list[dict], chunks: list[dict], paper_id: str) -> list[GraphEdge]:
    """Build (Chunk)-[:REFERENCES]->(Chunk) edges from a citing text/list chunk to the figure/table/equation chunk it
    mentions.

    1. Label-exhaustive (method="label"): for every figure/chart/table item with a resolved figure_label/table_label and
    every equation with an equation_tag, every text/list item mentioning that label (the enrichment's own matching,
    find_all_label_reference_indices / find_all_equation_reference_indices) gets an edge: no cap, no proximity bound.
    2. Positional fallback (method="positional"): for items without a resolved label, the link is recovered by matching
    `enriched_context` against candidate chunk text, bounded by the enrichment's top-3 excerpts."""
    idx_to_chunk = {c["item_idx"]: c for c in chunks}

    candidates: list[tuple[str, str]] = []
    for c in chunks:
        if c["type"] in {"text", "list"}:
            t = (c["item"].get("text") or "").strip()
        elif c["type"] in IMAGE_TYPES:
            t = (
                " ".join(c["item"].get("image_caption") or [])
                if c["type"] == "image"
                else " ".join(c["item"].get("chart_caption") or [])
            ).strip()
        else:
            continue
        if t:
            candidates.append((c["chunk_id"], t))

    edges: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()

    def _add(cand_id: str, target_id: str, via_label: Optional[str], method: str) -> None:
        if cand_id == target_id:
            return
        key = (cand_id, target_id)
        if key in seen:
            return
        seen.add(key)
        props: dict = {"paper_id": paper_id, "method": method}
        if via_label:
            props["via_label"] = via_label
        edges.append(GraphEdge(
            start_id=cand_id, start_label="Chunk",
            end_id=target_id, end_label="Chunk",
            type="REFERENCES", props=props,
        ))

    for c in chunks:
        if c["type"] in IMAGE_TYPES:
            label = c["item"].get("figure_label") or c["item"].get("table_label")
        elif c["type"] in TABLE_TYPES:
            label = c["item"].get("table_label")
        elif c["type"] == "equation":
            label = c["item"].get("equation_tag")
        else:
            continue

        if label:
            if c["type"] == "equation":
                ref_indices = find_all_equation_reference_indices(items, label, exclude_idx=c["item_idx"])
            else:
                ref_indices = find_all_label_reference_indices(items, label, exclude_idx=c["item_idx"])
            for ref_idx in ref_indices:
                ref_chunk = idx_to_chunk.get(ref_idx)
                if ref_chunk is None:
                    continue
                _add(ref_chunk["chunk_id"], c["chunk_id"], label, "label")
            if ref_indices:
                continue  # exhaustive label match found — skip positional fallback

        # Positional fallback (figures/equations only — matches prior
        # behavior, where tables had no enrichment/REFERENCES at all).
        if c["type"] not in IMAGE_TYPES | {"equation"}:
            continue
        enriched = (c["item"].get("enriched_context") or "").strip()
        if not enriched:
            continue
        for excerpt in enriched.split("\n"):
            excerpt = excerpt.strip()
            if len(excerpt) < MIN_REF_EXCERPT_CHARS:
                continue
            for cand_id, cand_text in candidates:
                if cand_id == c["chunk_id"]:
                    continue
                if excerpt in cand_text:
                    _add(cand_id, c["chunk_id"], None, "positional")

    return edges


# ── Paper/section/chunk discovery ──────────────────────────────────────────
_OUTPUT_DIR: Path = OUTPUT_DIR   # set from --output-dir by main()
_SUBDIR: str = MINERU_OUT_SUBDIR


def _find_paper_dir(paper_id: str) -> Optional[Path]:
    from memento.ingestion.run_pipeline import find_paper_dir_in
    return find_paper_dir_in(_OUTPUT_DIR, paper_id, _SUBDIR)


def read_papers_file(path: Path) -> list[str]:
    """One paper id per line, kept verbatim (ids may contain inner and trailing spaces); blank lines skipped."""
    return [l for l in Path(path).read_text().splitlines() if l.strip()]


def list_collection_sources(collection: str) -> list[str]:
    """Every distinct paper id (`source`) embedded in a Qdrant collection."""
    from qdrant_client import QdrantClient
    from memento.config.settings import QDRANT_URL
    c = QdrantClient(url=QDRANT_URL, timeout=120)
    sources, off = set(), None
    while True:
        pts, off = c.scroll(collection, limit=10000, offset=off, with_payload=["source"])
        sources |= {p.payload.get("source") for p in pts if p.payload.get("source")}
        if off is None:
            return sorted(sources)


def _find_content_list(paper_dir: Path) -> Optional[Path]:
    matches = [p for p in paper_dir.glob("*_content_list.json")
               if "enriched" not in p.name and "v2" not in p.name]
    if not matches:
        matches = [p for p in paper_dir.glob("*_content_list_v2.json") if "enriched" not in p.name]
    return matches[0] if matches else None


# ── Build one paper ─────────────────────────────────────────────────────────
def build_paper(
    paper_id: str,
    *,
    store: GraphStore,
    items: Optional[list[dict]] = None,
    paper_dir: Optional[Path] = None,
    use_llm: bool = True,
    clear: bool = True,
    doc_info: Optional[dict] = None,
    orphan_cleanup: bool = True,
    gliner: Optional[dict] = None,
) -> dict:
    """Build or refresh the graph for one document; returns summary counts. `gliner` (the G2-lite-gliner rung, no LLM
    calls): {chunk_id: {"entities": [{"name", "type"}], "relations": []}} from evaluation/RQ2/build/rq2_gliner_extract.py, used in
    place of the LLM extraction and passed through the same fusion (registry aliases, junk filters, type check); needs
    use_llm=False, and every chunk with extraction text must have an entry. `items`/`paper_dir` can be injected (tests);
    otherwise `paper_id` is resolved under OUTPUT_DIR and the enriched content list is (re)built, as in both ingestion
    paths."""
    from memento.graph.documents import work_id
    # every parsed and embedded document gets entities: the graph covers exactly the files the retrieval index covers
    make_entities = True
    llm_stats = {"n_llm_calls": 0, "n_cache_hits": 0}
    if items is None:
        resolved_dir = paper_dir or _find_paper_dir(paper_id)
        if resolved_dir is None:
            raise FileNotFoundError(f"No parsed output found for paper '{paper_id}' under {_OUTPUT_DIR}")
        paper_dir = resolved_dir
        cl_path = _find_content_list(paper_dir)
        if cl_path is None:
            raise FileNotFoundError(f"No content_list.json in {paper_dir}")
        enriched_path = Path(enrich_content_list(cl_path))
        items = json.loads(enriched_path.read_text())
        # a standalone rebuild never went through run_pipeline.py's ingest_paper(), so item['llm_extraction'] is not
        # stored yet: run the same combined batched pass here, so the loop below reads the cache instead of one live
        # call per chunk. Skipped for --no-llm and when the combined pass is off, as in run_pipeline.py.
        if use_llm and GRAPH_ENRICH_COMBINED:
            # the extractor sees each chunk's own text (caption + context for
            # figures), never MinerU's hallucination-prone `content`
            pre_chunks = _embeddable_chunks(items, paper_dir, paper_id)
            snippets = {c["item_idx"]: c["extract_text"] for c in pre_chunks if c["extract_text"]}
            from memento.graph.build.extraction_cache import RawResponseCache
            cache = RawResponseCache(paper_dir / f"{paper_id}_graph_llm_cache.json")
            llm_enrich_items(items, want_context=False, want_entities=True, images_dir=paper_dir,
                             label=paper_id, snippets=snippets, cache=cache)
            cache.save()
            llm_stats = {"n_llm_calls": cache.misses, "n_cache_hits": cache.hits}
        write_debug_file = True
    else:
        write_debug_file = paper_dir is not None
        if paper_dir is None:
            paper_dir = Path(".")  # only used for images/ existence checks below

    chunks = _embeddable_chunks(items, paper_dir, paper_id)
    if gliner is not None:
        if use_llm:
            raise ValueError("gliner entities replace the LLM pass: call with use_llm=False")
        missing = [c["chunk_id"] for c in chunks if c["extract_text"] and c["chunk_id"] not in gliner]
        if missing:
            raise RuntimeError(f"{paper_id}: {len(missing)} chunks have no GLiNER entry "
                               f"(e.g. {missing[0]}); re-run evaluation/RQ2/build/rq2_gliner_extract.py")

    if clear:
        try:
            store.clear_paper(paper_id, orphan_cleanup=orphan_cleanup)
        except TypeError:          # stores without the parameter (test fakes)
            store.clear_paper(paper_id)

    # ── Paper node (+ best-effort front-matter metadata) ────────────────
    metadata = extract_paper_metadata(items)
    paper_props: dict = {"source": paper_id}
    parsed_title = metadata["title"]
    ledger_title = _ledger_title(paper_id)
    if ledger_title:
        paper_props["title"] = ledger_title
        if parsed_title and parsed_title != ledger_title:
            paper_props["parsed_title"] = parsed_title
    elif parsed_title:
        paper_props["title"] = parsed_title
    if metadata["authors"]:
        paper_props["authors"] = metadata["authors"]
    year = metadata["year"]
    if year is None:
        year = _ledger_year(paper_id)
    if year is not None:
        paper_props["year"] = year
    if metadata["doi"]:
        paper_props["doi"] = metadata["doi"]
    if doc_info:
        paper_props["doc_type"] = doc_info.get("type")
        if doc_info.get("student"):
            paper_props["student"] = doc_info["student"]
    paper_props["work_id"] = work_id(paper_id, doc_info)

    nodes: list[GraphNode] = [GraphNode(id=paper_id, label="Paper", props=paper_props)]
    edges: list[GraphEdge] = []

    # ── Authors ──────────────────────────────────────────────────────────
    n_authors = 0
    for author_name in metadata["authors"]:
        author_id = normalize(author_name)
        if not author_id:
            continue
        nodes.append(GraphNode(id=author_id, label="Author", props={"name": author_name}))
        edges.append(GraphEdge(
            start_id=paper_id, start_label="Paper", end_id=author_id, end_label="Author",
            type="AUTHORED_BY",
        ))
        n_authors += 1

    # ── Sections ─────────────────────────────────────────────────────────
    section_ids: dict[str, str] = {}
    for c in chunks:
        section = c["section"]
        if section not in section_ids:
            sid = f"{paper_id}::{section}"
            section_ids[section] = sid
            nodes.append(GraphNode(id=sid, label="Section", props={"name": section, "paper_id": paper_id}))
            edges.append(GraphEdge(
                start_id=paper_id, start_label="Paper", end_id=sid, end_label="Section",
                type="HAS_SECTION",
            ))

    # ── Chunks ───────────────────────────────────────────────────────────
    for c in chunks:
        sid = section_ids[c["section"]]
        props = {"type": c["type"], "page": c["page"], "text": c["disp_text"], "source": paper_id}
        if c["page_end"] is not None:
            props["page_end"] = c["page_end"]
        if c["type"] in IMAGE_TYPES:
            img_name = c["item"].get("img_path") or c["item"].get("image_path")
            if img_name:
                props["img_path"] = img_name
        nodes.append(GraphNode(id=c["chunk_id"], label="Chunk", props=props))
        edges.append(GraphEdge(
            start_id=c["chunk_id"], start_label="Chunk", end_id=sid, end_label="Section",
            type="IN_SECTION",
        ))

    # ── Entities & relations ────────────────────────────────────────────
    entities: dict[str, GraphNode] = {}
    n_relations = 0
    n_llm_eligible = 0
    n_precomputed = 0
    debug_records: list[dict] = []

    def _prepare_chunk(c):
        text = c["extract_text"]
        if not text:
            return None
        prose = c["type"] in PROSE_TYPES
        if gliner is not None:
            # precomputed GLiNER entities through the LLM fusion path (never a live call)
            return text, True, gliner[c["chunk_id"]]
        precomputed = c["item"].get("llm_extraction") if isinstance(c.get("item"), dict) else None
        llm_enabled = use_llm and (prose or precomputed is not None)
        return text, llm_enabled, precomputed

    def _extract_chunk(c):
        prep = _prepare_chunk(c)
        if prep is None:
            return c, None
        text, llm_enabled, precomputed = prep
        result = extract_entities(text, use_llm=llm_enabled, precomputed=precomputed)
        return c, result

    n_workers = max(1, EXTRACT_WORKERS) if use_llm and not GRAPH_ENRICH_COMBINED else 1
    eligible = [c for c in chunks if _prepare_chunk(c) is not None] if make_entities else []

    if n_workers > 1:
        chunk_results = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_extract_chunk, c): c for c in eligible}
            for f in as_completed(futures):
                chunk_results.append(f.result())
    else:
        chunk_results = [_extract_chunk(c) for c in eligible]

    for c, result in chunk_results:
        if result is None:
            continue
        prep = _prepare_chunk(c)
        if prep is not None:
            _, llm_enabled, precomputed = prep
            if llm_enabled:
                n_llm_eligible += 1
                if precomputed is not None:
                    n_precomputed += 1

        for ent in result.entities:
            if ent.id not in entities:
                entity_def = get_by_id(ent.id) if ent.canonical else None
                aliases = list(entity_def.aliases) if entity_def else []
                entities[ent.id] = GraphNode(id=ent.id, label="Entity", props={
                    "name": ent.name, "type": ent.type, "canonical": ent.canonical, "aliases": aliases,
                })
            edges.append(GraphEdge(
                start_id=c["chunk_id"], start_label="Chunk", end_id=ent.id, end_label="Entity",
                type="MENTIONS", props={"paper_id": paper_id},
            ))

        for rel in result.relations:
            edges.append(GraphEdge(
                start_id=rel.source, start_label="Entity", end_id=rel.target, end_label="Entity",
                type="RELATION", props={"rel_type": rel.rel_type, "paper_id": paper_id, "chunk_id": c["chunk_id"]},
            ))
            n_relations += 1

        if result.entities or result.relations:
            debug_records.append({
                "chunk_id": c["chunk_id"],
                "type": c["type"],
                "dict_entities": [{"id": e.id, "name": e.name} for e in result.entities if e.canonical],
                "llm_entities": [{"id": e.id, "name": e.name, "type": e.type}
                                  for e in result.entities if not e.canonical],
                "relations": [{"source": r.source, "target": r.target, "rel_type": r.rel_type}
                              for r in result.relations],
            })

    nodes.extend(entities.values())
    n_dict_entities = sum(1 for n in entities.values() if n.props.get("canonical"))
    n_llm_entities = len(entities) - n_dict_entities

    # ── References (figure/table/equation -> citing text) ──────────────
    reference_edges = _reference_edges(items, chunks, paper_id)
    edges.extend(reference_edges)

    # ── Bibliography + in-text citations ────────────────────────────────
    bib_references = parse_bibliography(items)
    reference_node_ids: dict[int, str] = {}
    for ref in bib_references:
        ref_id = f"{paper_id}::ref::{ref['marker']}"
        reference_node_ids[ref["marker"]] = ref_id
        ref_props: dict = {"marker": ref["marker"], "raw": ref["raw"], "paper_id": paper_id}
        if ref["year"] is not None:
            ref_props["year"] = ref["year"]
        if ref["doi"]:
            ref_props["doi"] = ref["doi"]
        if ref["title"]:
            ref_props["title"] = ref["title"]
        if ref["authors"]:
            ref_props["authors"] = ref["authors"]
        nodes.append(GraphNode(id=ref_id, label="Reference", props=ref_props))
        edges.append(GraphEdge(
            start_id=paper_id, start_label="Paper", end_id=ref_id, end_label="Reference",
            type="HAS_REFERENCE",
        ))

    n_citations = 0
    for chunk_id, marker in link_intext_citations(chunks, bib_references):
        edges.append(GraphEdge(
            start_id=chunk_id, start_label="Chunk", end_id=reference_node_ids[marker], end_label="Reference",
            type="CITES", props={"paper_id": paper_id},
        ))
        n_citations += 1

    n_nodes = store.upsert_nodes(nodes)
    n_edges = store.upsert_edges(edges)

    # only the LLM build writes the debug file: the G2-lite rungs must not overwrite it
    if debug_records and write_debug_file and use_llm and gliner is None:
        try:
            debug_path = Path(paper_dir) / f"{paper_id}_graph_debug.json"
            debug_path.write_text(json.dumps({
                "paper_id": paper_id,
                "n_dict_entities": n_dict_entities,
                "n_llm_entities": n_llm_entities,
                "n_relations": n_relations,
                "chunks": debug_records,
            }, indent=2))
        except Exception:
            pass  # debugging aid only — never let this block the real build

    return {
        "paper_id":     paper_id,
        "n_chunks":     len(chunks),
        "n_sections":   len(section_ids),
        "n_entities":   len(entities),
        "n_relations":  n_relations,
        "n_references": len(reference_edges),
        "n_authors":    n_authors,
        "n_bib_references": len(bib_references),
        "n_citations":  n_citations,
        "n_nodes":      n_nodes,
        "n_edges":      n_edges,
        "n_edges_dropped": getattr(store, "last_dropped", 0),
        "n_llm_eligible": n_llm_eligible,
        "n_precomputed":  n_precomputed,
        "n_dict_entities": n_dict_entities,
        "n_llm_entities":  n_llm_entities,
        **llm_stats,
    }


# ── cross-paper citation resolution ──
# Conservative on purpose (precision over recall): a wrong Reference→Paper edge harms cross-document traversal more than
# a missing one. 0.92 requires near-complete word-set overlap.
_TITLE_MATCH_THRESHOLD = 0.92

# A title whose tokens are a subset of a reference string scores 1.0, so
# titles built from common words match indiscriminately. Token count alone
# does not separate them ("Entropy-stabilized oxides" is three tokens and
# correct); corpus document frequency does. The fraction is of the corpus's
# own reference count, so it carries across corpus sizes and domains.
_MIN_TITLE_TOKENS = 3
_TITLE_DF_FRACTION = 0.005
_TITLE_DF_FLOOR = 20

_TITLE_NORM_RE = re.compile(r'[^a-z0-9\s]')


def _normalize_title(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return re.sub(r'\s+', ' ', _TITLE_NORM_RE.sub(' ', text.lower())).strip()


def _token_set_ratio(a: str, b: str) -> float:
    """Order- and duplicate-independent similarity in [0, 1].

    Mirrors thefuzz's token_set_ratio without adding a dependency: scores
    1.0 when one string's word set is a subset of the other's, which is the
    common case here -- a parsed Paper.title's words all appear, in order,
    somewhere inside the much longer `Reference.raw` bibliography-entry
    string (authors + title + journal + year)."""
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = ' '.join(sorted(tokens_a & tokens_b))
    sorted_a = ' '.join(sorted(tokens_a))
    sorted_b = ' '.join(sorted(tokens_b))
    return max(
        SequenceMatcher(None, intersection, sorted_a).ratio(),
        SequenceMatcher(None, intersection, sorted_b).ratio(),
        SequenceMatcher(None, sorted_a, sorted_b).ratio(),
    )


def resolve_cross_paper_citations(store: GraphStore) -> dict:
    """Connect each document's Reference nodes to other documents in the corpus by fuzzy matching the reference's raw
    bibliography text against Paper.title (extract_metadata.parse_bibliography leaves Reference.title unset, but a title
    appears nearly verbatim inside other documents' Reference.raw). For each Reference whose best match clears
    _TITLE_MATCH_THRESHOLD (and agrees on year within 1 when both are known): (Reference)-[:REFERS_TO {score}]->(Paper)
    and (Paper)-[:CITES_PAPER {via_ref}]->(Paper). Never a self-citation. Run after all documents are built (--all, then
    --resolve)."""
    from memento.graph.build.citations import resolve_citations
    references = store.all_references()
    res = resolve_citations(store.all_papers(), references, threshold=_TITLE_MATCH_THRESHOLD,
                            min_title_tokens=_MIN_TITLE_TOKENS, df_fraction=_TITLE_DF_FRACTION,
                            df_floor=_TITLE_DF_FLOOR)
    edges: list[GraphEdge] = []
    for m in res["matches"]:
        edges.append(GraphEdge(start_id=m["ref_id"], start_label="Reference", end_id=m["cited"],
                               end_label="Paper", type="REFERS_TO", props={"score": m["score"]}))
        edges.append(GraphEdge(start_id=m["citing"], start_label="Paper", end_id=m["cited"],
                               end_label="Paper", type="CITES_PAPER", props={"via_ref": m["ref_id"]}))
    n_edges = store.upsert_edges(edges)
    return {"n_references": len(references), "n_resolved": len(res["matches"]),
            "n_edges": n_edges, "n_edges_dropped": getattr(store, "last_dropped", 0),
            "n_candidates": res["n_candidates"], "n_nondistinctive": res["n_nondistinctive"],
            "df_cutoff": res["df_cutoff"]}


# ── CO_OCCURS edge building ────────────────────────────────────────────
_COOCCUR_MAX_ENTS = 25
_COOCCUR_BATCH = 5000


def build_cooccurrence_edges(store: GraphStore, min_weight: int = 1) -> dict:
    """Build CO_OCCURS edges between entities sharing a chunk; corpus-wide, after all documents are built. The
    relation-free co-occurrence graph (LinearRAG's design) encodes the same adjacency as the LLM RELATION layer without
    extraction error, so the two can be compared."""
    rows = [r.data() for r in store._run(
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
        "RETURN c.id AS chunk, e.id AS entity"
    ).records]
    by_chunk: dict[str, set[str]] = {}
    for r in rows:
        by_chunk.setdefault(r["chunk"], set()).add(r["entity"])

    pair_counts: Counter = Counter()
    skipped = 0
    for ents in by_chunk.values():
        if len(ents) < 2:
            continue
        if len(ents) > _COOCCUR_MAX_ENTS:
            skipped += 1
            continue
        for a, b in combinations(sorted(ents), 2):
            pair_counts[(a, b)] += 1

    kept = {p: w for p, w in pair_counts.items() if w >= min_weight}

    store._run("MATCH ()-[r:CO_OCCURS]->() DELETE r")
    items = [{"a": a, "b": b, "w": w} for (a, b), w in kept.items()]
    for i in range(0, len(items), _COOCCUR_BATCH):
        store._run(
            "UNWIND $rows AS row "
            "MATCH (a:Entity {id: row.a}), (b:Entity {id: row.b}) "
            "MERGE (a)-[r:CO_OCCURS]->(b) SET r.weight = row.w",
            rows=items[i:i + _COOCCUR_BATCH])

    n = store._run("MATCH ()-[r:CO_OCCURS]->() RETURN count(r) AS n").records[0]["n"]
    return {"n_pairs": len(pair_counts), "n_kept": len(kept),
            "n_skipped_chunks": skipped, "n_edges": n}


# ── CLI ──────────────────────────────────────────────────────────────────
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the RQ2 dual-graph KG into Neo4j")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("paper_id", nargs="?", help="Paper id (PDF stem) to build")
    group.add_argument("--all", action="store_true",
                       help="Build every paper embedded in --collection")
    p.add_argument("--domain", required=True, help="corpus domain (config/domains/<name>.yaml)")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help=f"parsed-output folder of the corpus (default: the domain's OUTPUT_ROOT/<corpus_dir>, "
                        f"here {OUTPUT_DIR})")
    p.add_argument("--collection", default=COLLECTION,
                   help=f"Qdrant collection whose papers are built (default {COLLECTION})")
    p.add_argument("--papers-file", type=Path,
                   help="only build the paper ids listed in this file (one per line)")
    p.add_argument("--no-llm", action="store_true",
                   help="Dictionary match only, no relation extraction (ablation D)")
    p.add_argument("--gliner-dir", type=Path,
                   help="with --no-llm: take entities from <dir>/<paper_id>.json "
                        "(evaluation/RQ2/build/rq2_gliner_extract.py) -- the G2-lite-gliner rung")
    p.add_argument("--no-clear", action="store_true",
                   help="Don't clear the paper's existing graph before rebuilding")
    p.add_argument("--force", action="store_true",
                   help="With --all: rebuild papers that already have a graph too "
                        "(default is to skip them -- see incremental-build note below)")
    p.add_argument("--resolve", action="store_true",
                   help="Run Phase 3 cross-paper citation resolution over the whole "
                        "corpus after building (or alone, with no paper_id/--all, to "
                        "(re-)resolve without rebuilding -- e.g. after adding new papers)")
    p.add_argument("--cooccurrence", action="store_true",
                   help="Build CO_OCCURS edges between entities sharing a chunk "
                        "(corpus-wide, run after --all completes)")
    p.add_argument("--cooccurrence-min-weight", type=int, default=1,
                   help="Only keep pairs co-occurring in >= N chunks (default 1)")
    p.add_argument("--workers", type=int, default=1,
                   help="With --all: process N papers concurrently so the vLLM "
                        "batch queue stays full (each worker gets its own Neo4j "
                        "connection). Default 1 (sequential).")
    p.add_argument("--work-links", action="store_true",
                   help="corpus-wide SAME_WORK/SAME_AUTHOR edges from the document registry "
                        "and near-identical titles (run after building)")
    p.add_argument("--manifest", type=Path,
                   help="write a build manifest (config, git state, hashes, totals) to this JSON file")
    p.add_argument("--title-dup-threshold", type=float, default=0.95,
                   help="token_set_ratio for SAME_WORK title duplicates (default 0.95)")
    return p


def main():
    # the root logger, so extract_entities.py / llm_enrich.py warnings show up in this log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = _parser()
    args = p.parse_args()

    if (not args.all and not args.paper_id and not args.papers_file
            and not args.resolve and not args.cooccurrence and not args.work_links):
        p.error("one of paper_id, --all, --papers-file, --resolve, --cooccurrence or --work-links is required")
    if args.gliner_dir and not args.no_llm:
        p.error("--gliner-dir replaces the LLM extraction: add --no-llm")

    from memento.graph.documents import load_document_registry
    registry = load_document_registry(args.domain)

    from memento.config.cli_domain import domain_banner
    domain_banner()
    print(f"[build_graph] Neo4j: {NEO4J_URI}  collection: {args.collection}  "
          f"output dir: {args.output_dir}", flush=True)

    global _OUTPUT_DIR
    _OUTPUT_DIR = args.output_dir.resolve()
    paper_ids: list[str] = []
    if args.all or args.papers_file:
        paper_ids = list_collection_sources(args.collection)
        if args.papers_file:
            wanted = set(read_papers_file(args.papers_file))
            missing = wanted - set(paper_ids)
            if missing:
                p.error(f"{len(missing)} papers-file ids are not in {args.collection}: {sorted(missing)[:5]}")
            paper_ids = [pid for pid in paper_ids if pid in wanted]
        if not paper_ids:
            print(f"[ERROR] No papers in {args.collection}")
            sys.exit(1)
    elif args.paper_id:
        paper_ids = [args.paper_id]

    def _build_one(paper_id: str, store: GraphStore, use_llm: bool, clear: bool,
                   incremental: bool, force: bool) -> dict:
        if incremental and not force and store.paper_exists(paper_id):
            print(f"[SKIP] {paper_id}: already built (use --force to rebuild)", flush=True)
            return {"paper_id": paper_id, "skipped": True}
        try:
            gliner = None
            if args.gliner_dir:
                gpath = args.gliner_dir / f"{paper_id}.json"
                if not gpath.exists():
                    raise RuntimeError(f"no GLiNER file {gpath}")
                gliner = json.loads(gpath.read_text())
            summary = build_paper(
                paper_id, store=store,
                use_llm=use_llm, clear=clear,
                doc_info=registry.get(paper_id),
                orphan_cleanup=not parallel,
                gliner=gliner,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {paper_id}: {e}", flush=True)
            return {"paper_id": paper_id, "failed": True, "error": str(e)}
        except Exception as e:
            import traceback
            print(f"[FAIL] {paper_id}", flush=True)
            traceback.print_exc()
            return {"paper_id": paper_id, "failed": True, "error": repr(e)}
        print(
            f"[{paper_id}] chunks={summary['n_chunks']} "
            f"sections={summary['n_sections']} "
            f"entities={summary['n_entities']} "
            f"relations={summary['n_relations']} "
            f"references={summary['n_references']} "
            f"authors={summary['n_authors']} "
            f"bib_refs={summary['n_bib_references']} "
            f"citations={summary['n_citations']} "
            f"edges_dropped={summary['n_edges_dropped']} "
            f"llm_precomputed={summary['n_precomputed']}/{summary['n_llm_eligible']} "
            f"entities(dict={summary['n_dict_entities']},llm={summary['n_llm_entities']}) "
            f"llm_calls={summary['n_llm_calls']} cache_hits={summary['n_cache_hits']}",
            flush=True,
        )
        return summary

    batch = bool(args.all or args.papers_file)
    n_workers = max(1, args.workers) if batch else 1
    parallel = n_workers > 1
    summaries: list[dict] = []
    import time as _time
    t_start = _time.time()

    if n_workers > 1 and paper_ids:
        # Each worker gets its own Neo4j connection so concurrent MERGE
        # writes don't contend on a single driver session.
        with get_store() as schema_store:
            schema_store.ensure_schema()

        def _worker(pid: str) -> dict:
            with get_store() as wstore:
                return _build_one(pid, wstore, use_llm=not args.no_llm,
                                  clear=not args.no_clear, incremental=batch,
                                  force=args.force)

        print(f"[build_graph] {len(paper_ids)} papers, {n_workers} workers")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            summaries = list(pool.map(_worker, paper_ids))
        # per-document orphan sweeps are off in parallel mode (they race with other workers' new nodes): one sweep once
        # all writers are done
        with get_store() as sweep_store:
            sweep_store.delete_orphans()
    elif paper_ids:
        with get_store() as store:
            store.ensure_schema()
            summaries = [_build_one(paper_id, store, use_llm=not args.no_llm,
                                    clear=not args.no_clear, incremental=batch,
                                    force=args.force)
                         for paper_id in paper_ids]

    if args.resolve or args.cooccurrence or args.work_links:
        with get_store() as store:
            if args.resolve:
                resolve_summary = resolve_cross_paper_citations(store)
                print(
                    f"[resolve] references={resolve_summary['n_references']} "
                    f"resolved={resolve_summary['n_resolved']} "
                    f"edges={resolve_summary['n_edges']} "
                    f"candidates={resolve_summary['n_candidates']} "
                    f"nondistinctive={resolve_summary['n_nondistinctive']} "
                    f"df_cutoff={resolve_summary['df_cutoff']}"
                )
            if args.work_links:
                from memento.graph.build.work_links import title_duplicate_edges, work_link_edges
                papers = store.all_papers()
                present = {pp["id"] for pp in papers}
                wl = work_link_edges(registry, present) + title_duplicate_edges(papers, args.title_dup_threshold)
                n_wl = store.upsert_edges(wl)
                print(f"[work-links] same_work={sum(x.type == 'SAME_WORK' for x in wl)} "
                      f"same_author={sum(x.type == 'SAME_AUTHOR' for x in wl)} written={n_wl} "
                      f"dropped={getattr(store, 'last_dropped', 0)}")
            if args.cooccurrence:
                cooccur = build_cooccurrence_edges(
                    store, min_weight=args.cooccurrence_min_weight)
                print(
                    f"[cooccurrence] pairs={cooccur['n_pairs']:,} "
                    f"kept={cooccur['n_kept']:,} "
                    f"skipped_chunks={cooccur['n_skipped_chunks']} "
                    f"edges={cooccur['n_edges']:,}"
                )

    # every write step gets a new build id: memento/graph/ppr_cache.py must never score
    # a changed graph with a stale matrix
    import time
    with get_store() as store:
        store.set_meta("build_id", f"build:{time.time_ns()}")

    if args.manifest:
        from memento.graph.build.manifest import make_manifest
        man = make_manifest(domain=args.domain, args={k: str(v) for k, v in vars(args).items()},
                            summaries=[x for x in summaries if not x.get("skipped")],
                            extra={"wall_s": round(_time.time() - t_start, 1),
                                   "n_skipped": sum(1 for x in summaries if x.get("skipped"))})
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        print(f"[build_graph] manifest {man['manifest_id']} -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
