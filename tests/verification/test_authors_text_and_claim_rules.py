"""The authors' text next to tables and equations, and the model-free claim checks chosen per library.

- Tables: the passages citing a table by its label (`table_context`) are embedded with caption and body and reach the
  answer model, the claim checker and the rules; keyword search and the displayed/cited text stay caption + body.
- Equations: the sentences around the formula (`equation_context`, already in search) are stored and reach the answer
  model, the checker and the rules; the cited text stays the formula.
- Hits without these fields (the frozen RQ3 path, stored hits) read exactly as before.
- `claim_rules` in the domain profile chooses the model-free checks; RQ3 ran all three.
"""
import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from qdrant_client.models import SparseVector
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import library_state, prompts, request
from memento.query.scope import documents
from memento.query.flows import gap_report
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

from memento.retrieval.answer_pipeline import LINKED_TEXT_ADDENDUM, build_context, linked_text_addendum, source_text  # noqa: E402
from memento.ingestion.upload import workflow

TABLE = {"type": "table", "text": "Table 2: Density\nA | 3.2 g/cm³", "source": "P", "page": 1,
         "table_context": "Table 2 shows that sample A, sintered longest, reached 98 % relative density."}
EQUATION = {"type": "equation", "text": "$$\\sigma = E\\varepsilon$$", "source": "P", "page": 2,
            "equation_context": "where σ is the stress, E the Young's modulus and ε the strain."}
PLAIN = {"type": "text", "text": "Alumina reached 99 % density.", "source": "P", "page": 0}


# ── what the answer model reads ───────────────────────────────────────────────
def test_a_table_carries_the_authors_text_citing_it():
    assert source_text(TABLE) == (TABLE["text"] + "\n\nAuthors' text about this table (not part of the table):\n"
                                  + TABLE["table_context"])


def test_an_equation_carries_the_authors_text_around_it():
    assert source_text(EQUATION) == (EQUATION["text"] + "\n\nAuthors' text about this equation (not part of the "
                                     "formula):\n"
                                     + EQUATION["equation_context"])


@pytest.mark.parametrize("hit", [PLAIN, {**TABLE, "table_context": None}, {**EQUATION, "equation_context": ""},
                                 {k: v for k, v in TABLE.items() if k != "table_context"},
                                 {**PLAIN, "table_context": "not a table"}])
def test_hits_without_linked_text_read_as_before(hit):
    assert source_text(hit) == hit["text"]


def test_the_rule_keeping_table_and_authors_apart_comes_only_with_linked_text():
    assert linked_text_addendum([PLAIN, TABLE]) == LINKED_TEXT_ADDENDUM
    assert linked_text_addendum([EQUATION]) == LINKED_TEXT_ADDENDUM
    assert linked_text_addendum([PLAIN, {**TABLE, "table_context": None}]) == ""
    assert linked_text_addendum([{**PLAIN, "table_context": "not a table"}]) == ""
    assert linked_text_addendum([]) == ""
    assert "authors" in LINKED_TEXT_ADDENDUM and "table" in LINKED_TEXT_ADDENDUM


def test_stored_answers_keep_their_keys():
    """The stored summaries' key is built from the base answer prompt
    (cache.summary_prompt_version(_answer_prompt(dom)));
    the rule is added per answer, outside it, so no stored answer goes stale."""
    from test_rq3_answer_pipeline import _load_pipe
    mod = _load_pipe()
    for dom in ("polymer-engineering", "computational-materials", "generic"):
        assert LINKED_TEXT_ADDENDUM.strip() not in prompts.answer_prompt(request.domain(dom))


def test_the_context_block_uses_the_linked_text():
    assert build_context([TABLE]) == "[1] P, page 2\n" + source_text(TABLE)


def test_search_hits_carry_the_stored_linked_text():
    from memento.retrieval.search import point_to_hit
    assert point_to_hit({"type": "table", "text": "t", "table_context": "c"})["table_context"] == "c"
    assert point_to_hit({"type": "equation", "text": "e", "equation_context": "d"})["equation_context"] == "d"
    hit = point_to_hit({"type": "text", "text": "x"})
    assert hit["table_context"] is None and hit["equation_context"] is None


def test_the_claim_checker_sees_what_the_answer_model_saw():
    from memento.agentic_validation.verifiers import _source_text
    assert source_text(TABLE) in _source_text([TABLE], [1])
    assert source_text(EQUATION) in _source_text([PLAIN, EQUATION], [2])


def test_a_value_from_the_authors_text_about_a_table_is_found_by_the_number_rule():
    from memento.agentic_validation.scientific_checks import run_checks
    assert run_checks("c1", "Sample A reached 98 % relative density", [TABLE]) == []
    no_ctx = {k: v for k, v in TABLE.items() if k != "table_context"}
    assert [i.kind for i in run_checks("c1", "Sample A reached 98 % relative density", [no_ctx])] \
        == ["number_not_in_source"]


# ── claim rules per library ───────────────────────────────────────────────────
def test_claim_rules_default_to_the_three_rq3_rules():
    from memento.agentic_validation.scientific_checks import CLAIM_RULES
    from memento.agentic_validation.strategies import ArmContext
    assert CLAIM_RULES == ("numbers", "units", "phases")
    assert ArmContext(None, "m", "S", "q", [], None).claim_rules == CLAIM_RULES


def test_run_checks_applies_only_the_chosen_rules():
    from memento.agentic_validation.scientific_checks import run_checks
    hit = {"type": "text", "text": "γ-Al2O3 was observed after 2 h."}
    claim = "The fracture toughness of 4.5 GPa came with α-Al2O3"
    assert {i.kind for i in run_checks("c1", claim, [hit])} == {"number_not_in_source", "unit_mismatch",
                                                                "phase_contradicts_source"}
    assert {i.kind for i in run_checks("c1", claim, [hit], rules=("numbers",))} == {"number_not_in_source"}
    assert {i.kind for i in run_checks("c1", claim, [hit], rules=("units", "phases"))} == {"unit_mismatch",
                                                                                           "phase_contradicts_source"}
    assert run_checks("c1", claim, [hit], rules=()) == []


def test_an_unknown_claim_rule_is_an_error():
    from memento.agentic_validation.scientific_checks import claim_rules
    assert claim_rules(["numbers", "phases"]) == ("numbers", "phases")
    with pytest.raises(ValueError, match="unknown claim rule"):
        claim_rules(["numbers", "spelling"])


@pytest.mark.parametrize("path", sorted((ROOT / "config/domains").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_library_profile_names_valid_claim_rules(path):
    from memento.agentic_validation.scientific_checks import CLAIM_RULES, claim_rules
    rules = claim_rules(yaml.safe_load(path.read_text())["claim_rules"])
    assert rules == (("numbers",) if path.stem == "generic" else CLAIM_RULES)


def test_the_check_uses_the_contexts_claim_rules(monkeypatch):
    import memento.agentic_validation.strategies as st
    from memento.agentic_validation.claims import Claim
    claim = Claim("c1", "The fracture toughness of 4.5 GPa came with α-Al2O3", (1,), (0, 10))
    monkeypatch.setattr(st, "split_claims", lambda *a, **k: [claim])
    monkeypatch.setattr(st, "verify_claims", lambda *a, **k: {})
    hits = [{"type": "text", "text": "γ-Al2O3 was observed after 2 h.", "source": "P", "page": 0}]
    full = st.ArmContext(None, "m", "S", "q", hits, None)
    _, _, issues = st.check_answer(full, None, "a", hits)
    assert {i.kind for i in issues["c1"]} == {"number_not_in_source", "unit_mismatch", "phase_contradicts_source"}
    only_numbers = st.ArmContext(None, "m", "S", "q", hits, None, claim_rules=("numbers",))
    _, _, issues = st.check_answer(only_numbers, None, "a", hits)
    assert {i.kind for i in issues["c1"]} == {"number_not_in_source"}


def test_thorough_check_uses_the_librarys_claim_rules(tmp_path, monkeypatch):
    from test_rq3_answer_pipeline import _run_pipe_with_validation
    from memento.agentic_validation.strategies import ArmResult
    seen = {}

    def fake(name, ctx, *, p1_answer=None, max_rounds=2):
        seen["rules"] = ctx.claim_rules
        return ArmResult("A", "Verified answer [1].", hits=ctx.hits, trace=[], rounds=1)
    _run_pipe_with_validation(tmp_path, monkeypatch, fake, VALIDATION_STRATEGY="a")     # the generic library
    assert seen["rules"] == ("numbers",)


# ── the live pipe sends the linked text to the answer model ──────────────────
def test_the_live_pipe_sends_the_linked_text(tmp_path, monkeypatch):
    import asyncio
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    from memento.retrieval.answer_pipeline import build_messages, system_prompt_for
    from memento.retrieval.search import point_to_hit
    from test_rq3_answer_pipeline import _load_pipe, _RecordingAsyncOpenAI
    from memento.query.prompts import PLAIN_FORMULAS
    mod = _load_pipe()
    payloads = [{k: v for k, v in h.items()} for h in (PLAIN, TABLE, EQUATION)]
    points = [types.SimpleNamespace(payload=p, score=0.9 - i / 10, id=i) for i, p in enumerate(payloads)]
    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: None)
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", RERANK_ENABLED=True, GRAPH_RETRIEVAL=False,
                        VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True, THINKING_AUTO=False)
    question = "Which density did sample A reach?"

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": question}]}, {"id": "t"})]
    asyncio.run(run())
    sent = [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")]
    assert len(sent) == 1
    hits = [point_to_hit(pt.payload, pt.score) for pt in points]
    dom_prompt = yaml.safe_load((ROOT / "config/domains/generic.yaml").read_text())["system_prompt"].rstrip() \
        + " " + PLAIN_FORMULAS
    assert sent[0]["messages"] == build_messages(system_prompt_for(dom_prompt, False) + LINKED_TEXT_ADDENDUM, hits,
                                                 question, max_images=p.valves.MAX_IMAGES_TO_LLM)
    user = sent[0]["messages"][-1]["content"]
    assert "Authors' text about this table (not part of the table):\n" + TABLE["table_context"] in user
    assert "Authors' text about this equation (not part of the formula):\n" + EQUATION["equation_context"] in user
    assert sent[0]["messages"][0]["content"].endswith(LINKED_TEXT_ADDENDUM)      # the rule, as the sources need it


# ── both ingestion paths store and embed the linked text ─────────────────────
ITEMS = [
    {"type": "table", "table_caption": ["Table 2: Density"], "table_body": "A | 3.2 g/cm³", "page_idx": 1,
     "bbox": [1, 2, 3, 4], "enriched_context": "Table 2 shows that sample A reached 98 %.",
     "enrichment_method": "label_match"},
    {"type": "table", "table_caption": ["Composition"], "table_body": "B | 5 wt%", "page_idx": 2, "bbox": [1, 2, 3, 5],
     "enriched_context": "Nearby text about something else.", "enrichment_method": "positional_fallback"},
    {"type": "equation", "text": "$$\\sigma = E\\varepsilon$$", "page_idx": 3, "bbox": [1, 2, 3, 6],
     "enriched_context": "where σ is the stress.", "enrichment_method": "positional"},
]
WANT_EMBED = ["Table 2 shows that sample A reached 98 %.\nTable 2: Density\nA | 3.2 g/cm³",
              "Composition\nB | 5 wt%",
              "where σ is the stress.\n$$\\sigma = E\\varepsilon$$"]
WANT_SPARSE = ["Table 2: Density\nA | 3.2 g/cm³", "Composition\nB | 5 wt%",
               "where σ is the stress.\n$$\\sigma = E\\varepsilon$$"]


def test_table_embedding_text_puts_label_cited_text_first():
    from memento.ingestion.embed import table_context, table_embed_text
    assert table_context(ITEMS[0]) == "Table 2 shows that sample A reached 98 %."
    assert table_context(ITEMS[1]) == "" and table_context({"type": "table"}) == ""
    assert table_embed_text("ctx", "Cap\nBody") == "ctx\nCap\nBody" and table_embed_text("", "Cap\nBody") == "Cap\nBody"


def _stored(points):
    return [{k: pt.payload.get(k) for k in ("type", "text", "table_context", "equation_context")} for pt in points]


WANT_PAYLOAD = [
    {"type": "table", "text": "Table 2: Density\nA | 3.2 g/cm³",
     "table_context": "Table 2 shows that sample A reached 98 %.", "equation_context": None},
    {"type": "table", "text": "Composition\nB | 5 wt%", "table_context": None, "equation_context": None},
    {"type": "equation", "text": "$$\\sigma = E\\varepsilon$$", "table_context": None,
     "equation_context": "where σ is the stress."},
]


class _FakeQdrant:
    def __init__(self, *a, **kw):
        self.points = []

    def count(self, *a, **kw):
        return types.SimpleNamespace(count=len(self.points))

    def upsert(self, collection_name, points):
        self.points.extend(points)


def test_command_line_ingestion_stores_and_embeds_the_linked_text(tmp_path, monkeypatch):
    import memento.ingestion.run_pipeline as rp
    (tmp_path / "doc_content_list.json").write_text(json.dumps(ITEMS))
    embedded, sparse = [], []
    monkeypatch.setattr(rp, "call_embedder", lambda inputs, url, text_only=False:
                        (embedded.extend(inputs) or [[0.1]] * len(inputs), 0.0))
    monkeypatch.setattr(rp, "sparse_encode", lambda texts: sparse.extend(texts) or
                        [SparseVector(indices=[1], values=[1.0]) for _ in texts])
    monkeypatch.setattr(rp, "_record_in_manifest", lambda *a: None)
    monkeypatch.setattr(rp, "_add_to_document_list", lambda paper_id, **kw: None)
    for flag in ("GAP_TAG_ENABLED", "GRAPH_BUILD_ENABLED", "CONTEXTUALIZE_ENABLED"):
        monkeypatch.setattr(rp, flag, False)
    client = _FakeQdrant()
    rp.ingest_paper(tmp_path, "doc", "http://embedder", client, enrichment_enabled=False)
    assert [e["text"] for e in embedded] == WANT_EMBED
    assert sparse == WANT_SPARSE
    assert _stored(client.points) == WANT_PAYLOAD


def test_upload_pipe_stores_and_embeds_the_linked_text(tmp_path, monkeypatch):
    ip = load_function(ROOT / "openwebui/functions/ingestion_pipe.py", "memento_ingestion")
    out = tmp_path / "doc"
    out.mkdir()
    (out / "doc_content_list.json").write_text(json.dumps(ITEMS))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    embedded, sparse, client = [], [], _FakeQdrant()
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: client))
    import memento.common.model_servers as vc
    import memento.ingestion.embed as sp
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: embedded.extend(inputs) or [[0.1]] * len(inputs))
    monkeypatch.setattr(sp, "sparse_encode", lambda texts: sparse.extend(texts) or
                        [SparseVector(indices=[1], values=[1.0]) for _ in texts])
    monkeypatch.setattr(document, "set_paper_in_progress", lambda *a: None)
    monkeypatch.setattr(document.Ingester, "register_document", lambda *a, **k: None)
    p = document.Ingester(ip.Pipe().valves)
    p.valves.DOMAIN_NAME = "computational-materials"
    p.valves.NO_ENRICH, p.valves.RESEARCH_GAP_TAG, p.valves.GRAPH_BUILD = True, False, False
    monkeypatch.setattr(p, "find_paper_dir", lambda stem: out)
    res = [r for r in p.embed_one(pdf) if isinstance(r, dict)][-1]
    assert res["ok"] and res["vectors"] == 3
    assert [e["text"] for e in embedded] == WANT_EMBED
    assert sparse == WANT_SPARSE
    assert _stored(client.points) == WANT_PAYLOAD


def test_the_research_gap_report_check_uses_the_librarys_claim_rules(monkeypatch):
    import memento.agentic_validation.strategies as st
    from test_rq3_answer_pipeline import _load_pipe
    mod = _load_pipe()
    seen = {}
    monkeypatch.setattr(st, "run_arm_c", lambda ctx, text: seen.setdefault("rules", ctx.claim_rules))
    p = mod.Pipe()
    p.valves.DOMAIN_NAME = "generic"
    gap_report.check_report(p.valves, "Draft [1].", [], None)
    assert seen["rules"] == ("numbers",)
