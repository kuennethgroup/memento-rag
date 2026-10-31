# A document or author restriction is kept through every search: the zero-hit retry, Thorough check's re-search, the
# "nothing relevant" reply and the related block, a document still being uploaded, and a resolver error never widen the
# search to the whole library. The real pipe answers each question (memento.retrieval.pipe_cli), with a fake vector store that
# applies the filters it gets.
import asyncio
import json
import types

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from memento.retrieval import pipe_cli
from memento.query import checking, library_state, request, texts
from memento.query.scope import conversation, documents, resolve
from memento.query.flows import answer as answer_flow, figures

PAYLOADS = ([{"text": f"Paper A passage {i} about hardness.", "source": "A", "page": i, "type": "text"} for i in range(5)]
            + [{"text": "Paper B says alumina reaches 99 GPa.", "source": "B", "page": 1, "type": "text"},
               {"text": "Paper B figure: a hardness map.", "source": "B", "page": 2, "type": "image"},
               {"text": "Paper C passage about hardness.", "source": "C", "page": 1, "type": "text"}])


def _cond(payload, c):
    v, m = payload.get(c.key), c.match
    if hasattr(m, "value"):
        return v == m.value
    if hasattr(m, "any"):
        return v in m.any
    raise AssertionError(f"unexpected match {m!r}")


def allows(flt, payload) -> bool:
    """Does a Qdrant filter let this payload through? (must / must_not / should, nested filters)"""
    from qdrant_client.models import Filter
    if flt is None:
        return True
    test = lambda c: allows(c, payload) if isinstance(c, Filter) else _cond(payload, c)
    if not all(test(c) for c in (flt.must or [])):
        return False
    if any(test(c) for c in (flt.must_not or [])):
        return False
    return not flt.should or any(test(c) for c in flt.should)


class _LLM:
    calls: list = []
    route: dict = {}

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        _LLM.calls.append(kw)
        if kw.get("stream"):
            async def gen():
                delta = types.SimpleNamespace(content="Answer [1].", reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return gen()
        msg = types.SimpleNamespace(content=json.dumps(_LLM.route))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _run(monkeypatch, question, route, *, in_progress=(), library=("A", "B", "C"), valves=None, thorough=False,
         resolve=None, named=None, score=0.8, fronts=None, history=None, uploads=None, target=None):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    mod = pipe_cli.load_pipe_module()
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    v = {"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic", "GRAPH_RETRIEVAL": False, "VALIDATION_ENABLED": False,
         "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False, "RELATED_DOCS": False}
    v.update(valves or {})
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps(v)})
    if fronts:
        from memento.library import front_matter as doc_authors, documents as library_db
        doc_authors.ensure_table(e)
        library_db.ensure_tables(e)          # an author list is stored only for the library's version
        for src in library:
            library_db.add_document(e, collection="memento-x-rag", entry={"source": src, "title": src},
                                    content_hash="h", uploaded_by="u", output_root=None)
        for src, front in fronts.items():
            doc_authors.put(e, collection="memento-x-rag", source=src, content_hash="h", front=front)
    p = pipe_cli.make_pipe(mod, e)
    rec = {"searches": [], "candidates": [], "check_filters": [], "related": 0, "nothing_found": 0}

    def query(client, collection, qvec, q, flt, limit, hybrid, k):
        rec["searches"].append(flt)
        pts = [types.SimpleNamespace(payload=d, score=0.9 - i / 100, id=i)
               for i, d in enumerate(PAYLOADS) if allows(flt, d)]
        return types.SimpleNamespace(points=pts)

    def rerank(q, hits, url, top, min_score, *a, **k):
        return [dict(h, rerank_score=score) for h in hits[:top]]

    def resolve_ref(query, candidates, dom=None):
        rec["candidates"].append(set(candidates))
        if resolve == "raise":
            raise RuntimeError("resolver down")
        return resolve if resolve in candidates else None

    def named_docs(query, candidates, dom=None):
        return [s for s in (named or []) if s in candidates]

    async def thorough_check(self, **kw):
        rec["check_filters"].append(kw.get("retrieve_filter"))
        return kw["answer"], None, None, False

    def related_heads(self, client, qvec, query, flt):
        rec["related"] += 1
        return [dict(PAYLOADS[5], rerank_score=0.9)]          # paper B, strongly related

    async def nothing_found_reply(self, *a, **k):
        rec["nothing_found"] += 1
        return "NOTHING-FOUND-REPLY"

    _LLM.calls, _LLM.route = [], dict({"reasoning": "deep", "suggest": False, "element": "none", "scope": "none",
                                       "author": ""}, **route)
    monkeypatch.setattr(openai, "AsyncOpenAI", _LLM)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    # every library row has a fingerprint, equal to the stored author lists'
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {s: {"content_hash": "h"} for s in library})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set(in_progress))
    monkeypatch.setattr(documents, "resolve_reference", resolve_ref)
    monkeypatch.setattr(documents, "named_documents", named_docs)
    monkeypatch.setattr(library_state, "qdrant_query", query)
    monkeypatch.setattr(checking, "thorough_check", thorough_check)
    monkeypatch.setattr(answer_flow, "related_heads", related_heads)
    monkeypatch.setattr(answer_flow, "nothing_found_reply", nothing_found_reply)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", rerank)
    if uploads is not None:                       # the asker's own latest uploads
        monkeypatch.setattr(documents, "latest_uploads", lambda *a: list(uploads))
    if target is not None:                        # the document the previous answer cited
        async def conversation_target(*a, **k):
            return target or None
        monkeypatch.setattr(conversation, "conversation_target", conversation_target)
    out = asyncio.run(pipe_cli.ask(p, question, thorough=thorough, history=history))
    answers = [c for c in _LLM.calls if c.get("stream")]
    model_input = "\n".join(json.dumps(c["messages"], ensure_ascii=False) for c in answers)
    return out, rec, model_input


def _only(flt, allowed):
    """The filter lets through exactly the passages of these documents."""
    return {d["source"] for d in PAYLOADS if allows(flt, d)} <= set(allowed)


# ── the zero-hit retry ───────────────────────────────────────────────────────────────────────────────────────────────

def test_figure_request_in_a_named_document_stays_in_it(monkeypatch):
    # A has no figure: the figure search finds nothing, the retry searches A's text, never B's figure
    out, rec, model_input = _run(monkeypatch, "Show me the figures of paper A",
                                 {"intent": "image_request", "query": "figures of paper A", "element": "figure",
                                  "scope": "document"}, resolve="A")
    assert rec["searches"] and all(_only(f, {"A"}) for f in rec["searches"])
    assert "Paper B" not in model_input and "Paper A passage" in model_input
    assert "I found no indexed figure for this in" in out["answer"]


# ── Thorough check's re-search ───────────────────────────────────────────────────────────────────────────────────────

def test_check_searches_within_the_named_document(monkeypatch):
    out, rec, _ = _run(monkeypatch, "What does paper A report about hardness?",
                       {"intent": "ranked_retrieval", "query": "paper A hardness", "scope": "document"},
                       resolve="A", valves={"VALIDATION_ENABLED": True}, thorough=True)
    assert rec["check_filters"] and _only(rec["check_filters"][0], {"A"})


def test_check_searches_within_two_named_documents(monkeypatch):
    out, rec, _ = _run(monkeypatch, "Compare paper A and paper C on hardness",
                       {"intent": "ranked_retrieval", "query": "compare A and C hardness", "scope": "documents"},
                       named=["A", "C"], valves={"VALIDATION_ENABLED": True}, thorough=True)
    assert rec["check_filters"] and _only(rec["check_filters"][0], {"A", "C"})


def test_check_searches_within_a_persons_documents(monkeypatch):
    out, rec, _ = _run(monkeypatch, "What did Jane Doe find about hardness?",
                       {"intent": "ranked_retrieval", "query": "Jane Doe hardness", "author": "Jane Doe"},
                       fronts={"A": "Title A\n\nJane Doe, Max Muster", "C": "Title C\n\nJane Doe",
                               "B": "Title B\n\nOther Person"},
                       valves={"VALIDATION_ENABLED": True}, thorough=True)
    assert rec["check_filters"] and _only(rec["check_filters"][0], {"A", "C"})


def test_check_for_a_library_question_has_no_source_restriction(monkeypatch):
    out, rec, _ = _run(monkeypatch, "What is known about hardness?",
                       {"intent": "ranked_retrieval", "query": "hardness", "scope": "library"},
                       valves={"VALIDATION_ENABLED": True}, thorough=True)
    assert rec["check_filters"] and not _only(rec["check_filters"][0], {"A"})      # B and C allowed too


# ── a person's documents: no library-wide reply or block ─────────────────────────────────────────────────────────────

def test_low_scores_about_a_persons_documents_stay_inside(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What did Jane Doe find about hardness?",
                                 {"intent": "ranked_retrieval", "query": "Jane Doe hardness", "author": "Jane Doe"},
                                 fronts={"A": "Title A\n\nJane Doe, Max Muster", "C": "Title C\n\nJane Doe",
                                         "B": "Title B\n\nOther Person"},
                                 valves={"RELATED_DOCS": True}, score=0.02)
    assert rec["nothing_found"] == 0 and "NOTHING-FOUND-REPLY" not in out["answer"]
    assert rec["related"] == 0                                    # no "other documents" block from the whole library
    assert "Paper B" not in model_input and "Paper A passage" in model_input


def test_author_lookup_error_does_not_widen(monkeypatch):
    import memento.library.front_matter as da
    monkeypatch.setattr(da, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    out, rec, model_input = _run(monkeypatch, "What did Jane Doe find about hardness?",
                                 {"intent": "ranked_retrieval", "query": "Jane Doe hardness", "author": "Jane Doe"})
    assert rec["searches"] == [] and model_input == ""
    assert "could not check the requested documents" in out["answer"]


# ── documents still being uploaded ───────────────────────────────────────────────────────────────────────────────────

def test_named_document_being_uploaded_gets_the_processing_reply(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What does paper A report about hardness?",
                                 {"intent": "ranked_retrieval", "query": "paper A hardness", "scope": "document"},
                                 resolve="A", in_progress={"A"})
    assert "A" in rec["candidates"][0]                            # resolved against the uploads in progress too
    assert rec["searches"] == [] and model_input == ""
    assert "is still being processed" in out["answer"]


def test_new_upload_named_by_file_name_gets_the_processing_reply(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What does NewPaper report?",
                                 {"intent": "ranked_retrieval", "query": "NewPaper", "scope": "document"},
                                 resolve="NewPaper", in_progress={"NewPaper"})
    assert rec["searches"] == [] and "is still being processed" in out["answer"]


def test_two_named_one_uploading_answers_from_the_other(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "Compare paper A and paper C on hardness",
                                 {"intent": "ranked_retrieval", "query": "compare A and C", "scope": "documents"},
                                 named=["A", "C"], in_progress={"C"})
    assert rec["searches"] and all(_only(f, {"A"}) for f in rec["searches"])
    assert "Paper C" not in model_input and "Paper A passage" in model_input
    assert "is still being processed" in out["answer"]


# ── resolver errors ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_resolver_error_on_a_named_document_does_not_search(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What does paper A report about hardness?",
                                 {"intent": "ranked_retrieval", "query": "paper A hardness", "scope": "document"},
                                 resolve="raise")
    assert rec["searches"] == [] and model_input == ""
    assert "could not check the requested documents" in out["answer"]


def test_resolver_error_without_a_named_document_searches_the_library(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What is known about hardness?",
                                 {"intent": "ranked_retrieval", "query": "hardness", "scope": "none"},
                                 resolve="raise")
    assert rec["searches"] and "Paper B" in model_input


def test_the_new_texts_have_german_forms():
    mod = pipe_cli.load_pipe_module()
    token = request.LANG.set("de")
    try:
        assert texts.tr(texts.SCOPE_FAILED_MSG) == "Entschuldigung, ich konnte die angefragten Dokumente gerade nicht prüfen. " \
                                                 "Bitte versuchen Sie es noch einmal."
    finally:
        request.LANG.reset(token)
    assert resolve.processing_msg(["A"], all_busy=True, de=True) == \
        "**A** wird noch verarbeitet, daher kann ich noch nicht daraus antworten. Bitte versuchen Sie es später noch einmal."
    assert resolve.processing_msg(["A", "B"], all_busy=False, de=False) == \
        "_**A** and **B** are still being processed and not included in this answer._\n\n"
    assert figures.no_kind_note("figure", "A", True) == "_Ich habe dazu keine indizierte Abbildung in A gefunden._\n\n"
