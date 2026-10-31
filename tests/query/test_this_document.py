# "This paper" after an earlier turn: the document is taken from what the previous answer cited (its [N] markers, mapped
# through the numbered names stored with its citations), and the previous request must agree; otherwise Memento asks.
# Passage counts never decide. Also: "the latest uploaded file", and a message about an attached file alone ends after
# the hand-over note in Library Chat.
import asyncio
import json
import types

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from memento.retrieval import pipe_cli
from memento.query import library_state
from memento.query.scope import conversation, documents
from memento.query.flows import gap_answer, summary

mod = pipe_cli.load_pipe_module()
LIB = {"A", "B", "C", "PolyBERT"}


def _src(n, source):
    """A citation as OpenWebUI stores it with an answer: the number is in metadata[0].name, not in source.name."""
    return {"source": {"name": f"{source.split('#')[0]} (p. {n})"},
            "metadata": [{"source": source, "name": f"[{n}] {source.split('#')[0]} (p. {n})"}]}


# ── the cited documents ──────────────────────────────────────────────────────────────────────────────────────────────

def test_cited_documents_normal_and_gap_suffixes():
    stored = [_src(1, "A#0"), _src(2, "B#1"), _src(3, "C#gap2")]
    assert conversation.cited_documents("A fact [1].", stored, LIB) == ({"A"}, False)
    assert conversation.cited_documents("A gap [3].", stored, LIB) == ({"C"}, False)
    assert conversation.cited_documents("Both [1][2].", stored, LIB) == ({"A", "B"}, False)


def test_cited_documents_by_name_not_position():
    stored = [_src(2, "B#1"), _src(1, "A#0")]                    # stored out of order (a replay, appended passages)
    assert conversation.cited_documents("A fact [1].", stored, LIB) == ({"A"}, False)


def test_cited_documents_unmapped_or_conflicting_is_ambiguous():
    stored = [_src(1, "A#0")]
    assert conversation.cited_documents("A fact [1] and another [3].", stored, LIB)[1] is True
    conflict = [_src(1, "A#0"), _src(1, "B#0")]
    assert conversation.cited_documents("A fact [1].", conflict, LIB)[1] is True


def test_literature_citations_never_count():
    stored = [_src(1, "paper#0"), _src(2, "A#1")]
    assert conversation.cited_documents("Published [1], library [2].", stored, LIB) == ({"A"}, False)
    assert conversation.cited_documents("Published [1].", stored, LIB) == (set(), False)


def test_markers_outside_the_answer_text_are_ignored():
    stored = [_src(1, "A#0"), _src(2, "B#1"), _src(3, "C#2")]
    content = ('<think>maybe [2]</think>The answer [1].\n```\nx[3]\n```\n'
               '<details type="memento_sources">\n<summary>Sources</summary>\n(2) B\n</details>\n'
               '<details type="memento_related">\n<summary>Other</summary>\n[3]\n</details>')
    assert conversation.cited_documents(content, stored, LIB) == ({"A"}, False)


# ── the decision ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_decision():
    d = conversation.this_document
    assert d({"A"}, False, "none", None) == "A"                  # previous answer about A alone
    assert d({"A", "B"}, False, "none", None) is None            # several documents → ask
    assert d(set(), False, "none", None) is None                 # nothing cited → ask
    assert d({"A"}, True, "none", None) is None                  # an unmapped number → ask
    assert d({"A"}, False, "several", None) is None              # a comparison, even if only A was cited → ask
    assert d({"A"}, False, "one", "A") == "A"                    # the previous request named A
    assert d({"B"}, False, "one", "A") is None                   # it named A but cited B → ask
    assert d({"A"}, False, "time", None) == "A"                  # "my latest upload": not re-run, the cited one
    assert d({"A"}, False, "unresolved", None) is None


def test_request_scope_classes(monkeypatch):
    monkeypatch.setattr(documents, "named_documents", lambda q, s, d=None: ["A", "B"] if "compare" in q.lower() else [])
    monkeypatch.setattr(documents, "resolve_reference",
                        lambda q, s, d=None: "PolyBERT" if "polybert" in q.lower() else ("A" if " A" in q else None))
    sc = lambda q: conversation.request_scope(q, sorted(LIB), None)
    assert sc("Compare A and B") == ("several", None)
    assert sc("Summarize paper A") == ("one", "A")
    assert sc("Summarize my latest upload") == ("time", None)
    assert sc("What is known about hardness?") == ("none", None)
    assert sc("Summarize the dielectrics paper") == ("unresolved", None)     # describes a document that matches none
    assert sc("Which other papers use polyBERT?") == ("one", "PolyBERT")      # the method resolves to its paper …


def test_polybert_case_asks(monkeypatch):
    # … so "this paper" after that library-wide answer (citing B and C) asks instead of naming PolyBERT
    assert conversation.this_document({"B", "C"}, False, "one", "PolyBERT") is None
    assert conversation.this_document({"B"}, False, "one", "PolyBERT") is None


# ── how readers say it ──

def test_latest_uploaded_file_means_my_upload():
    assert documents.mentions_my_uploads("Can you summarize the latest uploaded file here?")
    assert documents.mentions_my_uploads("Summarize the last uploaded paper")
    assert documents.mentions_my_uploads("Fasse die zuletzt hochgeladene Datei zusammen")
    assert not documents.mentions_my_uploads("Which documents were uploaded most recently?")       # the newest-list question


def test_attachment_only():
    assert conversation.attachment_only("Summarise this")
    assert conversation.attachment_only("What is this paper about?")
    assert conversation.attachment_only("Fasse die Datei bitte zusammen")
    assert not conversation.attachment_only("What does the library say about hafnia ferroelectrics?")


# ── the pipe ─────────────────────────────────────────────────────────────────────────────────────────────────────────

PAYLOADS = ([{"text": f"Paper A passage {i}.", "source": "A", "page": i, "type": "text"} for i in range(3)]
            + [{"text": "Paper B passage.", "source": "B", "page": 1, "type": "text"}])


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


def _ask(monkeypatch, question, route, previous, *, history=True):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    m = pipe_cli.load_pipe_module()
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT PRIMARY KEY, valves TEXT)"))
        c.execute(text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps(
            {"COLLECTION": "memento-x-rag", "DOMAIN_NAME": "generic", "GRAPH_RETRIEVAL": False,
             "VALIDATION_ENABLED": False, "ENABLE_QUERY_REWRITE": True, "THINKING_AUTO": False,
             "RELATED_DOCS": False})})
    p = pipe_cli.make_pipe(m, e)
    searched = []

    def query(client, collection, qvec, q, flt, limit, hybrid, k):
        searched.append(flt)
        src = {c.match.value for c in (getattr(flt, "must", None) or []) if getattr(c, "key", None) == "source"
               and hasattr(c.match, "value")}
        pts = [types.SimpleNamespace(payload=d, score=0.9, id=i) for i, d in enumerate(PAYLOADS)
               if not src or d["source"] in src]
        return types.SimpleNamespace(points=pts)

    async def stored_previous(metadata):
        return previous

    _LLM.calls, _LLM.route = [], dict({"reasoning": "deep", "suggest": False, "element": "none", "scope": "none",
                                       "author": ""}, **route)
    monkeypatch.setattr(openai, "AsyncOpenAI", _LLM)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {s: {} for s in ("A", "B")})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "resolve_reference", lambda q, s, d=None: "A" if "paper A" in q else None)
    monkeypatch.setattr(documents, "named_documents", lambda q, s, d=None: [])
    monkeypatch.setattr(library_state, "qdrant_query", query)
    monkeypatch.setattr(summary, "document_passages", lambda client, coll, src: [dict(d) for d in PAYLOADS
                                                                             if d["source"] == src])
    monkeypatch.setattr(conversation, "stored_previous_turn", stored_previous)
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, url, top, *a, **k: [dict(h, rerank_score=0.8)
                                                                               for h in hits[:top]])
    hist = [{"role": "user", "content": previous[0] if previous else "Hi"},
            {"role": "assistant", "content": previous[1] if previous else "Hello!"}] if history else []
    out = asyncio.run(pipe_cli.ask(p, question, history=hist))
    inputs = "\n".join(json.dumps(c["messages"], ensure_ascii=False) for c in _LLM.calls if c.get("stream"))
    return out, searched, inputs


def test_greeting_then_this_paper_asks(monkeypatch):
    out, searched, inputs = _ask(monkeypatch, "Summarize this paper", {"intent": "summary", "query": "Summarize this paper"},
                                 ("Hi", "Hello! How can I help?", []))
    assert "Could you tell me which document you mean?" in out["answer"] and not searched and not inputs


def test_previous_summary_of_a_then_this_paper(monkeypatch):
    prev = ("Summarize paper A", "Paper A finds X [1][2].", [_src(1, "A#0"), _src(2, "A#1")])
    out, searched, inputs = _ask(monkeypatch, "What are the methods of this paper?",
                                 {"intent": "ranked_retrieval", "query": "What are the methods of this paper?"}, prev)
    assert "Could you tell me which document you mean?" not in out["answer"]
    assert "Paper A passage" in inputs and "Paper B" not in inputs


def test_ambiguous_two_document_history_asks(monkeypatch):
    prev = ("What is known about X?", "A says [1], B says [2].", [_src(1, "A#0"), _src(2, "B#1")])
    out, searched, inputs = _ask(monkeypatch, "Summarize this paper", {"intent": "summary", "query": "Summarize this paper"},
                                 prev)
    assert "Could you tell me which document you mean?" in out["answer"] and not inputs


def test_gap_target_uses_the_cited_document(monkeypatch):
    # "suggest future work" after an answer about B that also read two passages of A: B's research gaps
    m = mod
    prev = ("Summarize paper B", "B finds Y [3].", [_src(1, "A#0"), _src(2, "A#1"), _src(3, "B#2")])

    async def stored_previous(metadata):
        return prev
    monkeypatch.setattr(conversation, "stored_previous_turn", stored_previous)
    monkeypatch.setattr(documents, "resolve_reference", lambda q, s, d=None: "B" if "paper B" in q else None)
    monkeypatch.setattr(documents, "named_documents", lambda q, s, d=None: [])
    target = asyncio.run(gap_answer.gap_target(m.Pipe().valves, None, "suggest future work", {"A", "B"}, None, {"chat_id": "c"}))
    assert target == "B"


BLOCK = '<attached_files>\n<file type="file" id="f1" name="paper.pdf"/>\n</attached_files>\n\n'


def test_attachment_alone_in_library_chat_gets_only_the_note(monkeypatch):
    out, searched, inputs = _ask(monkeypatch, BLOCK + "Summarise this", {"intent": "summary", "query": "Summarise this"},
                                 None, history=False)
    assert "Library Chat can't read attachments directly" in out["answer"]
    assert not searched and not inputs


def test_attachment_with_a_library_question_goes_on(monkeypatch):
    out, searched, inputs = _ask(monkeypatch, BLOCK + "What does the library say about hardness?",
                                 {"intent": "ranked_retrieval", "query": "hardness in the library"}, None, history=False)
    assert "Library Chat can't read attachments directly" in out["answer"]
    assert searched and "Paper A passage" in inputs


# ── a cited document that is no longer in the library: the previous answer stays about both, so "this paper" asks ──

def _stored(*docs):
    return [{"metadata": [{"name": f"[{n}] {s}", "source": f"{s}#0"}]} for n, s in enumerate(docs, start=1)]


def test_a_removed_cited_document_keeps_the_answer_ambiguous():
    cited, ambiguous = conversation.cited_documents("Findings [1] and [2].", _stored("A", "B"), {"A"})
    assert ambiguous and conversation.this_document(cited, ambiguous, "none", None) is None


def test_an_answer_citing_only_a_removed_document_asks():
    cited, ambiguous = conversation.cited_documents("Findings [1].", _stored("B"), {"A"})
    assert conversation.this_document(cited, ambiguous, "none", None) is None


def test_one_cited_document_still_in_the_library_is_used():
    cited, ambiguous = conversation.cited_documents("Findings [1] and again [1].", _stored("A"), {"A", "B"})
    assert (cited, ambiguous) == ({"A"}, False) and conversation.this_document(cited, ambiguous, "none", None) == "A"


def test_a_literature_paper_is_still_ignored():
    stored = _stored("A") + [{"metadata": [{"name": "[2] Some paper", "source": "paper#3"}]}]
    cited, ambiguous = conversation.cited_documents("Findings [1], literature [2].", stored, {"A"})
    assert (cited, ambiguous) == ({"A"}, False)
