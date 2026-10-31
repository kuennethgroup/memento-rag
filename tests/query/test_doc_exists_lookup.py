# "Is document X in the library?" answered from the document list: the model sees only a few passages, so for a missing
# title it would hedge. The name is matched against the library's documents (the resolver the questions use); found →
# "Yes" + its title; not found → "No document titled exactly X" + related documents from the per-document search
# (measured: missing titles score ≤ 0.03, related documents 0.14–0.83), no answer-model call. Topic questions ("is there
# a paper about X") are not caught: the normal path answers them well.
import asyncio
import types

import pytest

from test_rq3_answer_pipeline import _RecordingAsyncOpenAI, _load_pipe
from memento.query import library_state
from memento.query.scope import documents
from memento.query.flows import answer as answer_flow


@pytest.fixture(scope="module")
def mod():
    return _load_pipe()


@pytest.mark.parametrize("question, name, lang", [
    ("Is the paper 'Bioplastic design using multitask deep neural networks' in the library?",
     "Bioplastic design using multitask deep neural networks", "en"),
    ("Do we have the polyBERT paper?", "polyBERT", "en"),
    ("Is the paper 'Attention Is All You Need' in the library?", "Attention Is All You Need", "en"),
    ("Is “Attention Is All You Need” in the library?", "Attention Is All You Need", "en"),
    ("is attention is all you need in the library", "attention is all you need", "en"),
    ("Is there a paper called Attention Is All You Need?", "Attention Is All You Need", "en"),
    ("Is there a document titled 'Deep Residual Learning' in the library?", "Deep Residual Learning", "en"),
    ("Does the paper 'Attention Is All You Need' exist?", "Attention Is All You Need", "en"),
    ("Is bioplastic_design_multitask.pdf in the library?", "bioplastic_design_multitask.pdf", "en"),
    ("Do we have the document Polymer informatics with multi-task learning?",
     "Polymer informatics with multi-task learning", "en"),
    ("Does the library contain the paper 'Attention Is All You Need'?", "Attention Is All You Need", "en"),
    ("Gibt es das Paper 'Bioplastic design using multitask deep neural networks' in der Bibliothek?",
     "Bioplastic design using multitask deep neural networks", "de"),
    ("Gibt es das Paper „Attention Is All You Need“?", "Attention Is All You Need", "de"),
    ("Haben wir das polyBERT-Paper?", "polyBERT", "de"),
    ("Ist das Dokument 'Attention Is All You Need' in der Bibliothek?", "Attention Is All You Need", "de"),
    ("Ist ‚Attention Is All You Need‘ in der Bibliothek vorhanden?", "Attention Is All You Need", "de"),
    # titles that start like a pronoun or a topic word: quoted, after "called", or a file name, they are names
    ("Is the paper 'It's All Connected: Topology-Aware Structural Graph Encoding' in the library?",
     "It's All Connected: Topology-Aware Structural Graph Encoding", "en"),
    ("Is It's All Connected in the library?", "It's All Connected", "en"),
    ("Is the paper 'On the Origin of the Large Remanent Polarization in La:HfO₂' in the library?",
     "On the Origin of the Large Remanent Polarization in La:HfO₂", "en"),
    ("Is there a paper called On the Origin of the Large Remanent Polarization?",
     "On the Origin of the Large Remanent Polarization", "en"),
    ("Is On the Origin of the Large Remanent Polarization in LaHfO2.pdf in the library?",
     "On the Origin of the Large Remanent Polarization in LaHfO2.pdf", "en"),
    ("Gibt es das Paper „On the Origin of the Large Remanent Polarization“?",
     "On the Origin of the Large Remanent Polarization", "de"),
])
def test_existence_questions_are_recognised_with_the_name(mod, question, name, lang):
    assert documents.existence_name(question) == (name, lang)


@pytest.mark.parametrize("question", [
    "Is there a paper about ferroelectric hafnia?",
    "Is there a paper on bead foams?",
    "Is there a document about RAG-Anything in the library?",
    "Which papers by Ravi Ramaswami are in the library?",
    "Welche Arbeiten von Ravi Ramaswami sind in der Bibliothek?",
    "Gibt es ein Paper über Bead Foams?",
    "Do we have data on dielectric constants?",
    "What documents are in the library?",
    "Does a ferroelectric phase exist in HfO2?",
    "Is it in the library?",
    "Is this paper in the library?",
    "Is there anything about polyBERT in the library?",
    "Does the library contain information about tungsten carbide?",
    "Does the library have a paper on field cycling?",
    "Do we have any papers on polymer dielectrics?",
    "Are there papers on hafnia in the library?",
    "Do we have the paper by Ramaswami?",
    "Do we have the melting point of tungsten carbide?",
    "What is polyBERT and how is it trained?",
    "Haben wir Daten zur Dielektrizitätskonstante?",
    "Is field cycling discussed in the library?",
    "Do we have a paper on 'field cycling'?",
    "Is it's in the library?",
    "Ist Feldzyklieren in der Bibliothek beschrieben?",
])
def test_topic_and_content_questions_are_not_caught(mod, question):
    assert documents.existence_name(question) is None


def test_found_reply_says_yes_with_the_title(mod):
    out = documents.existence_reply("polyBERT", "en", title="polyBERT: a chemical language model (PolyBERT)", n_docs=25)
    assert out.startswith("**Yes**, *polyBERT: a chemical language model (PolyBERT)* is in the library.")
    assert "—" not in out


def test_missing_reply_lists_related_documents(mod):
    out = documents.existence_reply("Attention [1] Is *All*", "en", related=["Polymer informatics", "PolyBERT"], n_docs=25)
    assert out.startswith("**No** document titled exactly *Attention (1) Is \\*All\\** is in the library.")
    assert "Related documents:\n\n- Polymer informatics\n- PolyBERT" in out and "**Library Upload**" in out
    assert "—" not in out


def test_missing_reply_without_related_documents(mod):
    out = documents.existence_reply("Attention Is All You Need", "en", n_docs=25)
    assert out == ("**No** document titled exactly *Attention Is All You Need* is in the library (25 documents). "
                   "You can add it with **Library Upload**.")


def test_german_replies(mod):
    assert documents.existence_reply("X", "de", title="T", n_docs=3).startswith("**Ja**, *T* ist in der Bibliothek.")
    assert documents.existence_reply("X", "de", n_docs=3).startswith(
        "**Nein**, kein Dokument mit genau dem Titel *X* ist in der Bibliothek (3 Dokumente).")


# the library's names as the resolver sees them (file name, first-page title, PDF title)
NAMES = {
    "PolyBERT": ["PolyBERT", "polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer "
                             "informatics"],
    "A general-purpose material property data extraction pipeline": [
        "A general-purpose material property data extraction pipeline",
        "A general-purpose material property data extraction pipeline from large polymer corpora using natural "
        "language processing"],
    "Polymer informatics with multi-task learning": ["Polymer informatics with multi-task learning"],
    "Polymer informatics beyond homopolymers": ["Polymer informatics beyond homopolymers"],
    "Polymer informatics Current status and critical next steps": [
        "Polymer informatics Current status and critical next steps"],
    "Bioplastic design using multitask deep neural networks": [
        "Bioplastic design using multitask deep neural networks"],
    "On the Origin of the Large Remanent Polarization in LaHfO2": [
        "On the Origin of the Large Remanent Polarization in LaHfO2",
        "Dieses Dokument ist eine Zweitveröffentlichung (Postprint)",
        "On the Origin of the Large Remanent Polarization in La:HfO₂"],
    "Application of convolutional neural networks and ensemble methods": [
        "Application of convolutional neural networks and ensemble methods in the fiber volume content analysis of "
        "natural fiber composites"],
}


@pytest.fixture
def library(mod, monkeypatch):
    monkeypatch.setattr(documents, "_doc_aliases", lambda sources, domain: {s: NAMES[s] for s in sources})
    return list(NAMES)


@pytest.mark.parametrize("name, expected", [
    ("polyBERT", "PolyBERT"),
    ("Bioplastic design using multitask deep neural networks", "Bioplastic design using multitask deep neural networks"),
    ("bioplastic_design_using_multitask_deep_neural_networks.pdf",
     "Bioplastic design using multitask deep neural networks"),
    ("BERT", None),                         # a false "Yes" is worse than a false "No"
    ("Polymer informatics", None),          # shared by three documents: no single one is meant
    ("Attention Is All You Need", None),
    # a cut-off title: the start of exactly one document's title, four words or more
    ("Polymer informatics Current status and", "Polymer informatics Current status and critical next steps"),
    ("polyBERT a chemical language model", "PolyBERT"),
    ("Polymer informatics with", None),      # three words: too short to stand for a title
    ("On the Origin of the Large", "On the Origin of the Large Remanent Polarization in LaHfO2"),
    ("Application of convolutional neural networks and", "Application of convolutional neural networks and ensemble methods"),
    ("Polymer informatics Current status", "Polymer informatics Current status and critical next steps"),
])
def test_lookup_uses_the_resolver_without_false_yes(mod, library, name, expected):
    assert documents.lookup_document(name, library, {"name": "generic"}) == expected


# ── the pipe: answered from the list, no answer model ──────────────────────────
def _run(mod, monkeypatch, question, *, resolved=None, heads=(), dc_titles=None):
    import openai
    import memento.retrieval.search as search
    import memento.common.model_servers as vc
    payloads = [{"text": "passage", "source": "P", "page": 1, "type": "text"}]
    points = [types.SimpleNamespace(payload=p, score=0.5, id=i) for i, p in enumerate(payloads)]
    _RecordingAsyncOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _RecordingAsyncOpenAI)
    monkeypatch.setattr(library_state, "collection_status", lambda c, coll: "ok")
    monkeypatch.setattr(library_state, "load_documents", lambda *a: {"P": {}, "Q": {}, "R": {}})
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d=None: resolved)
    monkeypatch.setattr(documents, "display_titles", lambda s, d: {x: f"Title of {x}" for x in s})
    monkeypatch.setattr(library_state, "qdrant_query", lambda *a, **k: types.SimpleNamespace(points=points))
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0] * 8)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [dict(h, rerank_score=0.6) for h in hits])
    seen = {}

    def related(self, client, qvec, query, query_filter):
        seen["query"] = query
        return list(heads)
    monkeypatch.setattr(answer_flow, "related_heads", related)
    p = mod.Pipe()
    p.valves = p.Valves(DOMAIN_NAME="generic", COLLECTION="x", RERANK_ENABLED=True, GRAPH_RETRIEVAL=False,
                        VALIDATION_ENABLED=False, ENABLE_QUERY_REWRITE=True, THINKING_AUTO=False, RELATED_DOCS=False)

    async def run():
        return [c async for c in p.pipe({"messages": [{"role": "user", "content": question}]}, {"id": "t"})]
    out = "".join(c for c in asyncio.run(run()) if isinstance(c, str))
    return out, [c for c in _RecordingAsyncOpenAI.calls if c.get("stream")], seen


def test_pipe_found_document_answers_yes_without_the_model(mod, monkeypatch):
    out, streamed, _ = _run(mod, monkeypatch, "Do we have the polyBERT paper?", resolved="Q")
    assert streamed == [] and out.strip().startswith("**Yes**, *Title of Q* is in the library.")


def test_pipe_missing_document_lists_related_ones_above_the_cut_off(mod, monkeypatch):
    heads = [{"source": "P", "rerank_score": 0.45}, {"source": "R", "rerank_score": 0.12},
             {"source": "Q", "rerank_score": 0.03}]
    out, streamed, seen = _run(mod, monkeypatch, "Is the paper 'Attention Is All You Need' in the library?",
                               heads=heads)
    assert streamed == [] and seen["query"] == "Attention Is All You Need"     # searched with the name alone
    assert out.strip().startswith("**No** document titled exactly *Attention Is All You Need* is in the library.")
    assert "- Title of P\n- Title of R" in out and "Title of Q" not in out


def test_pipe_missing_document_nothing_related(mod, monkeypatch):
    heads = [{"source": "P", "rerank_score": 0.027}, {"source": "R", "rerank_score": 0.02}]
    out, streamed, _ = _run(mod, monkeypatch, "Is the paper 'Attention Is All You Need' in the library?", heads=heads)
    assert streamed == [] and "(3 documents)" in out and "Related documents" not in out


def test_pipe_reranker_down_gives_no_related_list(mod, monkeypatch):
    heads = [{"source": "P", "rerank_score": 0.9, "rerank_fallback": True}]
    out, streamed, _ = _run(mod, monkeypatch, "Is the paper 'Attention Is All You Need' in the library?", heads=heads)
    assert streamed == [] and "(3 documents)" in out and "Title of P" not in out


def test_pipe_topic_question_goes_the_normal_way(mod, monkeypatch):
    out, streamed, _ = _run(mod, monkeypatch, "Is there a paper about ferroelectric hafnia?")
    assert len(streamed) == 1 and "**No** document" not in out and "**Yes**" not in out
