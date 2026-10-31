# Processing one new document for the gap ledger (memento/research_gaps/update.py): the per-document lock, and the order of the
# steps under it.
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, update
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import ledger as gl, update as gu
from memento.library import documents as ldb  # noqa: E402


@pytest.fixture
def engine():
    from sqlalchemy.pool import StaticPool
    # one in-memory database for every thread (the steps run side by side, as with Postgres live)
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    gu.ensure_tables(e)
    gl.ensure_tables(e)
    ldb.ensure_tables(e)
    return e


def test_a_document_is_processed_by_one_process_at_a_time(engine):
    assert gu.claim(engine, "col", "D", "updater") is True
    assert gu.claim(engine, "col", "D", "report") is False and gu.in_progress(engine, "col", "D")
    assert gu.claim(engine, "col", "E", "report") is True                       # another document: free
    gu.release(engine, "col", "D")
    assert not gu.in_progress(engine, "col", "D") and gu.claim(engine, "col", "D", "report") is True


def test_a_lock_left_by_a_dead_process_is_taken_over(engine):
    gu.claim(engine, "col", "D", "updater")
    with engine.begin() as c:
        c.execute(update(gu.JOBS).values(started_at=datetime.now(timezone.utc) - timedelta(minutes=30)))
    assert gu.claim(engine, "col", "D", "report") is True


def _ctx(engine, log):
    return gu.Ctx(engine=engine, qdrant=None, llm_client=None, model="m", collection="col", embedder_url="",
                  embedder_model="", reranker_url="", instruction="", log=log.append)


def test_the_steps_run_in_order_under_the_lock_and_report_progress(engine, monkeypatch):
    ldb.add_document(engine, collection="col", entry={"source": "D", "title": "Doc D"}, content_hash="h",
                     uploaded_by="u", output_root=None)
    calls, said, log = [], [], []

    def extract(ctx, source, title, chash):
        assert gu.in_progress(engine, "col", "D")                                 # under the lock
        calls.append(("extract", source, title, chash))
        gid = gl.add_gap(engine, "col", "New?", "limitation")
        gl.add_mention(engine, gid, "col", source="D", point_id="1", page=0, signal="s", claim="c", passage="p",
                       judge="supported")
        return [gid]
    monkeypatch.setattr(gu, "extract", extract)
    monkeypatch.setattr(gu, "check_against", lambda ctx, s, gaps: calls.append(("against", [g["statement"] for g in
                                                                                         gaps])) or {})
    monkeypatch.setattr(gu, "check_library", lambda ctx, gaps, n: calls.append(("library", [g["statement"] for g in
                                                                                        gaps], n)) or {})
    monkeypatch.setattr(gu, "check_literature", lambda ctx, gaps, t, d: calls.append(("literature", len(gaps))) or {})
    old = gl.add_gap(engine, "col", "Old?", "limitation")
    r = gu.process_document(_ctx(engine, log), "D", owner="report", status=said.append)
    # the document's own questions first; then the existing ones against it, and its new ones checked, side by side
    assert calls[0][0] == "extract" and {c[0] for c in calls[1:]} == {"against", "library", "literature"}
    assert [c[0] for c in calls if c[0] != "against"] == ["extract", "library", "literature"]
    assert calls[0][1:] == ("D", "Doc D", "h")
    by = {c[0]: c for c in calls}
    assert by["against"][1] == ["Old?"] and by["library"][1:] == (["New?"], 1)     # new questions not "against"
    assert said[0] == "extract" and set(said) == {"extract", "against", "library", "literature"}
    assert r["questions"] == 1
    assert not gu.in_progress(engine, "col", "D") and old


def test_a_document_another_process_has_is_left_alone(engine, monkeypatch):
    gu.claim(engine, "col", "D", "updater")
    monkeypatch.setattr(gu, "extract", lambda *a: pytest.fail("processed twice"))
    assert gu.process_document(_ctx(engine, []), "D", owner="report") is None


def test_the_lock_is_released_when_a_step_fails(engine, monkeypatch):
    monkeypatch.setattr(gu, "extract", lambda *a: (_ for _ in ()).throw(RuntimeError("model down")))
    with pytest.raises(RuntimeError):
        gu.process_document(_ctx(engine, []), "D", owner="report")
    assert not gu.in_progress(engine, "col", "D")


def test_model_calls_are_limited_to_the_workers(engine):
    import threading
    import time
    running, peak, lock = [0], [0], threading.Lock()

    class Client:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, **kw):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=Client(), model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="", workers=3)
    assert gu._pmap(lambda i: ctx.llm([]), list(range(12)), 8) == ["ok"] * 12
    assert peak[0] == 3


def test_answered_questions_count_a_new_document_without_a_check(engine, monkeypatch):
    # adding and deleting must be symmetric: deletion counts a document off every question that does not raise it
    import memento.common.model_servers as vc
    a = gl.add_gap(engine, "col", "Answered?", "limitation")
    gl.add_mention(engine, a, "col", source="X", point_id="1", page=0, signal="s", claim="c", passage="p",
                   judge="supported")
    gl.set_library_status(engine, a, status="addressed", docs_checked=24,
                          evidence=[dict(source="Y", point_id="2", page=0, relation="addresses", reason="r")])
    monkeypatch.setattr(vc, "vl_embed_many", lambda *a, **k: pytest.fail("no search for an answered question"))
    out = gu.check_against(_ctx(engine, []), "New", gl.list_gaps(engine, "col"))
    g = gl.get_gap(engine, a)
    assert (g["library_status"], g["library_docs_checked"]) == ("addressed", 25) and out == {"counted": 1}


def test_the_literature_check_also_searches_the_offline_index(engine, monkeypatch):
    from memento.research_gaps import literature as al, jobs as gj, literature as ls
    import memento.retrieval.search as search
    w = ls.Work("Offline paper", None, "10.1/off", "An abstract that answers it.", "https://doi.org/10.1/off",
                "alexandria", "10.1/off", {"alexandria"})
    monkeypatch.setattr(al, "available", lambda *a, **k: True)
    monkeypatch.setattr(al, "search", lambda q, **k: [w])
    monkeypatch.setattr(ls, "search_openalex", lambda *a, **k: ([], None))
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    gid = gl.add_gap(engine, "col", "Does X work?", "limitation")
    gl.add_mention(engine, gid, "col", source="D", point_id="1", page=0, signal="s", claim="c", passage="p",
                   judge="supported")

    class Client:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, messages, **kw):
            text = "x work phrase" if "search query" in messages[0]["content"] else \
                '[{"i": 1, "relation": "partly", "reason": "r"}]'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=Client(), model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="")
    out = gu.check_literature(ctx, gl.list_gaps(engine, "col"), [], [])
    g = gl.get_gap(engine, gid)
    assert out == {"partly": 1} and "alexandria" in g["web_sources"]
    assert [e["origin"] for e in gl.evidence(engine, gid)] == ["alexandria"]


def test_new_questions_count_only_documents_the_ledger_has_processed(engine, monkeypatch):
    # a new document's questions are counted against the documents the ledger has processed, not the whole library
    # (documents uploaded with it would be counted again when processed)
    for s in ("D", "E", "F", "P"):
        ldb.add_document(engine, collection="col", entry={"source": s, "title": f"Doc {s}"}, content_hash="h" + s,
                         uploaded_by="u", output_root=None)
    gl.set_document(engine, "col", "P", content_hash="hP", status="done")          # processed earlier
    got = {}
    monkeypatch.setattr(gu, "extract", lambda ctx, source, title, chash: [])
    monkeypatch.setattr(gu, "check_against", lambda ctx, s, gaps: {})
    monkeypatch.setattr(gu, "check_library", lambda ctx, gaps, n: got.setdefault("n", n) or {})
    monkeypatch.setattr(gu, "check_literature", lambda ctx, gaps, t, d: {})
    gid = gl.add_gap(engine, "col", "New?", "limitation")
    gl.add_mention(engine, gid, "col", source="D", point_id="1", page=0, signal="s", claim="c", passage="p",
                   judge="supported")
    monkeypatch.setattr(gu, "extract", lambda ctx, source, title, chash: [gid])
    gu.process_document(_ctx(engine, []), "D", owner="report", status=lambda s: None)
    assert got["n"] == 2        # P (processed) + D itself; E and F are counted when they are processed


# ── an incomplete judgment sets no status: never a false "open" / "nothing found" ──
def _client(replies):
    """A model whose answers come from `replies(messages)` (the literature check also asks for a search phrase)."""
    calls = []

    class Client:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, messages, **kw):
            calls.append(messages)
            text = "x phrase" if "search query" in messages[0]["content"] else replies(messages)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    return Client(), calls


def _search_fakes(monkeypatch, passages):
    import memento.retrieval.search as search
    import memento.common.model_servers as vc

    def point(i, src):
        return SimpleNamespace(id=i, score=0.9, payload={"text": f"passage {i}", "source": src, "page": 0,
                                                         "type": "text"})
    monkeypatch.setattr(vc, "vl_embed", lambda *a, **k: [0.0])
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, items, *a, **k: [[0.0] for _ in items])
    monkeypatch.setattr(search, "qdrant_query",
                        lambda *a, **k: SimpleNamespace(points=[point(i, s) for i, s in passages]))
    monkeypatch.setattr(search, "related_heads", lambda *a, **k: [])
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: [{**h, "rerank_score": 0.5} for h in hits])


def _gap(engine, statement="Does X work?", src="D"):
    gid = gl.add_gap(engine, "col", statement, "limitation")
    gl.add_mention(engine, gid, "col", source=src, point_id="1", page=0, signal="s", claim="c", passage="p",
                   judge="supported")
    return gid


def test_a_library_check_with_unjudged_passages_sets_no_status(engine, monkeypatch):
    _search_fakes(monkeypatch, [(11, "E"), (12, "F")])
    gid = _gap(engine)
    # the model judges passage 11 and never passage 12 (on the retry passage 12 alone is asked: no answer)
    client, calls = _client(lambda m: '[{"i": 1, "relation": "not_concern", "reason": "r"}]'
                            if "passage 11" in m[-1]["content"] else "[]")
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=client, model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="")
    out = gu.check_library(ctx, gl.list_gaps(engine, "col"), 5)
    g = gl.get_gap(engine, gid)
    assert out == {"incomplete": 1} and g["library_status"] is None and g["library_checked_at"] is None
    assert len(calls) == 2                                      # asked once more, for the unjudged passage only


def test_a_complete_library_check_still_sets_its_status(engine, monkeypatch):
    _search_fakes(monkeypatch, [(11, "E"), (12, "F")])
    gid = _gap(engine)
    client, _ = _client(lambda m: '[{"i": 1, "relation": "not_concern"}, {"i": 2, "relation": "not_concern"}]')
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=client, model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="")
    assert gu.check_library(ctx, gl.list_gaps(engine, "col"), 5) == {"open": 1}
    assert gl.get_gap(engine, gid)["library_status"] == "open"


def test_a_literature_check_with_unjudged_papers_sets_no_status(engine, monkeypatch):
    from memento.research_gaps import literature as al, literature as ls
    import memento.retrieval.search as search
    works = [ls.Work(f"Paper {k}", None, f"10.1/p{k}", "An abstract.", "", "alexandria", f"10.1/p{k}", {"alexandria"})
             for k in range(2)]
    monkeypatch.setattr(al, "available", lambda *a, **k: True)
    monkeypatch.setattr(al, "search", lambda q, **k: works)
    monkeypatch.setattr(ls, "search_openalex", lambda *a, **k: ([], None))
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    gid = _gap(engine)
    client, _ = _client(lambda m: "[]")                         # nothing judged, twice
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=client, model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="")
    out = gu.check_literature(ctx, gl.list_gaps(engine, "col"), [], [])
    g = gl.get_gap(engine, gid)
    assert out == {"incomplete": 1} and g["web_status"] is None and gl.evidence(engine, gid) == []


def test_a_new_document_with_unjudged_passages_changes_and_counts_nothing(engine, monkeypatch):
    _search_fakes(monkeypatch, [(21, "New"), (22, "New")])
    gid = _gap(engine)
    gl.set_library_status(engine, gid, status="open", docs_checked=24, evidence=[])
    client, _ = _client(lambda m: '[{"i": 2, "relation": "addresses"}]' if "passage 22" in m[-1]["content"]
                        and "passage 21" in m[-1]["content"] else "[]")
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=client, model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="")
    out = gu.check_against(ctx, "New", gl.list_gaps(engine, "col"))
    g = gl.get_gap(engine, gid)
    assert out == {"incomplete": 1} and (g["library_status"], g["library_docs_checked"]) == ("open", 24)
