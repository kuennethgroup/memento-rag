# Both libraries' copies in one Open WebUI process: the query and the upload pipe are each loaded twice from the same
# file, as Open WebUI loads them (function_memento_query / function_memento_query_polymer, function_memento_ingestion /
# function_memento_ingestion_polymer), and every per-library setting is checked with both copies present, called in
# turn, so a value one copy leaves in a shared module would show up in the other.
import asyncio
import contextvars
import os
import sys
import time
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.query import generation, library_state, request, service, texts, workers
from memento.document_chat import chat, cleanup
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_two_libraries_worker import load_copy  # noqa: E402

QUERY = ROOT / "openwebui/functions/query_pipe.py"


def copy_state(pipe_module, *, cached: bool = True, role: str = "query"):
    """A copy's state as that copy reads it: its id and its database access."""
    return library_state.copy_state(role, cached=cached, function_id=pipe_module._OWN_ID,
                                    db_engine=pipe_module._db_engine)


@pytest.fixture(autouse=True)
def _own_generations(monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, "_memento_warmup_gen", {}, raising=False)


# ── X2: both copies' hourly cleaners share the report folder ─────────────────────────────────────────────────────────

def test_a_report_removed_by_the_other_copy_between_listing_and_stat_is_skipped(tmp_path, monkeypatch):
    qp = load_copy(QUERY, "memento_query_polymer", monkeypatch)
    d = tmp_path / "_runtime" / "reports"
    d.mkdir(parents=True)
    gone, old = d / ("a" * 24 + ".pdf"), d / ("b" * 24 + ".pdf")
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 31 * 86400,) * 2)
    real_glob = Path.glob
    # the listing still names a file the other copy's cleaner has just removed
    monkeypatch.setattr(Path, "glob", lambda self, pat: [gone, *real_glob(self, pat)] if self == d else
                        real_glob(self, pat))
    assert cleanup.remove_old_reports(str(tmp_path), days=30) == 1 and not old.exists()


# ── both libraries' copies in one process ────────────────────────────────────────────────────────────────────────────
import json  # noqa: E402
import types  # noqa: E402

from memento.library import documents as db  # noqa: E402
from memento.document_chat import store as td  # noqa: E402
from memento.library import warmup_queue as wq  # noqa: E402
from test_two_libraries_worker import INGEST, METRICS, engine, put, study_valves  # noqa: E402
from test_two_libraries_doc_chat import COMP as TEMP_COMP, POLY as TEMP_POLY, _ask, _claim, study  # noqa: E402,F401
from test_query_pipe_document_chat import CHAT, PASSAGES, _drive  # noqa: E402

COMP, POLY = "memento-comp-mat-rag", "memento-polymer-rag"
STUDY_IDS = dict(QUERY_MODEL_ID="memento-query-polymer", UPLOAD_MODEL_ID="memento-ingestion-polymer",
                 DOC_CHAT_MODEL_ID="memento-document-chat-polymer")
LINKS = "[Library Upload](/?model=memento-ingestion-prod), [Document Chat](/?model=memento-document-chat)"


@pytest.fixture
def both(monkeypatch):
    """The default and the second query and upload copies, loaded from the same two files into this one process, the
    second copy's Valves saved as the deployment saves them."""
    qc, qs = load_copy(QUERY, "memento_query", monkeypatch), load_copy(QUERY, "memento_query_polymer", monkeypatch)
    ic, is_ = load_copy(INGEST, "memento_ingestion", monkeypatch), load_copy(INGEST, "memento_ingestion_polymer",
                                                                             monkeypatch)
    sq = study_valves(qs, "query", LLM_MODEL="qwen-flash-next", LLM_API_KEY_ENV="QWEN_FLASH_NEXT_API_KEY",
                      METRICS_URL="http://llm.test:8001/metrics", METRICS_API_KEY_ENV="QWEN_FLASH_NEXT_API_KEY",
                      VARIANTS_FILE="query_models_study.yaml", **STUDY_IDS)
    si = study_valves(is_, "ingestion", LLM_API_KEY_ENV="QWEN_FLASH_NEXT_API_KEY", **STUDY_IDS)
    put("memento_query_polymer", sq)
    put("memento_ingestion_polymer", si)
    return types.SimpleNamespace(qc=qc, qs=qs, ic=ic, is_=is_, sq=sq, si=si, vs=workers.valves_from(qs.Pipe.Valves, sq),
                                 vc=qc.Pipe.Valves(DOMAIN_NAME="computational-materials", COLLECTION=COMP))


def test_each_copy_knows_its_id_its_readiness_and_its_own_worker(both):
    b = both
    assert [m._OWN_ID for m in (b.qc, b.qs, b.ic, b.is_)] == [
        "memento_query", "memento_query_polymer", "memento_ingestion", "memento_ingestion_polymer"]
    assert copy_state(b.qc).enabled and copy_state(b.qs, cached=False).enabled
    assert copy_state(b.ic, role="ingestion").enabled and copy_state(b.is_, role="ingestion").enabled
    put("memento_query_polymer", b.sq, active=False)                     # the second copy switched off
    assert not copy_state(b.qs, cached=False).enabled and copy_state(b.qc, cached=False).enabled
    today, study_gen = workers.next_gen("memento_query"), workers.next_gen("memento_query_polymer")
    workers.next_gen("memento_query_polymer")                              # the second copy's code reloaded
    assert workers.gen_current("memento_query", today)                     # today's worker runs on
    assert not workers.gen_current("memento_query_polymer", study_gen)


def test_each_copy_uses_its_own_domain_keys_models_and_links_in_turn(both, monkeypatch):
    b = both
    monkeypatch.delenv("EMBED_INSTRUCTION", raising=False)
    monkeypatch.setenv("REASONING_API_KEY", "key-comp")       # the default copy: the answer model's key
    monkeypatch.setenv("QWEN_FLASH_NEXT_API_KEY", "key-study")
    poly = request.domain("polymer-engineering")["rerank_instruction"]
    comp = request.domain("computational-materials")["rerank_instruction"]
    assert poly and comp and poly != comp
    up_today, up_study = b.ic.Pipe.Valves(), types.SimpleNamespace(LLM_API_KEY_ENV="QWEN_FLASH_NEXT_API_KEY",
                                                                   **STUDY_IDS)
    for _ in range(2):              # in turn, twice: nothing one copy leaves in a shared module reaches the other
        assert request.rerank_instruction(request.domain(b.vs.DOMAIN_NAME)) == poly
        assert request.rerank_instruction(request.domain(b.vc.DOMAIN_NAME)) == comp
        assert (request.llm_key(b.vs), request.llm_key(b.vc)) == ("key-study", "key-comp")
        assert (document.llm_key(up_study), document.llm_key(up_today)) == ("key-study", "key-comp")
        assert [v.name for v in generation.copy_variants(b.vs) if not v.is_cloud] == ["Library Chat (Qwen 3.8 Flash-Next)"]
        # both libraries on the local model
        assert [v.name for v in generation.copy_variants(b.vc) if not v.is_cloud] == ["Library Chat (Qwen 3.8 Flash-Next)"]
        assert texts.relink_for_copy(LINKS, b.vs) == LINKS.replace("memento-ingestion-prod", "memento-ingestion-polymer") \
            .replace("memento-document-chat", "memento-document-chat-polymer")
        assert b.is_._relink(LINKS, up_study) == texts.relink_for_copy(LINKS, b.vs)
        assert texts.relink_for_copy(LINKS, b.vc) == LINKS and b.ic._relink(LINKS, up_today) == LINKS
    # a request of the second copy in progress, in its own context as every request runs
    contextvars.copy_context().run(request.LINK_IDS.set, texts.copy_link_ids(b.vs))
    assert texts.relink_for_copy(LINKS) == LINKS                                  # today's copy's own request is unchanged


def test_each_worker_claims_asks_and_records_only_in_its_own_library(both, monkeypatch):
    b = both
    monkeypatch.setenv("QWEN_FLASH_NEXT_API_KEY", "key-study")
    e = engine()
    db.ensure_tables(e)
    wq.ensure_table(e)
    for coll, src, title in ((COMP, "c1", "Comp doc"), (POLY, "p1", "Poly doc")):
        db.add_document(e, collection=coll, entry={"source": src, "title": title}, content_hash="h", uploaded_by="u",
                        output_root=None)
        wq.enqueue(e, coll, src, ["summary"])
    asked = []
    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout=None, headers=None: asked.append((url, headers)) or
                        types.SimpleNamespace(ok=True, text=METRICS % "0.0"))
    for m in (b.qc, b.qs):
        monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())

    seen = {"memento_query_polymer": [], "memento_query": []}
    keys = {"memento_query_polymer": "key-poly", "memento_query": "key-comp"}

    async def handle(body, user, emit, files, metadata, task, *, function_id, **k):   # each copy's answer path
        seen[function_id].append((body["model"], body["messages"][-1]["content"]))
        metadata["warmup_result"].update(result="stored", reason="", key=keys[function_id])
        yield "an answer"
    monkeypatch.setattr(service, "handle", handle)

    def step(valves, function_id):                  # one round of the copy's worker
        return asyncio.run(workers.warmup_step(valves, e, function_id=function_id, db_engine=engine))
    assert step(b.vs, "memento_query_polymer") == "done"
    assert step(b.vc, "memento_query") == "done"
    assert step(b.vs, "memento_query_polymer") == "empty"                 # today's library's job is never its own
    assert seen["memento_query_polymer"] == [("memento_query_polymer.local", 'Summarize the document "Poly doc"')]
    assert seen["memento_query"] == [("memento_query.local", 'Summarize the document "Comp doc"')]
    with e.connect() as c:
        rows = {r["collection"]: (r["status"], r["cache_key"]) for r in c.execute(wq.JOBS.select()).mappings()}
    assert rows == {POLY: ("done", "key-poly"), COMP: ("done", "key-comp")}
    assert asked[:2] == [("http://llm.test:8001/metrics", {"Authorization": "Bearer key-study"}),
                         (b.vc.METRICS_URL, None)]


def test_both_copies_publish_their_own_list_and_models_without_dropping_the_others(both):
    b = both
    e = engine()
    db.ensure_tables(e)
    for coll, src, title in ((COMP, "c1", "Comp doc"), (POLY, "p1", "Poly doc")):
        db.add_document(e, collection=coll, entry={"source": src, "title": title}, content_hash="h", uploaded_by="u",
                        output_root=None)
    root = Path(os.environ["OUTPUT_ROOT"])
    for _ in range(2):              # the second copy writes last: the default copy's picker file must still be its own
        workers.publish_runtime(b.vc, function_id="memento_query", engine=e)
        workers.publish_runtime(b.vs, function_id="memento_query_polymer", engine=e)

    def titles(p):
        return [d["title"] for d in json.loads(p.read_text())["documents"]]
    assert titles(root / "_runtime/library" / f"{POLY}.json") == ["Poly doc"]
    assert titles(root / "_runtime/library" / f"{COMP}.json") == ["Comp doc"]
    assert titles(root / "_runtime/library.json") == ["Comp doc"]        # today's picker file: today's library only
    m = json.loads((root / "_runtime/query_models.json").read_text())
    assert list(m["copies"]) == ["memento_query", "memento_query_polymer"]
    for fid, coll, ids in (("memento_query", COMP, ("memento-query", "memento-ingestion-prod", "memento-document-chat")),
                           ("memento_query_polymer", POLY, tuple(STUDY_IDS.values()))):
        entries = m["copies"][fid]
        assert {x["collection"] for x in entries} == {coll}
        assert {(x["siblings"]["query"], x["siblings"]["upload"], x["siblings"]["doc"]) for x in entries} == {ids}
        assert all(x["id"].startswith(ids[0]) for x in entries if x["kind"] == "query")
    assert [x["id"] for x in m["models"]][:1] == ["memento-query"]      # today's loader: today's model first


def test_both_upload_copies_keep_their_own_seen_files_and_never_take_the_others_name(both, monkeypatch, tmp_path):
    b = both
    inp, out = tmp_path / "input", tmp_path / "upload-output"
    inp.mkdir()
    out.mkdir()
    monkeypatch.setattr(document, "MANIFEST_FILE", tmp_path / "manifest.json")    # one manifest, as deployed
    pc = document.Ingester(b.ic.Pipe.Valves(COLLECTION=COMP, INPUT_ROOT=str(inp), OUTPUT_ROOT=str(out)))
    ps = document.Ingester(b.is_.Pipe.Valves(COLLECTION=POLY, DOMAIN_NAME="polymer-engineering", INPUT_ROOT=str(inp),
                                             OUTPUT_ROOT=str(out)))
    for p in (pc, ps):
        p.in_library = lambda stem: False
    pc.save_seen_ids(pc.load_seen_ids() | {"f1"})                      # an attachment added to today's library
    assert "f1" in pc.load_seen_ids() and "f1" not in ps.load_seen_ids()
    a, s = tmp_path / "a.pdf", tmp_path / "s.pdf"
    a.write_text("today's text")
    s.write_text("the study's text")
    [d] = pc.decide([("report.pdf", a)], "alex-id", "Alex")
    [d2] = ps.decide([("report.pdf", s)], "alex-id", "Alex")
    assert (d.action, d.stem) == ("new", "report") and (d2.action, d2.stem) == ("renamed", "report__alex")
    docs = json.loads((tmp_path / "manifest.json").read_text())["documents"]
    assert docs["report"]["collection"] == COMP and docs["report__alex"]["collection"] == POLY


def test_one_chat_stays_in_its_library_while_both_copies_answer_in_turn(study, monkeypatch):
    today = load_copy(QUERY, "memento_query", monkeypatch)
    pc = today.Pipe()
    pc.valves = pc.Valves(DOMAIN_NAME="computational-materials", COLLECTION=COMP, QDRANT_URL="http://127.0.0.1:9")
    src = _claim(study.engine, TEMP_POLY)                               # the chat's file, in the second library
    monkeypatch.setitem(PASSAGES[0], "source", src)
    monkeypatch.setitem(PASSAGES[1], "source", src)
    assert "98 % [1]." in _ask(study)
    monkeypatch.setattr(chat, "doc_chat_read_one", lambda *a, **k: pytest.fail("a file was read"))
    out = _drive(pc, {"model": "memento_query.document-chat", "messages": [{"role": "user", "content": "q?"}]},
                 {"id": "u1"}, {"chat_id": CHAT}, files=[{"id": "f2", "name": "f2.pdf"}])[0]
    assert out.strip() == texts.DOC_CHAT_OTHER_LIBRARY_MSG             # today's Document Chat refuses the chat
    assert td.chat_documents(study.engine, CHAT, "u1", collection=TEMP_COMP) == []
    assert "98 % [1]." in _ask(study)                                   # and the second copy goes on


def test_a_polite_error_from_the_real_pipe_is_retried_not_counted(both, monkeypatch):
    # against the real pipe: the second copy answers its warm-up with a polite line only
    b = both
    p = b.qs.Pipe()
    p.valves = b.vs

    async def polite(turn):
        yield texts.STUDY_NOT_READY_MSG
    monkeypatch.setattr(service, "run", polite)
    out = asyncio.run(workers.run_warmup_job(p.valves, {"action": "summary"}, "Poly doc", function_id=p.function_id,
                                             db_engine=p.engine))
    assert out["result"] == "error"
