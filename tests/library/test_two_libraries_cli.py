# The command-line tools with two libraries. A command-line upload adds to the start-screen list of the copy that serves
# its collection (given with --function-id, else the one query copy whose saved COLLECTION is the run's; none: skipped;
# several: an error); the question CLI asks the copy named with --function-id. Encrypted (unreadable) Valves are
# refused, never replaced by defaults.
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.library import copies as cs  # noqa: E402

COMP, POLY = "memento-comp-mat-rag", "memento-polymer-rag"


def _db(rows):
    e = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with e.begin() as c:
        c.execute(text("CREATE TABLE function (id TEXT, valves TEXT)"))
        for fid, v in rows:
            c.execute(text("INSERT INTO function VALUES (:i, :v)"),
                      {"i": fid, "v": v if isinstance(v, str) else json.dumps(v)})
    return e


TWO = [("memento_query", {"COLLECTION": COMP}), ("memento_query_polymer", {"COLLECTION": POLY}),
       ("memento_ingestion_polymer", {"COLLECTION": POLY}), ("memento_thorough", {})]


def test_the_copy_serving_a_collection_is_found_by_its_saved_collection():
    e = _db(TWO)
    assert cs.find_copy(e, POLY) == "memento_query_polymer"            # the ingestion copy does not count
    assert cs.find_copy(e, COMP) == "memento_query"
    assert cs.find_copy(e, "memento-ceramics-rag-ocr") is None          # an evaluation collection: nobody's


def test_a_named_copy_is_used_only_for_its_own_collection():
    e = _db(TWO)
    assert cs.find_copy(e, POLY, "memento_query_polymer") == "memento_query_polymer"
    assert cs.find_copy(e, COMP, "memento_query_polymer") is None
    assert cs.find_copy(e, POLY, "memento_query_gone") is None


def test_two_copies_naming_one_collection_are_an_error():
    e = _db(TWO + [("memento_query_copy", {"COLLECTION": POLY})])
    with pytest.raises(ValueError, match="memento_query_copy"):
        cs.find_copy(e, POLY)


def test_encrypted_valves_are_refused_not_replaced_by_defaults():
    e = _db([("memento_query", "gAAAAAB-encrypted"), ("memento_query_polymer", {"COLLECTION": POLY})])
    with pytest.raises(ValueError, match="memento_query"):
        cs.find_copy(e, POLY)


# ── the command-line upload's list update (memento/ingestion/run_pipeline.py) ─────────

def _run(monkeypatch, rows, collection, function_id=None):
    import memento.ingestion.run_pipeline as rp
    import sqlalchemy
    from memento.library import documents as db
    e = _db(rows)
    db.ensure_tables(e)
    db.add_document(e, collection=collection, entry={"source": "old", "title": "old"}, content_hash="h",
                    uploaded_by="u", output_root=None)
    monkeypatch.setattr(rp, "_doc_list", {})
    monkeypatch.setattr(rp, "COLLECTION", collection)
    monkeypatch.setattr(rp, "FUNCTION_ID", function_id)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda u: e)
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    rp._add_to_document_list("paper_a", client=types.SimpleNamespace(scroll=lambda **k: ([], None)))
    return sorted(r["source"] for r in db.list_documents(e, collection) if r["source"] != "old")


def test_a_polymer_command_line_upload_updates_the_polymer_list(monkeypatch):
    assert _run(monkeypatch, TWO, POLY) == ["paper_a"]
    assert _run(monkeypatch, TWO, POLY, function_id="memento_query_polymer") == ["paper_a"]


def test_no_copy_or_several_copies_leave_the_list_alone(monkeypatch):
    assert _run(monkeypatch, TWO, "memento-ceramics-rag-ocr") == []
    assert _run(monkeypatch, TWO + [("memento_query_copy", {"COLLECTION": POLY})], POLY) == []
    assert _run(monkeypatch, TWO, POLY, function_id="memento_query") == []      # named copy serves comp-mat


# ── the question CLI (memento/retrieval/pipe_cli.py) ──────────────────────────────────

def test_the_question_cli_asks_the_named_copy(monkeypatch):
    from memento.retrieval import pipe_cli
    e = _db(TWO)
    mod = pipe_cli.load_pipe_module()
    p = pipe_cli.make_pipe(mod, e, "memento_query_polymer")
    assert p.valves.COLLECTION == POLY and p.function_id == "memento_query_polymer"
    seen = []

    async def fake(body, user, emit, files, meta):
        seen.append(body["model"])
        yield "x"
    p.pipe = fake
    asyncio.run(pipe_cli.ask(p, "q"))
    asyncio.run(pipe_cli.ask(p, "q", model="qwen35"))
    assert seen == ["memento_query_polymer.local", "memento_query_polymer.qwen35"]


def test_the_question_cli_refuses_encrypted_valves():
    from memento.retrieval import pipe_cli
    e = _db([("memento_query", "gAAAAAB-encrypted")])
    with pytest.raises(SystemExit, match="encrypted"):
        pipe_cli.make_pipe(pipe_cli.load_pipe_module(), e)


def test_the_question_cli_has_a_function_id_option():
    from memento.retrieval import pipe_cli
    src = Path(pipe_cli.__file__).read_text()
    assert '"--function-id"' in src
