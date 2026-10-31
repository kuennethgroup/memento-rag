# Reply links name the copy's own models. The default copy's replies link Library Chat, Library Upload and Document Chat
# by its model ids, which an account of the second copy cannot open. Each copy names its models in Valves
# (QUERY_MODEL_ID, UPLOAD_MODEL_ID, DOC_CHAT_MODEL_ID); the fixed texts and their German twins keep the default ids (the
# German table matches the English text exactly), and the pipe's output rewrites every link to the copy's ids after
# translating. The default copy's output is unchanged, byte for byte.
import ast
import asyncio
import re
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT
from memento.query import presentation, request, texts
from memento.query.scope import conversation
from memento.document_chat import chat
from memento.ingestion.upload import messages

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from test_two_libraries_worker import INGEST, QUERY, load_copy  # noqa: E402
from memento.ingestion.upload import workflow

TODAY = {"QUERY_MODEL_ID": "memento-query", "UPLOAD_MODEL_ID": "memento-ingestion-prod",
         "DOC_CHAT_MODEL_ID": "memento-document-chat"}
STUDY = {"QUERY_MODEL_ID": "memento-query-polymer", "UPLOAD_MODEL_ID": "memento-ingestion-polymer",
         "DOC_CHAT_MODEL_ID": "memento-document-chat-polymer"}
LINK = re.compile(r"/\?model=([\w-]+)")


@pytest.fixture
def qp(monkeypatch):
    return load_copy(QUERY, "memento_query", monkeypatch)


@pytest.fixture
def ip(monkeypatch):
    return load_copy(INGEST, "memento_ingestion", monkeypatch)


def _query_modules(qp) -> tuple:
    """The query pipe and every module of memento.query its texts moved to."""
    import importlib, pkgutil
    import memento.query as q
    return (qp,) + tuple(importlib.import_module(m.name) for m in pkgutil.walk_packages(q.__path__, "memento.query."))


def _texts(mods: tuple, table_mod, table: str) -> list[str]:
    """Every module-level text with a model link (in any of `mods`), and the German twins of the fixed texts."""
    out = [v for mod in mods for k, v in vars(mod).items() if isinstance(v, str) and "/?model=" in v]
    out += [v for v in getattr(table_mod, table).values() if isinstance(v, str) and "/?model=" in v]
    return out


def test_the_valves_default_to_todays_ids(qp, ip):
    v = qp.Pipe.Valves()
    assert {k: getattr(v, k) for k in TODAY} == TODAY and v.INGESTION_FUNCTION_ID == "memento_ingestion"
    w = ip.Pipe.Valves()
    assert {k: getattr(w, k) for k in TODAY} == TODAY


def test_todays_ids_leave_every_text_unchanged(qp, ip):
    fixed = _texts(_query_modules(qp), texts, "DE_FIXED")
    assert len(fixed) >= 5               # NO_UPLOADS_MSG (+ German), GENERAL_SYSTEM_PROMPT, DOC_CHAT_LIBRARY_NOTE, …
    for t in fixed:
        assert texts.relink_for_copy(t, qp.Pipe.Valves()) == t
    for t in _texts((ip, messages), messages, "_DE_UPLOAD_FIXED"):
        assert ip._relink(t, ip.Pipe.Valves()) == t


def test_the_study_ids_reach_every_link_of_every_text(qp, ip):
    for mods, (table_mod, table), relink, v in ((_query_modules(qp), (texts, "DE_FIXED"), texts.relink_for_copy, qp.Pipe.Valves(**STUDY)),
                                                ((ip, messages), (messages, "_DE_UPLOAD_FIXED"), ip._relink, ip.Pipe.Valves(**STUDY))):
        for t in _texts(mods, table_mod, table):
            out = relink(t, v)
            assert set(LINK.findall(out)) <= set(STUDY.values()), out[:120]
            assert relink(out, v) == out                                     # twice = once


def test_a_question_link_keeps_its_question(qp):
    out = texts.relink_for_copy(chat.doc_chat_no_file("Was ist die Dichte?"), qp.Pipe.Valves(**STUDY))
    assert "/?model=memento-query-polymer&q=Was%20ist%20die%20Dichte%3F" in out


def test_every_link_in_the_pipe_code_is_one_the_output_rewrites():
    # a new link with another id would escape the rewrite: only today's three ids may be written in the code
    query_side = [QUERY] + sorted((ROOT / "memento/query").rglob("*.py")) + sorted((ROOT / "memento/document_chat").rglob("*.py"))
    for paths in (query_side, [INGEST] + sorted((ROOT / "memento/ingestion/upload").rglob("*.py"))):
        ids = set()
        for path in paths:
            for n in ast.walk(ast.parse(path.read_text())):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    ids |= set(LINK.findall(n.value))
        assert ids and ids <= set(TODAY.values()), (paths[0].name, ids)


# ── the output of a second copy ──

def _drive(mod, pipe, texts, content="What does the file say?", model="m.x"):
    from memento.query import service

    async def fake(turn):
        for t in texts:
            yield t
    original = service.run
    service.run = fake
    try:
        async def go():
            return [c async for c in pipe.pipe({"model": model, "messages": [{"role": "user", "content": content}]},
                                               {"id": "u", "role": "user"}, None, None, {})]
        return "".join(c for c in asyncio.run(go()) if isinstance(c, str))
    finally:
        service.run = original


def test_a_study_copys_replies_link_its_own_models(qp):
    p = qp.Pipe()
    p.valves = p.Valves(**STUDY)
    out = _drive(qp, p, [chat.doc_chat_no_file("density?"), "\n", conversation.unnamed_document_msg(False), "\n",
                         texts.NO_UPLOADS_MSG])
    assert set(LINK.findall(out)) == {"memento-query-polymer", "memento-document-chat-polymer",
                                      "memento-ingestion-polymer"}


def test_a_german_reply_is_translated_first_then_relinked(qp):
    p = qp.Pipe()
    p.valves = p.Valves(**STUDY)
    out = _drive(qp, p, [texts.NO_UPLOADS_MSG], content="Welche Dokumente habe ich in die Bibliothek hochgeladen?")
    assert out == texts.NO_UPLOADS_DE.replace("/?model=memento-ingestion-prod", "/?model=memento-ingestion-polymer")
    assert "/?model=memento-ingestion-polymer" in out


def test_todays_copy_replies_exactly_as_before(qp):
    p = qp.Pipe()
    replies = [chat.doc_chat_no_file("density?"), conversation.unnamed_document_msg(False), texts.NO_UPLOADS_MSG]
    assert _drive(qp, p, replies) == "".join(replies)


def test_a_replayed_answer_is_relinked_before_it_is_cut_into_pieces(qp):
    text = "See [Document Chat](/?model=memento-document-chat) for your own files. " * 3
    token = request.LINK_IDS.set(texts.copy_link_ids(qp.Pipe.Valves(**STUDY)))
    try:
        async def go():
            return [p async for p in presentation.type_out(text, rate=10_000, sleep=lambda s: asyncio.sleep(0))]
        pieces = asyncio.run(go())
    finally:
        request.LINK_IDS.reset(token)
    assert len(pieces) > 3 and "".join(pieces).count("/?model=memento-document-chat-polymer)") == 3


def test_the_ingestion_copys_replies_link_its_own_models(ip, monkeypatch):
    p = ip.Pipe()
    p.valves = p.Valves(**STUDY)

    async def fake(*a, **k):
        yield messages.ONLY_UPLOADS_MSG + messages.ask_line("What is a foam?")
    monkeypatch.setattr(workflow, "run", fake)

    async def go(content):
        return [c async for c in p.pipe({"model": "x", "messages": [{"role": "user", "content": content}]},
                                        {"id": "u", "role": "user"}, [])]
    out = "".join(asyncio.run(go("What is a foam?")))
    assert set(LINK.findall(out)) == {"memento-document-chat-polymer", "memento-query-polymer"}
    de = "".join(asyncio.run(go("Welche Methoden werden zur Charakterisierung von Schaumstoffen eingesetzt?")))
    assert "Bibliothek" in de and set(LINK.findall(de)) <= set(STUDY.values())


# ── Document Chat reads the linked ingestion copy's settings ─────────────────

def test_document_chat_reads_the_linked_ingestion_copys_settings():
    # its reading and its cleanup (memento/document_chat): the linked upload copy's saved Valves, never its own
    calls = [n for f in sorted((ROOT / "memento/document_chat").glob("*.py")) for n in ast.walk(ast.parse(f.read_text()))
             if isinstance(n, ast.Call) and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "saved_valves"
             and n.args and ast.unparse(n.args[0]) != "fid"]          # chat.answer hands the read a function of fid
    assert len(calls) >= 2
    for c in calls:
        assert ast.unparse(c.args[0]).endswith("valves.INGESTION_FUNCTION_ID"), ast.unparse(c)


def test_a_study_document_chat_id_is_recognised(qp):
    assert chat.is_doc_chat("memento_query_polymer.document-chat")
    assert chat.is_doc_chat("memento-document-chat-polymer", "memento-document-chat-polymer")
    assert not chat.is_doc_chat("memento-query-polymer", "memento-document-chat-polymer")
    assert chat.is_doc_chat("memento-document-chat") and not chat.is_doc_chat("memento-query")
