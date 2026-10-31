"""A message's context (question language, the copy's link ids, Document Chat's shown names) is set at the request
boundary and reset however the request ends; the people lock is one per copy."""
import asyncio

import pytest

from memento.query import library_state, request


def _values():
    return request.LANG.get(None), request.LINK_IDS.get(None), request.SHOWN_NAMES.get(None)


@pytest.mark.parametrize("error", [asyncio.CancelledError, ValueError])
def test_request_scope_resets(error):
    before = _values()
    with pytest.raises(error):
        with request.scope(lang="de", link_ids=("a", "b")), request.shown_names({"Doc-3fa2": "Report.pdf"}):
            assert _values() == ("de", ("a", "b"), {"Doc-3fa2": "Report.pdf"})
            raise error()
    assert _values() == before


def test_request_scope_resets_on_cancel():
    before = _values()
    with pytest.raises(asyncio.CancelledError):
        with request.scope(lang="de"):
            raise asyncio.CancelledError()
    assert _values() == before


def test_request_scope_resets_on_error():
    before = _values()
    with pytest.raises(ValueError):
        with request.scope(lang="de"):
            raise ValueError()
    assert _values() == before


def test_people_lock_per_copy():
    assert library_state.people_lock("a") is library_state.people_lock("a")
    assert library_state.people_lock("a") is not library_state.people_lock("b")


def test_a_turn_carries_one_message():
    import dataclasses
    t = request.Turn(body={}, user={}, files=None, metadata=None, emit=None, function_id="memento_query",
                     valves=object(), engine=lambda: None, rec=None, doc_chat=None)
    assert [f.name for f in dataclasses.fields(request.Turn)][:10] == [
        "body", "user", "files", "metadata", "emit", "function_id", "valves", "engine", "rec", "doc_chat"]
    assert t.done is False and t.status is None                    # set by the coordinator as the message goes on
    assert t.query_intent is None and t.hits is None and t.paper is None   # each step's own fields start empty


def test_request_imports_no_later_part_of_the_query_path():
    # Turn's annotations name cache.Recorder and DocChat as text only: request is imported by both
    import ast
    from memento.config.paths import PROJECT_ROOT
    tree = ast.parse((PROJECT_ROOT / "memento/query/request.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} \
        | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"cache", "chat", "memento.query.cache", "memento.document_chat.chat",
                           "memento.document_chat"}
