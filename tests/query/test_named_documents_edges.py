"""Edge cases of named documents, author lists and stored answers; no live services."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from memento.library import documents as ldb
from memento.research_gaps import passages as gr
from test_scope_kept import _run
from test_locks_scopes_stored_answers import pipe_databases
from memento.config.paths import PROJECT_ROOT


def test_new_missing_document_must_not_become_previous_document(monkeypatch):
    out, rec, model_input = _run(monkeypatch,
        'What does the Missing Zebra paper report about hardness?',
        {'intent': 'ranked_retrieval', 'query': 'Missing Zebra paper hardness', 'scope': 'document'},
        history=[{'role': 'user', 'content': 'Tell me about paper A'},
                 {'role': 'assistant', 'content': 'Paper A studies hardness [1].'}], target='A')
    assert not rec['searches'], (out['answer'], model_input)


def test_no_own_uploads_must_not_resolve_somebody_elses_paper(monkeypatch):
    out, rec, model_input = _run(monkeypatch,
        'What do my latest uploads of Paper B report about hardness?',
        {'intent': 'ranked_retrieval', 'query': 'Paper B hardness', 'scope': 'document'},
        uploads=[], resolve='B')
    assert not rec['searches'], (out['answer'], model_input)


@pytest.mark.parametrize('emphasis', ['_', '**'])
def test_emphasized_numbered_deliberation_is_rejected(emphasis):
    raw = f"### Limitations\n1. {emphasis}Let's think about classification.{emphasis}\n"
    assert gr.parse_gap_answer(raw).rejected


@pytest.mark.parametrize('emphasis', ['_', '**'])
def test_stored_emphasized_numbered_deliberation_is_rejected(emphasis):
    raw = f"### Limitations\n1. {emphasis}Let's think about classification.{emphasis}\n"
    assert not gr.stored_gap_ok(gr.calm_gap_answer(raw))


@pytest.mark.parametrize('relative,client_name,source_name', [
    ('memento/ingestion/upload/document.py', 'qc', 'stem'),
    ('memento/ingestion/run_pipeline.py', 'client', 'paper_id'),
])
def test_production_source_callback_does_not_resurrect_removed_upload(monkeypatch, relative, client_name, source_name):
    # Execute the actual nested callback, extracted unchanged to avoid importing ingestion services.
    tree = ast.parse((PROJECT_ROOT / relative).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'read_sources')
    vectors = {'Doc'}
    client = SimpleNamespace(scroll=lambda **kw: ([SimpleNamespace(payload={'source': s}) for s in vectors], None))
    namespace = {client_name: client, source_name: 'Doc', 'COLLECTION': 'c',
                 'self': SimpleNamespace(valves=SimpleNamespace(COLLECTION='c'))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    e = create_engine('sqlite://')
    ldb.ensure_tables(e)
    ldb.add_document(e, collection='c', entry={'source': 'Doc', 'title': 'Doc'},
                     content_hash='h1', uploaded_by='u', output_root=None)
    original = ldb.document_entry
    def entry(*a, **kw):
        result = original(*a, **kw)
        vectors.clear()
        ldb.remove_document(e, collection='c', source='Doc', output_root=None)
        return result
    monkeypatch.setattr(ldb, 'document_entry', entry)
    ldb.rebuild_documents(e, collection='c', sources=namespace['read_sources'], in_dir=None, out_dir=None,
                         section_hints=None, manifest={'documents': {'Doc': {'content_hash': 'h1'}}}, output_root=None)
    assert ldb.get_document(e, 'c', 'Doc') is None


def test_refresh_must_not_replace_newer_upload_with_prelock_metadata(monkeypatch):
    e = create_engine('sqlite://')
    ldb.ensure_tables(e)
    ldb.add_document(e, collection='c', entry={'source': 'Doc', 'title': 'Doc'},
                     content_hash='h1', uploaded_by='u', output_root=None)
    original = ldb.document_entry
    def entry(*a, **kw):
        result = original(*a, **kw)
        ldb.add_document(e, collection='c', entry={'source': 'Doc', 'title': 'Updated Doc'},
                         content_hash='h2', uploaded_by='u', output_root=None)
        return result
    monkeypatch.setattr(ldb, 'document_entry', entry)
    ldb.rebuild_documents(e, collection='c', sources=lambda: {'Doc'}, in_dir=None, out_dir=None,
                         section_hints=None, manifest={'documents': {'Doc': {'content_hash': 'h1'}}},
                         output_root=None, refresh=True)
    assert ldb.get_document(e, 'c', 'Doc')['content_hash'] == 'h2'
