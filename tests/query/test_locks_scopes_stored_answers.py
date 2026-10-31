"""Contract checks of the library locks, scopes and stored answers; no live services."""
import pytest
from sqlalchemy import create_engine, text
from memento.library import documents as ldb, front_matter as da
from memento.research_gaps import passages as gr
from memento.retrieval import pipe_cli
from test_scope_kept import _run
from memento.query.scope import conversation, documents


@pytest.fixture(autouse=True)
def pipe_databases(monkeypatch):
    """Supply the parent/cache tables omitted by the author's fake-pipe fixtures."""
    original = pipe_cli.make_pipe
    engines = []
    def make(mod, engine, *a, **kw):
        ldb.ensure_tables(engine)
        for source, fingerprint in [('A','h1'), ('B','h2'), ('C','h3')]:
            ldb.add_document(engine, collection='memento-x-rag', entry={'source':source,'title':source},
                             content_hash=fingerprint, uploaded_by='probe', output_root=None)
        engines.append(engine)
        return original(mod, engine, *a, **kw)
    monkeypatch.setattr(pipe_cli, 'make_pipe', make)
    return engines


def test_empty_explicit_document_does_not_search_the_library(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What does the missing paper report about hardness?",
        {"intent": "ranked_retrieval", "query": "missing paper hardness", "scope": "document"})
    assert not rec['searches'], (out['answer'], model_input)


def test_empty_own_uploads_does_not_search_the_library(monkeypatch):
    original = pipe_cli.load_pipe_module
    def loaded():
        mod = original()
        monkeypatch.setattr(documents, 'latest_uploads', lambda *a: [])
        return mod
    monkeypatch.setattr(pipe_cli, 'load_pipe_module', loaded)
    out, rec, model_input = _run(monkeypatch, "What do my latest uploads report about hardness?",
        {"intent": "ranked_retrieval", "query": "hardness in my uploads", "scope": "documents"})
    assert not rec['searches'], (out['answer'], model_input)


def test_rebuild_does_not_resurrect_a_document_removed_after_source_read(monkeypatch):
    e = create_engine('sqlite://')
    ldb.ensure_tables(e)
    ldb.add_document(e, collection='c', entry={'source':'Doc','title':'Doc'}, content_hash='h1',
                     uploaded_by='u', output_root=None)
    original = ldb.document_entry
    vectors = {'Doc'}
    def entry(*a, **kw):
        result = original(*a, **kw)
        # The rebuild reads its sources through a function, called again under the collection lock, so a removal that
        # completes before the lock is visible there (removal deletes the vectors first).
        vectors.discard('Doc')
        ldb.remove_document(e, collection='c', source='Doc', output_root=None)
        return result
    monkeypatch.setattr(ldb, 'document_entry', entry)
    ldb.rebuild_documents(e, collection='c', sources=lambda: set(vectors), in_dir=None, out_dir=None,
                         section_hints=None, manifest={'documents':{'Doc':{'content_hash':'h1'}}}, output_root=None)
    assert ldb.get_document(e, 'c', 'Doc') is None


def test_unknown_hash_job_cannot_overwrite_current_author_list():
    e = create_engine('sqlite://')
    ldb.ensure_tables(e); da.ensure_table(e)
    ldb.add_document(e, collection='c', entry={'source':'Doc','title':'Doc'}, content_hash='h2',
                     uploaded_by='u', output_root=None)
    da.put(e, collection='c', source='Doc', content_hash='h2', front='Bob New')
    da.put(e, collection='c', source='Doc', content_hash=None, front='Alice Old')
    assert da.load(e,'c',with_hash=True)['Doc'] == ('Bob New','h2')


def test_author_write_after_removal_is_rejected():
    e = create_engine('sqlite://')
    ldb.ensure_tables(e); da.ensure_table(e)
    assert da.put(e, collection='c', source='Gone', content_hash='h1', front='Old author') is False


def test_italic_deliberation_is_not_an_accepted_pipe_note():
    raw = "### Limitations\n1. Only one material was tested [1].\n\n_Let's think about classification._\n"
    assert gr.parse_gap_answer(raw).rejected


def test_cached_italic_deliberation_is_rejected():
    good = gr.calm_gap_answer("### Limitations\n1. Only one material was tested [1].\n")
    assert not gr.stored_gap_ok(good + "\n_Let's think about classification._\n")


def test_uncertain_planning_is_not_a_no_gap_statement():
    assert gr.parse_gap_answer('I am not sure which limitations to report.').kind == 'rejected'


def test_pipe_does_not_display_and_cache_uncertain_planning(monkeypatch, pipe_databases):
    from test_gap_parse import _gap
    planning = 'I am not sure which limitations to report.'
    answer, recorder, streams = _gap(monkeypatch, [planning, planning])
    with pipe_databases[-1].connect() as conn:
        stored = conn.execute(text('SELECT answer FROM memento_answer_cache')).scalars().all()
    assert planning not in answer and not recorder.clean and not stored, (answer, recorder.clean, stored)


def test_similar_full_first_names_are_not_confirmed_identity():
    assert da.match_person('# A study\n\nAnnabelle Smith, Bob Jones\n', 'Anna Smith') != 'author'


def test_deleted_cited_document_does_not_remove_historical_ambiguity():
    mod = pipe_cli.load_pipe_module()
    stored = [{'metadata':[{'name':f'[{n}] {s}', 'source':f'{s}#0'}]} for n,s in [(1,'A'),(2,'B')]]
    cited, ambiguous = conversation.cited_documents('Findings [1] and [2].', stored, {'A'})
    assert conversation.this_document(cited, ambiguous, 'none', None) is None
