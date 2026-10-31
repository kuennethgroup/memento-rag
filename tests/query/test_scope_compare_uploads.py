"""Edge cases of scoped searches and stored citations."""
from test_scope_kept import _run, allows, PAYLOADS
from test_locks_scopes_stored_answers import pipe_databases  # noqa: F401  (autouse)


def test_compare_two_own_uploads_retains_both(monkeypatch):
    out, rec, model_input = _run(monkeypatch,
        'Compare my latest uploads Paper A and Paper B on hardness',
        {'intent': 'ranked_retrieval', 'query': 'compare Paper A and Paper B hardness', 'scope': 'documents'},
        uploads=['A', 'B'], resolve='A', named=['A', 'B'])
    assert rec['searches']
    # two named documents are searched one by one, each with its own share, as in any comparison: both are covered
    # across the searches and reach the answer, and nothing else
    selected = {p['source'] for flt in rec['searches'] for p in PAYLOADS if p['type'] == 'text' and allows(flt, p)}
    assert selected == {'A', 'B'}, (selected, model_input)
    assert 'Paper A passage' in model_input and 'Paper B' in model_input and 'Paper C' not in model_input
