# When a named document limits the search: the router's scope decides; without one, list questions are never limited to
# one document. Questions about a named document keep their filter.
import pytest

from memento.query.scope.resolve import search_whole_library as _search_whole_library


@pytest.mark.parametrize("scope, intent, want", [
    ("library", "ranked_retrieval", True),      # "has X been used in other research?"
    ("library", "set_query", True),             # "which other papers in the library use X?"
    ("none", "set_query", True),                # no scope from the router: a list question is never one document
    ("document", "set_query", False),           # "list the tables in the X paper"
    ("document", "ranked_retrieval", False),    # "what does the X paper report about Y?"
    ("documents", "ranked_retrieval", False),   # "compare A with B"
    ("none", "ranked_retrieval", False),        # an older reply without the field: as before
    ("none", "summary", False),                 # "summarize the X paper"
])
def test_when_the_search_ignores_a_named_document(scope, intent, want):
    assert _search_whole_library(scope, intent) is want
