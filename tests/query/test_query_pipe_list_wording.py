# List and count answers say they come from the passages found: the answer sees about 20 passages, not every document.
import openwebui.functions.query_pipe as qp
from memento.query.flows import lists


def test_list_questions_get_the_wording_rule():
    add = lists.list_addendum("set_query", whole_doc=False)
    assert add == lists.LIST_ADDENDUM
    assert "not every document in the library" in add
    assert "never claim completeness" in add and '"all"' in add
    # a positive, honest close
    assert "I found N documents on this topic" in add
    assert "If one you expected is missing, ask about it by name and I'll check it." in add


def test_other_questions_do_not():
    for intent in ("ranked_retrieval", "multi_hop", "image_request", "summary", "gap_query"):
        assert lists.list_addendum(intent, whole_doc=False) == "", intent
    assert lists.list_addendum("set_query", whole_doc=True) == ""      # a whole-document read sees everything
