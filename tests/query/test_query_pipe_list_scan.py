# List questions see the best passage of every matching document (reranked best passage per document; threshold relative
# to the best document), calibrated on a small library; the few unrelated ones the model skips.
import openwebui.functions.query_pipe as qp
from memento.query.flows import lists


def _h(src, score, kind="text", bbox=None):
    return {"source": src, "rerank_score": score, "type": kind, "text": f"passage of {src}", "page": 1,
            "img_path": "/x.png" if kind != "text" else None, "bbox": bbox}


def test_documents_close_to_the_best_are_added_once():
    hits = [_h("A", 0.3), _h("A", 0.2), _h("B", 0.1)]
    heads = [_h("A", 0.28), _h("C", 0.05), _h("D", 0.02), _h("E", 0.01), _h("B", 0.09)]
    extra = lists.list_extra(hits, heads, ratio=0.05, cap=15)
    assert [h["source"] for h in extra] == ["C", "D"]          # A and B are already in; E below 5 % of 0.28


def test_the_number_added_is_capped():
    heads = [_h(f"D{i}", 0.3 - i * 0.01) for i in range(20)]
    assert len(lists.list_extra([], heads, ratio=0.05, cap=15)) == 15


def test_icons_and_unscored_heads_are_not_added():
    heads = [_h("A", 0.3), _h("Icon", 0.2, kind="image", bbox=[0, 0, 40, 40]), {**_h("F", 0.25), "rerank_fallback": True}]
    assert lists.list_extra([], heads, ratio=0.05, cap=15) == []                   # reranker failed: no judgement
    heads = [_h("A", 0.3), _h("Icon", 0.2, kind="image", bbox=[0, 0, 40, 40])]
    assert [h["source"] for h in lists.list_extra([], heads, ratio=0.05, cap=15)] == ["A"]


def test_only_library_wide_list_questions_scan():
    ok = dict(enabled=True, dc=False, paper=None, named_docs=[], image_intent=False)
    assert lists.list_scan_applies("set_query", **ok) is True
    assert lists.list_scan_applies("ranked_retrieval", **ok) is False
    assert lists.list_scan_applies("set_query", **{**ok, "paper": "PolyBERT"}) is False
    assert lists.list_scan_applies("set_query", **{**ok, "dc": True}) is False
    assert lists.list_scan_applies("set_query", **{**ok, "enabled": False}) is False
    assert lists.list_scan_applies("set_query", **{**ok, "image_intent": True}) is False


def test_valves():
    v = qp.Pipe().valves
    assert v.LIST_SCAN is True and v.LIST_SCAN_RATIO == 0.08 and v.LIST_SCAN_MAX == 50
