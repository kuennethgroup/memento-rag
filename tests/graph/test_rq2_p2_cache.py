# tests/graph/test_rq2_p2_cache.py
import json
import re
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.extraction_cache import RawResponseCache  # noqa: E402
from memento.graph.build.llm_enrich import build_enrich_prompt, llm_enrich_items  # noqa: E402


class FakeLLM:
    def __init__(self, bad=False):
        self.calls, self.bad = 0, bad
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        user = kw["messages"][-1]["content"]
        ids = re.findall(r"^\[(\d+)\]", user, re.M)
        body = "not json" if self.bad else json.dumps(
            {i: {"entities": [{"name": "alumina", "type": "Material"}], "relations": []} for i in ids})
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": body})})]})


ITEMS = [{"type": "text", "text": "Alumina was sintered by SPS at 1500 C to full density.", "page_idx": 0},
         {"type": "text", "text": "Zirconia toughened the matrix considerably.", "page_idx": 0}]


def _run(fake, cache):
    items = json.loads(json.dumps(ITEMS))
    llm_enrich_items(items, want_context=False, want_entities=True, client=fake, model="m",
                     workers=1, cache=cache)
    return items


def test_second_run_uses_cache_and_gives_identical_extraction(tmp_path):
    path = tmp_path / "c.json"
    first = FakeLLM()
    c1 = RawResponseCache(path)
    a = _run(first, c1)
    c1.save()
    second = FakeLLM()
    c2 = RawResponseCache(path)
    b = _run(second, c2)
    assert first.calls == 1 and second.calls == 0 and c2.hits == 1
    assert [i["llm_extraction"] for i in a] == [i["llm_extraction"] for i in b]


def test_malformed_response_is_not_cached(tmp_path):
    path = tmp_path / "c.json"
    c = RawResponseCache(path)
    _run(FakeLLM(bad=True), c)
    c.save()
    assert RawResponseCache(path).entries == {}


def test_entities_only_prompt_has_no_relations_field():
    p = build_enrich_prompt("overview", [(0, "text")], want_context=False,
                            want_entities=True, want_relations=False)
    assert '"entities"' in p and '"relations"' not in p


def test_explicit_empty_answer_means_no_entities_and_is_cached(tmp_path):
    class EmptyLLM(FakeLLM):
        def create(self, **kw):
            self.calls += 1
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "{}"})})]})
    path = tmp_path / "c.json"
    fake = EmptyLLM()
    c = RawResponseCache(path)
    items = json.loads(json.dumps(ITEMS))
    llm_enrich_items(items, want_context=False, want_entities=True, client=fake, model="m", workers=1, cache=c)
    c.save()
    assert fake.calls == 1                                              # no retry
    assert all(i["llm_extraction"] == {"entities": [], "relations": []} for i in items)
    assert RawResponseCache(path).entries                               # cached
