# tests/graph/test_rq2_p2_extraction.py
import json
import re
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.extract_entities import extract_entities  # noqa: E402
from memento.graph.build.llm_enrich import llm_enrich_items  # noqa: E402


class FakeLLM:
    """Answers every numbered excerpt with one 'alumina' entity; records prompts."""
    def __init__(self, bad=False):
        self.calls, self.bad, self.prompts = 0, bad, []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        user = kw["messages"][-1]["content"]
        user = user if isinstance(user, str) else user[0]["text"]
        self.prompts.append(user)
        ids = re.findall(r"^\[(\d+)\]", user, re.M)
        body = "not json" if self.bad else json.dumps(
            {i: {"entities": [{"name": "alumina", "type": "Material"}], "relations": []} for i in ids})
        msg = type("M", (), {"content": body})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_precomputed_dict_is_consumed():
    r = extract_entities("The epoxy resin was cured with DDS at 180 C and tested by DMA.",
                         use_llm=True,
                         precomputed={"entities": [{"name": "epoxy resin", "type": "Material"}],
                                      "relations": []})
    assert "epoxy" in [e.id for e in r.entities]


def test_snippets_override_replaces_item_text():
    items = [{"type": "image", "img_path": "a.jpg", "content": "mermaid junk zirconia",
              "image_caption": ["Fig. 1 SEM of alumina"], "page_idx": 0},
             {"type": "text", "text": "Alumina was sintered at 1500 C.", "page_idx": 0}]
    fake = FakeLLM()
    llm_enrich_items(items, want_context=False, want_entities=True, client=fake, model="m",
                     workers=1, snippets={0: "Fig. 1 SEM of alumina"})
    assert "mermaid" not in fake.prompts[0] and "Fig. 1 SEM of alumina" in fake.prompts[0]
    assert "llm_extraction" in items[0] and "llm_extraction" not in items[1]   # only overridden idxs
