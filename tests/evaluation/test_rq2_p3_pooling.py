# tests/evaluation/test_rq2_p3_pooling.py
import json
import sys
from memento.config.paths import PROJECT_ROOT

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.questions.pooling import judge_facts, pool_and_judge  # noqa: E402


class FakeJudge:
    """Candidate refs are 'c0', 'c1', ... in prompt order; returns scripted fact lists, can omit refs."""
    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        import re
        user = kw["messages"][-1]["content"]
        text = user if isinstance(user, str) else " ".join(p.get("text", "") for p in user if isinstance(p, dict))
        refs = re.findall(r"--- candidate (c\d+) ---", text)
        rows = [{"ref": r, "facts": self.script[r]} for r in refs if r in self.script]
        msg = type("M", (), {"content": json.dumps(rows)})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


C = [{"source": "P1", "page": 0, "type": "text", "text": "x"}, {"source": "P2", "page": 3, "type": "text", "text": "y"},
     {"source": "P3", "page": 1, "type": "text", "text": "z"}]


def test_pool_unions_lists_and_records_who_found_each_page():
    r = pool_and_judge(FakeJudge({"c0": ["a"], "c1": [], "c2": ["b"]}), "m", {"a": "fa", "b": "fb"},
                       {"g0": C[:2], "bm25": [C[1], C[2]]})
    gold = {(g["source"], g["page"]): g for g in r["gold"]}
    assert set(gold) == {("P1", 0), ("P3", 1)} and gold[("P3", 1)]["found_by"] == ["bm25"]
    assert r["n_candidates"] == 3


def test_unparsed_judgements_are_reported_not_labelled():
    r = judge_facts(FakeJudge({"c0": ["a"]}), "m", {"a": "fa"}, C)        # c1, c2 missing from the reply
    assert set(r["unparsed"]) == {("P2", 3), ("P3", 1)} and r["labels"] == {("P1", 0): {"a"}}


def test_known_judgements_are_not_rejudged():
    first = pool_and_judge(FakeJudge({"c0": ["a"], "c1": []}), "m", {"a": "fa"}, {"g0": C[:2]})
    j = FakeJudge({"c0": ["a"]})
    second = pool_and_judge(j, "m", {"a": "fa"}, {"g0": C[:2], "arm": [C[2]]}, known=first)
    assert j.calls == 1 and {(g["source"], g["page"]) for g in second["gold"]} == {("P1", 0), ("P3", 1)}


def test_g0_settings_match_the_production_query_pipe():
    import re
    from pathlib import Path
    import evaluation.RQ2.questions.pooling as pl
    src = (PROJECT_ROOT / "openwebui/functions/query_pipe.py").read_text()
    for name in ("RETRIEVE_K", "RERANK_TOP", "RERANK_MIN_SCORE", "IMAGE_RERANK_RATIO"):
        m = re.search(rf"^\s+{name}:\s*\w+\s*=\s*([0-9.]+)", src, re.M)
        assert m and float(m.group(1)) == float(getattr(pl, name)), name
