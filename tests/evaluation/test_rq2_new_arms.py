# tests/evaluation/test_rq2_new_arms.py — G2-lite arms (G2 code on the no-LLM graphs) and the fact-sheets arm
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness import arms, router  # noqa: E402
from evaluation.RQ2.harness.view import QuestionView  # noqa: E402
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from test_rq2_p4_arms import Store, _ctx, _h, _patch  # noqa: E402

TERMS = {"terms": [{"term": "HfO2", "variants": ["HfO2"]}, {"term": "EXAFS", "variants": ["EXAFS"]}],
         "combine": "and", "relation": None}


class LiteStore(Store):
    def papers_for_entities(self, eids, need=1):
        return [{"src": "P3", "hits": 2}]


def test_lite_arms_run_g2_on_their_own_graph(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    ctx, v = _ctx(stores={"g2l": LiteStore()}), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.corpus_lookup(arms.lite_ctx(ctx, "g2l"), v, g0, "g2",
                             {"terms": [{"term": "HfO2", "variants": ["HfO2"]}], "combine": "or", "relation": None},
                             name="g2l")
    assert out.arm == "g2l" and out.docs == ["P3"]
    assert arms.corpus_lookup(ctx, v, g0, "g2", {"terms": [{"term": "HfO2", "variants": ["HfO2"]}],
                                                 "combine": "or", "relation": None}).docs == ["P1", "P2"]


def test_lite_graph_arm_is_labelled_with_its_own_name(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)], graph_ids=[40, 41])
    ctx, v = _ctx(stores={"g2lg": LiteStore()}), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.arm_graph(arms.lite_ctx(ctx, "g2lg"), v, g0, "g2", {"terms": [], "combine": "or", "relation": None},
                         name="g2lg")
    assert out.arm == "g2lg" and {h["origin"] for h in out.context} == {"g0", "g2lg"}


def test_fact_sheet_arm_soft_and_scope_and_ranking(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    hits = {"HfO2": {"P1": 2, "P2": 5, "OUT": 9}, "EXAFS": {"P1": 1, "P3": 1}}
    monkeypatch.setattr(arms, "_sheet_matches", lambda ctx, variants: hits[variants[0]])
    ctx, v = _ctx(sheets="coll", qdrant=type("Q", (), {"collection_exists": lambda self, n: True})()), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.corpus_fs(ctx, v, g0, TERMS)
    assert out.arm == "fs" and out.docs == ["P1"] and out.block.startswith("Documents found by fact sheets (1)")
    none = {"HfO2": {"P2": 5}, "EXAFS": {"P3": 1}}
    monkeypatch.setattr(arms, "_sheet_matches", lambda ctx, variants: none[variants[0]])
    relaxed = arms.corpus_fs(ctx, v, g0, TERMS)
    assert relaxed.docs == ["P2", "P3"] and relaxed.info["and_relaxed_to"] == 1   # soft AND, then by mentions


def test_fact_sheet_arm_abstains_without_terms_or_matches(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])
    monkeypatch.setattr(arms, "_sheet_matches", lambda ctx, variants: {})
    ctx, v = _ctx(sheets="coll", qdrant=type("Q", (), {"collection_exists": lambda self, n: True})()), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    assert arms.corpus_fs(ctx, v, g0, {"terms": [], "combine": "or", "relation": None}).info["abstained"]
    out = arms.corpus_fs(ctx, v, g0, TERMS)
    assert out.info["abstained"] and out.context == g0.context


def test_routing_of_the_new_arms():
    assert router.effective_arm("g2l", "multihop") == ("g2l", "multihop")
    assert router.effective_arm("g2lg", "corpus") == ("g2lg", "corpus")
    assert router.effective_arm("fs", "corpus") == ("fs", "corpus")
    assert router.effective_arm("fs", "multihop") == ("g0", "rag")
    assert router.effective_arm("fs", "single") == ("g0", "rag")


def test_match_counts_counts_matching_names(request, throwaway_qdrant):
    import uuid
    import pytest
    from memento.graph import canonicalize
    from memento.graph.build import fact_sheets as fs
    client = throwaway_qdrant
    name = f"test-fact-sheets-{uuid.uuid4().hex[:8]}"
    request.addfinalizer(lambda: client.delete_collection(name))
    sheets = [{"source": "A", "characterization": ["EXAFS", "EXAFS spectroscopy", "XRD"]},
              {"source": "B", "characterization": ["in-situ EXAFS"], "materials": ["HfO2"]}]
    fs.index_sheets(client, name, [fs.payload(s, canonicalize) for s in sheets])
    assert fs.match_counts(client, name, ["EXAFS"], canonicalize) == {"A": 2, "B": 1}


def test_policy_table_runs_new_arms_only_where_they_apply():
    sys.argv = [sys.argv[0], "--domain", "ceramics"]
    from evaluation.RQ2.harness import run
    rows = run.policy_table([{"qid": "c1", "qtype": "corpus_list", "route_real": "corpus"},
                             {"qid": "m1", "qtype": "multihop", "route_real": "multihop"}])
    keys = {(r["qid"], r["policy"], r["arm"]): r["key"] for r in rows}
    assert keys[("c1", "true", "fs")] == "c1|fs|corpus" and keys[("m1", "true", "fs")] == "m1|g0|rag"
    assert keys[("m1", "true", "g2lg")] == "m1|g2lg|multihop" and keys[("c1", "true", "g2l")] == "c1|g2l|corpus"


def test_round_robin_llm_alternates_servers(monkeypatch):
    from evaluation.RQ2.harness import run

    class C:
        def __init__(self, name):
            self.name, self.chat, self.completions = name, self, self

        def create(self, **kw):
            return self.name
    rr = run.RoundRobinLLM([C("a"), C("b"), C("c")])
    assert [rr.chat.completions.create(model="m") for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]
    monkeypatch.setattr(run, "LOCAL_URLS", ["http://x:1/v1"])
    assert not isinstance(run.local_llm(5), run.RoundRobinLLM)
    monkeypatch.setattr(run, "LOCAL_URLS", ["http://x:1/v1", "http://y:2/v1"])
    assert isinstance(run.local_llm(5), run.RoundRobinLLM)


def test_reranker_calls_alternate_over_listed_servers():
    from evaluation.RQ2.harness import scope
    ctx = _ctx(reranker_url="http://a:1, http://b:2")
    got = {scope.reranker_for(ctx) for _ in range(4)}
    assert got == {"http://a:1", "http://b:2"}
    assert scope.reranker_for(_ctx(reranker_url="http://only:3")) == "http://only:3"


def test_for_each_question_parallel_processes_every_item_once(monkeypatch, tmp_path):
    from evaluation.RQ2.harness import run
    monkeypatch.setattr(run, "QUESTION_WORKERS", 4)
    ck = run.Checkpoint(tmp_path / "x.jsonl", key="qid")
    run.for_each_question(lambda i: ck.add({"qid": i}), list(range(50)))
    assert sorted(r["qid"] for r in ck.rows) == list(range(50)) and len(ck.done) == 50
    assert len((tmp_path / "x.jsonl.partial").read_text().splitlines()) == 50


def test_fact_sheet_arm_skips_when_the_corpus_has_no_sheets(monkeypatch):
    _patch(monkeypatch, [_h(i) for i in range(20)])

    class Q:
        def collection_exists(self, name):
            return False
    ctx, v = _ctx(sheets="missing", qdrant=Q()), QuestionView("q", "Q?", "en")
    g0 = arms.arm_g0(ctx, v)
    out = arms.corpus_fs(ctx, v, g0, TERMS)
    assert out.info["abstained"] and "no fact sheets" in out.info["why"]


def test_round_robin_llm_fails_over_on_connection_errors():
    import httpx
    import openai
    from evaluation.RQ2.harness import run

    class Dead:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, **kw):
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    class Ok:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, **kw):
            return "ok"
    rr = run.RoundRobinLLM([Dead(), Ok()])
    assert [rr.create(model="m") for _ in range(4)] == ["ok"] * 4


def test_sheets_points_is_none_without_a_collection():
    from evaluation.RQ2.harness import run

    class Q:
        def collection_exists(self, name):
            return False
    assert run.sheets_points(Q(), "missing") is None
