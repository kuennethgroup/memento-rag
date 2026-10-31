# tests/evaluation/test_rq2_p4_run.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.harness import run  # noqa: E402


def test_strict_modes_are_on():
    import os
    assert os.environ["GRAPH_STRICT"] == "1" and os.environ["MEMENTO_STRICT_RERANK"] == "1"


def test_policy_table_covers_every_policy_and_reuses_g0():
    routes = [{"qid": "m1", "qtype": "multihop", "route_true": "multihop", "route_real": "single"},
              {"qid": "c1", "qtype": "control", "route_true": "single", "route_real": "single"}]
    table = run.policy_table(routes, arms=("g0", "g2", "d2", "lx"))
    rows = {(r["qid"], r["policy"], r["arm"]): r["key"] for r in table}
    assert rows[("m1", "true", "g2")] == "m1|g2|multihop" and rows[("m1", "real", "g2")] == "m1|g0|rag"
    assert rows[("c1", "allgraph", "g2")] == "c1|g2|multihop" and rows[("c1", "true", "g2")] == "c1|g0|rag"
    assert rows[("m1", "true", "lx")] == "m1|g0|rag"
    assert {r["key"] for r in table} == {"m1|g0|rag", "m1|g2|multihop", "m1|d2|multihop", "c1|g0|rag",
                                         "c1|g2|multihop", "c1|d2|multihop"}


def test_outputs_start_with_a_manifest(tmp_path):
    p = tmp_path / "x.jsonl"
    run.write_rows(p, [{"a": 1}], manifest={"stage": "arms"})
    first = json.loads(p.read_text().splitlines()[0])
    assert first["_manifest"]["stage"] == "arms" and json.loads(p.read_text().splitlines()[1]) == {"a": 1}
    assert run.read_rows(p) == [{"a": 1}]


def test_cloud_key_is_found_without_the_environment(monkeypatch, tmp_path):
    # the judge stage loads config/.env (else 401)
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # A checkout must not need the developer's real credentials. Exercise actual
    # dotenv file parsing against a disposable repository-shaped fixture.
    fake_module = tmp_path / "evaluation" / "RQ2" / "harness" / "run.py"
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("CLOUD_API_KEY=fixture-key-only\n")
    monkeypatch.setattr(run, "__file__", str(fake_module))
    assert run._cloud_key() == "fixture-key-only"


def test_stats_skip_arms_routed_to_g0_for_a_family():
    rows = [{"qid": "m1", "qtype": "multihop", "arm": "lx", "key": "m1|g0|rag"},
            {"qid": "m1", "qtype": "multihop", "arm": "g2", "key": "m1|g2|multihop"},
            {"qid": "c1", "qtype": "corpus_list", "arm": "lx", "key": "c1|lx|corpus"}]
    assert run.arms_for_family(rows, "multihop") == ["g2"]
    assert run.arms_for_family(rows, "corpus") == ["lx"]


def test_corpus_answers_are_judged_with_their_citation_map():
    from evaluation.RQ2.harness.arms import ArmOut
    o = ArmOut("g2", "corpus", "c1", [{"source": "D1"}, {"source": "D2"}])
    assert run.judge_text("corpus_list", "See [2].", o).endswith("Documents cited: [2] D2")
    assert run.judge_text("multihop", "See [2].", o) == "See [2]."


def test_bh_family_excludes_secondary_measures():
    comps = [{"family": "corpus", "kind": "h6", "arm": "g2", "base": "lx", "metric": m, "p": 0.01}
             for m in ("f1", "abs", "cited_f1")]
    fam = run.primary_family(comps)
    assert sorted(k.split(":")[-1] for k in fam) == ["abs", "f1"]


def test_checkpoint_resumes_done_rows(tmp_path):
    p = tmp_path / "answers.jsonl"
    ck = run.Checkpoint(p, key="key")
    ck.add({"key": "a", "v": 1})
    ck2 = run.Checkpoint(p, key="key")               # relaunch
    assert ck2.done == {"a"} and ck2.rows == [{"key": "a", "v": 1}]
    ck2.add({"key": "b", "v": 2})
    ck2.finish({"stage": "answers"})
    assert run.read_rows(p) == [{"key": "a", "v": 1}, {"key": "b", "v": 2}] and not ck2.partial.exists()


def test_force_discards_a_stale_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "out_dir", lambda a: tmp_path)
    (tmp_path / "answers.jsonl.partial").write_text('{"key": "old"}\n')
    called = {}
    monkeypatch.setitem(run.STAGES, "answers", lambda a: called.setdefault("ran", True))
    monkeypatch.setattr(sys, "argv", ["run.py", "--domain", "ceramics", "--universe", "pilot", "--run", "x",
                                      "--force", "answers"])
    run.main()
    assert called["ran"] and not (tmp_path / "answers.jsonl.partial").exists()
