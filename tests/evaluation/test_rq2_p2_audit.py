# tests/evaluation/test_rq2_p2_audit.py
import importlib.util
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("ga", PROJECT_ROOT / "evaluation/RQ2/graph_audit.py")
ga = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], "--domain", "ceramics"]
spec.loader.exec_module(ga)


def test_verdict_rules():
    assert ga.verdict(170, 200, 0.80) == "PASS"
    assert ga.verdict(120, 200, 0.80) == "FAIL"            # upper bound 0.66 < 0.80
    assert ga.verdict(150, 200, 0.80) == "INCONCLUSIVE"    # 0.75, CI reaches 0.80


def test_wilson_and_kappa():
    p, lo, hi = ga.wilson_ci(8, 10)
    assert p == 0.8 and lo < 0.8 < hi
    assert ga.cohen_kappa(list("yyynn"), list("yyynn")) == 1.0
    assert ga.cohen_kappa(list("yyyy"), list("yyyy")) == 1.0      # degenerate agreement stays 1.0


def test_parse_labels_drops_unknown_and_duplicates():
    raw = '[{"ref":"a","label":"yes"},{"ref":"a","label":"no"},{"ref":"b","label":"maybe"},{"ref":"c","label":"no"}]'
    assert ga.parse_labels(raw, ["a", "b", "c"], {"yes", "no"}) == {"c": "no"}


def test_negative_controls_are_clean():
    items = [{"ref": f"r{i}", "paper_id": f"p{i % 4}", "entity_name": f"entity{i}",
              "passage": f"text about entity{i} only"} for i in range(20)]
    ctrl = ga.negative_controls(items, n=10, seed=0, field="entity_name")
    assert ctrl and all(c["is_control"] for c in ctrl)
    for c in ctrl:
        assert c["entity_name"].lower() not in c["passage"].lower()


import json  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402


@pytest.mark.skipif(os.getenv("NEO4J_TEST_URI") is None, reason="set NEO4J_TEST_URI=bolt://localhost:7690")
def test_audit_main_end_to_end_on_test_graph(tmp_path, monkeypatch):
    from memento.graph.store import GraphEdge, GraphNode, Neo4jStore
    s = Neo4jStore(uri=os.environ["NEO4J_TEST_URI"])
    s._run("MATCH (n) DETACH DELETE n")
    nodes, edges = [], []
    for p in range(4):
        nodes.append(GraphNode(f"P{p}", "Paper", {"source": f"P{p}", "title": f"Paper {p}", "student": f"S{p % 2}, X"}))
        for c in range(3):
            cid = f"P{p}c{c}"
            nodes.append(GraphNode(cid, "Chunk", {"source": f"P{p}", "page": c, "type": "text",
                                                  "text": f"Alumina {p} was sintered with zirconia{c} at 1500 C."}))
            edges += [GraphEdge(cid, "Chunk", "alumina", "Entity", "MENTIONS", {"paper_id": f"P{p}"}),
                      GraphEdge(cid, "Chunk", f"zr{c}", "Entity", "MENTIONS", {"paper_id": f"P{p}"})]
    nodes += [GraphNode("alumina", "Entity", {"name": "Alumina", "type": "Material", "canonical": True})]
    nodes += [GraphNode(f"zr{c}", "Entity", {"name": f"zirconia{c}", "type": "Material"}) for c in range(3)]
    edges += [GraphEdge("alumina", "Entity", "zr0", "Entity", "RELATION",
                        {"rel_type": "compared_to", "paper_id": "P0", "chunk_id": "P0c0"}),
              GraphEdge("P0", "Paper", "P1", "Paper", "SAME_WORK", {"kind": "duplicate"})]
    s.upsert_nodes(nodes)
    s.upsert_edges(edges)
    mp = tmp_path / "merge.json"
    mp.write_text(json.dumps({"threshold": 0.9, "mapping": {"zr1": "zr0"}}))

    class FakeJudge:
        def __init__(self, *a, **k):
            self.calls = 0

        def judge(self, kind, items, domain=""):
            self.calls += 1
            return {i["ref"]: ({"label": "no"} if i.get("is_control") else {"label": "yes", "informative": "yes"})
                    for i in items}

    class Ctx:
        def __enter__(self):
            return s

        def __exit__(self, *e):
            return False

    monkeypatch.setattr(ga, "Judge", FakeJudge)
    monkeypatch.setattr("memento.graph.store.get_store", lambda: Ctx())
    monkeypatch.setattr(sys, "argv", ["x", "--domain", "ceramics", "--merge-map", str(mp),
                                      "--out-dir", str(tmp_path), "--cross-family", "4",
                                      "--cross-family-url", "http://x/v1"])
    ga.main()
    rep = json.loads((tmp_path / "graph_audit_ceramics.json").read_text())
    assert rep["metrics"]["entity"]["verdict"] == "PASS" and rep["metrics"]["entity"]["n"] > 0
    assert rep["metrics"]["relation"]["n"] == 1
    assert rep["metrics"]["merge"]["by_threshold"]["0.9"]["n"] == 1
    assert rep["metrics"]["same_work"]["n"] == 1
    assert rep["census"]["n_chunks"] == 12 and 0 < rep["census"]["coverage"] <= 1
    assert rep["census"]["n_entities"] == 4 and rep["census"]["n_mentions"] == 24
    assert rep["cross_family"]["by_kind"] and (tmp_path / "graph_audit_ceramics.md").exists()
    s._run("MATCH (n) DETACH DELETE n")
    s.close()


def test_parallel_judge_equals_sequential_and_overlaps(monkeypatch):
    import threading
    import time as _t
    items = [{"ref": f"e{i}", "entity_name": f"n{i}", "entity_type": "Material", "passage": "p"} for i in range(35)]
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake_chat(self, system, user):
        import re as _re
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        _t.sleep(0.05)
        with lock:
            state["now"] -= 1
        refs = _re.findall(r"--- item (\S+) ---", user)
        return json.dumps([{"ref": r, "label": "yes" if int(r[1:]) % 2 else "no"} for r in refs])

    monkeypatch.setattr(ga.Judge, "_chat", fake_chat)
    monkeypatch.setattr(ga.Judge, "__init__", lambda self, *a, **k: None)
    seq, par = ga.Judge(), ga.Judge()
    seq.workers, par.workers = 1, 8
    a = seq.judge("entity", items)
    assert state["peak"] == 1
    b = par.judge("entity", items)
    assert a == b and len(b) == 35
    assert state["peak"] >= 3          # 4 batches of <= 10 ran concurrently


def test_cloud_key_prefers_cloud_api_key(monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "cloud")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    assert ga.cloud_api_key() == "cloud"
    monkeypatch.delenv("CLOUD_API_KEY")
    assert ga.cloud_api_key() == "openai"


def test_entity_judging_shows_aliases_and_counts_conditions():
    it = {"ref": "e1", "entity_name": "Scanning electron microscopy", "entity_type": "Characterization",
          "aliases": ["SEM", "Rasterelektronenmikroskopie"], "passage": "SEM images show grains."}
    shown = ga.render("entity", it)
    assert "SEM" in shown and "also written as" in shown
    p = ga.PROMPTS["entity"].lower()
    assert "abbreviation" in p and "temperature" in p and "person" in p


def test_open_relation_sample_is_random_not_database_order():
    rows = [{"_uid": f"u{i}", "rel_type": f"free text {i}", "chunk_id": f"c{i}", "head": "a", "tail": "b",
             "paper_id": "p", "passage": "x"} for i in range(2000)]
    picked = ga.sample_relations(rows, 200, seed=0)
    idx = sorted(int(r["_uid"][1:]) for r in picked)
    assert len(picked) == 200 and idx[-1] > 1000          # not just the first 200 rows


def test_entity_controls_carry_the_donor_aliases_and_type():
    # controls do not keep the original entity's aliases (the judge would see the original abbreviation)
    items = [{"ref": "r0", "paper_id": "p0", "entity_name": "alumina", "aliases": ["Al2O3"],
              "entity_type": "Material", "passage": "Al2O3 was sintered."},
             {"ref": "r1", "paper_id": "p1", "entity_name": "X-ray diffraction", "aliases": ["XRD"],
              "entity_type": "Characterization", "passage": "XRD patterns were recorded."},
             {"ref": "r2", "paper_id": "p2", "entity_name": "zirconia", "aliases": ["ZrO2"],
              "entity_type": "Material", "passage": "Al2O3 powders were checked by XRD."}]
    ctrl = ga.negative_controls(items, n=3, seed=0, field="entity_name")
    by_ref = {c["ref"]: c for c in ctrl}
    for c in ctrl:
        donor = next(i for i in items if i["entity_name"] == c["entity_name"])
        assert c["aliases"] == donor["aliases"] and c["entity_type"] == donor["entity_type"]
        for name in [c["entity_name"], *c["aliases"]]:           # no donor surface form in the passage
            assert name.lower() not in c["passage"].lower()
    assert "ctrl_r2" not in by_ref                                # every donor's alias occurs in r2's passage
