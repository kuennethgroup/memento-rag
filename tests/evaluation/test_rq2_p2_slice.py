# tests/evaluation/test_rq2_p2_slice.py
import importlib.util
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("sl", PROJECT_ROOT / "evaluation/RQ2/build/rq2_pilot_slice.py")
sl = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], "--domain", "ceramics"]
spec.loader.exec_module(sl)


def test_ceramics_slice_prefers_citation_clusters():
    pairs = [(f"a{i}", f"a{i+1}") for i in range(30)] + [("b0", "b1")]
    sizes = {f"a{i}": 100 + i for i in range(31)} | {"b0": 50, "b1": 60} | {f"u{i}": 10 * i for i in range(40)}
    r = sl.pick_ceramics_slice(pairs, sizes, n=20, seed=1)
    assert len(r["ids"]) == 20 and r["n_linked"] >= 16 and r["n_edges_inside"] >= 15
    assert len(set(r["ids"])) == 20


def test_polymer_slice_takes_whole_folders_and_some_non_research():
    reg = {}
    for s in range(12):
        st = f"S{s}, X"
        reg[f"Diss_{s}"] = {"type": "dissertation", "student": st}
        reg[f"Vortrag_{s}"] = {"type": "defense", "student": st, "parent_diss": f"Diss_{s}"}
        reg[f"0{s}_Kap"] = {"type": "chapter", "student": st, "parent": f"Diss_{s}"}
    reg |= {f"img{i}": {"type": "image_artifact"} for i in range(10)}
    reg |= {f"21  {i}  T": {"type": "student_thesis"} for i in range(5)}
    r = sl.pick_polymer_slice(reg, n=20, seed=3)
    ids = set(r["ids"])
    for st in r["students"]:
        s = st.split(",")[0][1:]
        assert {f"Diss_{s}", f"Vortrag_{s}", f"0{s}_Kap"} <= ids
    assert r["types"].get("image_artifact", 0) >= 1 and r["types"].get("student_thesis", 0) >= 1
    assert r["n_same_work_pairs"] >= 2 * len(r["students"])


def test_polymer_slice_caps_chapters_per_student():
    reg = {}
    for s in range(10):
        st = f"S{s}, X"
        reg[f"Diss_{s}"] = {"type": "dissertation", "student": st}
        reg[f"Vortrag_{s}"] = {"type": "defense", "student": st, "parent_diss": f"Diss_{s}"}
        for c in range(12):
            reg[f"K{c}_{s}"] = {"type": "chapter", "student": st, "parent": f"Diss_{s}"}
    r = sl.pick_polymer_slice(reg, n=40, seed=0)
    per = {}
    for i in r["ids"]:
        if i.startswith("K"):
            per[i.split("_")[1]] = per.get(i.split("_")[1], 0) + 1
    assert max(per.values()) <= 4 and len(r["students"]) >= 5
