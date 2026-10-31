# tests/evaluation/test_rq2_p3_build.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.questions.control import from_rq1_v4  # noqa: E402


def test_control_conversion(tmp_path):
    gt = {"queries": [{"query_id": 7, "query": "What is HV?", "category": "factual", "target_paper": "P1",
                       "target_type": "equation", "relevant_chunks": [{"paper_id": "P1", "page": 3, "type": "equation"},
                                                                      {"paper_id": "P2", "page": 1, "type": "text"}]}]}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    qs = from_rq1_v4(p, "ceramics", "full")
    assert len(qs) == 1 and qs[0].qtype == "control" and qs[0].qid == "ceramics:control:7"
    assert [(g["source"], g["page"]) for g in qs[0].gold_pages] == [("P1", 3), ("P2", 1)]


def test_build_cli_lists_subcommands():
    import subprocess
    r = subprocess.run([sys.executable, "evaluation/RQ2/questions/build.py", "--domain", "ceramics", "--help"],
                       capture_output=True, text=True)
    for sub in ("multihop", "corpus-level", "control", "hardness", "crosscheck", "report"):
        assert sub in r.stdout
