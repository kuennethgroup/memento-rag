# tests/evaluation/test_rq2_build_scripts.py: the RQ2 input builders in evaluation/RQ2/build/ (fact sheets, document
# registry)
import json
import subprocess
import sys

import pytest

from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def test_failover_round_robin_skips_a_dead_server():
    import importlib.util
    import openai
    spec = importlib.util.spec_from_file_location("fsb", "evaluation/RQ2/build/rq2_fact_sheets_build.py")
    fsb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fsb)

    class Ok:
        def __init__(self, name):
            self.name, self.chat, self.completions = name, self, self

        def create(self, **kw):
            return self.name

    class Dead:
        def __init__(self):
            self.chat, self.completions = self, self

        def create(self, **kw):
            raise openai.APIConnectionError(request=__import__("httpx").Request("POST", "http://x"))

    class Bad:
        def __init__(self):
            self.chat, self.completions = self, self

        def create(self, **kw):
            raise ValueError("400 context too long")
    rr = fsb.FailoverRoundRobin([Ok("a"), Dead(), Ok("c")])
    assert [rr.create(model="m") for _ in range(6)] == ["a", "c", "c", "a", "c", "c"]
    with pytest.raises(ValueError):
        fsb.FailoverRoundRobin([Bad(), Ok("a")]).create(model="m")


def test_duplicates_need_half_text_containment_in_either_direction():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("bdr", PROJECT_ROOT / "evaluation/RQ2/build/build_document_registry.py")
    bdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bdr)
    same = ["one two three four five six seven eight nine ten eleven twelve"]
    draft = same + ["totally different words that never occur in the final version at all here"]
    other = ["a defence talk with slides about something else entirely and no shared text"]
    texts = {"final": same, "copy": same, "draft": draft, "talk": other}
    src = {"final": {"type": "dissertation", "document_id": "final"},
           "copy": {"type": "dissertation", "duplicate_of": "final", "document_id": "final"},
           "draft": {"type": "dissertation", "duplicate_of": "final", "document_id": "final"},
           "talk": {"type": "dissertation", "duplicate_of": "final", "document_id": "final"}}
    bdr.confirm_duplicates(src, lambda s: texts[s])
    assert src["copy"]["duplicate_of"] == "final"
    assert src["draft"]["duplicate_of"] == "final"          # final fully contained in the draft
    assert "duplicate_of" not in src["talk"] and src["talk"]["document_id"] == "talk"
    assert src["talk"]["duplicate_rejected_overlap"] == 0.0


def test_registry_finds_persons_from_the_folder_tree(tmp_path):
    # no per-domain folder key; walk the corpus recursively, keep the substructure, person = first folder in the path
    # that looks like a name (memento.graph.documents.is_person_folder)
    root = tmp_path / "PolymerEngineering"
    for rel, body in (("8 Doktoranden/Garcia, Pablo/a.pdf", b"one"), ("8 Doktoranden/Garcia, Pablo/Diss/b.pdf", b"b"),
                      ("8 Doktoranden/Lang, M/a.pdf", b"two"), ("1_Ordnung/c.pdf", b"c")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(body)
    code = ("import sys, json; sys.argv=['x','--domain','polymer-engineering']; "
            "from memento.config.cli_domain import apply_domain_arg; apply_domain_arg(); sys.path.insert(0,'evaluation/RQ2/build'); "
            "import build_document_registry as r; from pathlib import Path; "
            f"print(json.dumps(r.build_stem_to_student(Path({str(root)!r}))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {"8_Doktoranden__Garcia,_Pablo__a": "Garcia, Pablo", "b": "Garcia, Pablo",
                   "8_Doktoranden__Lang,_M__a": "Lang, M"}          # collision names = the ingested stems
