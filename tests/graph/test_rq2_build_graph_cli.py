# tests/graph/test_rq2_build_graph_cli.py
import sys
sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.ingestion.run_pipeline import find_paper_dir_in  # noqa: E402


def _cl(d):
    d.mkdir(parents=True)
    (d / "x_content_list.json").write_text("[]")


def test_find_paper_dir_in_prefers_backend_then_office(tmp_path):
    _cl(tmp_path / "P1" / "vlm")
    _cl(tmp_path / "P2" / "office")
    _cl(tmp_path / "P3" / "hybrid_auto")
    assert find_paper_dir_in(tmp_path, "P1", "vlm") == tmp_path / "P1" / "vlm"
    assert find_paper_dir_in(tmp_path, "P2", "vlm") == tmp_path / "P2" / "office"
    assert find_paper_dir_in(tmp_path, "P3", "vlm") == tmp_path / "P3" / "hybrid_auto"
    assert find_paper_dir_in(tmp_path, "missing", "vlm") is None


def test_build_graph_requires_domain():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "memento.graph.build.build_graph", "--all"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "--domain is required" in (r.stdout + r.stderr)


def test_papers_file_ids_keep_surrounding_spaces(tmp_path):
    from memento.graph.build.build_graph import read_papers_file
    f = tmp_path / "ids.txt"
    f.write_text("02_Prüfung Promotionsvortrag Albrecht \n21  1000001  Hallmann, Jan\n\n")
    assert read_papers_file(f) == ["02_Prüfung Promotionsvortrag Albrecht ", "21  1000001  Hallmann, Jan"]
