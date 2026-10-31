# evaluation/README.md can be followed as written: RQ1's commands pass each step's output to the next (and the
# scripts' defaults do the same), and the RQ2/RQ3 sections name every setting fixed in their scripts (corpora,
# ground-truth files, models, endpoints), each still present in the code.
import re

from memento.config.paths import PROJECT_ROOT

EV = PROJECT_ROOT / "evaluation"
README = (EV / "README.md").read_text()


def section(title):
    start = README.index(f"## {title}")
    end = README.find("\n## ", start + 1)
    return README[start:end if end > 0 else None]


def test_rq1_commands_pass_each_output_to_the_next_step():
    rq1 = section("RQ1")
    produced = set()
    for m in re.finditer(r"`([^`]*\.py[^`]*)`", rq1):
        cmd = m.group(1)
        if (src := re.search(r"--in (\S+)", cmd)) and produced:
            assert src.group(1) in produced, cmd
        if out := re.search(r"--out (\S+)", cmd):
            produced.add(out.group(1))
    assert {"evaluation/RQ1/ground_truth_auto.json", "evaluation/RQ1/ground_truth_auto_pooled_all.json",
            "evaluation/RQ1/rq1_results.json"} <= produced


def _default(script, flag):
    line = next(l for l in (EV / "RQ1" / script).read_text().splitlines() if f'add_argument("{flag}"' in l)
    return re.search(r'"([\w.]+\.json)"\)', line).group(1)


def test_rq1_script_defaults_chain():
    assert _default("auto_annotate.py", "--out") == _default("pool_relevance_all.py", "--in")
    assert _default("pool_relevance_all.py", "--out") == _default("rq1_eval.py", "--in")
    assert _default("rq1_eval.py", "--out") == _default("analyze_rq1_results.py", "--in") \
        == _default("plot_rq1_results.py", "--in")


def test_plot_reads_the_results_next_to_the_scripts():
    src = (EV / "RQ1/plot_rq1_results.py").read_text()
    assert 'Path(__file__).parent / "rq1_results.json"' in src


def test_rq3_section_names_what_is_fixed_in_its_scripts():
    rq3 = section("RQ3")
    assert "adapt" in rq3.lower()
    code = {p: (EV / "RQ3" / p).read_text() for p in ("eval/run_eval.py", "eval/select_questions.py",
                                                      "eval/build_nuggets.py", "eval/freeze.py")}
    for fixed, where in (("localhost:8090", "eval/run_eval.py"), ("qwen3.8-27b", "eval/run_eval.py"),
                         ("localhost:8095", "eval/run_eval.py"), ("gemma-4-12b", "eval/run_eval.py"),
                         ("COLL", "eval/run_eval.py"), ("GT", "eval/select_questions.py"),
                         ("data/rq2/runs/final", "eval/select_questions.py"),
                         ("ground_truth_auto_v4_qwenjudge.json", "eval/build_nuggets.py"),
                         ("--qwen-url", "eval/run_eval.py")):
        assert fixed in rq3, fixed
        assert fixed in code[where], (fixed, where)


def test_rq2_section_names_its_fixed_inputs():
    rq2 = section("RQ2")
    build = (EV / "RQ2/questions/build.py").read_text()
    run = (EV / "RQ2/harness/run.py").read_text()
    for fixed, src in (("GT_V4", build), ("RQ2_LLM_URLS", run), ("qwen3.8-27b", run), ("LITE_PORTS", run)):
        assert fixed in rq2, fixed
        assert fixed in src, fixed


# The application is published before the evaluation: its README carries the evaluation parts between
# <!-- evaluation --> and <!-- /evaluation --> (rows: a trailing <!-- evaluation -->), which the application's
# release drops with STRIP; the evaluation's release keeps them.
STRIP = ["sed", "-e", r"/^<!-- evaluation -->$/,/^<!-- \/evaluation -->$/d", "-e", r"/<!-- evaluation -->$/d"]


def _app_readme():
    import subprocess
    return subprocess.run(STRIP + [str(PROJECT_ROOT / "README.md")], capture_output=True, text=True, check=True).stdout


def test_the_application_readme_has_no_evaluation_links():
    app = _app_readme()
    assert "](evaluation/" not in app and "`evaluation/`" not in app
    assert "Reproducing the evaluations" not in app and "[Publication status](#publication-status)" in app
    full = (PROJECT_ROOT / "README.md").read_text()
    assert full.count("](evaluation/README.md)") >= 2 and "## Reproducing the evaluations" in full


def test_the_markers_keep_both_readmes_well_formed():
    full = (PROJECT_ROOT / "README.md").read_text().splitlines()
    assert sum(l == "<!-- evaluation -->" for l in full) == sum(l == "<!-- /evaluation -->" for l in full) >= 2
    for l in full:                                    # a marked table row keeps its cells; the marker is extra
        if l.startswith("|") and l.endswith("<!-- evaluation -->"):
            assert l[:-len("<!-- evaluation -->")].rstrip().endswith("|")
    app = _app_readme()
    assert "evaluation -->" not in app
    assert "\n\n\n\n" not in app
