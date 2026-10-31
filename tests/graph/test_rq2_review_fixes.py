# RQ2 graph store, build and retrieval regressions.
import os
import subprocess
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


def _py(code, **env):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, **env))


def test_c1_build_graph_importable_as_library_without_domain_flag():
    # OpenWebUI's ingestion_pipe imports build_paper/resolve_cross_paper_citations;
    # a SystemExit there escapes its `except Exception`.
    r = _py("import memento.graph.build.build_graph as b; print(b.build_paper.__name__)", DOMAIN_NAME="ceramics")
    assert r.returncode == 0, r.stderr[-500:]
    assert "build_paper" in r.stdout


def test_i1_extract_text_excludes_situating_context_and_mineru_content(tmp_path):
    code = f"""
import sys; sys.argv=['x','--domain','ceramics']
from memento.config.cli_domain import apply_domain_arg; apply_domain_arg()
from pathlib import Path
from memento.graph.build.build_graph import _embeddable_chunks
d = Path({str(tmp_path)!r}); (d/'images').mkdir(); (d/'images'/'a.jpg').write_bytes(b'x')
items = [
  {{'type':'text','text':'The density increased.','page_idx':0,'chunk_context':'Paper on SPS of 3Y-TZP and alumina'}},
  {{'type':'equation','text':'E = mc^2','enriched_context':'Young modulus of alumina','page_idx':0}},
  {{'type':'image','img_path':'images/a.jpg','content':'mermaid: zirconia peak 30','page_idx':1}},
]
for c in _embeddable_chunks(items, d, 'P'): print(repr(c['extract_text']))
"""
    r = _py(code)
    assert r.returncode == 0, r.stderr[-800:]
    out = r.stdout.splitlines()
    assert out == ["'The density increased.'", "'Young modulus of alumina\\nE = mc^2'", "''"], out


def test_i6_domain_without_own_neo4j_refuses_default_store():
    r = _py("from memento.graph.store import get_store\n"
            "try:\n    get_store()\nexcept RuntimeError as e:\n    print('refused:', e)\n", DOMAIN_NAME="generic")
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.startswith("refused:") and "neo4j_uri" in r.stdout
