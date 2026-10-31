# tests/graph/test_rq2_neo4j_domain.py
import os
import subprocess
import sys
from memento.config.paths import PROJECT_ROOT

CODE = "from memento.config.settings import NEO4J_URI; print(NEO4J_URI)"


def _uri(domain):
    env = dict(os.environ, DOMAIN_NAME=domain)
    return subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True,
                          text=True, check=True, cwd=str(PROJECT_ROOT)).stdout.strip()


def test_domain_selects_neo4j_uri():
    assert _uri("ceramics") == "bolt://localhost:7688"
    assert _uri("polymer-engineering") == "bolt://localhost:7689"
