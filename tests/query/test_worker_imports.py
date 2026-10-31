# The query pipe's background worker starts while Open WebUI goes on loading the other functions. Everything the worker
# imports later is imported before it starts, in the loading thread: a first import of qdrant_client in the worker at
# the same time as the ingestion function's load ended in an import deadlock, and Open WebUI switched Library Upload off.
import json
import subprocess
import sys

from memento.config.paths import PROJECT_ROOT

PROBE = r"""
import json, sys, threading
import memento.query.workers as workers
seen = {}
class FakeThread:
    def __init__(self, *a, **k): seen.update({m: m in sys.modules for m in WANTED})
    def start(self): pass
WANTED = ["qdrant_client", "qdrant_client.http", "qdrant_client.http.models", "qdrant_client.models",
          "memento.library.documents", "memento.library.front_matter", "memento.library.people",
          "memento.library.warmup_queue", "memento.query.models"]
workers.threading.Thread = FakeThread
workers.start("a_copy_that_is_not_a_default", object, lambda: None)
print(json.dumps(seen))
"""


def test_the_worker_starts_after_its_imports():
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert seen and all(seen.values()), [m for m, ok in seen.items() if not ok]
