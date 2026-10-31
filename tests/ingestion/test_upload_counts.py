# The upload bar counts stored, unchanged and failed documents apart. The status line carries the three counts
# (parse-only uploads say "read"), the bar shows them and says "Done with errors" when a file failed; status lines of
# older chats (no counts) say "processed", never "stored". Producer (Python) and formatter (loader.js in Node) tested
# together.
import json
import re
import shutil
import subprocess
import time

import pytest

import openwebui.functions.ingestion_pipe as ip
from memento.ingestion.upload import messages
from memento.ingestion.upload import document, workflow


def _labels(status: str):
    """loader.js's own uploadLabels() on a status line: [left, right, errors] as the bar shows them."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = open("openwebui/loader.js").read()
    pick = lambda pat: re.search(pat, js, re.S).group(0)
    src = "\n".join([pick(r"const RUN = /.+?/;"), pick(r"const DONE = /.+?/;"),
                     pick(r"const uploadLabels = .+?\n  };"),
                     f"const t = {json.dumps(status)};",
                     "console.log(JSON.stringify(uploadLabels(RUN.exec(t), DONE.exec(t))));"])
    return json.loads(subprocess.run([node, "-e", src], capture_output=True, text=True, check=True).stdout)


# ── producer ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_running_line_carries_the_counts():
    line = messages.batch_status(2, 7, 4, 1, 1.2, 45, counts=(1, 0, 1, "stored"))
    assert line == "Uploading... 2 of 7 ready (4 parsing, 1 storing) · 1 stored · 0 unchanged · 1 failed · 1m elapsed" \
                   " · 45% done"
    assert messages.batch_status(3, 5, None, 2, 0.4, counts=(2, 1, 0, "stored")) == \
        "Uploading... 3 of 5 ready (2 storing) · 2 stored · 1 unchanged · 0 failed · 0m elapsed"


def test_done_line_carries_the_counts():
    assert messages.done_status(7, 7, 190, counts=(5, 1, 1, "stored")) == \
        "Done · 5 of 7 stored · 1 unchanged · 1 failed · 3m elapsed"
    assert messages.done_status(3, 3, 61, counts=(3, 0, 0, "read")) == "Done · 3 of 3 read · 0 unchanged · 0 failed · 1m elapsed"


# ── formatter (loader.js) ────────────────────────────────────────────────────────────────────────────────────────────

def test_bar_all_failed():
    assert _labels(messages.done_status(1, 1, 20, counts=(0, 0, 1, "stored"))) == \
        ["Done with errors", "0 of 1 document stored · 1 failed · under 1 min", True]


def test_bar_mixed():
    assert _labels(messages.done_status(7, 7, 190, counts=(5, 1, 1, "stored"))) == \
        ["Done with errors", "5 of 7 documents stored · 1 failed · 1 unchanged · 3 min", True]


def test_bar_unchanged_only_is_no_error():
    assert _labels(messages.done_status(2, 2, 5, counts=(0, 2, 0, "stored"))) == \
        ["Done", "0 of 2 documents stored · 2 unchanged · under 1 min", False]


def test_bar_parse_only_says_read():
    assert _labels(messages.done_status(3, 3, 61, counts=(3, 0, 0, "read"))) == ["Done", "3 of 3 documents read · 1 min", False]


def test_bar_running_with_counts():
    # the third value says a document failed; the amber class is set only once the upload is done
    assert _labels(messages.batch_status(2, 7, 4, 1, 1.2, 45, counts=(1, 0, 1, "stored"))) == \
        ["45%", "1 of 7 documents stored · 1 failed · 1 min", True]
    # storing only (documents read before): drawn as a bar too
    assert _labels(messages.batch_status(3, 5, None, 2, 0.4, counts=(2, 1, 0, "stored"))) == \
        ["60%", "2 of 5 documents stored · 1 unchanged · under 1 min", False]


def test_bar_old_lines_say_processed():
    assert _labels("Done · 1 of 1 processed · 1m elapsed") == ["Done", "1 of 1 document processed · 1 min", False]
    assert _labels("Uploading... 3 of 25 ready (8 parsing, 2 storing) · 12m elapsed") == \
        ["12%", "3 of 25 documents processed · 12 min", False]
    assert _labels("Uploading... 1 of 7 ready (4 parsing, 2 storing) · 1m elapsed · 62% done") == \
        ["62%", "1 of 7 documents processed · 1 min", False]


def test_bar_marks_errors_with_a_class():
    js = open("openwebui/loader.js").read()
    assert "classList.toggle('memento-upload-errors', !!done && errors)" in js
    css = open("openwebui/custom.css").read()
    assert "div.memento-upload-errors" in css


# ── the batch loop (parse only, fake reader: no MinerU, no Qdrant, no library) ───────────────────────────────────────

def _run(monkeypatch, tmp_path, outcomes):
    pdfs = []
    for i, _ in enumerate(outcomes):
        p = tmp_path / f"doc{i}.pdf"
        p.write_bytes(b"%PDF-1.4")
        pdfs.append(p)
    pipe = document.Ingester(ip.Pipe().valves)
    pipe.valves.SKIP_EMBED = True
    monkeypatch.setattr(messages, "page_count", lambda p: 1)
    monkeypatch.setattr(document.Ingester, "load_manifest", lambda self: {"documents": {}})
    monkeypatch.setattr(document.Ingester, "save_manifest", lambda self, m: None)
    plan = dict(zip(pdfs, outcomes))

    def fake_parse(self, p, msg_q, results):
        msg_q.put(("start", "parse", p))
        time.sleep(0.05)
        results[p] = {"ok": True, "pages": 1} if plan[p] == "ok" else {"ok": False, "error": "broken"}
        msg_q.put(("file", "parse", p, results[p]))
        msg_q.put(None)
    monkeypatch.setattr(workflow, "parse_worker", fake_parse)
    return [x.text for x in workflow.run_batch(pipe, pdfs) if isinstance(x, messages.Status)]


def test_loop_counts_a_failed_file_as_failed(monkeypatch, tmp_path):
    statuses = _run(monkeypatch, tmp_path, ["fail"])
    assert statuses[-1].startswith("Done · 0 of 1 read · 0 unchanged · 1 failed · ")


def test_loop_counts_mixed_results(monkeypatch, tmp_path):
    statuses = _run(monkeypatch, tmp_path, ["ok", "fail", "ok"])
    assert statuses[-1].startswith("Done · 2 of 3 read · 0 unchanged · 1 failed · ")
    assert all(" read · " in s for s in statuses if s.startswith("Uploading"))
