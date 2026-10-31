# The upload bar moves while a long document is read. Reading has no real percentage from MinerU, so it is estimated
# from the page count (1.05 s per page measured on a long book, slower when several documents are read at once, at most
# 95 % until done); storing has a real one.
import re

import openwebui.functions.ingestion_pipe as ip
from memento.ingestion.upload import messages
from memento.ingestion.upload import document, workflow


def test_reading_is_estimated_from_pages_and_never_shown_done():
    assert messages.doc_progress("parse", elapsed=0, pages=282, reading=1) == 0
    mid = messages.doc_progress("parse", elapsed=148, pages=282, reading=1)
    assert 0.25 < mid < 0.35                                   # half the reading time = half of the 60 % share
    assert messages.doc_progress("parse", elapsed=10_000, pages=282, reading=1) == round(0.6 * messages.READ_SHOWN_MAX, 3)
    slower = messages.doc_progress("parse", elapsed=148, pages=282, reading=3)
    assert slower < mid                                         # three documents read at once: each slower
    assert 0 < messages.doc_progress("parse", elapsed=30, pages=None, reading=1) < 0.6     # no page count: a guess


def test_storing_uses_its_real_percentage():
    assert messages.doc_progress("embed", store_pct=0) == 0.6
    assert messages.doc_progress("embed", store_pct=50) == 0.8
    assert messages.doc_progress("embed", store_pct=100) == 1.0


def test_batch_percent_and_status_line():
    assert messages.batch_percent(completed=0, n=1, in_progress=[0.3]) == 30
    assert messages.batch_percent(completed=3, n=7, in_progress=[0.5, 0.25]) == 54
    line = messages.batch_status(0, 1, 1, 0, 3.4, pct=30)
    assert line == "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 3m elapsed · 30% done"
    assert messages.batch_status(0, 1, 1, 0, 3.4) == "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 3m elapsed"


def test_loader_reads_the_new_status_line():
    js = open("openwebui/loader.js").read()
    m = re.search(r"const RUN = /(.+)/;", js)
    rx = m.group(1).replace("\\/", "/").replace("(?<", "(?P<")             # JS named groups in Python's syntax
    assert re.match(rx, "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 3m elapsed · 30% done")
    assert re.match(rx, "Uploading... 3 of 25 ready (8 parsing, 2 storing) · 12m elapsed")       # older chats
    assert re.match(rx, messages.batch_status(0, 1, 1, 0, 3.4, 30, counts=(0, 0, 0, "stored")))       # with counts
    assert "g.pct" in js                                                     # the bar uses the percentage


def test_the_drain_loop_tracks_progress_and_pushes_it():
    import inspect
    src = inspect.getsource(workflow.run_batch)
    assert "messages.batch_percent(" in src and "messages.doc_progress(" in src and "STATUS_EVERY_S" in src


def test_the_status_line_moves_while_a_document_is_read(monkeypatch, tmp_path):
    # the batch loop with a fake reader (parse only: no MinerU, no Qdrant, no library): the status line carries a
    # growing "% done" before the document is finished, and 100 % at the end
    import time
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pipe = document.Ingester(ip.Pipe().valves)
    pipe.valves.SKIP_EMBED = True
    monkeypatch.setattr(messages, "STATUS_EVERY_S", 0.05)
    monkeypatch.setattr(messages, "page_count", lambda p: 1)                  # 1 page: the estimate grows fast
    monkeypatch.setattr(document.Ingester, "load_manifest", lambda self: {"documents": {}})
    monkeypatch.setattr(document.Ingester, "save_manifest", lambda self, m: None)

    def fake_parse(self, p, msg_q, results):
        msg_q.put(("start", "parse", p))
        for _ in range(6):
            time.sleep(0.1)
            msg_q.put(("tick", "parse", p))                           # anything that wakes the loop
        results[p] = {"ok": True, "pages": 1}
        msg_q.put(("file", "parse", p, results[p]))
        msg_q.put(None)
    monkeypatch.setattr(workflow, "parse_worker", fake_parse)
    statuses = [x.text for x in workflow.run_batch(pipe, [pdf]) if isinstance(x, messages.Status)]
    pcts = [int(m.group(1)) for t in statuses for m in [re.search(r"· (\d+)% done$", t)] if m]
    assert pcts and max(pcts[:-1] or [0]) > 0                             # moved before the end
    assert pcts[-1] == 100 and pcts == sorted(pcts)


def test_the_bar_never_goes_back():
    # when more documents start reading, each reading estimate is divided by more, so the batch percent could drop; the
    # shown one holds
    assert messages.shown_percent(previous=27, pct=22) == 27
    assert messages.shown_percent(previous=27, pct=31) == 31
    import inspect
    assert "messages.shown_percent(" in inspect.getsource(workflow.run_batch)


def _labels(status: str):
    # runs loader.js's own uploadLabels() in Node on a status line: [left, right] as the bar shows them
    import json, shutil, subprocess, pytest
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


def test_the_bar_text_is_percent_then_documents_stored():
    # the combined percent in bold on the left, "N of M documents stored · time" on the right; no parsing / storing
    # counts. The stored count comes from the line's own counts; lines without counts (older chats) say "processed"
    # (tests/ingestion/test_upload_counts.py)
    assert _labels(messages.batch_status(1, 7, 4, 2, 1.2, 62, counts=(1, 0, 0, "stored"))) == \
        ["62%", "1 of 7 documents stored · 1 min", False]
    assert _labels(messages.batch_status(0, 1, 1, 0, 0.2, 6, counts=(0, 0, 0, "stored"))) == \
        ["6%", "0 of 1 document stored · under 1 min", False]
    assert _labels(messages.done_status(7, 7, 190, counts=(7, 0, 0, "stored"))) == ["Done", "7 of 7 documents stored · 3 min", False]
    # older chats without a percent: by finished documents
    assert _labels("Uploading... 3 of 25 ready (8 parsing, 2 storing) · 12m elapsed") == \
        ["12%", "3 of 25 documents processed · 12 min", False]
