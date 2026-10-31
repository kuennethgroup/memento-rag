# Elapsed time in Library Upload: one helper for the completion summary (minutes and seconds from the rounded total,
# never "2 min 60 s") and whole minutes floored in every status line.
import inspect

import openwebui.functions.ingestion_pipe as ip
from memento.ingestion.upload import messages
from memento.ingestion.upload import workflow


def test_took_splits_the_rounded_total():
    assert messages.took(0) == "0 s"
    assert messages.took(59) == "59 s"
    assert messages.took(59.6) == "1 min 0 s"
    assert messages.took(60) == "1 min 0 s"
    assert messages.took(90) == "1 min 30 s"
    assert messages.took(119.6) == "2 min 0 s"
    assert messages.took(162) == "2 min 42 s"
    assert messages.took(463) == "7 min 43 s"


def test_status_minutes_are_floored():
    line = lambda mins: messages.batch_status(0, 1, 1, 0, mins)
    assert line(0.5) == "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 0m elapsed"      # 30 s: under 1 min
    assert line(0.99) == "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 0m elapsed"     # 59 s
    assert line(1.0) == "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 1m elapsed"
    assert line(2.7) == "Uploading... 0 of 1 ready (1 parsing, 0 storing) · 2m elapsed"
    assert messages.batch_status(3, 5, None, 2, 1.9) == "Uploading... 3 of 5 ready (2 storing) · 1m elapsed"


def test_done_line_minutes_are_floored():
    assert messages.done_status(3, 5, 59) == "Done · 3 of 5 processed · 0m elapsed"
    assert messages.done_status(3, 5, 119.6) == "Done · 3 of 5 processed · 1m elapsed"
    assert messages.done_status(5, 5, 463) == "Done · 5 of 5 processed · 7m elapsed"


def test_both_paths_use_the_helpers():
    # the parse-and-store batch and the store-only batch both end with these lines; no inline formatting left
    src = inspect.getsource(workflow)
    assert "elapsed / 60:.0f" not in src and "elapsed % 60:.0f" not in src
    assert src.count("messages.took(") >= 2 and src.count("messages.done_status(") >= 2
