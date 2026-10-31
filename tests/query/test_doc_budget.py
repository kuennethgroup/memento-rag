# What goes to the model for research gaps and long summaries.
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.query import doc_budget as db  # noqa: E402


def gap(page, gap_type, n_chars=300, text=None):
    return {"page": page, "gap_type": gap_type, "text": text or f"{gap_type} p{page} " + "x" * n_chars}


def sec(page, section="Discussion", n_chars=300, text=None):
    return {"page": page, "gap_type": "", "section": section, "text": text or f"{section} p{page} " + "y" * n_chars}


def test_gap_budget_small_doc_sends_all():
    gaps = [gap(3, "future_work"), gap(1, "limitation"), gap(2, "uncertainty")]
    sections = [sec(4), sec(5, "Conclusion")]
    chosen, stats = db.select_gap_passages(gaps, sections, budget_tokens=20000)
    assert len(chosen) == 5 and [c["page"] for c in chosen] == [1, 2, 3, 4, 5]        # everything, in reading order
    assert stats["candidates"] == 5 and stats["chosen"] == 5


def test_gap_budget_caps():
    gaps = [gap(p, "limitation", n_chars=900) for p in range(500)]                      # ~300 tokens each
    chosen, stats = db.select_gap_passages(gaps, [], budget_tokens=20000)
    assert stats["tokens"] <= 20000 and 55 <= len(chosen) <= 70 and stats["candidates"] == 500


def test_gap_priority_order():
    # budget for exactly two passages: the two most important types win, whatever their page
    gaps = [gap(p, t, text=f"{p:02d}" + "x" * 298) for p, t in
            ((1, "uncertainty"), (2, "future_work"), (3, "untested_condition"), (4, "limitation"))]   # equal length
    per = len(gaps[0]["text"]) / 3
    chosen, _ = db.select_gap_passages(gaps, [sec(5)], budget_tokens=int(per * 2) + 1)
    assert [c["gap_type"] for c in chosen] == ["untested_condition", "limitation"]      # reading order: p3, p4


def test_gap_sections_dedupe():
    same = "Discussion text shared by both"
    chosen, stats = db.select_gap_passages([gap(4, "limitation", text=same)], [sec(4, text=same), sec(6)], 20000)
    assert len(chosen) == 2 and stats["candidates"] == 2


def passage(i, section, n_chars=600):
    return {"page": i // 3, "section": section, "text": f"{section or 'none'} {i} " + "z" * n_chars}


def long_doc():
    secs = ["Abstract"] * 2 + ["Introduction"] * 20 + ["Methods"] * 150 + ["Results"] * 200 + \
           ["Discussion"] * 100 + ["Conclusion"] * 6
    return [passage(i, s) for i, s in enumerate(secs)]


def test_summary_budget_priorities():
    doc = long_doc()
    chosen, stats = db.select_summary_passages(doc, budget_tokens=40000)
    got = {c["text"].split()[0] for c in chosen}
    assert stats["tokens"] <= 40000 and stats["chosen"] < len(doc)
    ids = {int(c["text"].split()[1]) for c in chosen}
    assert {0, 1} <= ids                                           # abstract
    assert set(range(len(doc) - 6, len(doc))) <= ids               # every conclusion passage
    assert set(range(2, 22)) <= ids                                # introduction
    assert {"Methods", "Results", "Discussion"} <= got             # the start of every other section at least


def test_summary_budget_no_sections():
    doc = [passage(i, None) for i in range(600)]
    chosen, stats = db.select_summary_passages(doc, budget_tokens=40000)
    ids = [int(c["text"].split()[1]) for c in chosen]
    assert stats["tokens"] <= 40000 and len(ids) > 150
    assert min(ids) < 20 and max(ids) > 580                        # spread over the whole document, not its start


def test_summary_budget_reading_order():
    chosen, _ = db.select_summary_passages(long_doc(), budget_tokens=40000)
    ids = [int(c["text"].split()[1]) for c in chosen]
    assert ids == sorted(ids)
