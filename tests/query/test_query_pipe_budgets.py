# The Query pipe's budgets for research gaps and long summaries.
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import generation, presentation, prompts
from memento.query.scope import documents
from memento.query.flows import gap_answer, summary

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
qp = load_function(ROOT / "openwebui/functions/query_pipe.py", "memento_query")


def _gap(page, t, n=900):
    return {"text": f"{t} {page} " + "x" * n, "source": "doc", "page": page, "section": "Results", "gap_type": t,
            "gap_signal": "s", "type": "text"}


def _sec(page, n=300):
    return {"text": f"sec {page} " + "y" * n, "source": "doc", "page": page, "section": "Discussion", "gap_type": "",
            "gap_signal": "section-fallback", "type": "text"}


def test_gap_passages_small_doc_all(monkeypatch):
    import memento.research_gaps.passages as gr
    monkeypatch.setattr(gr, "fetch_gap_chunks", lambda c, col, p: [_gap(2, "future_work"), _gap(1, "limitation")])
    monkeypatch.setattr(gr, "fetch_fallback_sections", lambda c, col, p: [_sec(3)])
    chunks, only_sections = gap_answer.gap_passages(None, "col", "doc", budget_tokens=20000)
    assert [c["page"] for c in chunks] == [1, 2, 3] and only_sections is False


def test_gap_passages_long_doc_capped_and_logged(monkeypatch, caplog):
    import logging
    import memento.research_gaps.passages as gr
    monkeypatch.setattr(gr, "fetch_gap_chunks", lambda c, col, p: [_gap(i, "limitation") for i in range(400)])
    monkeypatch.setattr(gr, "fetch_fallback_sections", lambda c, col, p: [_sec(i) for i in range(400, 450)])
    with caplog.at_level(logging.INFO):
        chunks, _ = gap_answer.gap_passages(None, "col", "doc", budget_tokens=20000)
    assert sum(len(c["text"]) for c in chunks) / 3 <= 20000 and len(chunks) < 400
    assert any("gaps: doc candidates=450" in r.getMessage() for r in caplog.records)


def test_gap_passages_sections_only(monkeypatch):
    import memento.research_gaps.passages as gr
    monkeypatch.setattr(gr, "fetch_gap_chunks", lambda c, col, p: [])
    monkeypatch.setattr(gr, "fetch_fallback_sections", lambda c, col, p: [_sec(3), _sec(4)])
    chunks, only_sections = gap_answer.gap_passages(None, "col", "doc", budget_tokens=20000)
    assert len(chunks) == 2 and only_sections is True


def _passages(n, chars=600):
    secs = ["Abstract"] * 2 + ["Introduction"] * 10 + ["Results"] * (n - 16) + ["Conclusion"] * 4
    return [{"text": f"p{i} " + "z" * chars, "source": "doc", "page": i // 3, "section": s} for i, s in enumerate(secs)]


def test_summary_hits_whole_when_it_fits():
    hits, mode = summary.summary_hits(_passages(40), history_tokens=0, max_tokens=40000, prompt_tokens=46000)
    assert mode == "whole" and len(hits) == 40                                          # one source per passage


def test_summary_hits_partial_for_long_docs(caplog):
    import logging
    with caplog.at_level(logging.INFO):
        hits, mode = summary.summary_hits(_passages(600), history_tokens=2000, max_tokens=40000, prompt_tokens=46000)
    assert mode == "partial" and 0 < sum(len(h["text"]) for h in hits) / 3 <= 40000
    assert [h["page"] for h in hits] == sorted(h["page"] for h in hits)
    assert any("summary: doc passages=" in r.getMessage() and "partial=True" in r.getMessage() for r in caplog.records)


def test_summary_sources_are_passages_with_their_boxes_and_no_images():
    # a summary citation marks its passage
    ps = [{"text": "Fig. 2: the phase diagram", "source": "doc", "page": 1, "section": "Results", "type": "image",
           "bbox": [10, 20, 300, 400], "img_path": "/x/fig2.png"},
          {"text": "We find a stable phase.", "source": "doc", "page": 1, "section": "Results", "type": "text",
           "bbox": [10, 420, 300, 500], "img_path": None}]
    hits, mode = summary.summary_hits(ps, history_tokens=0, max_tokens=40000, prompt_tokens=46000)
    assert mode == "whole" and [h["bbox"] for h in hits] == [[10, 20, 300, 400], [10, 420, 300, 500]]
    assert all(h["img_path"] is None for h in hits) and ps[0]["img_path"] == "/x/fig2.png"   # input untouched
    url = presentation.source_file_url("http://f", "doc", 1, [hits[1]["bbox"]], text=hits[1]["text"])
    assert "bbox=10,420,300,500" in url


def test_summary_budget_counts_labels():
    # 900 passages of 120 characters: 36,000 tokens of text alone would fit 40,000; with their labels they do not
    ps = [{"text": "x" * 120, "source": "doc", "page": i // 9, "section": "Results"} for i in range(900)]
    hits, mode = summary.summary_hits(ps, history_tokens=0, max_tokens=40000, prompt_tokens=46000)
    assert mode == "partial"
    assert sum(len(h["text"]) + prompts.SUMMARY_LABEL_CHARS for h in hits) / 3 <= 40000


def test_summary_partial_prompt_says_not_to_mention_the_selection():
    assert "COMPLETE" not in prompts.SUMMARY_PARTIAL_ADDENDUM and "do not mention" in prompts.SUMMARY_PARTIAL_ADDENDUM.lower()


def test_picked_document_exact(tmp_path, monkeypatch):
    from memento.library import documents as ll
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    ll.write_library(tmp_path, [{"source": "openalex_W1", "title": "Grain growth of $B_2O_3$-doped ZnO"},
                                {"source": "openalex_W2", "title": "NRC Publications Archive"},
                                {"source": "openalex_W3", "title": "NRC Publications Archive"},
                                {"source": "q", "title": 'The "best" ceramics'}])
    srcs = {"openalex_W1", "openalex_W2", "openalex_W3", "q"}
    assert documents.picked_document('Summarize the document "Grain growth of $B_2O_3$-doped ZnO"', srcs) == \
        ("openalex_W1", "summary")
    assert documents.picked_document('What research gaps does the document "openalex_W3" name?', srcs) == \
        ("openalex_W3", "gap_query")
    assert documents.picked_document('Summarize the document "NRC Publications Archive"', srcs) is None     # shared title
    assert documents.picked_document("Summarize the document \"The 'best' ceramics\"", srcs) == ("q", "summary")
    assert documents.picked_document('Summarize the document "openalex_W9"', srcs) is None                  # not in library
    assert documents.picked_document("Summarize the grain growth paper", srcs) is None


def test_gap_citations_have_unique_numbered_names():
    # two passages on one page get distinct names (Open WebUI drops one of two equal names and shifts [N])
    hs = [_gap(2, "limitation"), _gap(2, "future_work"), _sec(4)]
    evs = [presentation.gap_citation(i, h, "http://f") for i, h in enumerate(hs)]
    names = [e["data"]["metadata"][0]["name"] for e in evs]
    assert names == ["[1] doc (p. 3)", "[2] doc (p. 3)", "[3] doc (p. 5)"] and len(set(names)) == 3
    assert all(e["data"]["source"]["embed_url"].startswith("http://f/view/doc/") for e in evs)


def test_gap_prompt_labels_untagged_passages():
    from memento.research_gaps.passages import build_gap_prompt, GAP_SYSTEM_PROMPT
    msg = build_gap_prompt([_gap(1, "limitation"), _sec(3)], "What gaps?")
    assert "Pre-tagged as: limitation" in msg and "Not gap-tagged: from the Discussion section" in msg
    assert "section-fallback" not in msg and "Not gap-tagged" in GAP_SYSTEM_PROMPT


def test_gap_thinking_cap_local_only_and_thorough_lifts_it():
    assert generation.gap_thinking_budget(False, 2000) == 2000
    assert generation.gap_thinking_budget(True, 2000) == 2000          # Thorough check keeps the cap
    assert generation.gap_thinking_budget(False, 0) is None            # 0 = no gap-specific cap
    pipe = qp.Pipe()
    pipe.valves.THINKING_BUDGET = 12288
    assert generation.gen_extra(pipe.valves, True, 2000)["thinking_token_budget"] == 2000
    assert generation.gen_extra(pipe.valves, True, None)["thinking_token_budget"] == 12288
    assert "thinking_token_budget" not in generation.gen_extra(pipe.valves, False, 2000)


def test_open_answer_caps_only_the_local_model(monkeypatch):
    import asyncio
    sent = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    sent.append(kw)
                    raise RuntimeError("stop here")
    pipe = qp.Pipe()
    monkeypatch.setattr(generation, "client", lambda *a: FakeClient)
    local = next(v for v in generation.copy_variants() if not v.is_cloud)
    try:
        asyncio.run(generation.open_answer(pipe.valves, local, messages=[{"role": "user", "content": "x"}], think=True,
                                      local_thinking_budget=2000))
    except RuntimeError:
        pass
    assert sent and sent[0]["extra_body"]["thinking_token_budget"] == 2000


def test_gap_prompt_fixes_the_layout():
    from memento.research_gaps.passages import GAP_SYSTEM_PROMPT as P
    for h in ("### Limitations", "### Uncertainties", "### Untested conditions", "### Future work"):
        assert h in P
    assert "future_work |" not in P and "No bold text" in P and "numbering continues" in P
