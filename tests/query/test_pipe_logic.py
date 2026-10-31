"""Unit tests for pipe logic that needs no live services."""

import pytest
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from memento.query import library_state, presentation, request, texts
from memento.query.scope import conversation, documents
from memento.query.flows import figures, gap_answer, gap_report, general

pytestmark = pytest.mark.unit

# the upload pipe and the modules its code moved to
_PIPE_SRC = "\n".join(p.read_text() for p in [PROJECT_ROOT / "openwebui" / "functions" / "ingestion_pipe.py"]
                       + sorted((PROJECT_ROOT / "memento" / "ingestion" / "upload").glob("*.py")))

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from memento.query.prompts import rewrite_query as _rewrite_query  # noqa: E402
from memento.query.texts import sanitize_assistant_text as _sanitize_assistant_text  # noqa: E402
from memento.query.flows.library_listing import _wants_corpus_listing, listing_markdown as _listing_markdown
from memento.query.flows.summary import document_passages as _document_passages
from memento.query.flows.verbatim import section_span as _section_span
from memento.query.scope.conversation import history_text as _history_text
from memento.query.presentation import page_label as _page_label
from memento.ingestion.upload.messages import file_summary_line as _file_summary_line, Status as _Status, DocLines as _DocLines


# ── multimodal content handling ───────────────────────────────────────────────

def _extract_text(content):
    """Mirror of the content-extraction logic in ingestion_pipe.pipe()."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text").strip()
    return content.strip()


def test_plain_string_content():
    assert _extract_text("BMW-Group-Bericht-2023-de") == "BMW-Group-Bericht-2023-de"


def test_plain_string_strips_whitespace():
    assert _extract_text("  data/input  ") == "data/input"


def test_multimodal_list_extracts_text_part():
    content = [
        {"type": "text", "text": "data/input"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]
    assert _extract_text(content) == "data/input"


def test_multimodal_list_with_no_text_returns_empty():
    content = [{"type": "image_url", "image_url": {"url": "..."}}]
    assert _extract_text(content) == ""


def test_multimodal_list_joins_multiple_text_parts():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    assert _extract_text(content) == "hello world"


# ── supported extensions ──────────────────────────────────────────────────────

def test_all_documented_extensions_present_in_pipe():
    expected = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    for ext in expected:
        assert f'"{ext}"' in _PIPE_SRC, f"Extension {ext!r} missing from ingestion_pipe.py SUPPORTED_EXTENSIONS"


# ── data/test and data/rough not referenced ───────────────────────────────────

def test_scratch_dirs_not_in_pipe():
    assert "data/test" not in _PIPE_SRC, "data/test should not be used in production code"
    assert "data/rough" not in _PIPE_SRC, "data/rough should not be used in production code"


# ── env-var defaults are portable (no hardcoded host paths) ──────────────────

def test_no_hardcoded_home_paths_in_pipe():
    assert "/home/" not in _PIPE_SRC, "hardcoded home path found in ingestion_pipe.py"


# ── ingestion output: per-file line + live status (ingestion_pipe) ──

def test_file_summary_line_success_is_collapsible_with_numbering():
    result = {"ok": True, "vectors": 352, "elapsed": 12.3, "notices": ["🔄 File changed — re-storing (10 old entries removed)"],
              "type_counts": {"text": 300, "image": 40, "chart": 3, "equation": 7, "table": 1, "list": 1}}
    line = _file_summary_line(7, 229, "embed", "alumina.pdf", result, pages=40)
    assert "[7/229]" in line and "alumina.pdf" in line
    assert "<details>" in line and "<summary>" in line
    summary = line.split("<summary>")[1].split("</summary>")[0]
    assert "40 pages" in summary and "12 s" in summary and "352" not in summary     # no "N sections"
    assert ("Stored: 300 text passages, 1 list, 7 equations, 1 table, 40 figures/images, 3 charts/plots" in line)
    assert "File changed" in line and "text: 300" not in line


def test_file_summary_line_failure_is_visible_not_collapsed():
    result = {"ok": False, "error": "embedding failed: timeout"}
    line = _file_summary_line(3, 229, "embed", "broken_scan.pdf", result)
    # the headline failure must be outside any <details> so it can't be missed
    assert line.split("<details>")[0].strip().startswith("❌ [3/229]")
    assert "broken_scan.pdf" in line
    assert "timeout" in line


def test_file_summary_line_skipped():
    result = {"ok": True, "skipped": True, "vectors": 0, "type_counts": {}, "elapsed": 0.0}
    line = _file_summary_line(5, 229, "embed", "unchanged.pdf", result)
    assert "[5/229]" in line
    assert "skipped, already in the library and unchanged" in line


def test_file_summary_line_no_result_returned():
    line = _file_summary_line(1, 10, "parse", "weird.pdf", None)
    assert "❌ [1/10]" in line
    assert "no result returned" in line


def test_status_marker_carries_text_and_done_flag():
    s = _Status("⏳ 5/10 done", done=False)
    assert s.text == "⏳ 5/10 done"
    assert s.done is False
    s2 = _Status("✅ 10/10 done", done=True)
    assert s2.done is True


# ── citation page-range label (query_pipe._page_label) ────────────────────────

def test_page_label_single_page_when_no_page_end():
    assert _page_label(0, None) == "p. 1"


def test_page_label_single_page_when_page_end_equal():
    assert _page_label(0, 0) == "p. 1"


def test_page_label_range_when_page_end_greater():
    assert _page_label(0, 1) == "pp. 1–2"
    assert _page_label(3, 5) == "pp. 4–6"


# ── conversation history helpers (multi-turn chat) ─────────────────────────────

def test_message_text_plain_string():
    assert request.message_text("what about its sintering temperature?") == "what about its sintering temperature?"


def test_message_text_multimodal_list():
    content = [
        {"type": "text", "text": "compare these"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]
    assert request.message_text(content) == "compare these"


def test_message_text_none_returns_empty():
    assert request.message_text(None) == ""


def test_sanitize_assistant_text_strips_base64_image():
    text = "The XRD pattern shows this [1].\n\n![Paper p.3](data:image/jpeg;base64,AAAA/BBBB==)"
    assert _sanitize_assistant_text(text) == "The XRD pattern shows this [1]."


def test_sanitize_assistant_text_strips_think_block():
    text = "<think>let me reason about this</think>The answer is 1400°C [2]."
    assert _sanitize_assistant_text(text) == "The answer is 1400°C [2]."


def test_sanitize_assistant_text_strips_both_and_keeps_prose():
    text = ("<think>reasoning...</think>Grain growth is suppressed [3]."
            "\n\n![Fig](data:image/jpeg;base64,ZZZZ==)")
    assert _sanitize_assistant_text(text) == "Grain growth is suppressed [3]."


def test_sanitize_assistant_text_plain_prose_unchanged():
    assert _sanitize_assistant_text("Sintering occurs at 1400°C [1].") == "Sintering occurs at 1400°C [1]."


# ── query rewrite (multi-turn chat) ─────────────────────────────────────────────

def test_rewrite_query_no_history_returns_unchanged_without_llm_call():
    """First turn of a chat: no history means no rewrite LLM call at all —
    single-shot queries pay nothing extra."""
    import asyncio

    class _ExplodingLLM:
        class chat:
            class completions:
                @staticmethod
                async def create(*a, **k):
                    raise AssertionError("LLM should not be called when history is empty")

    query = "what is the boehmite to alumina reaction?"
    result = asyncio.run(_rewrite_query(_ExplodingLLM(), "some-model", query, []))
    assert result == query


# ── corpus-listing intent ───────────────────────────────────────────────────

def test_wants_corpus_listing_english_phrasing():
    assert _wants_corpus_listing("list all the different paper titles in database", None)
    assert _wants_corpus_listing("what documents are in the corpus?", None)
    assert _wants_corpus_listing("which papers are available in the collection", None)


def test_wants_corpus_listing_german_phrasing():
    assert _wants_corpus_listing("liste alle Paper in der Datenbank auf", None)
    assert _wants_corpus_listing("welche Dokumente sind im Korpus?", None)


def test_wants_corpus_listing_false_when_paper_detected():
    # a named paper means this is a content question about that paper, not
    # a corpus-listing request — even if phrased with "list"/"papers"
    assert not _wants_corpus_listing(
        "list the figures in BMW-Group-Bericht-2023-de", "BMW-Group-Bericht-2023-de"
    )


def test_wants_corpus_listing_false_for_analysis_questions():
    # contains "paper"/"database" but is an analysis request, not a listing one
    assert not _wants_corpus_listing("compare the two papers in the database", None)
    assert not _wants_corpus_listing("explain how this differs from the other documents", None)


def test_wants_corpus_listing_false_for_normal_questions():
    assert not _wants_corpus_listing("what is the boehmite to alumina reaction?", None)
    assert not _wants_corpus_listing("how does sintering temperature affect porosity?", None)


# ── corpus listing: first 20 shown, the rest behind "Show 50 more" / "Show all" ──

def _names(n):
    return [f"doc{i:03d}" for i in range(1, n + 1)]


def test_listing_short_library_has_no_collapsibles():
    text = _listing_markdown(_names(12))
    assert "<details" not in text
    assert text.startswith("1. doc001\n") and "12. doc012" in text


def test_listing_shows_first_20_then_nested_steps_of_50():
    text = _listing_markdown(_names(135))
    visible = text.split("<details", 1)[0]
    assert "20. doc020" in visible and "21. doc021" not in visible
    assert "<summary>Show 50 more (21–70 of 135)</summary>" in text
    assert "<summary>Show 50 more (71–120 of 135)</summary>" in text
    assert "<summary>Show 15 more (121–135 of 135)</summary>" in text
    chain = text.split('<details type="memento_list_all">', 1)[0]
    # each step sits inside the one before: three opens, then three closes at the end of the chain
    assert chain.count("<details>") == 3 and chain.rstrip().endswith("</details>\n\n</details>\n\n</details>")
    assert all(f"{i}. doc{i:03d}" in chain for i in range(1, 136))


def test_listing_show_all_only_when_more_than_one_step_remains():
    assert "Show all" not in _listing_markdown(_names(70))          # 50 remain: "Show 50 more" is the whole rest
    text = _listing_markdown(_names(71))
    all_block = text.split('<details type="memento_list_all">', 1)[1]
    assert "<summary>Show all 71</summary>" in all_block
    assert "21. doc021" in all_block and "71. doc071" in all_block and "20. doc020" not in all_block


def test_listing_is_sorted_and_blocks_are_blank_line_separated():
    text = _listing_markdown(["b", "a", "c"] + _names(80))
    assert text.startswith("1. a\n2. b\n3. c\n")
    assert "</details>\n\n<details" in text                       # renderers need the blank line (memory note)


def test_history_drops_the_show_all_duplicate_but_keeps_the_chain():
    text = "The corpus currently contains **100** papers:\n\n" + _listing_markdown(_names(100))
    h = _history_text(text)
    assert "Show all" not in h and "memento_list_all" not in h
    assert h.count("100. doc100") == 1 and "Show 30 more (71–100 of 100)" in h


# ── whole-document summary: reading order from the point ids, one source per page ──

def _fake_doc(n_pages=3, per_page=4, source="paper"):
    """Points as ingestion writes them: id = chunk_point_id(source, page, type, index in document order)."""
    import sys as _s
    _s.path.insert(0, str(PROJECT_ROOT))
    from memento.ingestion.embed import chunk_point_id
    from types import SimpleNamespace
    pts, i = [], 0
    for page in range(n_pages):
        for k in range(per_page):
            typ = "image" if k == 2 else "text"
            # two-column page: the right column's first passage sits HIGHER than the left column's last one
            bbox = [0, 900 - 100 * k, 10, 10] if k % 2 else [0, 100 * k, 10, 10]
            pts.append(SimpleNamespace(id=chunk_point_id(source, page, typ, i), payload={
                "source": source, "page": page, "type": typ, "text": f"p{page}-{k}", "bbox": bbox, "section": "Results"}))
            i += 1
    return pts


class _FakeClient:
    def __init__(self, points, page_size=5):
        self.points, self.page_size, self.filters = points, page_size, []

    def scroll(self, collection_name, scroll_filter, with_payload, with_vectors, limit, offset=None):
        self.filters.append(scroll_filter)
        start = offset or 0
        batch = self.points[start:start + self.page_size]
        nxt = start + self.page_size
        return batch, (nxt if nxt < len(self.points) else None)


def test_document_passages_follow_the_ingestion_order_not_the_bbox():
    import random
    from types import SimpleNamespace
    pts = _fake_doc()
    extra = [SimpleNamespace(id=_fake_doc(1, 1, "x")[0].id + 7, payload={"source": "paper", "page": 1, "type": "text",
                                                                        "text": "kw", "section": "Keywords"})]
    shuffled = pts[:]; random.Random(1).shuffle(shuffled)
    out = _document_passages(_FakeClient(shuffled + extra), "c", "paper")
    assert [h["text"] for h in out] == [p.payload["text"] for p in pts]          # Keywords dropped, order exact


def test_document_passages_unmatched_ids_go_last_by_page():
    from types import SimpleNamespace
    pts = _fake_doc(n_pages=2, per_page=2)
    stray = SimpleNamespace(id=12345, payload={"source": "paper", "page": 0, "type": "text", "text": "stray"})
    out = _document_passages(_FakeClient([stray] + pts), "c", "paper")
    assert [h["text"] for h in out] == ["p0-0", "p0-1", "p1-0", "p1-1", "stray"]


def test_router_accepts_summary_intent():
    from memento.query.router import RouterResponse, pipe_intent
    r = RouterResponse(intent="summary", query="summarize the polyBERT paper")
    assert r.intent == "summary" and pipe_intent(r.intent) == "summary"


# ── verbatim section by heading ──

HINTS = {"Abstract": ["abstract", "kurzfassung", "zusammenfassung"], "Methods": ["method", "experimental"],
         "Conclusion": ["conclusion", "fazit", "summary", "outlook", "ausblick"]}


def _thesis(source="Thesis_Experimental_Foams"):
    layout = [("Kurzfassung", 2), ("Abstract", 2), ("1 Introduction", 2), ("5 Methods", 1), ("5.2.1 Sandblasting", 2),
              ("5.2.2 Etching", 2), ("5.3 Plating", 1), ("6 Results", 1), ("6.1 Adhesion Strength", 2), ("Fazit", 2),
              ("6.2 Failure", 3), ("Fazit", 1), ("7.1 Zusammenfassung", 3), ("7.2 Ausblick", 2), ("8 Literatur", 2)]
    out = []
    for head, n in layout:
        out += [{"source": source, "text": f"{head}/{len(out) + i}", "section_raw": head, "page": len(out) + i}
                for i in range(n)]
    return out


def _heads(span):
    return list(dict.fromkeys(h["section_raw"] for h in span))


def test_conclusion_takes_the_final_chapter_with_summary_siblings():
    doc = _thesis()
    assert _heads(_section_span(doc, "give me the conclusion", HINTS, [])) == ["7.1 Zusammenfassung", "7.2 Ausblick"]
    assert _heads(_section_span(doc, "Gib mir das Fazit", HINTS, [])) == ["7.1 Zusammenfassung", "7.2 Ausblick"]


def test_a_unique_exact_heading_wins_and_german_words_work():
    doc = _thesis()
    assert _heads(_section_span(doc, "zeig mir die Kurzfassung", HINTS, [])) == ["Kurzfassung"]
    assert _heads(_section_span(doc, "give me the abstract", HINTS, [])) == ["Abstract"]


def test_reranker_top5_picks_among_same_named_headings():
    doc = _thesis()
    ranked = [{"text": "Fazit/20", "section_raw": "Fazit"}]         # the "Fazit" run after 6.2
    span = _section_span(doc, "show the conclusion", HINTS, ranked)
    assert [h["text"] for h in span] == ["Fazit/20"]
    deep = [{"text": f"x{i}", "section_raw": "6.1 Adhesion Strength"} for i in range(5)] + ranked
    assert _heads(_section_span(doc, "show the conclusion", HINTS, deep)) == ["7.1 Zusammenfassung", "7.2 Ausblick"]


def test_section_number_takes_its_subsections():
    doc = _thesis()
    assert _heads(_section_span(doc, "show section 5.2", HINTS, [])) == ["5.2.1 Sandblasting", "5.2.2 Etching"]
    # the unnumbered "Fazit" after 6.1 is inside chapter 6 (6.2 follows); the one after 6.2 is not decidable → stop
    assert _heads(_section_span(doc, "show section 6", HINTS, [])) == ["6 Results", "6.1 Adhesion Strength", "Fazit", "6.2 Failure"]


def test_document_name_words_and_single_shared_words_do_not_select_a_section():
    doc = _thesis()
    # "experimental" is in the document's name, not a request for the Methods section
    assert _heads(_section_span(doc, "give me the conclusion of the experimental foams thesis", HINTS, [])) == [
        "7.1 Zusammenfassung", "7.2 Ausblick"]
    assert _section_span(doc, "tell me about adhesion", {}, []) == []                  # 1 of the heading's 2 words
    assert _heads(_section_span(doc, "show the adhesion strength part", {}, [])) == ["6.1 Adhesion Strength"]
    assert _heads(_section_span(doc, "show the sandblasting part", {}, [])) == ["5.2.1 Sandblasting"]   # its only word
    assert _section_span(doc, "what is sintering?", HINTS, []) == []


# ── uploads: every document streams as it finishes, the newest at the bottom ──
# (a streamed reply cannot fold earlier lines away in Open WebUI, so every line streams and the newest is last)

def _run_upload(n, fail_at=()):
    d = _DocLines()
    streamed = [d.add(f"<details>\n<summary>[{i}/{n}] doc{i}</summary>\n\nx\n</details>\n\n", i not in fail_at)
                for i in range(1, n + 1)]
    return [x for x in streamed if x], d.rest()


def test_every_document_streams_in_order():
    for n in (1, 20, 25, 130):
        shown, rest = _run_upload(n, fail_at={min(n, 23)})
        assert [int(x.split("[")[1].split("/")[0]) for x in shown] == list(range(1, n + 1)) and rest == ""
        assert not any("listed at the end" in x or "Show" in x for x in shown)


def test_upload_header_and_status_wording():
    from memento.ingestion.upload.messages import batch_header as _batch_header, batch_status as _batch_status
    # the header stays true after the upload
    assert _batch_header(25) == "\nUploading **25** files:\n\n"          # loader.js shows "Uploaded" once done
    assert _batch_header(1) == "\nUploading **1** file:\n\n"
    assert _batch_status(12, 25, 3, 2, 4.4) == "Uploading... 12 of 25 ready (3 parsing, 2 storing) · 4m elapsed"
    assert _batch_status(1, 2, None, 0, 0.2) == "Uploading... 1 of 2 ready (0 storing) · 0m elapsed"   # storing only


# ── library listing shows titles ──

def test_plausible_title_rejects_headings_notes_and_cut_titles():
    from memento.query.scope.documents import _plausible_title
    hints = {"Results": ["results", "ergebnis"], "Methods": ["experimentell", "experimental"]}
    assert _plausible_title("polyBERT: a chemical language model to enable fully machine-driven polymer informatics", hints)
    assert _plausible_title("1 **Experimenteller Teil**", hints) is None
    assert _plausible_title("**Ergebnisse und Diskussion** der Arbeit", hints) is None
    assert _plausible_title("Gefördert durch:", hints) is None
    assert _plausible_title("Transparente PC/PMMA Blends:", hints) is None
    assert _plausible_title("2 wt.% CB-Interleaf laminate", hints) is None
    assert _plausible_title(None, hints) is None


def test_listing_shows_title_with_file_name_when_different():
    import openwebui.functions.query_pipe as qp
    srcs = ["PolyBERT", "Bioplastic design using multitask deep neural networks", "Diss_Brandauer"]
    titles = {"PolyBERT": ("polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics", None),
              "Bioplastic design using multitask deep neural networks": ("Bioplastic design using multitask deep neural networks", None),
              "Diss_Brandauer": ("1 Einleitung 4", None)}
    for src in srcs:
        library_state.ALIASES[("None", src)] = [src]
        library_state.TITLES[("None", src)] = titles[src]
    shown = documents.display_titles(srcs, None)
    assert shown["PolyBERT"].startswith("polyBERT: a chemical") and shown["PolyBERT"].endswith("(PolyBERT)")
    assert shown["Bioplastic design using multitask deep neural networks"] == "Bioplastic design using multitask deep neural networks"
    assert shown["Diss_Brandauer"] == "Diss_Brandauer"                         # no plausible title → file name


def test_listing_titles_match_the_picker():
    # the listing compares plain text, like the picker: a formula written two ways is one name, and a cover-page line is
    # no title
    import openwebui.functions.query_pipe as qp
    srcs = ["Impact of field cycling on HfO2 based non-volatile memory devices",
            "On the Origin of the Large Remanent Polarization in LaHfO2"]
    titles = {srcs[0]: ("Impact of field cycling on $\\mathrm{HfO_2}$ based non-volatile memory devices", None),
              srcs[1]: ("Dieses Dokument ist eine Zweitveröffentlichung (Postprint)",
                        "On the Origin of the Large Remanent Polarization in La:HfO₂")}
    for src in srcs:
        library_state.ALIASES[("None", src)] = [src]
        library_state.TITLES[("None", src)] = titles[src]
    shown = documents.display_titles(srcs, None)
    assert shown[srcs[0]] == srcs[0]
    assert shown[srcs[1]] == "On the Origin of the Large Remanent Polarization in La:HfO₂"


# ── conversation history budget ──

def test_shorten_cuts_at_a_paragraph_and_marks_it():
    from memento.query.request import _shorten, SHORTENED_NOTE
    assert _shorten("short", 100) == "short"
    text = "A" * 60 + "\n\n" + "B" * 60 + "\n\n" + "C" * 60
    out = _shorten(text, 130)
    assert out.endswith(SHORTENED_NOTE) and "B" * 60 in out and "C" not in out


def test_history_is_bounded_per_message_and_in_total():
    import openwebui.functions.query_pipe as qp
    per = int(request.HISTORY_MESSAGE_TOKENS * request.CHARS_PER_TOKEN)
    total = int(request.HISTORY_TOTAL_TOKENS * request.CHARS_PER_TOKEN)
    verbatim = "\n\n".join(f"Passage {i}: " + "word " * 60 for i in range(400))       # ~a 47k-token chapter
    hist = []
    for i in range(3):
        hist += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": verbatim}]
    out = request.budget_history(hist)
    assert all(len(m["content"]) <= per + len(request.SHORTENED_NOTE) for m in out)
    assert sum(len(m["content"]) for m in out) <= total
    assert out[0]["role"] == "user" and out[-1]["content"].startswith("Passage 0")
    assert [m["content"] for m in out if m["role"] == "user"][-1] == "question 2"   # the newest turn is kept
    small = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a short answer"}]
    assert request.budget_history(small) == small
    assert all(len(m["content"]) <= request.ROUTER_TURN_CHARS + len(request.SHORTENED_NOTE) for m in request.router_history(hist))


def test_upload_reports_are_recognised_answers_are_not():
    from memento.query.request import INGEST_REPORT_RE
    assert INGEST_REPORT_RE.search("\nProcessing **25** file(s)...\n\n<details>…")
    assert INGEST_REPORT_RE.search("\nUploading **25** files to the library...\n\n<details>…")     # an older wording
    assert INGEST_REPORT_RE.search("\nUploading **1** file to the library...\n\n")
    assert INGEST_REPORT_RE.search("\nUpload of **25** files:\n\n<details>…")                    # an older wording
    assert INGEST_REPORT_RE.search("\nUpload of **1** file:\n\n")
    assert INGEST_REPORT_RE.search("\nUploading **25** files:\n\n<details>…")                     # the current wording
    assert not INGEST_REPORT_RE.search("The upload of **3** samples failed in the study [1].")
    assert INGEST_REPORT_RE.search("\n---\n✅ Done · 25 stored · 0 failed · 9 min 52 s\n")
    assert INGEST_REPORT_RE.search("Checking **3** file(s) against the library...")
    assert not INGEST_REPORT_RE.search("polyBERT is a chemical language model [1]. Processing times drop 100-fold [2].")


def test_table_of_contents_runs_are_never_a_section():
    from memento.query.flows.verbatim import _toc_like
    assert _toc_like(["4.2 Nanostructure formation via compatibilisation 123",
                      "4.3 Microstructural control via addition of polystyrene ..... 163"])
    assert not _toc_like(["Die Prüfung nach UL 94 ist geeignet, um das Brandverhalten zu bewerten."])
    doc = [{"source": "t", "text": "6.1 Adhesion ..... 70", "section_raw": "6 Results 68", "page": 2},
           {"source": "t", "text": "6.2 Failure ..... 75", "section_raw": "6 Results 68", "page": 2},
           {"source": "t", "text": "Intro prose.", "section_raw": "1 Introduction", "page": 5},
           {"source": "t", "text": "Adhesion rose by 40 %.", "section_raw": "6.1 Adhesion", "page": 70},
           {"source": "t", "text": "Cracks started at the interface.", "section_raw": "6.2 Failure", "page": 75}]
    span = _section_span(doc, "show section 6", {}, [])
    assert [h["text"] for h in span] == ["Adhesion rose by 40 %.", "Cracks started at the interface."]
    # named only in the table of contents → the chapter by its number
    span = _section_span(doc, "give me the results", {"Results": ["results"]}, [])
    assert [h["text"] for h in span] == ["Adhesion rose by 40 %.", "Cracks started at the interface."]


# ── document list belongs to this library; every reply closes its status ──

def test_documents_load_per_collection_and_fill_title_caches(monkeypatch):
    from sqlalchemy import create_engine
    from memento.library import documents as db, documents as ll
    import openwebui.functions.query_pipe as qp
    e = create_engine("sqlite://")
    db.ensure_tables(e)
    db.add_document(e, collection="mine", entry={"source": "p1", "title": "T", "parsed_title": "Parsed One",
                                                 "pdf_title": None}, content_hash="h", uploaded_by="u", output_root=None)
    db.add_document(e, collection="other", entry={"source": "x", "title": "x"}, content_hash="h", uploaded_by="u",
                    output_root=None)
    monkeypatch.setattr(qp, "_db_engine", lambda: e)
    monkeypatch.setattr(library_state, "ALIASES", {})
    monkeypatch.setattr(library_state, "TITLES", {})
    assert set(library_state.load_documents("mine", None, qp._db_engine)) == {"p1"}                # another library's rows are not this one's
    monkeypatch.setattr(ll, "raw_titles", lambda *a: (_ for _ in ()).throw(AssertionError("read from disk")))
    # names from the row and the listed title, no file reads
    assert documents._doc_aliases(["p1"], None)["p1"] == ["p1", "Parsed One", "T"]


def test_empty_list_is_rebuilt_from_the_collection(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from memento.library import documents as db
    import openwebui.functions.query_pipe as qp
    e = create_engine("sqlite://")
    db.ensure_tables(e)
    monkeypatch.setattr(qp, "_db_engine", lambda: e)
    monkeypatch.setenv("INPUT_ROOT", str(tmp_path / "in"))
    library_state.rebuild_documents("mine", {"corpus_dir": "C"}, {"b", "a"}, qp._db_engine)
    assert [r["source"] for r in db.list_documents(e, "mine")] == ["a", "b"]


def _steps_handed_the_reply(path) -> list:
    """(file, function) of each step a module hands the reply to: `contextlib.aclosing(<module>.<function>(turn))`,
    or a step of the same module, `contextlib.aclosing(<function>(turn))`."""
    import ast
    import re
    text = path.read_text()
    modules = {}
    for n in ast.parse(text).body:
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("memento."):
            for a in n.names:
                modules[a.asname or a.name] = PROJECT_ROOT / (f"{n.module}.{a.name}".replace(".", "/") + ".py")
    return [(modules[m.group(1)] if m.group(1) else path, m.group(2))
            for m in re.finditer(r"contextlib\.aclosing\((?:(\w+)\.)?(\w+)\(turn\)\)", text)]


def _function_lines(path, name):
    import ast
    code = path.read_text()
    fn = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
              and n.name == name)
    return fn.lineno, code.splitlines()[fn.lineno - 1:fn.end_lineno]


def test_every_reply_after_the_first_status_closes_it():
    import re
    pipe = PROJECT_ROOT / "openwebui" / "functions" / "query_pipe.py"
    service = PROJECT_ROOT / "memento" / "query" / "service.py"
    open_ends = []
    # the coordinator: every return after the first status line closes it, or follows a step that did
    for path, name in ((service, "run"), (service, "_setup")):
        first, lines = _function_lines(path, name)
        started = name == "run"                          # the first status line is _setup's
        for i, line in enumerate(lines):
            if 'status("Looking for answer...")' in line:
                started = True
            before = "\n".join(lines[max(0, i - 4):i])
            if started and re.match(r"\s+return\s*$", line) and "done=True" not in before \
                    and "contextlib.aclosing(" not in before and not re.match(r"\s+if turn\.done:\s*$", lines[i - 1]):
                open_ends.append(f"{path.name} {name}: line {first + i}")
    # a step that finishes the reply closes the status line itself, before it says so (turn.done)
    steps = {s for p in (pipe, service) for s in _steps_handed_the_reply(p)}
    for path, func in sorted(steps):
        first, lines = _function_lines(path, func)
        after_status = func != "_setup"
        for k, line in enumerate(lines):
            if 'status("Looking for answer...")' in line:
                after_status = True
            if after_status and line.strip() == "turn.done = True" \
                    and "done=True" not in "\n".join(lines[max(0, k - 4):k]):
                open_ends.append(f"{path.name} {func}: line {first + k}")
    assert open_ends == [], f"replies that leave 'Looking for answer...' open at lines {open_ends}"


# ── "documents", not "papers": uploads are Word, PowerPoint, Excel and images too ──
def test_listing_says_documents_in_the_library():
    from memento.query.flows.library_listing import listing_intro as _listing_intro
    from memento.config.domain_loader import load_domain
    assert _listing_intro(24, {}) == "The library currently contains **24** documents:\n\n"
    assert _listing_intro(1, {}) == "The library currently contains **1** document:\n\n"
    for name in ("computational-materials", "ceramics", "polymer-engineering", "generic"):
        dom = load_domain(name)
        assert dom["document_noun"] == {"singular": "document", "plural": "documents"}, name
        assert _listing_intro(3, dom) == "The library currently contains **3** documents:\n\n"


def test_user_facing_texts_do_not_say_paper():
    import re
    src = "".join(p.read_text() for p in [PROJECT_ROOT / "openwebui/functions/query_pipe.py",
                                           *sorted((PROJECT_ROOT / "memento/query").rglob("*.py"))])
    assert "for this document yet" in src and "for this paper yet" not in src
    assert 'nouns.get("singular", "paper")' not in src


def test_final_done_status_is_hidden():
    # Open WebUI hides the status line when its last event is hidden, so no grey "Done" stays above an answer
    from memento.query.presentation import status_event as _status_event
    assert _status_event("Done", True) == {"type": "status", "data": {"description": "Done", "done": True, "hidden": True}}
    assert _status_event("Looking for answer...", False) == {"type": "status", "data": {"description": "Looking for answer...", "done": False}}
    assert "hidden" not in _status_event("The cloud model stopped", True)["data"]     # other final notes stay visible


def test_images_attached_to_the_message_are_uploaded(tmp_path):
    # an image-only upload is still read: with vision on (needed so the chat box accepts images), Open WebUI moves
    # raster images out of the pipe's file list into the message; the upload is still in the user message's files, and
    # its path comes from Open WebUI's file table
    import asyncio
    from types import SimpleNamespace
    from memento.ingestion.upload.intake import message_images as _message_images
    img = tmp_path / "chart.png"; img.write_bytes(b"png")
    records = {"img1": SimpleNamespace(id="img1", user_id="u1", filename="chart.png", path=str(img), meta={"name": "chart.png"}),
               "other": SimpleNamespace(id="other", user_id="u2", filename="theirs.png", path=str(img), meta={})}

    async def lookup(fid):
        return records.get(fid)
    msg = {"user_message": {"files": [
        {"type": "file", "id": "img1", "name": "chart.png", "content_type": "image/png"},
        {"type": "file", "id": "pdf1", "name": "paper.pdf", "content_type": "application/pdf"},     # in __files__
        {"type": "file", "id": "other", "name": "theirs.png", "content_type": "image/png"},       # someone else's
        {"type": "file", "id": "doc1", "name": "notes.docx", "content_type": "application/msword"}]}}
    files = [{"id": "pdf1", "name": "paper.pdf", "file": {"path": "/x/paper.pdf"}}]
    got = asyncio.run(_message_images(msg, files, {"id": "u1"}, lookup))
    assert [(f["id"], f["name"], f["file"]["path"]) for f in got] == [("img1", "chart.png", str(img))]
    assert asyncio.run(_message_images({}, files, {"id": "u1"}, lookup)) == []
    # already in the file list (an OpenWebUI that keeps images there): not added twice
    assert asyncio.run(_message_images(msg, files + [{"id": "img1"}], {"id": "u1"}, lookup)) == []


def test_figure_under_an_answer_names_its_source():
    # figures under an answer get a caption with the citation number in the sources list's form "(5)" ("[5]" would
    # become a citation chip), markdown in file names escaped
    from memento.query.flows.figures import figure_markdown as _figure_markdown
    h = {"source": "memento_image_test", "page": 0, "type": "image", "section_raw": None}
    assert _figure_markdown(4, h, "data:image/png;base64,xx") == (
        "\n\n![memento_image_test p.1](data:image/png;base64,xx)\n\n*Figure (5): memento\\_image\\_test, p. 1*")


def test_figure_caption_shows_the_whole_section_heading():
    # pills keep the short tag; the caption under a figure has room for the full unnumbered heading
    from memento.query.flows.figures import figure_markdown as _figure_markdown
    h = {"source": "PolyBERT", "page": 1, "type": "image",
         "section_raw": "polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics"}
    assert _figure_markdown(3, h, "u").endswith(
        "*Figure (4): PolyBERT, p. 2, polyBERT: a chemical language model to enable fully machine-driven ultrafast "
        "polymer informatics*")
    h = {"source": "Doc", "page": 5, "type": "image", "section_raw": "2.4 | Results"}
    assert _figure_markdown(0, h, "u").endswith("*Figure (1): Doc, p. 6, §2.4*")


def test_per_paper_cap_applies_only_to_open_questions():
    # the soft per-document cap is for questions across the library; a named document, a figure request and "show the
    # section" are about one document on purpose
    from memento.query.search_step import balance_applies as _balance_applies
    assert _balance_applies(6, paper=None, image_intent=False, query_intent="ranked_retrieval")
    assert _balance_applies(6, paper=None, image_intent=False, query_intent="set_query")
    assert not _balance_applies(0, paper=None, image_intent=False, query_intent="ranked_retrieval")      # switched off
    assert not _balance_applies(6, paper="PolyBERT", image_intent=False, query_intent="ranked_retrieval")
    assert not _balance_applies(6, paper=None, image_intent=True, query_intent="image_request")
    assert not _balance_applies(6, paper=None, image_intent=False, query_intent="verbatim_section")


def test_thorough_check_runs_for_normal_answers_summaries_and_research_gaps():
    # with "Thorough check" on, summaries and research-gap answers get the statement check too
    from memento.query.checking import verify_applies as _verify_applies
    assert _verify_applies(True, thorough=True, query_intent="ranked_retrieval")
    assert _verify_applies(True, thorough=True, query_intent="summary")
    assert _verify_applies(True, thorough=True, query_intent="gap_query")
    assert not _verify_applies(True, thorough=False, query_intent="summary")
    assert not _verify_applies(False, thorough=True, query_intent="summary")          # VALIDATION_ENABLED off
    assert not _verify_applies(True, thorough=True, query_intent="corpus_listing")


def test_the_checks_re_searches_stay_in_the_document_for_summaries_and_gaps():
    from memento.query.scope.resolve import check_search_filter as _check_search_filter
    f = _check_search_filter("PolyBERT", {"busy_doc"})
    assert [c.key for c in f.must] == ["source"] and f.must[0].match.value == "PolyBERT"
    assert {c.key for c in f.must_not} == {"source", "section"}                       # mid-upload docs, Keywords lines
    f = _check_search_filter(None, set())                                              # a normal answer: the library
    assert not f.must and [c.key for c in f.must_not] == ["section"]


def test_a_checked_answer_is_stored_only_when_the_check_finished():
    from memento.query.cache import storable as _storable
    ok = dict(keyed=True, got_content=True, interrupted=False, note=None, by_chosen_model=True)
    assert _storable(**ok, verify=False, check_ok=False)                              # normal answer, as before
    assert _storable(**ok, verify=True, check_ok=True)                                # checked: replaces the stored one
    assert not _storable(**ok, verify=True, check_ok=False)                           # check timed out / failed
    assert not _storable(**{**ok, "interrupted": True}, verify=False, check_ok=False)
    assert not _storable(**{**ok, "note": "fallback"}, verify=False, check_ok=False)
    assert not _storable(**{**ok, "by_chosen_model": False}, verify=False, check_ok=False)
    assert not _storable(**{**ok, "keyed": False}, verify=True, check_ok=True)


_LIB = ["PolyBERT", "Polymer informatics with multi-task learning",
        "Impact of field cycling on HfO2 based non-volatile memory devices",
        "Domain Pinning Comparison of Hafnia and PZT Based Ferroelectrics",
        "Copolymer Informatics with Multi-Task Deep Neural Networks"]


def test_a_question_naming_two_documents_finds_both():
    # "compare A with B" resolves both documents and searches both
    from memento.query.scope.documents import named_documents as _named_documents
    dom = {"name": "test"}
    got = _named_documents("Compare the polyBERT paper with the paper Polymer informatics with multi-task learning.",
                           _LIB, dom)
    assert set(got) == {"PolyBERT", "Polymer informatics with multi-task learning"}
    got = _named_documents("How do the results of Impact of field cycling on HfO2 based non-volatile memory devices "
                           "differ from Domain Pinning Comparison of Hafnia and PZT Based Ferroelectrics?", _LIB, dom)
    assert set(got) == {_LIB[2], _LIB[3]}                        # "and" inside a title does not lose it
    got = _named_documents("Compare the field cycling paper and the domain pinning paper.", _LIB, dom)
    assert set(got) == {_LIB[2], _LIB[3]}


def test_one_or_no_named_document_is_not_a_set():
    from memento.query.scope.documents import named_documents as _named_documents
    dom = {"name": "test"}
    assert len(_named_documents("What does the polyBERT paper say about fingerprints?", _LIB, dom)) < 2
    assert _named_documents("How do wake-up and fatigue relate in hafnia?", _LIB, dom) == []
    assert len(_named_documents("Explain wake-up and fatigue in the field cycling paper.", _LIB, dom)) < 2


def test_each_named_document_gets_a_fair_share_of_the_passages():
    from memento.query.scope.documents import share_cap as _share_cap
    assert _share_cap(6, 20, 0) == 6 and _share_cap(6, 20, 1) == 6            # no set: the normal cap
    assert _share_cap(6, 20, 2) == 10 and _share_cap(6, 20, 3) == 7 and _share_cap(6, 20, 5) == 6
    assert _share_cap(0, 20, 2) == 10                                          # cap off: a named set still shares


def test_the_answer_instruction_comes_from_the_config_files():
    # no second copy of the instruction in code; a domain without one gets generic.yaml's
    from memento.query.prompts import answer_prompt as _answer_prompt, SYSTEM_PROMPT
    import yaml
    root = PROJECT_ROOT
    generic = " ".join(yaml.safe_load(open(root / "config/domains/generic.yaml"))["system_prompt"].split())
    from memento.query.prompts import PLAIN_FORMULAS
    assert " ".join(_answer_prompt({"system_prompt": "Domain own."}).split()) == "Domain own. " + PLAIN_FORMULAS
    assert " ".join(_answer_prompt({}).split()) == generic + " " + PLAIN_FORMULAS
    assert "polymer" not in SYSTEM_PROMPT.lower() and len(SYSTEM_PROMPT) < 300      # last resort, one short line


def test_thorough_replays_only_a_checked_answer():
    from memento.query.cache import replay_allowed as _replay_allowed
    assert _replay_allowed(checked=False, regenerated=False, thorough=False)        # normal question: any stored one
    assert _replay_allowed(checked=True, regenerated=False, thorough=True)          # already checked: instant
    assert not _replay_allowed(checked=False, regenerated=False, thorough=True)     # unchecked: write + check anew
    assert not _replay_allowed(checked=True, regenerated=True, thorough=False)      # Regenerate: always new


def test_a_background_check_uses_one_model_slot_at_a_time():
    # the warm-up's check must leave the local model's other slots to users (it runs 4 calls at once otherwise)
    import threading, time
    from concurrent.futures import ThreadPoolExecutor
    from memento.query.generation import GuardedLLM as _GuardedLLM
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    class Fake:
        def __init__(self):
            self.chat = self; self.completions = self

        def create(self, **kw):
            with lock:
                state["now"] += 1; state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return "ok"
    llm = _GuardedLLM(Fake(), threading.Event(), time.monotonic() + 60, slots=threading.Semaphore(1))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: llm.chat.completions.create(model="m"), range(8)))
    assert state["max"] == 1
    free = _GuardedLLM(Fake(), threading.Event(), time.monotonic() + 60)          # live: unchanged, parallel
    state.update(now=0, max=0)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: free.chat.completions.create(model="m"), range(8)))
    assert state["max"] > 1


def _grp(src, score, page=0, text="passage text"):
    return {"source": src, "page": page, "rerank_score": score, "text": text, "type": "text", "bbox": None,
            "section_raw": None}


def test_related_documents_lists_only_relevant_documents_the_answer_did_not_use():
    # documents the answer did not read but that have a relevant passage are listed under it (positives are checkable
    # through citations, negatives are not)
    from memento.query.presentation import related_documents as _related_documents
    # thresholds from live scores: relevant unused documents scored at least ~12 % of the best passage
    heads = [_grp("A", 0.80), _grp("B", 0.40, 3), _grp("C", 0.11), _grp("D", 0.06), _grp("E", 0.095)]
    kw = dict(used={"A"}, nothing_found=False, top_score=0.8, ratio=0.12, min_score=0.05, near=3)
    assert [h["source"] for h in _related_documents(heads, limit=5, **kw)] == ["B", "C"]   # D, E too weak; A used
    assert [h["source"] for h in _related_documents(heads, limit=1, **kw)] == ["B"]
    low = [_grp("X", 0.3), _grp("Y", 0.06)]                                       # a low best score: the floor holds
    assert [h["source"] for h in _related_documents(low, used=set(), nothing_found=False, top_score=0.3, ratio=0.12,
                                                     min_score=0.07, limit=5, near=3)] == ["X"]


def test_a_not_found_answer_shows_the_closest_passages_of_any_document():
    from memento.query.presentation import related_documents as _related_documents
    from memento.query.presentation import nothing_found as _nothing_found
    heads = [_grp("A", 0.09), _grp("B", 0.06), _grp("C", 0.055), _grp("D", 0.01)]
    got = _related_documents(heads, used={"A"}, nothing_found=True, top_score=0.09, ratio=0.12, min_score=0.05,
                             limit=5, near=3)
    assert [h["source"] for h in got] == ["A", "B", "C"]                         # the near-misses, used or not
    # "closest" passages near 0.004 are contact details and e-mail lists: below the floor, none listed
    junk = [_grp("A", 0.004), _grp("B", 0.003)]
    assert _related_documents(junk, used=set(), nothing_found=True, top_score=0.007, ratio=0.12, min_score=0.05,
                              limit=5, near=3) == []
    # nothing found = the answer cites nothing, or the best passage scores low (live: unanswerable 0.007 / 0.042,
    # answered 0.68–0.79), whatever language the answer is written in
    assert _nothing_found(cited=set(), top_score=0.7, low=0.1)
    assert _nothing_found(cited={0, 1}, top_score=0.007, low=0.1)
    assert not _nothing_found(cited={0}, top_score=0.68, low=0.1)


def test_formulas_are_asked_for_as_plain_text():
    # LaTeX like "HfO$_2$" breaks Open WebUI's maths rendering: formulas are asked for as plain text
    from memento.query.prompts import answer_prompt as _answer_prompt, PLAIN_FORMULAS
    assert "$" in PLAIN_FORMULAS and "HfO₂" in PLAIN_FORMULAS and "μC cm⁻²" in PLAIN_FORMULAS
    assert _answer_prompt({"system_prompt": "X"}).endswith(PLAIN_FORMULAS)


def test_no_figures_under_an_answer_that_found_nothing():
    # an answer that says "not covered" shows no figures underneath (they are cited only to say what the library does
    # cover)
    from memento.query.flows.figures import show_figures as _show_figures
    hits = [{"rerank_score": 0.007}, {"rerank_score": 0.004}]
    assert not _show_figures(hits, cited={0, 1}, low=0.1)
    assert not _show_figures([{"rerank_score": 0.7}], cited=set(), low=0.1)
    assert _show_figures([{"rerank_score": 0.7}, {"rerank_score": 0.02}], cited={1}, low=0.1)


def test_related_block_wording_links_and_history():
    # worded from separate facts and the documents the search reached (tests/query/test_related_footer.py)
    from memento.query.scope.conversation import history_text as _history_text
    from memento.query.presentation import related_footer as _related_footer
    items = [_grp("Doc B", 0.4, 2, "Wake-up was observed after 1e4 cycles in all samples."), _grp("Doc C", 0.3)]
    titles = {"Doc B": "Title B", "Doc C": "Title C"}
    f = lambda **kw: _related_footer(**dict(dict(items=items, titles=titles, searched=26,
                                                 files_url="http://f/files"), **kw))
    md = f(cited=True, weak=False)
    assert md.count('<details type="memento_related">') == 1 and "Other documents with related passages (2)" in md
    import re as _re
    assert "Title B, p. 3" in md and "http://f/files/view/" in md
    assert not _re.search(r"\[\d+\]", md)                                          # no [N]: it would become a pill
    md2 = f(cited=False, weak=True)
    assert "Searched 26 documents" in md2 and "closest passages" in md2.lower()
    assert "Wake-up was observed after 1e4 cycles" in md2                         # a short quote shows why it is close
    assert not _re.search(r"\[\d+\]", md2)
    assert "—" not in md and "—" not in md2                                       # no em dashes in user text
    assert "Title B" not in _history_text("The answer [1]." + md)                 # not context for the next turn
    assert f(items=[], cited=True, weak=False) == ""
    none = f(items=[], cited=False, weak=True)                                    # nothing close at all
    assert "Searched 26 documents" in none and "no passage came close" in none
    assert "Title" not in _history_text("Not covered." + none)


def test_related_documents_only_for_questions_across_the_library():
    from memento.query.presentation import related_applies as _related_applies
    base = dict(paper=None, named_docs=[], image_intent=False, whole_doc=False, query_intent="ranked_retrieval")
    assert _related_applies(True, **base)
    assert not _related_applies(False, **base)                                       # valve off
    assert not _related_applies(True, **{**base, "paper": "PolyBERT"})
    assert not _related_applies(True, **{**base, "named_docs": ["A", "B"]})
    assert not _related_applies(True, **{**base, "image_intent": True})
    assert not _related_applies(True, **{**base, "whole_doc": True})
    assert not _related_applies(True, **{**base, "query_intent": "verbatim_section"})


def _src(name, text):
    return {"document": [text], "metadata": [{"source": f"{name}#0", "name": f"[x] {name}"}], "source": {"name": name}}


def test_rework_prompt_carries_the_previous_answer_and_its_numbered_sources():
    # a rework reshapes the answer just read, from its own sources, with the same [N], instead of a fresh search
    from memento.query.flows.rework import rework_messages as _rework_messages
    sources = [_src("PolyBERT (p. 2)", "polyBERT is 215 times faster."), _src("Doc B (p. 5)", "B text.")]
    msgs = _rework_messages("System.", "polyBERT is 215× faster [1]. B says X [2].", sources,
                            "Summarize that in one sentence.")
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("System.")
    assert "add no new facts" in msgs[0]["content"].lower() and "same numbers" in msgs[0]["content"].lower()
    # a summary of several files must not link files that do not refer to each other
    assert "do not link them" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1] PolyBERT (p. 2)\npolyBERT is 215 times faster." in user and "[2] Doc B (p. 5)\nB text." in user
    assert "Previous answer:\npolyBERT is 215× faster [1]. B says X [2]." in user
    assert user.rstrip().endswith("Request: Summarize that in one sentence.")
    no_src = _rework_messages("S.", "A plain list.", [], "As a table.")[1]["content"]
    assert "Sources:" not in no_src and "Previous answer:\nA plain list." in no_src


def test_rework_sends_only_the_cited_sources_with_their_numbers():
    # a rework sends only the sources the answer cites, so a summary of several whole files stays under REWORK_MAX_CHARS
    from memento.query.flows.rework import rework_messages as _rework_messages, rework_size as _rework_size
    sources = [_src("A (p. 1)", "a" * 50), _src("B (p. 2)", "b" * 50000), _src("C (p. 3)", "c" * 50)]
    previous = "First point [1]. Third point [3]."
    user = _rework_messages("S.", previous, sources, "Shorter.")[1]["content"]
    assert "[1] A (p. 1)" in user and "[3] C (p. 3)" in user and "[2] B" not in user and "bbbb" not in user
    assert _rework_size(previous, sources) < 1000                 # the uncited 50,000 characters do not count


def test_previous_answer_for_rework_is_the_text_the_user_read():
    from memento.query.flows.rework import previous_answer as _previous_answer
    chat = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>x</think>The answer [1].\n\n_Answered by Qwen 3.8 27B (Local)._\n\n"
                                             '<details type="memento_sources">\n<summary>Cited</summary>\n\n- (1) Doc\n</details>\n'
                                             '<details type="memento_related">\n<summary>Other</summary>\n\n- B\n</details>\n'},
            {"role": "user", "content": "Summarize that."}]
    assert _previous_answer(chat) == "The answer [1]."                           # [N] kept, notes and blocks gone
    assert _previous_answer([{"role": "user", "content": "first question"}]) == ""


def test_rework_sources_marker_uses_the_stored_names():
    from memento.query.presentation import names_marker as _names_marker
    md = _names_marker(["PolyBERT (p. 2, §3)", "Doc B (p. 5)"], {0}, docs=["polybert.pdf", "docb.pdf"])
    # one cited document: named once, page and section per line
    assert "1 passage of this document" in md and "All from **PolyBERT**." in md and "- (1) p. 2, §3" in md
    # without document identities, the names alone do not decide it
    assert "1 of the 2 sources" in _names_marker(["PolyBERT (p. 2, §3)", "Doc B (p. 5)"], {0})
    md = _names_marker(["PolyBERT (p. 2, §3)", "Doc B (p. 5)"], {0, 1})
    assert "all 2 sources" in md and "(1) PolyBERT, p. 2, §3" in md


def test_router_knows_rework_requests():
    from memento.query import router as qr
    assert "rework" in qr._VALID_INTENTS and qr.pipe_intent("rework") == "rework"
    p = qr.ROUTER_SYSTEM_PROMPT
    assert '"rework"' in p and "as a table" in p and "tell me more" in p.lower()      # and what is NOT rework


def test_the_router_sees_enough_of_the_previous_answer():
    # "tell me more about the second one" resolved because the list started early; later items need more context
    from memento.query.request import ROUTER_TURN_CHARS
    assert ROUTER_TURN_CHARS >= 2500


def test_every_image_label_in_the_pipe_starts_on_its_own_line():
    # the chat template joins text parts without a separator; all three answer paths (one answer, one part per file,
    # Document Chat) build their labels with the same helper
    import re
    import openwebui.functions.query_pipe as qp
    src = (PROJECT_ROOT / "openwebui/functions/query_pipe.py").read_text()
    assert presentation.image_label(3) == "\n\nImage for Source [3]:"
    assert not re.search(r'f"Image for Source', src)


def test_upload_hints_no_longer_offer_images():
    # image uploads are off; the hints name only the document types
    import openwebui.functions.query_pipe as qp
    for text in (texts.DOC_CHAT_NO_FILE_MSG, texts.DOC_CHAT_FILES_NONE, general.GENERAL_SYSTEM_PROMPT):
        assert "image" not in text.lower(), text[:80]


def test_icons_and_title_page_images_are_not_shown_unless_a_figure_is_asked_for(tmp_path):
    # a page-1 image (journal icons, an uncaptioned graphical abstract) is not shown under an author question: its
    # stored text is the page text around it
    import openwebui.functions.query_pipe as qp
    from PIL import Image
    png = tmp_path / "f.png"
    Image.new("RGB", (400, 300), "white").save(png)
    fig = {"type": "chart", "img_path": str(png), "page": 4, "bbox": [100, 100, 600, 500]}
    assert figures.shown_figure(fig)
    assert not figures.shown_figure({**fig, "bbox": [449, 241, 472, 256]})                     # an icon
    assert not figures.shown_figure({**fig, "page": 0})                                         # title page
    assert figures.shown_figure({**fig, "page": 0}, figure_asked=True)                          # unless asked for
    assert not figures.shown_figure({**fig, "bbox": [449, 241, 472, 256]}, figure_asked=True)   # icons never
    assert figures.shown_figure({**fig, "bbox": None})                                          # no box: size unknown


def test_every_domain_asks_for_lists_and_titles_when_documents_are_asked_for():
    # a question asking for several documents gets a list with the documents' titles, not one long sentence
    import yaml
    for f in sorted((PROJECT_ROOT / "config/domains").glob("*.yaml")):
        p = yaml.safe_load(f.read_text())["system_prompt"]
        assert "bulleted list" in p and "name each by its title" in p, f.name


def test_a_figure_request_shows_the_figures_it_cites_even_with_low_scores():
    # a figure request shows its cited figure even though figures score low against such a question
    from memento.query.flows.figures import show_figures as _show_figures
    hits = [{"rerank_score": 0.094}, {"rerank_score": 0.06}]
    assert _show_figures(hits, cited={0}, low=0.1, figure_asked=True)
    assert not _show_figures(hits, cited={0}, low=0.1)                      # other questions keep the guard
    assert not _show_figures(hits, cited=set(), low=0.1, figure_asked=True)  # nothing cited: nothing to show


def test_a_figure_request_shows_the_cited_figures_after_the_first_paragraph():
    # a figure request brings the figure right below the first paragraph
    from memento.query.flows.figures import first_paragraph_figures as _first_paragraph_figures
    text = "Yes, **Figure 1** shows the pipeline [1].\n\nIt compares two strategies [1][3]."
    assert _first_paragraph_figures("Yes, **Figure 1** shows the pipe") is None          # paragraph not finished yet
    assert _first_paragraph_figures(text) == {0}                                         # only what it cites
    assert _first_paragraph_figures("\n\nYes, Figure 2 [2].\n\nMore [1].") == {1}       # leading blank lines skipped
    assert _first_paragraph_figures("Figure 1 is the pipeline.\n\nIt shows [1].") == set()  # first paragraph cites nothing


# ── typing out a finished answer: small pieces, so stored answers read like live ones ──
def test_a_finished_answer_is_cut_into_small_pieces_but_blocks_and_figures_stay_whole():
    from memento.query.presentation import _type_pieces
    text = ("Intro line [1].\n\n<details type=\"memento_sources\">\n<summary>Sources</summary>\n\n- a\n</details>\n\n"
            "More text.\n\n![Doc p.2](data:image/png;base64,AAAA)\n\n*Figure (1): Doc*")
    pieces = _type_pieces(text)
    assert "".join(pieces) == text
    assert any(p.startswith("<details") and p.endswith("</details>") for p in pieces)
    assert "![Doc p.2](data:image/png;base64,AAAA)" in pieces
    assert max(len(p) for p in pieces if not p.startswith(("<details", "!["))) <= 4


def test_typing_out_takes_about_as_long_as_a_live_answer_and_can_be_capped():
    import asyncio
    from memento.query.presentation import type_out as _typed
    waits = []

    async def sleep(s):
        waits.append(s)

    async def run(text, rate, cap):
        waits.clear()
        return "".join([p async for p in _typed(text, rate, cap, sleep=sleep)])
    text = "x" * 2000 + "\n\n<details><summary>S</summary>\n\nlong block " + "y" * 3000 + "\n</details>"
    assert asyncio.run(run(text, 200, 0)) == text
    assert abs(sum(waits) - 2000 / 200) < 0.05                       # 200 chars/s; the block itself costs no time
    asyncio.run(run(text, 200, 4))
    assert abs(sum(waits) - 4) < 0.05                                 # capped at 4 s
    assert asyncio.run(run(text, 0, 0)) == text and not waits         # rate 0: at once, as before


def test_a_long_answer_types_out_its_start_and_sends_the_rest_at_once():
    # a long report is typed out only at its start; the rest arrives at once
    import asyncio
    from memento.query.presentation import type_out as _typed
    waits = []

    async def sleep(s):
        waits.append(s)

    async def run(text, limit):
        waits.clear()
        return [p async for p in _typed(text, 200, 0, limit, sleep=sleep)]
    text = "x" * 10000
    pieces = asyncio.run(run(text, 4000))
    assert "".join(pieces) == text and abs(sum(waits) - 4000 / 200) < 0.05 and pieces[-1] == "x" * 6000
    assert abs(sum(asyncio.run(run(text, 0)) and waits) - 10000 / 200) < 0.05         # 0: the whole answer typed
    short = asyncio.run(run("y" * 300, 4000))
    assert "".join(short) == "y" * 300 and abs(sum(waits) - 300 / 200) < 0.05


def test_a_replayed_answer_is_typed_out():
    import asyncio
    from memento.query.cache import replay as _replay
    events = []

    async def emit(e):
        events.append(e)

    async def run():
        return [p async for p in _replay({"citations": [{"type": "citation"}], "answer": "abcdefghij"}, emit,
                                         rate=1000, max_s=0)]
    pieces = asyncio.run(run())
    assert "".join(pieces) == "abcdefghij" and len(pieces) > 1 and events == [{"type": "citation"}]


# ── which document a gap question is about ("suggest future work for this study") ──
def _cited(source, i):
    return {"source": {"name": f"[{i}] {source}"}, "metadata": [{"source": f"{source}#{i}", "page": 1}]}


def test_conversation_document_needs_a_clear_majority():
    from memento.query.scope.conversation import _conversation_document
    lib = {"Thesis A", "Paper B"}
    assert _conversation_document([_cited("Thesis A", i) for i in range(5)] + [_cited("Paper B", 5)], lib) == "Thesis A"
    assert _conversation_document([_cited("Thesis A", i) for i in range(3)]
                                  + [_cited("Paper B", i) for i in range(3, 6)], lib) is None      # no majority
    assert _conversation_document([_cited("Thesis A", 0)], lib) is None                            # one citation
    assert _conversation_document([_cited("Gone", i) for i in range(4)] + [_cited("Paper B", 4)], lib) is None
    assert _conversation_document(None, lib) is None and _conversation_document([], lib) is None


def test_gap_target_prefers_the_named_document_then_the_conversation(monkeypatch):
    # the conversation's document is the one the previous answer cited, its question agreeing
    # (tests/query/test_this_document.py)
    import asyncio
    import openwebui.functions.query_pipe as qp
    p = qp.Pipe()
    stored = [{"source": {"name": "Thesis A (p. 1)"},
               "metadata": [{"source": f"Thesis A#{i}", "name": f"[{i + 1}] Thesis A (p. 1)", "page": 1}]}
              for i in range(4)]
    previous = ["Summarize the thesis", "The thesis finds X [1][2]."]

    async def previous_turn(meta):
        return (previous[0], previous[1], stored) if previous[1] else None
    monkeypatch.setattr(conversation, "stored_previous_turn", previous_turn)
    monkeypatch.setattr(documents, "_resolve_paper", lambda q, s, d: "Paper B" if "polyBERT" in q else None)
    lib = ["Thesis A", "Paper B"]
    run = lambda paper, q: asyncio.run(gap_answer.gap_target(p.valves, paper, q, lib, {}, {"chat_id": "c"}))
    assert run("Picked", "anything") == "Picked"
    assert run(None, "future work of the polyBERT paper") == "Paper B"
    assert run(None, "suggest future work for this study") == "Thesis A"
    previous[1] = ""
    assert run(None, "suggest future work for this study") is None


def test_the_no_questions_reply_points_to_the_limitations_button():
    import openwebui.functions.query_pipe as qp
    assert "Limitations & future work" in gap_report.GAP_REPORT_NONE_MSG


def test_stored_gap_answers_never_keep_the_suggestions():
    # the suggestions are stored on their own (action gaps_ext); the stored Limitations answer is the authors' part
    from memento.query.scope.conversation import without_suggestions as _without_suggestions
    authors = "| What the authors state | Points |\n|---|---|\n| Limitations | 1 |\n\n<details type=\"memento_gaps\">\n" \
              "<summary>Limitations (1)</summary>\n\n1. A [1].\n\n</details>\n"
    ideas = ("\n\n<details type=\"memento_ideas\">\n<summary>Possible extensions (suggested by the AI)</summary>\n\n"
             "_Suggested by the AI, not stated by the authors. Please verify._\n\n1. B [1].\n\n</details>\n\n")
    assert _without_suggestions(authors + ideas + "\n_Answered by X._") == authors + "\n_Answered by X._"
    assert _without_suggestions(authors + "\n\n_Suggestions could not be written this time._\n\n") .strip() == authors.strip()
    assert _without_suggestions(authors) == authors


def test_the_authors_part_is_always_asked_with_the_neutral_question():
    # the reader's wording goes to the suggestions only: in the gap prompt it made the model write its own ideas as a
    # plain list
    from memento.query.scope.conversation import gap_question as _gap_question
    en = _gap_question("Can you suggest future work for this study to expand on what was done?")
    assert en == "Which limitations, uncertainties, untested conditions and future work do the authors state?"
    de = _gap_question("Was könnte man darauf aufbauend untersuchen?")
    assert de == "Welche Einschränkungen, Unsicherheiten, ungetesteten Bedingungen und zukünftigen Arbeiten nennen die " \
                 "Autoren?"
    picker = 'What research gaps does the document "polyBERT: a chemical language model" name?'
    assert _gap_question(picker) == picker                      # the start-screen button's own question: unchanged


# ── the passage types an element request searches (a table request searches tables) ──
@pytest.mark.parametrize("element, image_intent, query, want", [
    ("table", False, "results table of the topology paper", ["table"]),
    ("table", True, "table of final results", ["table"]),                 # the router said figure AND table: table
    ("equation", False, "the loss equation", ["equation"]),
    ("figure", True, "pipeline figure", ["image", "chart"]),
    ("none", True, "pipeline figure", ["image", "chart"]),               # an older reply without the field: unchanged
    ("none", True, "table of final results in the paper", ["image", "chart", "table"]),   # the safety net
    ("none", True, "Tabelle der Ergebnisse", ["image", "chart", "table"]),
    ("none", False, "what are the final results", None),                 # a normal question: no type filter
    ("figure", False, "what does figure 3 show", None),                   # a normal question naming a figure: unchanged
])
def test_element_passage_types(element, image_intent, query, want):
    from memento.query.search_step import passage_types as _element_types
    assert _element_types(element, image_intent, query) == want


@pytest.mark.parametrize("element, intent, want", [("table", "image_request", False), ("equation", "image_request", False),
                                                   ("figure", "image_request", True), ("none", "image_request", True),
                                                   ("figure", "ranked_retrieval", False), ("none", "ranked_retrieval", False)])
def test_only_figures_are_figure_requests(element, intent, want):
    # a figure request shows the figures it cites; a table or equation request must not
    from memento.query.flows.figures import figure_request as _figure_request
    assert _figure_request(intent, element) is want


def test_table_and_equation_requests_add_their_passages_to_the_normal_search():
    # table and equation passages are added to the normal search (the text often holds the answer); figure requests keep
    # their figure-only search
    from types import SimpleNamespace as P
    from memento.query.search_step import element_union as _element_union, merge_points as _merge_points
    assert _element_union("table") and _element_union("equation")
    assert not _element_union("figure") and not _element_union("none")
    a, b = [P(id=1), P(id=2)], [P(id=2), P(id=3)]
    assert [p.id for p in _merge_points(a, b)] == [1, 2, 3] and _merge_points([], []) == []
