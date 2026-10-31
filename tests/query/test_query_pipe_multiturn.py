# Multi-turn answers and attachment handling.
import asyncio
from pathlib import Path
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import library_state, presentation
from memento.query.scope import conversation

qp = load_function(PROJECT_ROOT / "openwebui/functions/query_pipe.py", "memento_query")

HITS = [{"source": "Diss_Grueber", "page": 11}, {"source": "openalex_W123", "page": 0}, {"source": "Talk--X>y", "page": 4}]


def test_marker_is_a_collapsed_details_block_listing_only_cited_sources():
    m = presentation.sources_marker(HITS, {0, 2})
    # OpenWebUI's details tokenizer needs "<details ...>\n<summary>...</summary>\n" at the start of a block
    assert m.startswith('\n\n<details type="memento_sources">\n<summary>Cited in this answer: 2 of the ')
    assert m.rstrip().endswith("</details>") and m.count("</details>") == 1
    assert "- (1) Diss_Grueber, p. 12" in m and "- (3) Talk--X y, p. 5" in m and "openalex_W123" not in m
    assert presentation.sources_marker(HITS, set()) == ""


def test_history_text_names_stale_citations_and_drops_the_marker():
    answer = "Density was 3.9 g/cm3 [1], hardness 18 GPa [3]; see also [2]." + presentation.sources_marker(HITS, {0, 2})
    h = conversation.history_text(answer)
    assert "details" not in h and "Sources in this answer" not in h and "[1]" not in h and "[3]" not in h
    assert "(Diss_Grueber, p. 12)" in h and "(Talk--X y, p. 5)" in h
    assert "see also ." in h                             # an uncited/unknown [2] is dropped, not kept as a stale number


def test_history_text_without_marker_removes_bare_numbers():
    assert conversation.history_text("It was 5 MPa [4][5].") == "It was 5 MPa ."


class _Emitter:
    def __init__(self):
        self.events = []

    async def __call__(self, e):
        self.events.append(e)


def _run(pipe, body, files):
    async def go():
        out = []
        async for chunk in pipe.pipe(body, {"role": "user"}, _Emitter(), __files__=files):
            out.append(chunk)
            if len(out) >= 2:
                break
        return out
    return asyncio.run(go())


def test_attachment_template_is_not_mistaken_for_a_background_task(monkeypatch):
    pipe = qp.Pipe()
    pipe.valves.ENABLE_QUERY_REWRITE = False
    # stop right after the notice: the first retrieval step raising ends the generator early in the test
    monkeypatch.setattr(library_state, "load_documents", lambda *a: (_ for _ in ()).throw(RuntimeError("stop")))
    tpl = ("### Task:\nRespond to the user query using the provided context...\n<context>file text</context>\n"
           "<user_query>What is the density of A-17?</user_query>")
    out = _run(pipe, {"messages": [{"role": "user", "content": tpl}]}, [{"id": "f1", "name": "x.pdf"}])
    assert "can't read attachments directly" in out[0]


def test_citation_links_use_the_unified_view():
    u = presentation.source_file_url("http://localhost:8024/", "Diss X", 3, [[1, 2, 3, 4]], text="a b", section="2 Methods")
    assert u == "http://localhost:8024/view/Diss%20X/3?bbox=1,2,3,4&highlight=a%20b&section=2%20Methods"
    assert presentation.source_file_url("", "x", 0) is None


# ── [N] of earlier answers from Open WebUI's stored citations ──
def test_stored_citations_give_the_n_to_document_mapping():
    stored = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "sources": [
            {"source": {"name": "Becker 2026 (p. 6, §2.4.2)"}}, {"source": {"name": "Rao 2025 (pp. 2–3)"}}]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2 (no sources: chit-chat)"},
    ]
    names = presentation._names_per_answer(stored)
    assert names == [{"1": "Becker 2026, p. 6, §2.4.2", "2": "Rao 2025, pp. 2–3"}, {}]
    h = conversation.history_text("The MMD is a histogram [1]; the dataset has 771 points [2].", names[0])
    assert h == "The MMD is a histogram (Becker 2026, p. 6, §2.4.2); the dataset has 771 points (Rao 2025, pp. 2–3)."


def test_sources_list_uses_round_brackets_and_old_lists_still_resolve():
    m = presentation.sources_marker([{"source": "Becker 2026", "page": 5, "section_raw": "2.4.2 | MMD"}], {0})
    # one document: named once ("All from"), the line has page and section
    assert "All from **Becker 2026**." in m and "- (1) p. 6, §2.4.2" in m and "[1]" not in m
    assert conversation.history_text("It is a histogram [1]." + m, {}) == "It is a histogram (Becker 2026, p. 6, §2.4.2)."
    old = "Density was 3.9 g/cm3 [1]." + presentation.sources_marker(HITS, {0})               # an older stored form
    assert "memento_sources" not in conversation.history_text(old, {}) and "[1]" not in conversation.history_text(old, {})


def test_no_chat_id_means_no_stored_names():
    import asyncio
    assert asyncio.run(presentation.stored_citation_names({})) == []
    assert asyncio.run(presentation.stored_citation_names({"chat_id": "local:abc"})) == []


def test_cited_list_says_how_it_relates_to_openwebuis_sources_button():
    hits = [{"source": f"D{i}", "page": i} for i in range(20)]
    m = presentation.sources_marker(hits, {2, 4})
    assert "<summary>Cited in this answer: 2 of the 20 sources Memento read</summary>" in m
    # Open WebUI's own "Sources" button is hidden: no pointer to it
    assert "listed under" not in m and "- (3) D2, p. 3" in m
    assert conversation.history_text("x [3]." + m, {}) == "x (D2, p. 3)."                  # still parsed for follow-ups


def test_cited_list_heading_is_grammatical():
    # one source: no "1 of the 1 sources"
    one = [{"source": "D0", "page": 0}]
    # one cited document: its passages counted
    assert "<summary>Cited in this answer: 1 passage of this document</summary>" in presentation.sources_marker(one, {0})
    five = [{"source": f"D{i}", "page": i} for i in range(5)]
    assert "<summary>Cited in this answer: all 5 sources Memento read</summary>" in presentation.sources_marker(five, set(range(5)))
    assert "<summary>Cited in this answer: 2 of the 5 sources Memento read</summary>" in presentation.sources_marker(five, {1, 3})
    assert "<summary>Cited in this answer: 1 passage of this document</summary>" in presentation.sources_marker(five, {1})
