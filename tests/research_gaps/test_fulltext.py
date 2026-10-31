# Full texts of papers held offline, for the literature check. Every file here is generated; no publisher text.
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from memento.research_gaps import fulltext as ft
from memento.research_gaps import jobs as gj

SENTENCE = "The measured tensile modulus of the blend rose with filler content under dry conditions. "
BODY = "\n\n".join(f"Paragraph {k} about sample {k}. " + SENTENCE * 8 for k in range(12))


def index(path: Path, rows: list[tuple]) -> Path:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE papers (doi TEXT PRIMARY KEY, path TEXT)")
        db.executemany("INSERT INTO papers VALUES (?, ?)", rows)
    return path


def make_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    for start in range(0, len(text), 2400):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), text[start:start + 2400], fontsize=9)
    doc.save(path)
    doc.close()


def work(doi="10.9999/abc", abstract="An abstract about something else entirely.", **kw):
    return SimpleNamespace(title="A paper", year=2024, doi=doi, abstract=abstract, url="u", source="openalex",
                           work_id="w1", excerpts=[], **kw)


# ── switch and lookup ─────────────────────────────────────────

def test_off_by_default_and_needs_the_database_and_the_lookup(tmp_path, monkeypatch):
    idx = index(tmp_path / "paths.sqlite", [])
    monkeypatch.delenv("LITERATURE_FULLTEXT", raising=False)
    monkeypatch.setenv("ALEXANDRIA_DB", str(tmp_path))
    assert not ft.enabled(index=idx)
    monkeypatch.setenv("LITERATURE_FULLTEXT", "1")
    assert ft.enabled(index=idx)
    assert not ft.enabled(index=tmp_path / "missing.sqlite")
    monkeypatch.delenv("ALEXANDRIA_DB")
    assert not ft.enabled(index=idx)


@pytest.mark.parametrize("doi", ["10.9999/ABC", "https://doi.org/10.9999/abc", "doi:10.9999/Abc"])
def test_paper_folder_matches_doi_spellings(tmp_path, doi):
    (tmp_path / "pub/issn/one").mkdir(parents=True)
    idx = index(tmp_path / "paths.sqlite", [("10.9999/abc", "pub/issn/one")])
    assert ft.paper_folder(doi, index=idx, root=tmp_path) == tmp_path / "pub/issn/one"


def test_paper_folder_none_when_not_held_or_gone(tmp_path):
    idx = index(tmp_path / "paths.sqlite", [("10.9999/gone", "pub/issn/gone")])
    assert ft.paper_folder("10.9999/other", index=idx, root=tmp_path) is None
    assert ft.paper_folder("10.9999/gone", index=idx, root=tmp_path) is None
    assert ft.paper_folder("", index=idx, root=tmp_path) is None


def test_paper_folder_stays_inside_the_database(tmp_path):
    (tmp_path / "db").mkdir()
    (tmp_path / "outside").mkdir()
    idx = index(tmp_path / "paths.sqlite", [("10.9999/x", "../outside")])
    assert ft.paper_folder("10.9999/x", index=idx, root=tmp_path / "db") is None


# ── reading a paper ───────────────────────────────────────────

def test_markdown_comes_first_and_references_are_cut(tmp_path):
    (tmp_path / "paper.md").write_text(BODY + "\n\n# References\n\n[1] Someone. A cited title. 2020.\n")
    make_pdf(tmp_path / "Paper.pdf", "pdf text " * 900)
    kind, body = ft.read_body(tmp_path)
    assert kind == "md" and "Paragraph 3" in body and "A cited title" not in body


def test_a_word_references_early_in_the_text_is_not_a_cut(tmp_path):
    (tmp_path / "paper.md").write_text("References\n\n" + BODY)
    assert "Paragraph 11" in ft.read_body(tmp_path)[1]


def test_pdf_text_layer_when_no_markdown(tmp_path):
    make_pdf(tmp_path / "Paper.pdf", BODY)
    kind, body = ft.read_body(tmp_path)
    assert kind == "pdf" and "tensile modulus" in body


def test_pdfs_are_read_one_thread_at_a_time(tmp_path, monkeypatch):
    """PyMuPDF is not thread-safe, and questions are checked side by side."""
    state = {"now": 0, "most": 0}
    guard = threading.Lock()

    class Counting:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            with guard:
                state["now"] += 1
                state["most"] = max(state["most"], state["now"])
            time.sleep(0.01)
            return []

        def __exit__(self, *a):
            with guard:
                state["now"] -= 1
    monkeypatch.setattr(fitz, "open", Counting)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: ft._pdf(tmp_path), range(8)))
    assert state["most"] == 1


def test_xml_reads_body_paragraphs_only(tmp_path):
    paras = "".join(f"<p>{SENTENCE * 8}</p>" for _ in range(12))
    (tmp_path / "ab1c.xml").write_text(f"<article><front><abstract><p>Short.</p></abstract></front><body><sec>{paras}"
                                       "</sec></body><back><ref-list><p>A cited title</p></ref-list></back></article>")
    kind, body = ft.read_body(tmp_path)
    assert kind == "xml" and "tensile modulus" in body and "A cited title" not in body and "Short." not in body


def test_xml_with_namespaces_and_para_elements(tmp_path):
    paras = "".join(f"<ce:para>{SENTENCE * 8}</ce:para>" for _ in range(12))
    (tmp_path / "Paper.xml").write_text('<response xmlns:ce="http://example.org/ce"><body><ce:sections>'
                                        f"{paras}</ce:sections></body></response>")
    assert ft.read_body(tmp_path)[0] == "xml"


def test_a_file_without_a_body_is_not_full_text(tmp_path):
    (tmp_path / "Paper.xml").write_text("<article><front><title>Only metadata</title></front></article>")
    doc = fitz.open()
    doc.new_page()
    doc.save(tmp_path / "Paper.pdf")
    doc.close()
    assert ft.read_body(tmp_path) is None


def test_broken_files_fall_through_to_the_next_kind(tmp_path):
    (tmp_path / "Paper.pdf").write_bytes(b"not a pdf")
    paras = "".join(f"<p>{SENTENCE * 8}</p>" for _ in range(12))
    (tmp_path / "Paper.xml").write_text(f"<article><body>{paras}</body></article>")
    assert ft.read_body(tmp_path)[0] == "xml"
    (tmp_path / "Paper.xml").write_text("<article><body>")
    assert ft.read_body(tmp_path) is None


def test_reading_writes_nothing_into_the_folder(tmp_path):
    (tmp_path / "paper.md").write_text(BODY)
    before = sorted(os.listdir(tmp_path))
    tmp_path.chmod(0o555)
    try:
        assert ft.read_body(tmp_path)[0] == "md"
    finally:
        tmp_path.chmod(0o755)
    assert sorted(os.listdir(tmp_path)) == before


# ── passages ──────────────────────────────────────────────────

def test_passages_stay_within_the_limit_and_keep_offsets():
    body = "\n\n".join(["Short one."] * 3 + [SENTENCE * 30])
    spans = ft.passages(body, chars=500)
    assert spans and all(b - a <= 500 for a, b in spans)
    assert "Short one." in body[spans[0][0]:spans[0][1]]
    assert sum(b - a for a, b in spans) >= 0.9 * len(body)


def test_a_long_paragraph_splits_at_a_sentence_end():
    spans = ft.passages(SENTENCE * 30, chars=500)
    assert len(spans) > 1
    assert SENTENCE[:-1].endswith(".") and (SENTENCE * 30)[spans[0][0]:spans[0][1]].rstrip().endswith(".")


def test_a_paragraph_without_sentences_still_splits():
    spans = ft.passages("x" * 3000, chars=500)
    assert len(spans) == 6 and all(b - a <= 500 for a, b in spans)


def test_best_passages_prefilter_rerank_and_skip_copies_of_the_abstract():
    target = "Crack growth in humid air doubled the fatigue damage of the polyamide samples after aging. " * 9
    abstract = "We review processing of glass fibre composites and report their mechanical air properties in detail. " * 9
    other = "Unrelated text on furnaces and ovens and their maintenance schedules over many years. " * 9
    body = "\n\n".join([abstract, target, other] * 2)
    seen = {}

    def rank(texts):
        seen["texts"] = texts
        return sorted(texts, key=lambda t: "Crack" not in t)
    got = ft.best_passages("Does humid air speed up fatigue crack growth?", "fatigue crack humid air", abstract, body,
                           rank=rank, k=2)
    assert "Crack growth" in got[0]
    assert not any("glass fibre" in t for t in seen["texts"])
    assert len(got) <= 2


def test_best_passages_skip_a_title_block_or_page_header():
    head = "www.example-journal.de 2101297 (1 of 8) © 2021 Publisher GmbH Research Article Fatigue crack growth in air "
    text = "Crack growth in humid air doubled the fatigue damage of the polyamide samples after aging. " * 9
    body = "\n\n".join([head * 8, text])
    got = ft.best_passages("Does humid air speed up fatigue crack growth?", "fatigue crack", "", body, rank=lambda t: t)
    assert len(got) == 1 and got[0].startswith("Crack growth")


def test_a_header_goes_and_the_paragraph_after_it_stays():
    body = "\n\n".join(["www.example-journal.de 2101297 © 2021 Publisher GmbH",
                         "Crack growth in humid air doubled the fatigue damage of the samples."])
    got = ft.best_passages("Does humid air speed up crack growth?", "crack growth", "", body, rank=lambda t: t)
    assert len(got) == 1 and got[0].startswith("Crack growth")


def test_a_result_with_a_link_to_its_data_is_kept():
    body = "The tensile modulus was 4.2 GPa; the measurement dataset is at www.example.org/results."
    assert ft.best_passages("What tensile modulus was measured?", "tensile modulus", "", body, rank=lambda t: t)


def test_best_passages_empty_when_nothing_shares_a_word():
    assert ft.best_passages("zebra quagga", "zebra", "", "Plain words only here.\n\nMore plain words.",
                            rank=lambda t: t) == []


# ── the check reads the top papers ────────────────────────────

def test_attach_reads_held_papers_and_leaves_the_others_alone(tmp_path, monkeypatch):
    (tmp_path / "pub/one").mkdir(parents=True)
    (tmp_path / "pub/one/paper.md").write_text(BODY)
    idx = index(tmp_path / "paths.sqlite", [("10.9999/abc", "pub/one")])
    held, other = work(), work(doi="10.9999/none")
    n = ft.attach([held, other], "What is the tensile modulus of the blend?", "tensile modulus blend",
                  rank=lambda t: t, index=idx, root=tmp_path)
    assert n == 1 and 0 < len(held.excerpts) <= 3 and other.excerpts == []


def test_attach_survives_an_unreadable_paper(tmp_path):
    (tmp_path / "pub/one").mkdir(parents=True)
    (tmp_path / "pub/one/Paper.pdf").write_bytes(b"not a pdf")
    idx = index(tmp_path / "paths.sqlite", [("10.9999/abc", "pub/one")])
    w = work()
    assert ft.attach([w], "q words here", "q words", rank=lambda t: t, index=idx, root=tmp_path) == 0
    assert w.excerpts == []


def test_attach_survives_a_failing_ranker(tmp_path):
    (tmp_path / "pub/one").mkdir(parents=True)
    (tmp_path / "pub/one/paper.md").write_text(BODY)
    idx = index(tmp_path / "paths.sqlite", [("10.9999/abc", "pub/one")])

    def boom(texts):
        raise RuntimeError("no reranker")
    w = work()
    assert ft.attach([w], "tensile modulus", "tensile", rank=boom, index=idx, root=tmp_path) == 0
    assert w.excerpts == []


# ── the judge and the stored evidence ─────────────────────────

def test_judge_input_is_unchanged_without_excerpts():
    w = work()
    msgs = gj.web_check_messages("Q?", "claim", [w])
    assert msgs[0]["content"] == gj.WEB_CHECK_SYSTEM
    assert ft.MARK not in msgs[1]["content"]


def test_judge_input_labels_excerpts_and_adds_the_rule():
    w = work()
    w.excerpts = ["First excerpt.", "Second excerpt."]
    msgs = gj.web_check_messages("Q?", "claim", [w, work(doi="10.9999/b")])
    assert msgs[0]["content"] == gj.WEB_CHECK_SYSTEM + gj.WEB_CHECK_EXCERPTS_RULE
    assert ft.MARK in msgs[1]["content"] and "Second excerpt." in msgs[1]["content"]
    assert msgs[1]["content"].index("Second excerpt.") < msgs[1]["content"].index("[2]")


def test_objects_without_the_excerpts_field_still_work():
    w = SimpleNamespace(title="T", year=None, abstract="A.")
    assert "A." in gj.web_check_messages("Q?", "c", [w])[1]["content"]


def test_judged_text_is_what_the_judge_read():
    w = work(abstract="The abstract.")
    assert ft.judged_text(w) == "The abstract."
    w.excerpts = ["One.", "Two."]
    assert ft.judged_text(w) == f"The abstract.\n\n{ft.MARK}\nOne.\n\nTwo."
    assert ft.read_in_full(ft.judged_text(w)) and not ft.read_in_full("The abstract.")


# ── the report says what was read ─────────────────────────────

def _paper(text):
    return {"kind": "paper", "source": "A paper", "title": "A paper", "year": 2024, "url": "", "text": text}


def _gap():
    return {"web_status": "partly", "web_sources": "openalex", "web_checked_at": None}


def _refs():
    return {"papers": [(1, "partly")], "unnumbered": [], "reasons": {1: "because"}, "unnumbered_reasons": []}


def test_report_says_abstracts_when_only_abstracts_were_read():
    from memento.research_gaps import report as gr
    src = [_paper("An abstract.")]
    assert "judged from abstracts)" in gr.literature_line(_gap(), _refs(), {1: ("A paper", 2024)})
    assert "judged from abstracts)" in gr.literature_block(_gap(), _refs(), src)[0] + ")"


def test_report_says_so_when_excerpts_were_read():
    from memento.research_gaps import report as gr
    src = [_paper(f"An abstract.\n\n{ft.MARK}\nOne.")]
    refs = {**_refs(), "read": {1}}
    assert "judged from abstracts and full-text excerpts" in gr.literature_line(_gap(), refs, {1: ("A paper", 2024)})
    assert "judged from abstracts and full-text excerpts" in gr.literature_block(_gap(), refs, src)[0]


def test_writer_sees_the_label_of_what_a_source_is():
    from memento.research_gaps import report as gr
    msgs = gr.prose_messages([{"n": 1, "statement": "Q", "facts": "f", "sources": [1, 2]}],
                             [_paper("An abstract."), _paper(f"Abstract.\n\n{ft.MARK}\nOne.")])
    text = msgs[-1]["content"]
    assert "[1] A paper (2024), abstract\n" in text and "[2] A paper (2024), abstract and excerpts\n" in text


def test_how_it_was_made_follows_what_was_read():
    from memento.research_gaps import report as gr
    assert "abstracts only" in gr.papers_how_line(False)
    line = gr.papers_how_line(True)
    assert "abstracts only" not in line and "full text" in line


# ── the whole check ───────────────────────────────────────────

@pytest.fixture
def check(tmp_path, monkeypatch):
    """One question and one paper through check_literature, with the database in tmp_path; returns (run, seen)."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from memento.library import documents as ldb
    from memento.research_gaps import ledger as gl, literature as ls, update as gu
    import memento.retrieval.search as search
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    gu.ensure_tables(engine)
    gl.ensure_tables(engine)
    ldb.ensure_tables(engine)
    (tmp_path / "pub/one").mkdir(parents=True)
    (tmp_path / "pub/one/paper.md").write_text(BODY)
    monkeypatch.setattr(ft, "INDEX", index(tmp_path / "paths.sqlite", [("10.9999/abc", "pub/one")]))
    monkeypatch.setenv("ALEXANDRIA_DB", str(tmp_path))
    w = ls.Work("A paper", 2024, "10.9999/abc", "An abstract about ovens.", "https://doi.org/10.9999/abc", "openalex",
                "W1", {"openalex"})
    monkeypatch.setattr(ls, "search_openalex", lambda *a, **k: ([w], None))
    monkeypatch.setattr(ls, "available", lambda *a, **k: False)
    monkeypatch.setattr(search, "rerank", lambda q, hits, *a, **k: hits)
    gid = gl.add_gap(engine, "col", "What is the tensile modulus of the blend?", "limitation")
    gl.add_mention(engine, gid, "col", source="D", point_id="1", page=0, signal="s", claim="c", passage="p",
                   judge="supported")
    seen = []

    class Client:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, messages, **kw):
            seen.append(messages)
            text = "tensile modulus blend" if "search query" in messages[0]["content"] else \
                '[{"i": 1, "relation": "partly", "reason": "r"}]'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    ctx = gu.Ctx(engine=engine, qdrant=None, llm_client=Client(), model="m", collection="col", embedder_url="",
                 embedder_model="", reranker_url="", instruction="")

    def run():
        gu.check_literature(ctx, gl.list_gaps(engine, "col"), [], [])
        return gl.evidence(engine, gid)
    return run, seen


def test_the_check_reads_the_full_text_and_keeps_what_the_judge_read(check, monkeypatch):
    monkeypatch.setenv("LITERATURE_FULLTEXT", "1")
    run, seen = check
    ev = run()
    assert ft.MARK in seen[-1][1]["content"] and "tensile modulus" in seen[-1][1]["content"]
    assert ft.read_in_full(ev[0]["text"]) and ev[0]["text"].startswith("An abstract about ovens.")


def test_the_check_is_unchanged_when_full_text_is_off(check, monkeypatch):
    monkeypatch.delenv("LITERATURE_FULLTEXT", raising=False)
    run, seen = check
    ev = run()
    assert ft.MARK not in seen[-1][1]["content"] and ev[0]["text"] == "An abstract about ovens."


def test_a_question_judged_from_the_abstract_says_so_beside_one_judged_from_excerpts():
    # one paper, two questions: only the second verdict read its full text
    from memento.research_gaps import report as gr
    row = {"origin": "openalex", "source": "W1", "title": "A paper", "doi": "10.9999/x", "relation": "partly",
           "reason": "because"}
    evidence = {1: [{**row, "text": "An abstract."}], 2: [{**row, "text": f"An abstract.\n\n{ft.MARK}\nOne."}]}
    _, refs = gr.gather_sources([{"id": 1, "mentions": []}, {"id": 2, "mentions": []}], evidence, {})
    assert gr._read_from(refs[1]) == "abstracts" and gr._read_from(refs[2]) == "abstracts and full-text excerpts"


def test_the_method_line_counts_excerpts_behind_a_near_miss():
    from memento.research_gaps import report as gr
    assert gr.reads_full_text({1: [{"relation": "near_miss", "text": f"An abstract.\n\n{ft.MARK}\nOne."}]})
    assert not gr.reads_full_text({1: [{"relation": "partly", "text": "An abstract."}], 2: []})


def test_the_writers_are_told_only_what_their_cut_of_a_source_holds():
    from memento.research_gaps import proposal as gp, report as gr
    long = [_paper("A long abstract. " * 200 + f"\n\n{ft.MARK}\nOne.")]
    short = [_paper(f"An abstract.\n\n{ft.MARK}\nOne.")]
    for build in (lambda s: gr.prose_messages([], s), lambda s: gp.proposal_messages(["Q?"], ["F."], s)):
        assert "abstract and excerpts" not in build(long)[-1]["content"]
        assert "abstract and excerpts" in build(short)[-1]["content"]


def test_the_proposal_says_when_its_papers_were_read_in_part():
    from memento.research_gaps import proposal as gp
    args = dict(questions=["Q?"], facts=["F."], names={}, checked=True)
    assert "papers judged from their abstracts)" in gp.assemble({}, sources=[_paper("An abstract.")], **args)
    read = gp.assemble({}, sources=[_paper(f"An abstract.\n\n{ft.MARK}\nOne.")], **args)
    assert "papers judged from their abstracts)" not in read and "passages of their full text" in read


def test_a_paper_not_read_for_this_question_keeps_no_earlier_excerpts(monkeypatch):
    w = work()
    w.excerpts = ["left over from another question"]
    monkeypatch.setattr(ft, "paper_folder", lambda *a, **k: None)
    assert ft.attach([w], "Another question?", "other", rank=lambda t: t) == 0 and w.excerpts == []


def test_a_lookup_closes_its_connection(tmp_path, monkeypatch):
    idx = index(tmp_path / "paths.sqlite", [("10.9999/abc", "pub/one")])
    opened = []
    real = ft.sqlite3.connect

    def connect(*a, **k):
        opened.append(real(*a, **k))
        return opened[-1]
    monkeypatch.setattr(ft.sqlite3, "connect", connect)
    ft.paper_folder("10.9999/abc", index=idx, root=tmp_path)
    with pytest.raises(ft.sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
