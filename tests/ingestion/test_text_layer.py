# Look-alike characters in parsed text, corrected against the PDF's own text layer (synthetic PDFs only).
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz
import pytest

from memento.ingestion import text_layer as tl

SENTENCE = "The sample was mixed with water and stirred for two hours at room temperature before testing"


def pdf(path: Path, lines: list[str], *, invisible: bool = False, scan: bool = False) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    if scan:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 80), False)
        pix.set_rect(pix.irect, (250, 250, 250))
        page.insert_image(page.rect, pixmap=pix)
    y = 60
    for line in lines:
        page.insert_text((40, y), line, fontsize=9, render_mode=3 if invisible else 0)
        y += 14
    doc.save(path)
    doc.close()
    return path


def item(text: str, page: int = 0, kind: str = "text") -> dict:
    return {"type": kind, "text": text, "page_idx": page, "bbox": [0, 0, 1000, 1000]}


LAYER = [SENTENCE, "The compound C(=O)NC was tested in the qwen3-vl model together with AlN powder",
         "and the grade PA66-GF30 was stored in a dry room for a week"]


def test_letter_read_as_zero_is_corrected(tmp_path):
    items = [item("The compound C(=0)NC was tested in the qwen3-vl model together with AlN powder")]
    stats = tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["text"].startswith("The compound C(=O)NC was")
    assert [(c["old"], c["new"]) for c in stats["changes"]] == [("0", "O")]


def test_letter_read_as_one_is_corrected_inside_a_code(tmp_path):
    items = [item("The compound C(=O)NC was tested in the qwen3-v1 model together with AlN powder")]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert "qwen3-vl model" in items[0]["text"]


def test_capital_i_for_small_l_is_corrected_in_a_formula(tmp_path):
    items = [item("The compound C(=O)NC was tested in the qwen3-vl model together with AIN powder")]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert "with AlN powder" in items[0]["text"]


def test_identical_text_is_unchanged(tmp_path):
    items = [item(SENTENCE)]
    assert tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))["changes"] == []
    assert items[0]["text"] == SENTENCE


def test_a_different_word_is_never_taken_from_the_layer(tmp_path):
    text = SENTENCE.replace("mixed", "mashed")
    items = [item(text)]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["text"] == text


def test_a_different_number_is_never_taken_from_the_layer(tmp_path):
    text = "and the grade PA66-GF31 was stored in a dry room for a week"
    items = [item(text)]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["text"] == text


def test_inline_latex_is_left_alone(tmp_path):
    text = "The compound C(=O)NC was tested $\\mathrm{C}=0$ in the qwen3-vl model together with AlN powder"
    items = [item(text)]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["text"] == text


def test_table_cells_are_corrected_and_the_markup_is_kept(tmp_path):
    body = "<table><tr><td>the grade PA66-GF3O was stored in a dry room for a week</td></tr></table>"
    items = [{"type": "table", "table_body": body, "table_caption": [], "table_footnote": [], "page_idx": 0,
              "bbox": [0, 0, 1000, 1000]}]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["table_body"] == body.replace("GF3O", "GF30")


def test_list_items_and_captions_are_corrected(tmp_path):
    items = [{"type": "list", "list_items": ["the grade PA66-GF3O was stored in a dry room for a week"],
              "page_idx": 0, "bbox": [0, 0, 1000, 1000]},
             {"type": "image", "image_caption": ["the grade PA66-GF3O was stored in a dry room for a week"],
              "image_footnote": [], "page_idx": 0, "bbox": [0, 0, 100, 100]}]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert "GF30" in items[0]["list_items"][0] and "GF30" in items[1]["image_caption"][0]


def test_equations_are_never_touched(tmp_path):
    items = [item("the grade PA66-GF3O was stored in a dry room for a week", kind="equation")]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert "GF3O" in items[0]["text"]


def test_a_word_the_layer_itself_uses_elsewhere_is_left_alone(tmp_path):
    """A slip in the layer: the parsed word is right, and the layer shows it on another page or line."""
    layer = ["The binding energy was calibrated on the C ls line of the carbon signal in the survey spectrum",
             "using Al Kα radiation, and the C 1s line of the sample and the Al 2p line were then fitted again",
             SENTENCE]
    text = "The binding energy was calibrated on the C 1s line of the carbon signal in the survey spectrum"
    items = [item(text)]
    assert tl.correct_items(items, pdf(tmp_path / "a.pdf", layer))["changes"] == []
    assert items[0]["text"] == text


def test_a_short_string_without_context_is_left_alone(tmp_path):
    items = [item("GF3O")]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["text"] == "GF3O"


@pytest.mark.parametrize("kw", [{"invisible": True}, {"scan": True}], ids=["ocr overlay", "page image"])
def test_a_scanned_page_is_not_trusted(tmp_path, kw):
    items = [item("The compound C(=0)NC was tested in the qwen3-v1 model together with AlN powder")]
    text = items[0]["text"]
    stats = tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER, **kw))
    assert items[0]["text"] == text and stats["skipped"] == 1


def test_a_page_without_a_text_layer_is_skipped(tmp_path):
    doc = fitz.open()
    doc.new_page()
    doc.save(tmp_path / "a.pdf")
    doc.close()
    items = [item(SENTENCE)]
    assert tl.correct_items(items, tmp_path / "a.pdf")["changes"] == []


def test_a_document_with_too_many_changes_has_a_broken_layer(tmp_path):
    """A wrong font map turns characters systematically: the whole document is left as parsed."""
    layer = [" ".join(f"AlN{k}O word{k} other{k}" for k in range(i, i + 4)) for i in range(0, 12, 4)]
    text = " ".join(f"AIN{k}0 word{k} other{k}" for k in range(12))
    items = [item(text)]
    stats = tl.correct_items(items, pdf(tmp_path / "a.pdf", layer))
    assert items[0]["text"] == text and stats["changes"] == [] and stats["rejected"] > 0


def test_a_missing_or_broken_file_changes_nothing_and_does_not_raise(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    items = [item(SENTENCE)]
    assert tl.correct_items(items, tmp_path / "bad.pdf")["changes"] == []
    assert tl.correct_items(items, tmp_path / "gone.pdf")["changes"] == []
    assert items[0]["text"] == SENTENCE


def test_the_switch_turns_it_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT_LAYER_CHECK", "0")
    text = "The compound C(=0)NC was tested in the qwen3-vl model together with AlN powder"
    items = [item(text)]
    tl.correct_items(items, pdf(tmp_path / "a.pdf", LAYER))
    assert items[0]["text"] == text


def test_enrichment_corrects_the_text_and_records_each_change(tmp_path):
    from memento.ingestion.enrich_parsed_content_list import enrich_content_list
    items = [item("The compound C(=0)NC was tested in the qwen3-vl model together with AlN powder")]
    raw = tmp_path / "x_content_list.json"
    raw.write_text(json.dumps(items))
    out = enrich_content_list(raw, source_pdf=pdf(tmp_path / "a.pdf", LAYER))
    assert "C(=O)NC" in json.loads(Path(out).read_text())[0]["text"]
    assert "C(=0)NC" in json.loads(raw.read_text())[0]["text"]        # the parser's own output stays as it was
    assert json.loads((tmp_path / "text_layer_changes.json").read_text())[0]["old"] == "0"


def test_enrichment_without_a_pdf_works_as_before(tmp_path):
    from memento.ingestion.enrich_parsed_content_list import enrich_content_list
    raw = tmp_path / "x_content_list.json"
    raw.write_text(json.dumps([item(SENTENCE)]))
    out = enrich_content_list(raw)
    assert json.loads(Path(out).read_text())[0]["text"] == SENTENCE
    assert not (tmp_path / "text_layer_changes.json").exists()


class _CountingPdf:
    """Stands in for fitz.open and records how many documents are open at the same time."""
    open_now, most = 0, 0
    guard = threading.Lock()

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        with self.guard:
            type(self).open_now += 1
            type(self).most = max(type(self).most, type(self).open_now)
        time.sleep(0.01)
        return []

    def __exit__(self, *a):
        with self.guard:
            type(self).open_now -= 1


def test_pdfs_are_read_one_thread_at_a_time(tmp_path, monkeypatch):
    """PyMuPDF is not thread-safe, and uploads are enriched side by side."""
    monkeypatch.setattr(fitz, "open", _CountingPdf)
    _CountingPdf.most = 0
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda k: tl.correct_items([], tmp_path / f"{k}.pdf"), range(8)))
    assert _CountingPdf.most == 1


# ── notation the layer cannot overrule ───────────────────────

@pytest.mark.parametrize("word, layer", [("1s", "ls"), ("2p", "Zp"), ("6d", "Gd"), ("2p3", "Zp3")])
def test_an_orbital_label_is_notation_whatever_the_layer_spells(word, layer):
    text = f"The binding energy was calibrated on the C {word} line before the measurement"
    spelled = text.replace(word, layer).split()
    assert tl._edits(text, spelled, set(spelled)) == []


def test_a_look_alike_in_a_formula_is_still_corrected():
    assert tl.lookalike("Hf02", "HfO2") and tl.lookalike("A1N", "AlN")


def test_a_malformed_element_changes_nothing_and_does_not_raise(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    good = {"type": "text", "page_idx": 0, "text": "The film of Hf02 was grown on silicon at a low temperature"}
    monkeypatch.setattr(tl, "page_words", lambda page: ["The", "film", "of", "HfO2", "was", "grown"] * 4)

    class Doc(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(fitz, "open", lambda *a, **k: Doc([object()]))
    pdf.touch()
    stats = tl.correct_items([good, None], pdf)
    assert stats["changes"] == [] and "Hf02" in good["text"]


def test_a_run_without_changes_leaves_no_old_change_list(tmp_path, monkeypatch):
    from memento.ingestion import enrich_parsed_content_list as enrich
    listing = tmp_path / "paper_content_list.json"
    listing.write_text("[]")
    old = tmp_path / "text_layer_changes.json"
    old.write_text('[{"old": "a", "new": "b"}]')
    monkeypatch.setattr(enrich, "correct_items", lambda items, pdf: {"changes": []})
    enrich.enrich_content_list(listing, tmp_path / "paper.pdf")
    assert not old.exists()


# ── one PyMuPDF at a time ────────────────────────────────────

def _recording_lock(monkeypatch, tmp_path):
    seen = []

    class Doc:
        metadata = {"title": "The title of a synthetic paper"}

        def __enter__(self):
            seen.append(tl.PYMUPDF.locked())
            return self

        def __exit__(self, *a):
            return False

        def __getitem__(self, i):
            raise IndexError
    monkeypatch.setattr(fitz, "open", lambda *a, **k: Doc())
    return seen


def test_the_title_reader_holds_the_lock(tmp_path, monkeypatch):
    from memento.library import documents
    (tmp_path / "paper.pdf").touch()
    seen = _recording_lock(monkeypatch, tmp_path)
    documents.raw_titles("paper", tmp_path, tmp_path)
    assert seen == [True]


def test_rasterising_an_svg_holds_the_lock(tmp_path, monkeypatch):
    from memento.common.model_servers import image_data_url
    seen = _recording_lock(monkeypatch, tmp_path)
    with pytest.raises(IndexError):
        image_data_url(str(tmp_path / "figure.svg"))
    assert seen == [True] and not tl.PYMUPDF.locked()


def test_writing_a_proposal_pdf_holds_the_lock(tmp_path, monkeypatch):
    from memento.research_gaps import proposal as gp
    seen = []

    def story(**kw):
        seen.append(tl.PYMUPDF.locked())
        raise RuntimeError("stop here")
    monkeypatch.setattr(fitz, "Story", story)
    with pytest.raises(RuntimeError):
        gp.write_pdf("# A title", tmp_path / "out.pdf")
    assert seen == [True] and not tl.PYMUPDF.locked()
