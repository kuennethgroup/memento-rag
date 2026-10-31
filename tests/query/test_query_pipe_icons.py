# Icons are dropped from the search results so they do not take the model's image slots. Lenient on purpose: only clear
# icons go, both sides under 80 on the page's 1000 grid, or a QR code; a borderline image stays (the model and the
# display filter still judge it).
import pytest

import openwebui.functions.query_pipe as qp
from memento.query.flows import figures
from memento.query import presentation


def _img(w, h, kind="image"):
    return {"type": kind, "img_path": "/nonexistent.png", "bbox": [100, 100, 100 + w, 100 + h], "text": "t",
            "source": "doc", "page": 3}


@pytest.mark.parametrize("hit, icon", [
    (_img(75, 48), True),                       # "Check for updates" badge
    (_img(23, 15), True),                       # journal icon
    (_img(71, 76, "chart"), True),              # journal cover thumbnail
    (_img(138, 28), False),                     # a wide, thin strip: borderline, kept
    (_img(80, 18), False),                      # one side not small: kept
    (_img(328, 192), False),                    # a real figure
    ({"type": "text", "bbox": [0, 0, 10, 10], "text": "x"}, False),   # text is never an icon
    ({"type": "image", "img_path": "/nonexistent.png", "text": "t"}, False),   # no box: kept
])
def test_only_clear_icons_count(hit, icon, monkeypatch):
    monkeypatch.setattr("memento.common.qr_code.is_qr_code", lambda p: False)
    assert presentation.icon_image(hit) is icon


def test_a_qr_code_of_any_size_is_an_icon(monkeypatch):
    monkeypatch.setattr("memento.common.qr_code.is_qr_code", lambda p: True)
    assert presentation.icon_image(_img(150, 150)) is True


def test_icons_leave_the_results_and_the_order_stays(monkeypatch):
    monkeypatch.setattr("memento.common.qr_code.is_qr_code", lambda p: False)
    hits = [{"type": "text", "text": "a"}, _img(75, 48), _img(328, 192), {"type": "text", "text": "b"}, _img(23, 15)]
    kept = presentation.without_icons(hits)
    assert kept == [hits[0], hits[2], hits[3]]


def test_results_of_icons_only_are_kept_whole(monkeypatch):
    # never turn a search into "nothing found" because of this filter
    monkeypatch.setattr("memento.common.qr_code.is_qr_code", lambda p: False)
    hits = [_img(75, 48), _img(23, 15)]
    assert presentation.without_icons(hits) == hits


# ── author photos (References / author biography pages) are not shown as figures ──
@pytest.mark.parametrize("section, ref", [("References", True), ("ABOUT THE AUTHORS", True), ("Bibliography", True),
                                          ("Literaturverzeichnis", True), ("Author Biographies", True),
                                          ("Preferred Orientation", False), ("Results", False), (None, False)])
def test_images_from_reference_and_author_sections(section, ref, monkeypatch):
    monkeypatch.setattr("memento.common.qr_code.is_qr_code", lambda p: False)
    h = {**_img(124, 124), "section": section}
    assert presentation.reference_image(h) is ref
    assert (h in presentation.without_icons([h, _img(328, 192)])) is (not ref)
    assert figures.shown_figure({**h, "page": 10}, figure_asked=True) is (not ref)
