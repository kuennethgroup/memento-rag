# One passage per multi-panel figure. MinerU often saves each panel of a figure as its own image, with the real caption
# on one panel only. Fixtures in tests/fixtures/figure_panels/ are real pages from MinerU's output; the synthetic cases
# cover what must never be merged.
import copy
import json
import sys
from pathlib import Path

import fitz
import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.ingestion import enrich_parsed_content_list as en  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "figure_panels"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def figures(items):
    return [b for b in items if b.get("type") in ("image", "chart")]


def caption(b):
    return " ".join(b.get("image_caption") or b.get("chart_caption") or [])


def make_images(items, images_dir: Path, size=(60, 40)):
    """A small grey JPEG for every image item, at its img_path under images_dir's parent."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for b in figures(items):
        if b.get("img_path"):
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, *size), False)
            pix.clear_with(180)
            (images_dir.parent / b["img_path"]).parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(images_dir.parent / b["img_path"]))


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def item(kind, bbox, cap=None, page=0, img=None, **kw):
    b = {"type": kind, "bbox": bbox, "page_idx": page, **kw}
    if kind in ("image", "chart"):
        b[f"{kind}_caption"] = [cap] if cap else []
        b["img_path"] = img or f"images/{kind}_{bbox[0]}_{bbox[1]}.jpg"
    else:
        b["text"] = cap or "Body text of the page."
    return b


# ── real pages ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_hafnia_eleven_panels_one_figure(tmp_path):
    items = load("hafnia_p3")
    panels = figures(items)
    make_images(items, tmp_path / "images")
    stats = en.merge_figure_panels(items, tmp_path / "images")
    figs = figures(items)
    assert len(panels) == 11 and len(figs) == 1 and stats["groups"] == 1 and stats["panels"] == 11
    f = figs[0]
    assert f["bbox"] == union([p["bbox"] for p in panels])
    assert caption(f).startswith("(a) m No. 14 lowest energy") and "Figure 1. a) Monoclinic" in caption(f)
    assert [m["bbox"] for m in f["merged_panels"]] == [p["bbox"] for p in panels]
    assert f["img_path"].startswith("images/figure_") and (tmp_path / f["img_path"]).is_file()
    assert [b["type"] for b in items if b["type"] not in ("image", "chart")] == \
           [b["type"] for b in load("hafnia_p3") if b["type"] not in ("image", "chart")]     # text etc. untouched


def test_nature_style_letters(tmp_path):
    items = load("nature_letters")
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    figs = figures(items)
    assert len(figs) == 1 and figs[0]["type"] == "chart" and len(figs[0]["merged_panels"]) == 7
    assert "Fig. 4 Enhanced PT effect" in caption(figs[0])


def test_parenthesised_panels(tmp_path):
    items = load("parenthesised")
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    figs = figures(items)
    assert len(figs) == 1 and len(figs[0]["merged_panels"]) == 8 and "Figure 1. The procedure" in caption(figs[0])


def test_fig11_then_panels_of_fig12(tmp_path):
    # Figure 11 (one image, captioned) is followed by three captionless panels and the chart carrying Figure 12
    items = load("fig11_then_fig12")
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    figs = figures(items)
    assert len(figs) == 2
    assert caption(figs[0]).startswith("Figure 11.") and "merged_panels" not in figs[0]
    assert caption(figs[1]).startswith("Figure 12.") and len(figs[1]["merged_panels"]) == 4
    assert figs[1]["type"] == "image"                                    # images and a chart: an image


def test_mixed_image_and_chart(tmp_path):
    items = load("mixed_chart")
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    figs = figures(items)
    assert len(figs) == 1 and figs[0]["type"] == "image" and len(figs[0]["merged_panels"]) == 4
    assert "Fig. 11 Hydrogen-doped VO" in caption(figs[0])


# ── what must never merge ──────────────────────────────────────────────────────────────────────────────────────
def test_two_captioned_figures_stay_apart(tmp_path):
    items = [item("image", [100, 100, 450, 400], "Figure 1. Left."), item("image", [550, 100, 900, 400], "Figure 2. Right.")]
    before = copy.deepcopy(items)
    make_images(items, tmp_path / "images")
    stats = en.merge_figure_panels(items, tmp_path / "images")
    assert items == before and stats["groups"] == 0


def test_text_between_panels(tmp_path):
    items = [item("image", [100, 100, 450, 300], "(a)"), item("text", [100, 320, 900, 400]),
             item("image", [100, 420, 450, 600], "Figure 3. After the text.")]
    before = copy.deepcopy(items)
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    assert items == before


def test_union_covering_text_refused(tmp_path):
    # the panels are consecutive in reading order, but a text block sits between them on the page
    items = [item("image", [100, 100, 300, 300], "(a)"), item("image", [400, 100, 600, 300], "Figure 2. (a) x (b) y"),
             item("text", [310, 120, 390, 280])]
    before = copy.deepcopy(items)
    make_images(items, tmp_path / "images")
    stats = en.merge_figure_panels(items, tmp_path / "images")
    assert items == before and stats["refused"] == 1 and stats["groups"] == 0


def test_run_without_caption_untouched(tmp_path):
    items = [item("image", [100, 100, 300, 300]), item("image", [320, 100, 500, 300], "(b)")]
    before = copy.deepcopy(items)
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    assert items == before


def test_headers_between_panels_do_not_break_the_run(tmp_path):
    items = [item("image", [100, 100, 300, 300], "(a)"), {"type": "header", "bbox": [80, 10, 300, 40], "page_idx": 0},
             item("image", [320, 100, 500, 300], "(b) Figure 5. Two panels.")]
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    # the merged figure takes the captioned panel's place (design §3), after the header
    assert len(figures(items)) == 1 and [b["type"] for b in items] == ["header", "image"]


def test_panels_on_another_page_are_not_joined(tmp_path):
    items = [item("image", [100, 700, 300, 900], "(a)", page=0), item("image", [100, 100, 300, 300], "Figure 6. x", page=1)]
    before = copy.deepcopy(items)
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    assert items == before


def test_trailing_panels_join_previous(tmp_path):
    # caption above the figure: MinerU attaches it to the first panel
    items = [item("image", [100, 100, 300, 300], "Figure 7. Caption above."), item("image", [320, 100, 500, 300], "(b)")]
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images")
    figs = figures(items)
    assert len(figs) == 1 and caption(figs[0]).startswith("Figure 7.") and len(figs[0]["merged_panels"]) == 2


# ── the image ──────────────────────────────────────────────────────────────────────────────────────────────────
def _pdf(path: Path, pages=2):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=600, height=800)
        page.draw_rect(fitz.Rect(60, 80, 540, 320), color=(0, 0, 1), fill=(0.8, 0.8, 1))
    doc.save(str(path))
    doc.close()


def test_crop_from_pdf(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _pdf(pdf)
    items = [item("image", [100, 100, 450, 400], "(a)", page=1), item("image", [500, 100, 900, 400], "Figure 1. x", page=1)]
    stats = en.merge_figure_panels(items, tmp_path / "images", source_pdf=pdf)       # member images need not exist
    f = figures(items)[0]
    pix = fitz.Pixmap(str(tmp_path / f["img_path"]))
    # union 100..900 x 100..400, plus the margin for panel labels just outside the boxes (10 sideways, 20 up and
    # down): 90..910 x 80..420 of a 600x800 page = 492 x 272 points, at zoom 2; the highlight box stays the union
    assert stats["crop"] == 1 and abs(pix.width / pix.height - 492 / 272) < 0.02 and pix.width >= 900
    assert f["bbox"] == [100, 100, 900, 400]


def test_crop_margin_stops_at_text(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _pdf(pdf)
    items = [item("image", [100, 100, 450, 400], "(a)", page=1), item("image", [500, 100, 900, 400], "Figure 1. x", page=1),
             item("text", [100, 410, 900, 500], page=1)]
    en.merge_figure_panels(items, tmp_path / "images", source_pdf=pdf)
    pix = fitz.Pixmap(str(tmp_path / figures(items)[0]["img_path"]))
    # bottom margin stops at the text block's top (410): 90..910 x 80..410 = 492 x 264 points
    assert abs(pix.width / pix.height - 492 / 264) < 0.02


def test_canvas_fallback_without_pdf(tmp_path):
    items = [item("image", [100, 100, 450, 400], "(a)"), item("image", [500, 100, 900, 400], "Figure 1. x")]
    make_images(items, tmp_path / "images")
    stats = en.merge_figure_panels(items, tmp_path / "images", source_pdf=tmp_path / "missing.pdf")
    f = figures(items)[0]
    pix = fitz.Pixmap(str(tmp_path / f["img_path"]))
    assert stats["canvas"] == 1 and abs(pix.width / pix.height - 800 / 300) < 0.03


def test_existing_crop_is_kept_by_a_run_without_pdf(tmp_path):
    # build_graph.py and some scripts re-run the enrichment without the PDF: they must not overwrite the page crop
    pdf = tmp_path / "doc.pdf"
    _pdf(pdf)
    items = [item("image", [100, 100, 450, 400], "(a)", page=1), item("image", [500, 100, 900, 400], "Figure 1. x", page=1)]
    again = copy.deepcopy(items)
    make_images(items, tmp_path / "images")
    en.merge_figure_panels(items, tmp_path / "images", source_pdf=pdf)
    out = tmp_path / figures(items)[0]["img_path"]
    crop = out.read_bytes()
    stats = en.merge_figure_panels(again, tmp_path / "images")
    assert out.read_bytes() == crop and stats["reused"] == 1 and figures(again)[0]["img_path"] == figures(items)[0]["img_path"]


def test_merge_errors_leave_items_unchanged(tmp_path):
    # no PDF and no member images: no image can be made, so the panels stay as they were
    items = [item("image", [100, 100, 450, 400], "(a)"), item("image", [500, 100, 900, 400], "Figure 1. x")]
    before = copy.deepcopy(items)
    stats = en.merge_figure_panels(items, tmp_path / "images")
    assert items == before and stats["refused"] == 1


def test_same_members_same_image_name(tmp_path):
    a = load("hafnia_p3")
    b = load("hafnia_p3")
    make_images(a, tmp_path / "images")
    en.merge_figure_panels(a, tmp_path / "images")
    en.merge_figure_panels(b, tmp_path / "images")
    assert figures(a)[0]["img_path"] == figures(b)[0]["img_path"]          # re-enrichment reuses the file name


def test_dry_run_writes_nothing(tmp_path):
    items = load("hafnia_p3")
    stats = en.merge_figure_panels(items, tmp_path / "images", dry_run=True)
    assert stats["groups"] == 1 and not (tmp_path / "images").exists()


# ── inside enrich_content_list ─────────────────────────────────────────────────────────────────────────────────
def test_enrich_content_list_merges_and_keeps_old_caption_copy(tmp_path):
    out = tmp_path / "vlm"
    hafnia = load("hafnia_p3")
    refused = [item("image", [100, 100, 300, 300], "(a)", page=5), item("image", [400, 100, 600, 300],
               "Figure 2. (a) first (b) second", page=5), item("text", [310, 120, 390, 280], page=5)]
    items = hafnia + refused
    make_images(items, out / "images")
    cl = out / "doc_content_list.json"
    cl.write_text(json.dumps(items))
    enriched = json.loads(Path(en.enrich_content_list(cl)).read_text())
    figs = figures(enriched)
    assert len(figs) == 3                                                # hafnia merged into one, the refused two stay
    assert figs[0]["figure_label"] == "Figure 1" and len(figs[0]["merged_panels"]) == 11
    assert figs[1].get("caption_recovery_method") == "sibling_group"     # the old caption copy still helps
    assert "merged_panels" not in figs[1] and "merged_panels" not in figs[2]
    assert json.loads(cl.read_text()) == items                           # MinerU's own list is never rewritten


# ── QR codes ──
def test_qr_codes_are_flagged_and_never_join_a_figure(tmp_path):
    from test_qr_code import qr_png
    items = [item("image", [100, 100, 200, 200], img="images/qr.png"),
             item("image", [300, 100, 500, 300], img="images/panel_a.jpg"),
             item("image", [520, 100, 700, 300], cap="Figure 1. (a) SEM, (b) XRD.", img="images/panel_b.jpg")]
    make_images(items[1:], tmp_path / "images")
    qr_png(tmp_path / "images" / "qr.png")
    assert en.mark_qr_codes(items, tmp_path) == 1
    assert items[0]["qr_code"] and items[0]["low_info"] and not items[1].get("qr_code")
    groups = en._panel_groups(items)
    assert groups == [[1, 2]]                                          # the two panels, never the QR code
