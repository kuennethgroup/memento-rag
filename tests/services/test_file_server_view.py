# Unified citation view /view/<stem>/<page>.
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT



def _without_own_scripts(page: str) -> str:
    """The page minus the viewer's own scripts (theme listener, hover zoom) — what is left must carry no script."""
    import services.file_server as fsmod
    return page.replace(fsmod._THEME_JS, "").replace(fsmod._ZOOM_JS, "")

def _pdf(path: Path, pages: int = 2, text: str = "page"):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"{text} {i + 1}")
    doc.save(path)


def _cl(out: Path, corpus: str, stem: str, sub: str, entries: list):
    d = out / corpus / stem / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}_content_list.json").write_text(json.dumps(entries))
    (d / f"{stem}.md").write_text("# parsed")
    (d / "images").mkdir(exist_ok=True)
    (d / "images" / "fig1.jpg").write_bytes(b"jpg")
    return d


@pytest.fixture
def client(tmp_path, monkeypatch):
    host = tmp_path / "host_input"                       # what the staged symlinks point to (host paths)
    inp, out = tmp_path / "in", tmp_path / "out"
    (inp / "Corp" / "a").mkdir(parents=True)
    (inp / "Corp" / "b").mkdir(parents=True)
    _pdf(inp / "Corp" / "a" / "paper.pdf", 3)
    _cl(out, "Corp", "paper", "vlm", [{"type": "text", "text": "x", "page_idx": 0}])
    (inp / "Corp" / "a" / "photo.jpg").write_bytes(b"jpg")          # image: MinerU writes <stem>_origin.pdf
    d = _cl(out, "Corp", "photo", "vlm", [{"type": "image", "img_path": "images/fig1.jpg", "page_idx": 0}])
    _pdf(d / "photo_origin.pdf", 1)
    (inp / "Corp" / "a" / "photo.DSD").write_bytes(b"x")             # same stem, never parsed
    (inp / "Corp" / "a" / "talk.pptx").write_bytes(b"pptx")
    _cl(out, "Corp", "talk", "office", [
        {"type": "text", "text": "**Motivation**", "text_level": 1, "page_idx": 0},
        {"type": "list", "list_items": ["- alpha **beta**"], "page_idx": 0},
        {"type": "text", "text": "Slide two sintering at 1450 C in air", "page_idx": 1},
        {"type": "table", "table_body": "<table><tr><td onclick='x()'>1</td><script>alert(1)</script></tr></table>",
         "page_idx": 1}])
    (inp / "Corp" / "a" / "thesis.docx").write_bytes(b"docx")
    _cl(out, "Corp", "thesis", "office", [{"type": "text", "text": f"Paragraph {i} filler text here", "page_idx": 0}
                                          for i in range(30)]
        + [{"type": "text", "text": "The flexural strength reached 412 MPa after hot pressing", "page_idx": 0}])
    # collision-renamed legacy Word file: staged symlink with a HOST path target
    (host / "Corp" / "b").mkdir(parents=True)
    (inp / "Corp" / "b" / "Lebenslauf.doc").write_bytes(b"doc")
    (out / "_staged").mkdir(parents=True)
    (out / "_staged" / "b__Lebenslauf.doc").symlink_to(host / "Corp" / "b" / "Lebenslauf.doc")
    _cl(out, "Corp", "b__Lebenslauf", "office", [{"type": "text", "text": "Curriculum vitae", "page_idx": 0}])
    monkeypatch.setenv("INPUT_DIR", str(inp))
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    monkeypatch.setenv("HOST_INPUT_ROOT", str(host))
    sys.path.insert(0, str(ROOT / "services"))
    import file_server
    fs = importlib.reload(file_server)
    yield TestClient(fs.app), fs


def test_pdf_renders_the_cited_page_with_both_downloads(client):
    c, _ = client
    r = c.get("/view/paper/1", params={"bbox": "100,100,500,200"})
    assert r.status_code == 200 and "Page <b>2</b> of 3" in r.text and "PDF" in r.text
    # the page image is embedded (the side panel is a sandboxed frame without the login cookie) and is exactly what the
    # image endpoint renders, highlight included
    import base64
    embedded = r.text.split("src='data:image/png;base64,")[1].split("'")[0]
    direct = c.get("/view/paper/1/image", params={"bbox": "100,100,500,200"}).content
    assert base64.b64decode(embedded) == direct
    assert "/download-original/paper" in r.text and "/download-parsed/paper" in r.text
    img = c.get("/view/paper/1/image", params={"bbox": "100,100,500,200"})
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert c.get("/view/paper/9/image").status_code == 404


def test_image_uses_origin_pdf_as_jpeg_and_ignores_unparsed_same_stem_files(client):
    c, fs = client
    assert fs._original("photo").suffix == ".jpg"
    r = c.get("/view/photo/0")
    assert r.status_code == 200 and "Image" in r.text
    assert c.get("/view/photo/0/image").headers["content-type"] == "image/jpeg"


def test_pptx_reading_view_shows_one_slide_with_highlight_and_safe_table(client):
    c, _ = client
    r = c.get("/view/talk/1", params={"highlight": "sintering at 1450 C"})
    assert "Slide <b>2</b> of 2" in r.text and "PowerPoint" in r.text and "Text view of this PowerPoint" in r.text
    assert "Motivation" not in r.text                        # other slide not shown
    assert "class='hl'" in r.text and "<script" not in _without_own_scripts(r.text) and "onclick" not in r.text
    r0 = c.get("/view/talk/0")
    assert "<h2><b>" not in r0.text and ">Motivation</h2>" in r0.text and "<li>alpha <b>beta</b></li>" in r0.text


def test_docx_reading_view_windows_around_the_matched_paragraph(client):
    c, _ = client
    r = c.get("/view/thesis/0", params={"highlight": "flexural strength reached 412 MPa", "section": "3 Results"})
    assert "Section: 3 Results" in r.text and "Word" in r.text
    assert "earlier content" in r.text and "Paragraph 3 " not in r.text
    assert "class='hl'>The flexural strength" in r.text


def test_collision_renamed_file_resolves_through_host_path_symlink(client):
    c, fs = client
    orig = fs._original("b__Lebenslauf")
    assert orig is not None and orig.name == "Lebenslauf.doc" and orig.is_file()
    d = c.get("/download-original/b__Lebenslauf")
    assert d.status_code == 200 and d.content == b"doc" and "Lebenslauf.doc" in d.headers["content-disposition"]
    assert "Word" in c.get("/view/b__Lebenslauf/0").text


def test_parsed_zip_has_json_markdown_and_images(client):
    c, _ = client
    z = zipfile.ZipFile(io.BytesIO(c.get("/download-parsed/talk").content))
    assert sorted(z.namelist()) == ["talk/content_list.json", "talk/images/fig1.jpg", "talk/talk.md"]


def test_unknown_and_traversal_stems_are_404(client):
    c, _ = client
    assert c.get("/view/nope/0").status_code == 404
    assert c.get("/view/../0").status_code == 404
    assert c.get("/download-original/..").status_code == 404


def test_links_carry_the_proxy_prefix_and_bad_prefixes_are_ignored(client):
    # remote access: the proxy serves the files server under /files and says so in X-Forwarded-Prefix
    c, _ = client
    r = c.get("/view/paper/1", headers={"X-Forwarded-Prefix": "/files"})
    assert "src='data:image/png;base64," in r.text                      # page image embedded
    assert "href='/files/download-original/paper?t=" in r.text and "href='/files/download-parsed/paper?t=" in r.text
    r = c.get("/view/photo/0", params={"highlight": "x"}, headers={"X-Forwarded-Prefix": "/files/"})
    assert "href='/files/download-original/photo?t=" in r.text                  # trailing slash stripped
    r = c.get("/view/talk/0", headers={"X-Forwarded-Prefix": "/files"})
    assert "href='/files/download-original/talk?t=" in r.text
    r = c.get("/view/paper/1", headers={"X-Forwarded-Prefix": "/x'><script>"})
    assert "<script>" not in _without_own_scripts(r.text) and "href='/download-original/paper?t=" in r.text
    r = c.get("/view/paper/1")                                           # direct access: unchanged
    assert "href='/download-original/paper?t=" in r.text


def test_image_parsed_without_origin_pdf_renders_the_image_itself(client, tmp_path):
    # chat uploads go through MinerU's web API, which returns no <stem>_origin.pdf
    c, fs = client
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 30), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    pix.save(str(fs.INPUT_DIR / "Corp" / "a" / "scan.png"))
    _cl(fs.OUTPUT_DIR, "Corp", "scan", "vlm", [{"type": "image", "img_path": "images/fig1.jpg", "page_idx": 0}])
    r = c.get("/view/scan/0", params={"bbox": "0,0,500,500"})
    assert r.status_code == 200 and "Image" in r.text and "class='page'" in r.text
    img = c.get("/view/scan/0/image", params={"bbox": "0,0,500,500"})
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    (fs.INPUT_DIR / "Corp" / "a" / "broken.png").write_bytes(b"not an image")        # unreadable → text view
    _cl(fs.OUTPUT_DIR, "Corp", "broken", "vlm", [{"type": "text", "text": "caption", "page_idx": 0}])
    r = c.get("/view/broken/0")
    assert r.status_code == 200 and "Text view" in r.text


def test_side_panel_view_has_no_second_title(client):
    # Open WebUI's side panel header already names the document; standalone (new tab) keeps the title
    c, _ = client
    alone = c.get("/view/paper/1").text
    panel = c.get("/view/paper/1", params={"chat_id": "c1", "message_id": "m1", "source_id": "3"}).text
    assert "<h1 class='doc'>paper</h1>" in alone and "class='doc'" not in panel
    assert "Page <b>2</b> of 3" in panel and "PDF" in panel and "Download Original" in panel
    word = c.get("/view/thesis/0", params={"highlight": "flexural strength", "source_id": "1"}).text
    assert "class='doc'" not in word and "Text view of this Word file" in word


def test_side_panel_page_image_zooms_on_hover_and_new_tab_stays_plain(client):
    c, fs = client
    panel = c.get("/view/paper/1", params={"source_id": "2"}).text
    assert "<div class='zoom'><img class='page'" in panel and "addEventListener('mousemove'" in panel
    assert "Hover the page to zoom in, scroll to change the zoom.<br>Click the document name above to open it in a " \
           "new tab." in panel and "addEventListener('wheel'" in panel
    alone = c.get("/view/paper/1").text
    assert "class='zoom'" not in alone and "<script>" not in _without_own_scripts(alone) and "Hover the page" not in alone
    word = c.get("/view/thesis/0", params={"source_id": "1"}).text               # text view: readable, no zoom
    assert "class='zoom'" not in word


# ── Cloud quota for the model picker ──
def test_quota_endpoint_serves_the_state_file(client):
    c, fs = client
    p = fs.OUTPUT_DIR / "_runtime" / "cloud_quota.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"reserve": 3, "cloud_models": ["memento-query-qwen35"], "windows": {}}')
    r = c.get("/quota")
    assert r.json()["cloud_models"] == ["memento-query-qwen35"] and r.headers["cache-control"] == "no-store"


def test_quota_endpoint_missing_or_broken_is_empty(client):
    c, fs = client
    assert c.get("/quota").json() == {}
    p = fs.OUTPUT_DIR / "_runtime" / "cloud_quota.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{broken")
    assert c.get("/quota").json() == {}


# ── Library list for the start-screen document picker ──
def test_library_endpoint_serves_the_list_and_missing_is_empty(client):
    c, fs = client
    assert c.get("/library").json() == {}
    p = fs.OUTPUT_DIR / "_runtime" / "library.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"updated": "2026-09-29T12:00:00+00:00", "documents": [{"source": "a", "title": "A title"}]}')
    r = c.get("/library")
    assert r.json()["documents"] == [{"source": "a", "title": "A title"}] and r.headers["cache-control"] == "no-store"
    p.write_text("{broken")
    assert c.get("/library").json() == {}


# ── Model colours for the chat box and the model dropdown ──
def test_models_endpoint_serves_the_colours_and_missing_is_empty(client):
    c, fs = client
    assert c.get("/models").json() == {}
    p = fs.OUTPUT_DIR / "_runtime" / "query_models.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"models": [{"id": "memento-query-c", "cloud": true, "accent": "#0369a1"}]}')
    r = c.get("/models")
    assert r.json()["models"][0]["accent"] == "#0369a1" and r.headers["cache-control"] == "no-store"
    p.write_text("{broken")
    assert c.get("/models").json() == {}


# ── Side panel colours match the sidebar: Open WebUI's grey in light mode, the charcoal in dark ──
def test_side_panel_uses_the_sidebar_greys():
    import services.file_server as fsmod
    css = fsmod._VIEW_CSS
    light = css.split("@media")[0]
    dark = css.split(":root[data-theme='dark']{")[1].split("}")[0]
    assert "--desk:oklch(98% 0 0)" in light and "--sheet:#fff" in light
    assert "--desk:oklch(20% 0 0)" in dark and "--sheet:oklch(25% 0 0)" in dark
    assert "--accent:#009260" in light and "--accent:#2bb383" in dark              # the green accents stay


# ── A cited document that is no longer in the library ──
# After a library reset, an older answer still cites the removed document: a themed page, not bare JSON.
def test_removed_document_gets_a_themed_page_not_raw_json(client):
    c, fs = client
    for url in ("/view/old paper/3?source_id=1&chat_id=x", "/view/old paper/3"):
        r = c.get(url)
        assert r.status_code == 404 and r.headers["content-type"].startswith("text/html")
        page = r.text
        assert "This document is no longer in the library" in page and "old paper" in page
        assert fs._VIEW_CSS in page and fs._THEME_JS in page                     # same look, follows the theme
        assert "download-original" not in page and "download-parsed" not in page   # nothing left to download
        assert "—" not in page and "detail" not in page
    assert c.get("/view/<script>alert(1)</script>/0").status_code == 404
    assert "<script>alert(1)" not in c.get("/view/%3Cscript%3Ealert(1)%3C%2Fscript%3E/0").text   # the name is escaped


def test_side_panel_is_one_surface():
    # toolbar and download footer take the desk colour (the sidebar's grey), not a separate white band
    import services.file_server as fsmod
    css = fsmod._VIEW_CSS
    bar = css.split(".bar{")[1].split("}")[0]
    dock = css.split(".dock{")[1].split("}")[0]
    assert "background:var(--desk)" in bar and "var(--desk)" in dock and "var(--sheet)" not in bar + dock


def test_view_page_signs_its_downloads_for_the_side_panel(client):
    # Open WebUI's side panel is a sandboxed frame that sends no login cookie: the view page signs its two download
    # links; the proxy lets a signed request through without the login, and the server checks the signature (path-bound,
    # expiring)
    import re
    c, fs = client
    r = c.get("/view/paper/1", headers={"X-Forwarded-Prefix": "/files"})
    orig = re.search(r"href='/files(/download-original/paper\?t=[^']+)'", r.text).group(1)
    parsed = re.search(r"href='/files(/download-parsed/paper\?t=[^']+)'", r.text).group(1)
    assert c.get(orig).status_code == 200 and c.get(parsed).status_code == 200
    t = orig.split("t=")[1]
    assert c.get(f"/download-original/talk?t={t}").status_code == 403          # bound to its own path
    assert c.get(f"/download-parsed/paper?t={t}").status_code == 403
    assert c.get("/download-original/paper?t=99999999999.deadbeef").status_code == 403   # forged
    assert c.get("/download-original/paper?t=").status_code == 403
    assert c.get("/download-original/paper").status_code == 200                # no token: the proxy's login applies
    old = fs._sign("/download-original/paper", now=0)                           # expired long ago
    assert c.get(f"/download-original/paper?t={old}").status_code == 403


# ── draft proposals as PDF (memento/research_gaps/proposal.py) ──

def test_a_proposal_pdf_is_served_by_its_token_only(client):
    c, fs = client
    d = fs.OUTPUT_DIR / "_runtime" / "reports"
    d.mkdir(parents=True)
    token = "0123456789abcdef01234567"
    (d / f"{token}.pdf").write_bytes(b"%PDF-1.7 proposal")
    r = c.get(f"/report/{token}.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert r.headers["content-type"] == "application/pdf"
    assert 'filename="research_proposal_draft.pdf"' in r.headers["content-disposition"]
    for bad in ("0123456789abcdef0123456.pdf", "../reports/x.pdf", f"{token}.txt", "ffffffffffffffffffffffff.pdf"):
        assert c.get(f"/report/{bad}").status_code == 404


# ── A passage that runs on into the other column or onto the next page ──
# ingestion merges a paragraph MinerU split into blocks into its first block (merge_page_boundary_paragraphs): the
# passage's bbox is the first block's, the later blocks stay in the enriched list right after it, marked
# merged_continuation. The citation marks every block and shows the next page too.
HEAD, SAME, NEXT = [57, 907, 485, 935], [507, 394, 937, 569], [57, 66, 485, 210]
_box = lambda b: ",".join(map(str, b))


def _split_paper(fs, entries):
    _cl(fs.OUTPUT_DIR, "Corp", "paper", "vlm", entries)


def _png(fs, page: int, bbox) -> bytes:
    return fs._render_page("paper", page, bbox)[0]


def _embedded(page: str) -> list:
    import base64
    return [base64.b64decode(s.split("'")[0]) for s in page.split("src='data:image/png;base64,")[1:]]


def test_the_other_column_of_a_passage_is_marked_too(client):
    c, fs = client
    _split_paper(fs, [
        {"type": "text", "text": "Model optimization. The hyperparameters were optimized", "bbox": HEAD, "page_idx": 1},
        {"type": "image", "img_path": "images/fig1.jpg", "bbox": [100, 100, 900, 300], "page_idx": 1},
        {"type": "text", "text": "using the optuna library", "bbox": SAME, "page_idx": 1,
         "low_info": True, "merged_continuation": True}])
    r = c.get("/view/paper/1", params={"bbox": _box(HEAD)})
    both = _png(fs, 1, [_box(HEAD), _box(SAME)])
    assert _embedded(r.text) == [both] and "Page <b>2</b> of 3" in r.text
    assert c.get("/view/paper/1/image", params={"bbox": _box(HEAD)}).content == both    # the image link agrees
    assert both != _png(fs, 1, [_box(HEAD)])


def test_a_passage_continued_on_the_next_page_shows_both_pages(client):
    c, fs = client
    _split_paper(fs, [
        {"type": "text", "text": "Comparison with other models. There are", "bbox": HEAD, "page_idx": 0},
        {"type": "text", "text": "and parallel computing", "bbox": NEXT, "page_idx": 1,
         "low_info": True, "merged_continuation": True}])
    r = c.get("/view/paper/0", params={"bbox": _box(HEAD)})
    assert _embedded(r.text) == [_png(fs, 0, [_box(HEAD)]), _png(fs, 1, [_box(NEXT)])]
    assert "Pages <b>1–2</b> of 3" in r.text and "Continues on page 2" in r.text
    panel = c.get("/view/paper/0", params={"bbox": _box(HEAD), "source_id": "3"}).text
    assert panel.count("<div class='zoom'><img class='page'") == 2 and "querySelectorAll('.zoom')" in panel


def test_only_the_cited_passages_own_continuations_count(client):
    c, fs = client
    _split_paper(fs, [
        {"type": "text", "text": "First paragraph ends here.", "bbox": HEAD, "page_idx": 1},
        {"type": "text", "text": "Second paragraph starts", "bbox": [507, 100, 937, 300], "page_idx": 1},
        {"type": "text", "text": "and continues", "bbox": SAME, "page_idx": 1, "merged_continuation": True},
        {"type": "text", "text": "Heading", "text_level": 1, "bbox": [57, 50, 485, 60], "page_idx": 2},
        {"type": "text", "text": "after the heading", "bbox": NEXT, "page_idx": 2, "merged_continuation": True}])
    r = c.get("/view/paper/1", params={"bbox": _box(HEAD)})
    assert _embedded(r.text) == [_png(fs, 1, [_box(HEAD)])] and "Continues" not in r.text
    second = c.get("/view/paper/1", params={"bbox": "507,100,937,300"})
    assert _embedded(second.text) == [_png(fs, 1, ["507,100,937,300", _box(SAME)])]
    nobox = c.get("/view/paper/1")                                   # no box, an unknown box: the page as before
    assert _embedded(nobox.text) == [_png(fs, 1, None)]
    other = c.get("/view/paper/1", params={"bbox": "1,2,3,4"})
    assert _embedded(other.text) == [_png(fs, 1, ["1,2,3,4"])]


def test_a_long_run_on_shows_at_most_two_more_pages(client):
    c, fs = client
    _pdf(fs.INPUT_DIR / "Corp" / "a" / "paper.pdf", 5)
    _split_paper(fs, [{"type": "text", "text": "Starts", "bbox": HEAD, "page_idx": 0}] + [
        {"type": "text", "text": "and goes on", "bbox": NEXT, "page_idx": q, "merged_continuation": True}
        for q in (1, 3, 4)])
    r = c.get("/view/paper/0", params={"bbox": _box(HEAD)})
    assert len(_embedded(r.text)) == 3 and "Pages <b>1, 2, 4</b> of 5" in r.text
