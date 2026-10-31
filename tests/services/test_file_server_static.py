# The file server's open routes serve only what a page shows: figure images from the parsed output. Original files go
# out only through signed download links, parsed data only through its download route, and runtime state (the library
# list, reports, paper cards) only through its own routes; nothing under _runtime is open.
import sys

from test_file_server_view import client  # noqa: F401  (the same isolated file server)


def test_figure_images_are_served(client):
    c, _ = client
    r = c.get("/figures/Corp/photo/vlm/images/fig1.jpg")
    assert r.status_code == 200 and r.content == b"jpg"


def test_parsed_data_and_runtime_state_are_not_open(client):
    c, fs = client
    (fs.OUTPUT_DIR / "_runtime").mkdir(parents=True, exist_ok=True)
    (fs.OUTPUT_DIR / "_runtime" / "library.json").write_text("{}")
    (fs.OUTPUT_DIR / "_runtime" / "image.png").write_bytes(b"png")
    for path in ("/figures/Corp/paper/vlm/paper_content_list.json", "/figures/Corp/paper/vlm/paper.md",
                 "/figures/_runtime/library.json", "/figures/_runtime/image.png"):
        assert c.get(path).status_code == 404, path


def test_original_files_have_no_open_route(client):
    c, _ = client
    for path in ("/papers/Corp/a/paper.pdf", "/papers/Corp/a/photo.jpg", "/download/paper.pdf", "/pages/paper.pdf/1",
                 "/pages/paper.pdf/1/image"):
        assert c.get(path).status_code == 404, path


def test_the_content_page_still_shows_its_images(client):
    c, _ = client
    page = c.get("/content/photo/1").text
    assert "/figures/Corp/photo/vlm/images/fig1.jpg" in page


def test_the_content_page_offers_two_working_signed_downloads(client):
    # the text view (Office files): its buttons are signed like the page view's, so they also work in the side panel,
    # which sends no login cookie; no link points to a route that no longer exists
    import re
    c, _ = client
    page = c.get("/content/talk/1").text
    links = re.findall(r"href='([^']*/download[^']*)'", page)
    assert sorted(l.split("?")[0] for l in links) == ["/download-original/talk", "/download-parsed/talk"]
    for link in links:
        assert "?t=" in link and c.get(link.replace("&amp;", "&")).status_code == 200, link
