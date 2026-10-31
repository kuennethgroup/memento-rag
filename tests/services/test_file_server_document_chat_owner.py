# A Document Chat file opens only for the account that attached it. The file server asks Open WebUI whose login the
# request carries (the caller's own cookie, never a header someone could set) and compares it with the owner recorded
# in the chat's folder (INPUT_ROOT/_temp/<chat id>/.owner); another account, no login or no record: 404. Signed
# download links (the side panel sends no cookie) are issued only by a page view that passed this check. Library
# documents stay shared and need no lookup.
import importlib
import json
import re
import sys
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
CHAT, SRC = "3f2b8c1e-6a4d-4e0b-9c7a-1d2e3f4a5b6c", "Report-0123456789ab"
ALICE, BOB = {"cookie": "token=alice"}, {"cookie": "token=bob"}


def _pdf(path: Path, pages: int = 2):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path)


def _parsed(out: Path, corpus: str, stem: str):
    d = out / corpus / stem / "vlm"
    (d / "images").mkdir(parents=True)
    (d / f"{stem}_content_list.json").write_text(json.dumps([{"type": "text", "text": "x", "page_idx": 0},
                                                             {"type": "image", "img_path": "images/fig1.jpg",
                                                              "page_idx": 0}]))
    (d / "images" / "fig1.jpg").write_bytes(b"jpg")


@pytest.fixture
def server(tmp_path, monkeypatch):
    inp, out = tmp_path / "in", tmp_path / "out"
    (inp / "_temp" / CHAT).mkdir(parents=True)
    _pdf(inp / "_temp" / CHAT / f"{SRC}.pdf")
    (inp / "_temp" / CHAT / ".owner").write_text("alice")
    _parsed(out, "_temp", SRC)
    (inp / "Corp").mkdir()
    _pdf(inp / "Corp" / "paper.pdf")
    _parsed(out, "Corp", "paper")
    monkeypatch.setenv("INPUT_DIR", str(inp))
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    sys.path.insert(0, str(ROOT / "services"))
    import file_server
    fs = importlib.reload(file_server)
    asked = []

    def session_user(cookie, auth):          # Open WebUI's answer for a login: the approved account's id
        asked.append(cookie)
        return {"token=alice": "alice", "token=bob": "bob"}.get(cookie)
    monkeypatch.setattr(fs, "_session_user", session_user)
    return TestClient(fs.app), fs, asked


ROUTES = (f"/view/{SRC}/1", f"/view/{SRC}/1/image", f"/content/{SRC}/1",
          f"/figures/_temp/{SRC}/vlm/images/fig1.jpg", f"/download-original/{SRC}", f"/download-parsed/{SRC}")


def test_the_owner_opens_every_route(server):
    c, _, _ = server
    for path in ROUTES:
        assert c.get(path, headers=ALICE).status_code == 200, path


def test_another_account_or_no_login_gets_nothing(server):
    c, _, _ = server
    for headers in (BOB, {}):
        for path in ROUTES:
            assert c.get(path, headers=headers).status_code == 404, (path, headers)


def test_the_signed_links_of_the_owners_page_work_without_a_cookie(server):
    c, _, _ = server
    page = c.get(f"/view/{SRC}/1", headers=ALICE).text
    links = re.findall(r"/download-(?:original|parsed)/[^'\"]+\?t=[0-9a-f.]+", page)
    assert len(links) == 2
    for link in links:
        assert c.get(link).status_code == 200, link
    assert "?t=" not in c.get(f"/view/{SRC}/1", headers=BOB).text


def test_a_document_chat_file_without_an_owner_record_opens_for_nobody(server):
    c, fs, _ = server
    (fs.INPUT_DIR / "_temp" / CHAT / ".owner").unlink()
    assert c.get(f"/view/{SRC}/1", headers=ALICE).status_code == 404


def test_library_documents_stay_shared_without_a_lookup(server):
    c, _, asked = server
    for path in ("/view/paper/1", "/content/paper/1", "/figures/Corp/paper/vlm/images/fig1.jpg"):
        assert c.get(path, headers=BOB).status_code == 200, path
    assert asked == []


def test_a_login_is_looked_up_once_per_minute(server):
    c, _, asked = server
    for _ in range(3):
        c.get(f"/view/{SRC}/1/image", headers=ALICE)
    assert asked == ["token=alice"]


def test_the_owner_record_name_matches_document_chat():
    from memento.document_chat import intake
    sys.path.insert(0, str(ROOT / "services"))
    import file_server
    assert file_server.OWNER_FILE == intake.OWNER_FILE


def test_the_file_server_asks_open_webui_on_the_host_and_listens_on_localhost():
    import yaml
    files = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["files"]
    assert files["network_mode"] == "host" and "ports" not in files
    assert files["environment"]["FILES_HOST"] == "127.0.0.1"
    assert files["environment"]["OPENWEBUI_URL"] == "http://127.0.0.1:${OPENWEBUI_PORT:-8086}"
    src = (ROOT / "services/file_server.py").read_text()
    assert 'os.getenv("FILES_HOST", "127.0.0.1")' in src and 'host="0.0.0.0"' not in src


def test_a_figure_path_cannot_step_around_the_check(server):
    c, fs, _ = server
    (fs.OUTPUT_DIR / "_temp" / "loose.jpg").write_bytes(b"jpg")
    for path in (f"/figures/Corp/%2e%2e/_temp/{SRC}/vlm/images/fig1.jpg", "/figures/_temp/loose.jpg"):
        assert c.get(path, headers=BOB).status_code == 404, path
    assert c.get(f"/figures/Corp/%2e%2e/_temp/{SRC}/vlm/images/fig1.jpg", headers=ALICE).status_code == 200


def test_the_readme_says_who_opens_a_document_chat_file():
    flat = " ".join((ROOT / "README.md").read_text().split())
    assert "open only for the account that attached them" in flat and "any approved account that has" not in flat


def test_the_readme_says_how_long_a_download_link_works(server):
    _, fs, _ = server
    flat = " ".join((ROOT / "README.md").read_text().split())
    assert fs._LINK_TTL == 4 * 3600 and "download link" in flat and "four hours" in flat
