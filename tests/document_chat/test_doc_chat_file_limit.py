# Document Chat's file limit in the browser (loader.js): more files in the message box than Document Chat reads blocks
# sending (tests/openwebui/js/loader_file_limit.test.mjs), with the pipe's limit. The dimmed buttons' tip while a file uploads
# (tests/openwebui/js/loader_doc_tip.test.mjs); a click on one while its file uploads starts once the upload is done
# (tests/openwebui/js/loader_doc_wait.test.mjs).
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_browser_blocks_sending_more_files_than_document_chat_reads():
    r = subprocess.run(["node", str(ROOT / "tests/openwebui/js/loader_file_limit.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_browser_limit_is_the_pipes():
    js = re.search(r"const MAX_FILES = (\d+);", (ROOT / "openwebui/loader.js").read_text())
    py = re.search(r'DOC_CHAT_MAX_FILES: int = int\(os\.getenv\("DOC_CHAT_MAX_FILES", "(\d+)"\)\)',
                   (ROOT / "openwebui/functions/query_pipe.py").read_text())
    assert js and py and js.group(1) == py.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_dimmed_buttons_say_a_file_is_still_uploading():
    r = subprocess.run(["node", str(ROOT / "tests/openwebui/js/loader_doc_tip.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_click_during_the_upload_starts_once_it_is_done():
    r = subprocess.run(["node", str(ROOT / "tests/openwebui/js/loader_doc_wait.test.mjs")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
