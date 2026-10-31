# No key, token, secret or password is written into the code: a variable that holds one is read from the environment
# with an empty (or no) fallback, so a published file never carries a real value.
import re
import subprocess

from memento.config.paths import PROJECT_ROOT

FALLBACK = re.compile(r"""(?:getenv|environ\.get)\(\s*["'][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*["']\s*,"""
                      r"""\s*["'][A-Za-z0-9_\-]{16,}["']""")
SKIP = (".pdf", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".gz", ".zip", ".docx", ".xlsx", ".pptx")


def test_no_secret_variable_has_a_literal_fallback():
    files = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for name in files:
        if name.endswith(SKIP):
            continue
        try:
            text = (PROJECT_ROOT / name).read_text(errors="ignore")
        except (FileNotFoundError, IsADirectoryError):
            continue
        hits += [f"{name}:{text.count(chr(10), 0, m.start()) + 1}" for m in FALLBACK.finditer(text)]
    assert not hits, hits
