# The file server's image installs pinned packages, dependencies included, so a rebuild gets the versions it was
# tested with.
import re
import shlex

from memento.config.paths import PROJECT_ROOT


def _pip_packages(dockerfile: str) -> list[str]:
    """What the image's `pip install` lines install; a `-r <file>` adds that file's lines."""
    text = re.sub(r"\\\n", " ", dockerfile)
    out = []
    for cmd in re.findall(r"pip install ([^&\n]+)", text):
        tokens = shlex.split(cmd)
        for i, t in enumerate(tokens):
            if t == "-r":
                lines = (PROJECT_ROOT / tokens[i + 1]).read_text().splitlines()
                out += [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
            elif not t.startswith("-") and (i == 0 or tokens[i - 1] != "-r"):
                out.append(t)
    return out


def test_the_file_server_image_pins_every_package():
    packages = _pip_packages((PROJECT_ROOT / "docker" / "Dockerfile.fileserver").read_text())
    assert [p for p in packages if "==" not in p] == []
    pinned = {re.split(r"[=\[]", p)[0].lower() for p in packages}
    assert {"fastapi", "starlette", "pydantic", "uvicorn", "pymupdf"} <= pinned
