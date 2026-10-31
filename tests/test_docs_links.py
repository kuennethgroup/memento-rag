# Every relative link and anchor in the project's documentation leads somewhere.
import re

import pytest
from memento.config.paths import PROJECT_ROOT as ROOT

DOCS = ["README.md", "INSTALL.md", "REFERENCE.md", "openwebui/README.md", "memento/ingestion/README.md",
        "evaluation/README.md"]


def _anchors(path):
    heads = re.findall(r"(?m)^#{1,6}\s+(.+)$", path.read_text())
    return {re.sub(r"[^\w\- ]", "", re.sub(r"[`*]", "", h.strip().lower())).replace(" ", "-") for h in heads}


@pytest.mark.parametrize("doc", [d for d in DOCS if (ROOT / d).exists()])
def test_every_local_link_resolves(doc):
    path = ROOT / doc
    bad = []
    for link in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
        if link.startswith(("http", "mailto:")):
            continue
        name, _, anchor = link.partition("#")
        target = (path.parent / name).resolve() if name else path
        if not target.exists() or (anchor and target.suffix == ".md" and anchor not in _anchors(target)):
            bad.append(link)
    assert bad == []
