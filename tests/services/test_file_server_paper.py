# The paper card /paper/<token>: an outside paper cited by a report or proposal, shown in Open WebUI's side panel like a
# library page: title, year, journal, DOI, the services that found it, per question the verdict and the checker's
# reason, the whole abstract with the selected question's most related sentences marked. Written by the Query pipe as
# OUTPUT_DIR/_runtime/papers/<24 hex>.json.
import importlib
import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
TOKEN = "0123456789abcdef01234567"
CARD = {"title": "High-Tc superconductor candidates proposed by machine learning", "year": "2024",
        "journal": "Machine Learning Science and Technology", "doi": "10.1088/2632-2153/ae04c1",
        "url": "https://doi.org/10.1088/2632-2153/ae04c1",
        "services": [{"name": "Semantic Scholar", "url": "https://www.semanticscholar.org/paper/" + "a" * 40},
                     {"name": "Offline index", "url": ""}],
        "sentences": ["We train a model on 7000 compounds.", "It proposes ambient-pressure candidates.",
                      "Tc above 300 K is predicted."],
        "questions": [{"label": "Question 1", "statement": "How can ML find ambient-pressure superconductors?",
                       "verdict": "Partly", "reason": "Gives a search protocol, not the data curation.", "marks": [1]},
                      {"label": "Question 4", "statement": "Can models recognise Tc above 150 K?",
                       "verdict": "Possibly answers it", "reason": "Predicts Tc above 300 K.", "marks": [2]}]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "_runtime" / "papers").mkdir(parents=True)
    (out / "_runtime" / "papers" / f"{TOKEN}.json").write_text(json.dumps(CARD))
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    (tmp_path / "input").mkdir()
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))
    sys.path.insert(0, str(ROOT / "services"))
    import file_server
    fs = importlib.reload(file_server)
    return TestClient(fs.app), fs, out


def test_the_card_shows_the_paper_its_verdicts_and_the_whole_abstract(client):
    c, fs, _ = client
    r = c.get(f"/paper/{TOKEN}", params={"source_id": "x"})            # as the side panel loads it
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    t = r.text
    for s in ("Machine Learning Science and Technology", "2024", "doi.org/10.1088/2632-2153/ae04c1",
              "Semantic Scholar", "Offline index", "Question 1", "Partly",
              "Gives a search protocol, not the data curation.", "Question 4", "Possibly answers it",
              "What the check read", "selected by the reranker"):
        assert s in t, s
    assert all(x in t for x in CARD["sentences"])                      # the whole abstract
    assert re.search(r"<a [^>]*href='https://doi\.org/10\.1088/2632-2153/ae04c1'[^>]*target='_blank'", t)
    assert "href='https://www.semanticscholar.org/paper/" in t
    # the first question selected on open: its sentence marked, the other not
    assert re.search(r"<span class='s mark' data-i='1'>It proposes", t) and "<span class='s' data-i='2'>Tc" in t
    assert "data-marks='1'" in t and "data-marks='2'" in t


def test_the_card_escapes_everything_and_runs_only_its_own_script(client):
    c, fs, out = client
    bad = {**CARD, "title": "<script>alert(1)</script>", "url": "javascript:alert(1)", "doi": "10.1/<b>",
           "services": [{"name": "<i>x</i>", "url": "javascript:alert(2)"}],
           "sentences": ["<img src=x onerror=alert(3)>"],
           "questions": [{"label": "Question 1", "statement": "<b>q</b>", "verdict": "Partly",
                          "reason": "\"'><svg onload=alert(4)>", "marks": [0]}]}
    tok = "f" * 24
    (out / "_runtime" / "papers" / f"{tok}.json").write_text(json.dumps(bad))
    t = c.get(f"/paper/{tok}").text
    rest = t.replace(fs._THEME_JS, "").replace(fs._PAPER_JS, "")
    assert "<script" not in rest and "<img" not in rest and "<svg" not in rest and "<i>x" not in rest
    assert "<script>alert(1)" not in t and "&lt;script&gt;alert(1)&lt;/script&gt;" in t
    assert "javascript:" not in t and "&lt;img src=x onerror=alert(3)&gt;" in t and "<svg onload" not in t


@pytest.mark.parametrize("tok", ["0" * 23, "g" * 24, "../" + "0" * 21, "a" * 24])
def test_an_unknown_card_is_a_themed_not_found(client, tok):
    c, _, _ = client
    r = c.get(f"/paper/{tok}")
    assert r.status_code == 404
    if r.headers["content-type"].startswith("text/html"):
        assert "no longer available" in r.text


def test_a_question_without_marked_sentences_says_so(client):
    c, _, out = client
    tok = "e" * 24
    (out / "_runtime" / "papers" / f"{tok}.json").write_text(json.dumps(
        {**CARD, "questions": [{**CARD["questions"][0], "marks": []}]}))
    t = c.get(f"/paper/{tok}").text
    assert "No sentence stands out" in t and "class='s mark'" not in t


def test_the_card_says_when_the_check_read_more_than_the_abstract_it_shows(client):
    c, _, out = client
    assert "What the check read: the abstract</h2>" in c.get(f"/paper/{TOKEN}").text
    tok = "f" * 24
    (out / "_runtime" / "papers" / f"{tok}.json").write_text(json.dumps({**CARD, "full_text": True}))
    assert "the abstract and passages of the full text, not shown here" in c.get(f"/paper/{tok}").text
