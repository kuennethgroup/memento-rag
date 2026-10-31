# tests/evaluation/test_rq2_p3_common.py
import json
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from evaluation.RQ2.questions.common import (Question, content_chunks, lang_of, llm_json,  # noqa: E402
                                             read_jsonl, universe_ids, write_jsonl)


def test_jsonl_roundtrip(tmp_path):
    q = Question("polymer-engineering:multihop:1", "polymer-engineering", "multihop", "Q?", "A.", "en", "pilot",
                 [{"source": "X", "page": 3, "role": "A", "fact": "a"}], [], [], {"bridge": "DOPO"})
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [q])
    assert read_jsonl(p) == [q]


def test_universe_ids_keeps_ids_verbatim(tmp_path, monkeypatch):
    import evaluation.RQ2.questions.common as c
    f = tmp_path / "pilot_polymer-engineering.txt"
    f.write_text("02_Prüfung Promotionsvortrag Albrecht \nDiss_Klein\n")
    monkeypatch.setattr(c, "PILOT_DIR", tmp_path)
    assert universe_ids("polymer-engineering", "pilot") == {"02_Prüfung Promotionsvortrag Albrecht ", "Diss_Klein"}
    assert universe_ids("polymer-engineering", "full") is None


def test_content_chunks_filters_sections_documents_and_universe():
    chunks = [{"source": "A", "page": 0, "type": "text", "section": "Results", "text": "x"},
              {"source": "A", "page": 9, "type": "text", "section": "References", "text": "y"},
              {"source": "IMG", "page": 0, "type": "image", "section": "Results", "text": "z"},
              {"source": "COPY", "page": 0, "type": "text", "section": "Results", "text": "w"},
              {"source": "B", "page": 0, "type": "text", "section": "Results", "text": "v"}]
    reg = {"A": {"type": "dissertation"}, "IMG": {"type": "image_artifact"},
           "COPY": {"type": "chapter", "copy_of": "A"}, "B": {"type": "document"}}
    kept = content_chunks(chunks, reg, allowed={"A", "IMG", "COPY"})
    assert [(c["source"], c["page"]) for c in kept] == [("A", 0)]


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.chat = self
        self.completions = self

    def create(self, **kw):
        assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        msg = type("M", (), {"content": self.content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_llm_json_tolerates_fences_and_returns_none_on_garbage():
    assert llm_json(FakeLLM('```json\n{"a": 1}\n```'), "m", "s", "u") == {"a": 1}
    assert llm_json(FakeLLM("no json here"), "m", "s", "u") is None


def test_lang_of():
    assert lang_of(["Die Probe wird mit dem Epoxidharz und der Härter gemischt."]) == "de"
    assert lang_of(["The sample was mixed with the epoxy resin."]) == "en"


def test_list_sections_are_excluded_whatever_their_heading():
    # no question from an abbreviation list labelled "Nomenclature"
    from evaluation.RQ2.questions.common import is_excluded_section
    for s in ("Nomenclature", "Wiederkehrende Abkürzungen", "ABBREVIATIONS, EQUATIONS & INDICES",
              "Reoccurring notations and symbols", "Abbildungs- und Tabellenverzeichnis", "**Formelverzeichnis**",
              "GLOSSARY", "References", "Front Matter"):
        assert is_excluded_section(s), s
    for s in ("Results and Discussion", "A.1 Abkühlvorrichtung zur Untersuchung gedehnter Polymerschmelzen",
              "Space Group Notation", "Methods", None):
        assert not is_excluded_section(s), s
