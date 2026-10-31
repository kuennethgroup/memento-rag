# The research-gap fallback reads text and lists only. For a document without gap tags, the research-gap answer reads
# its Discussion / Conclusion / Outlook passages. Limitations and future work are sentences, so figures, tables and
# equations in those sections are left out.
import sys
import types
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.research_gaps import passages as gr  # noqa: E402


class FakeClient:
    def __init__(self, payloads):
        self.payloads = payloads

    def scroll(self, **kw):
        return [types.SimpleNamespace(payload=p) for p in self.payloads], None


def test_fallback_skips_figures_and_tables():
    payloads = [
        {"type": "text", "text": "A limitation is the small sample.", "source": "d", "page": 5, "section": "Discussion"},
        {"type": "list", "text": "- test above 1600 C", "source": "d", "page": 5, "section": "Discussion"},
        {"type": "image", "text": "Figure 3. Density versus temperature.", "source": "d", "page": 5, "section": "Discussion"},
        {"type": "chart", "text": "Figure 4. XRD patterns.", "source": "d", "page": 6, "section": "Discussion"},
        {"type": "table", "text": "Table 2. Hardness values.", "source": "d", "page": 6, "section": "Conclusion"},
        {"type": "equation", "text": "$$E = mc^2$$", "source": "d", "page": 6, "section": "Conclusion"},
    ]
    got = gr.fetch_fallback_sections(FakeClient(payloads), "col", "d")
    assert [c["type"] for c in got] == ["text", "list"]


def test_fallback_without_type_counts_as_text():
    got = gr.fetch_fallback_sections(FakeClient([{"text": "Future work: x.", "source": "d", "page": 1,
                                                  "section": "Outlook"}]), "col", "d")
    assert len(got) == 1


def test_gap_prompt_names_the_answer_language():
    # the prompt ends with the question's language (a document with a German cover page must not get a German answer to
    # an English question)
    from memento.research_gaps.passages import build_gap_prompt
    chunks = [{"source": "doc", "page": 0, "section": "Conclusion", "text": "Dieses Dokument ist eine Zweitveröffentlichung."}]
    en = build_gap_prompt(chunks, 'What research gaps does the document "Über Keramik" name?')
    de = build_gap_prompt(chunks, "Welche Forschungslücken nennt das Dokument?")
    assert en.rstrip().endswith("Write the answer in English.")
    assert de.rstrip().endswith("Write the answer in German.")
