# The research-gap answer of one document in the calm layout: a count table first, then one collapsed block per kind of
# point, and the model's own suggestions set apart with their label.
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import passages as gr  # noqa: E402

ANSWER = ("### Limitations\n1. Handcrafted fingerprints do not generalise [1].\n2. No direct comparison [2].\n"
          "3. Only random copolymers [9].\n\n### Future work\n4. Standardised benchmarks [4].\n"
          "5. A decoder for fingerprints [6].\n")


def test_calm_layout_groups_each_heading():
    out = gr.calm_gap_answer(ANSWER)
    assert out.index("| What the authors state | Points |") < out.index("<details")
    assert "| Limitations | 3 |" in out and "| Future work | 2 |" in out
    assert out.count('<details type="memento_gaps">') == 2
    assert "<summary>Limitations (3)</summary>" in out and "<summary>Future work (2)</summary>" in out
    assert "3. Only random copolymers [9]." in out and "4. Standardised benchmarks [4]." in out
    assert "\n\n<details" in out and "</details>\n\n" in out            # blank lines around every block


def test_calm_layout_is_idempotent():
    once = gr.calm_gap_answer(ANSWER)
    assert gr.calm_gap_answer(once) == once


def test_calm_layout_keeps_a_no_gaps_sentence():
    s = "The passages do not state any genuine research gaps."
    assert gr.calm_gap_answer(s) == s and gr.calm_gap_answer("") == ""


def test_calm_layout_german_headings():
    de = "### Einschränkungen\n1. Nur zufällige Copolymere [3].\n\n### Zukünftige Arbeiten\n2. Ein Decoder [6].\n"
    out = gr.calm_gap_answer(de)
    assert "| Was die Autoren nennen | Punkte |" in out and "| Einschränkungen | 1 |" in out
    assert "<summary>Zukünftige Arbeiten (1)</summary>" in out


def test_calm_layout_keeps_text_before_the_first_heading():
    out = gr.calm_gap_answer("The authors name these points.\n\n" + ANSWER)
    assert out.startswith("The authors name these points.") and out.count("<details") == 2


# ── the model's own suggestions ("Possible extensions") ──
def test_suggestions_must_build_on_a_stated_point():
    raw = ("1. Test the fingerprints on block copolymers, building on the untested condition [9].\n"
           "2. Something with no basis [12].\n3. A suggestion without any citation.\n"
           "4. Compare with handcrafted fingerprints on one benchmark [2].\n")
    items = gr.parse_suggestions(raw, allowed={1, 2, 4, 6, 9})
    assert items == ["Test the fingerprints on block copolymers, building on the untested condition [9].",
                     "Compare with handcrafted fingerprints on one benchmark [2]."]
    many = "".join(f"{i}. Idea {i} [1].\n" for i in range(1, 9))
    assert len(gr.parse_suggestions(many, allowed={1})) == 5


def test_suggestions_block_has_the_label():
    out = gr.suggestions_block(["Idea A [2].", "Idea B [9]."], german=False)
    assert out.startswith("\n\n<details type=\"memento_ideas\">\n<summary>Possible extensions (suggested by the AI)"
                          "</summary>\n\n_Suggested by the AI, not stated by the authors. Please verify._\n\n")
    assert "1. Idea A [2].\n2. Idea B [9]." in out and out.rstrip().endswith("</details>")
    de = gr.suggestions_block(["Idee [3]."], german=True)
    assert "Mögliche Erweiterungen (von der KI vorgeschlagen)" in de
    assert "_Von der KI vorgeschlagen, nicht von den Autoren. Bitte prüfen._" in de


def test_no_suggestions_no_block():
    assert gr.suggestions_block([], german=False) == ""


def test_suggest_messages_carry_the_authors_points_and_the_language():
    msgs = gr.suggest_messages(ANSWER, [{"source": "PolyBERT", "page": 1, "text": "passage"}], "Was könnte man tun?")
    assert msgs[0]["content"] == gr.SUGGEST_SYSTEM_PROMPT
    assert "Handcrafted fingerprints" in msgs[1]["content"] and "German" in msgs[1]["content"]


# ── the gaps a study addresses are not its limitations ──
EARLIER = "Gaps in earlier work it addresses"
EARLIER_DE = "Lücken in früheren Arbeiten, die sie schließt"
FIVE = ("### Limitations\n1. Only one alloy system [3].\n\n### Future work\n2. Test other systems [8].\n\n"
        f"### {EARLIER}\n3. Inverse design of multi-phase ceramics is unexplored [1].\n"
        "4. Bayesian optimisation needs many samples [2].\n")


def test_the_instructions_separate_the_study_s_limitations_from_gaps_in_earlier_work():
    p = gr.GAP_SYSTEM_PROMPT
    assert f'"### {EARLIER}"' in p and f'"### {EARLIER_DE}"' in p
    assert "THIS study" in p and "Literature review" in p and "other authors' methods" in p
    order = [p.index(f'"### {h}"') for h in ("Limitations", "Uncertainties", "Untested conditions", "Future work",
                                             EARLIER)]
    assert order == sorted(order)                      # the new heading comes last
    assert EARLIER_DE in gr.GAP_HEADINGS_DE


def test_calm_layout_counts_and_folds_the_gaps_in_earlier_work():
    out = gr.calm_gap_answer(FIVE)
    assert f"| {EARLIER} | 2 |" in out and f"<summary>{EARLIER} (2)</summary>" in out
    assert out.count('<details type="memento_gaps">') == 3


def test_an_answer_with_only_the_german_earlier_work_heading_is_german():
    out = gr.calm_gap_answer(f"### {EARLIER_DE}\n1. Inverses Design ist unerforscht [1].\n")
    assert "| Was die Autoren nennen | Punkte |" in out


def test_suggestions_build_only_on_the_study_s_own_points():
    own = gr.study_points(FIVE)
    assert "Only one alloy system [3]" in own and "Test other systems [8]" in own
    assert EARLIER not in own and "[1]" not in own and "[2]" not in own
    de = f"### Einschränkungen\n1. Nur ein System [3].\n\n### {EARLIER_DE}\n2. Unerforscht [1].\n"
    assert gr.study_points(de).strip() == "### Einschränkungen\n1. Nur ein System [3]."
    assert gr.study_points(ANSWER) == ANSWER            # an answer without the heading: unchanged
    assert gr.study_points("") == ""


# ── books and reviews have no study of their own, so no "gaps it addresses" ──
FIELD = "Open problems in the field"
FIELD_DE = "Offene Probleme des Fachgebiets"


def test_books_and_reviews_get_open_problems_of_the_field():
    p = gr.GAP_SYSTEM_PROMPT
    assert f'"### {FIELD}"' in p and f'"### {FIELD_DE}"' in p
    for kind in ("textbook", "review", "survey"):
        assert kind in p, kind
    assert "reports no study of its own" in p
    assert p.index(f'"### {EARLIER}"') < p.index(f'"### {FIELD}"')      # after the five, last
    assert FIELD_DE in gr.GAP_HEADINGS_DE


def test_calm_layout_counts_the_open_problems_block():
    out = gr.calm_gap_answer(f"### Future work\n1. More data [2].\n\n### {FIELD}\n2. Quantum sensors are costly [7].\n")
    assert f"| {FIELD} | 1 |" in out and f"<summary>{FIELD} (1)</summary>" in out


# ── a question about one kind of gap shows that kind open, first ──
@pytest.mark.parametrize("q, focus", [
    ("What future work does the polyBERT paper propose?", "Future work"),
    ("What limitations does the polyBERT paper mention?", "Limitations"),
    ("Which uncertainties remain in this study?", "Uncertainties"),
    ("Welche zukünftigen Arbeiten schlägt die Arbeit vor?", "Zukünftige Arbeiten"),
    ("What research gaps does the document name?", None),
    ("What are the limitations and future work of this thesis?", None),           # two kinds: the usual layout
])
def test_the_focus_of_a_gap_question(q, focus):
    assert gr.gap_focus(q) == focus


def test_the_asked_kind_is_shown_open_first():
    out = gr.calm_gap_answer(ANSWER, focus="Future work")
    assert out.startswith("**Future work (2)**\n\n4. Standardised benchmarks [4].")
    assert out.index("**Future work (2)**") < out.index("| What the authors state | Points |")
    assert "<summary>Future work" not in out and "<summary>Limitations (3)</summary>" in out
    assert gr.calm_gap_answer(ANSWER, focus="Uncertainties") == gr.calm_gap_answer(ANSWER)   # not in the answer


def test_without_focus_the_layout_is_unchanged():
    assert gr.calm_gap_answer(ANSWER, focus=None) == gr.calm_gap_answer(ANSWER)


def test_blocks_follow_the_fixed_heading_order():
    # "Gaps in earlier work it addresses" first in the model's answer
    mixed = (f"### {EARLIER}\n1. Earlier gap [1].\n\n### Future work\n2. More [2].\n\n### Limitations\n3. Small [3].\n")
    out = gr.calm_gap_answer(mixed)
    assert out.index("| Limitations |") < out.index("| Future work |") < out.index(f"| {EARLIER} |")
    assert out.index("<summary>Limitations") < out.index("<summary>Future work") < out.index(f"<summary>{EARLIER}")
    de = gr.calm_gap_answer(f"### {EARLIER_DE}\n1. Alt [1].\n\n### Einschränkungen\n2. Klein [2].\n")
    assert de.index("<summary>Einschränkungen") < de.index(f"<summary>{EARLIER_DE}")
