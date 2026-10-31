# The label, banner and section lists are the same in both domains. The upload code reads these lists once per process
# from config/.env's domain, so both copies' uploads use one list: both domain files carry the union, checked on the
# parsed corpora (no caption gains a false label).
import sys
from pathlib import Path

import pytest
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.config.domain_loader import load_domain  # noqa: E402

COMP, POLY = load_domain("computational-materials"), load_domain("polymer-engineering")
# the computational-materials section hints before the union: the titles they give must not change
TODAY_COMP_HINTS = {"Abstract": ["abstract", "kurzfassung", "zusammenfassung"], "Keywords": ["keywords", "key words", "index terms", "schlüsselwörter", "schlagwörter", "stichwörter", "stichworte"], "Introduction": ["introduction", "einleitung", "einführung"], "Background": ["background", "related work", "literature review", "previous work", "hintergrund", "stand der technik", "stand der forschung", "verwandte arbeiten"], "Methods": ["method", "methodology", "computational details", "simulation details", "computational methods", "materials and methods", "experimental", "methode", "methodik", "experimentell", "rechendetails", "simulationsdetails"], "Results": ["results", "ergebnis"], "Results and Discussion": ["results", "ergebnis", "discussion", "diskussion"], "Discussion": ["discussion", "diskussion"], "Conclusion": ["conclusion", "concluding", "fazit", "schlussfolgerung", "summary", "outlook", "ausblick", "future scope", "future work", "future direction", "future research", "zukünftige arbeit", "zukünftige forschung", "künftige forschung"], "Declarations": ["acknowledg", "author contributions", "data availability", "danksagung"], "References": ["references", "bibliography", "literatur", "literaturverzeichnis", "quellenverzeichnis", "referenzen"], "Appendix": ["appendix", "supplementary", "supporting information", "anhang", "zusatzmaterial"]}


def test_the_label_banner_and_section_lists_are_the_same_in_both_domains():
    assert COMP["structural_labels"] == POLY["structural_labels"]
    assert COMP["parsing"]["title_banner_tokens"] == POLY["parsing"]["title_banner_tokens"]
    assert COMP["section_hints"] == POLY["section_hints"]
    assert {"plot", "plate"} <= set(COMP["structural_labels"]["figure"])
    assert {"nomenklatur", "formelzeichen"} <= set(COMP["section_hints"]["Nomenclature"])


def test_the_union_lost_nothing_of_todays_comp_mat_lists():
    for key, words in TODAY_COMP_HINTS.items():
        assert set(words) <= set(COMP["section_hints"][key]), key


def test_plot_and_plate_are_families_of_their_own():
    from memento.ingestion import enrich_parsed_content_list as en
    fams = en._label_families("figure")
    assert {"plot"} in fams and {"plate"} in fams
    assert {"fig", "figs", "figure", "figures"} in fams and {"abb", "abbildung", "abbildungen"} in fams


@pytest.mark.parametrize("caption, label", [
    ("Figure 2: Band structure plot of the doped cell", "Figure 2"),
    ("Fig. 5: Charge density on the plate 3 nm above the surface", "Fig. 5"),
    ("Plate 3: Fracture surface after the tensile test", "Plate 3"),
])
def test_plate_and_plot_in_a_caption_keep_the_captions_own_label(caption, label):
    from memento.ingestion import enrich_parsed_content_list as en
    assert en._extract_figure_label(caption) == label


def test_a_nomenklatur_heading_is_never_the_title():
    from memento.ingestion import extract_metadata as em
    items = [{"type": "text", "text_level": 1, "text": "Nomenklatur"},
             {"type": "text", "text_level": 1, "text": "Rheologische Untersuchungen an thermoplastischen Schaumstoffen"}]
    assert em.extract_paper_metadata(items)["title"] == "Rheologische Untersuchungen an thermoplastischen Schaumstoffen"
