# Extraction noise rules.
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.extract_entities import _is_junk, extract_entities  # noqa: E402
from memento.graph.build.llm_enrich import build_enrich_prompt  # noqa: E402


def test_prompt_excludes_non_scientific_mentions():
    p = build_enrich_prompt("overview", [(0, "text")], want_context=False, want_entities=True)
    for phrase in ("person names", "grant", "software", "generic"):
        assert phrase in p


def test_person_names_and_grant_ids_are_junk():
    for s in ("Prof. John Halloran", "D.C. Sinclair", "Dr. Anna Weber", "DE-AC02-05CH11231",
              "EP/K008749/1"):
        assert _is_junk(s), s
    for s in ("Al2O3", "3Y-TZP", "spark plasma sintering", "Ni", "SiC", "HfB2", "8YSZ", "PA66", "SEM"):
        assert not _is_junk(s), s


def test_filtered_names_never_become_entities():
    r = extract_entities("Samples were supplied by Prof. John Halloran (grant DE-AC02-05CH11231) and sintered.",
                         use_llm=True, precomputed={"entities": [{"name": "Prof. John Halloran", "type": "Condition"},
                                                                 {"name": "DE-AC02-05CH11231", "type": "Condition"},
                                                                 {"name": "sintering", "type": "Process"}],
                                                    "relations": []})
    names = [e.name for e in r.entities]
    assert "Prof. John Halloran" not in names and "DE-AC02-05CH11231" not in names


def test_fractional_formulas_and_dotted_abbreviations_are_kept():
    # the grant-id rule keeps fractional stoichiometry, the initials-only person rule keeps dotted technique/phase
    # abbreviations
    for s in ("(Mg0.25(1-x)Co0.25Ni0.25Zn0.25)O", "(Hf1/3Zr1/3Ce1/3)O2", "(Ta1/3Zr1/3Nb1/3)C",
              "EP1-3_80/20", "C.S.H.", "B.E.T.", "T.E.M."):
        assert not _is_junk(s), s
