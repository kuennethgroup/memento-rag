# Documents by a person from positive evidence. A mention of the surname near a matching initial is not enough (a title,
# a supervisor, a similar name); the author lines are found first (lines of names only; never a role line, an
# affiliation, an e-mail), and each document is "author", "possible_identity" (only an initial or a short form of the
# first name matches), "possible_role" (the full name elsewhere in the front matter) or "no" (a title, a role, an
# institute, nothing).
import pytest

from memento.library import front_matter as da
from memento.query.flows import library_listing

BRUECKNER = "Christopher Brueckner"


# ── negatives ──

def test_a_title_mention_is_not_authorship():
    assert da.match_person("# Landau theory of phase transitions\n\nAlice Jones\n", "Landau") == "no"
    assert da.match_person("Landau theory of phase transitions\n\nAlice Jones\n", "Landau") == "no"     # no heading


def test_a_supervisor_is_not_an_author():
    front = "# A thesis on polymers\n\nAlice Jones\n\nSupervisor: Christopher Brückner\n"
    assert da.match_person(front, BRUECKNER) == "no"
    front = "Masterarbeit\n\nvorgelegt von\nMax Mustermann\n\nErstgutachter:\nProf. Dr. Christopher Brückner\n"
    assert da.match_person(front, BRUECKNER) == "no"
    assert da.match_person(front, "Max Mustermann") == "author"


def test_a_trailing_letter_needs_an_affiliation_starting_with_it():
    assert da.match_person("# T\n\nA. Schmidt, B. Other\n", "Alice Schmid") == "no"


def test_full_first_names_must_agree():
    assert da.match_person("# T\n\nCharles Brückner, B. Other\n", BRUECKNER) == "no"
    # a short form is not proof of the same person: possibly the same person
    assert da.match_person("# T\n\nChris Brueckner, B. Other\n", BRUECKNER) == "possible_identity"
    assert da.match_person("# A study\n\nAnnabelle Smith, Bob Jones\n", "Anna Smith") == "possible_identity"
    assert da.match_person("# A study\n\nAnna Smith, Bob Jones\n", "Anna Smith") == "author"
    assert da.match_person("# A study\n\nAnnika Smith, Bob Jones\n", "Anna Smith") == "no"


# ── more cases ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_an_institute_named_after_a_person_is_not_an_author():
    front = "# T\n\nAlice Jones\n\nMax Planck Institute for Polymer Research, Mainz, Germany\n"
    assert da.match_person(front, "Max Planck") == "no"


def test_same_person_as_author_and_supervisor_is_an_author():
    front = "Masterarbeit\n\nvorgelegt von\nAlice Jones\n\nBetreuer:\nProf. Dr. Alice Jones\n"
    assert da.match_person(front, "Alice Jones") == "author"


def test_name_in_running_text_is_role_unclear():
    # (an editor is a role: "edited by Alice Jones" is "no")
    assert da.match_person("Polymer informatics notes prepared with Alice Jones for the group\n", "Alice Jones") == \
        "possible_role"
    assert da.match_person("Polymer informatics edited by Alice Jones\n", "Alice Jones") == "no"


def test_an_initial_only_is_possibly_the_same_person():
    front = "# T\n\nU. Hoelzer $^{a}$ , C. Brückner $^{e}$ , T. Novakowski $^{a}$\n$^{a}$ NaMLab gGmbH, Dresden\n"
    assert da.match_person(front, BRUECKNER) == "possible_identity"
    assert da.match_person(front, "Brueckner") == "author"                    # surname asked: the surname decides


def test_e_mails_do_not_count():
    assert da.match_person("# T\n\nAlice Jones\n\ncontact: christopher.brueckner@example.org\n", BRUECKNER) == "no"


# ── real front matter ──

SINGLE_DOMAIN = ("# Evidence of single domain switching in Hafnium Oxide based FeFETs: enabler for multi-level FeFET "
                 "memory cells\nH. Begovic $_{a,f}$ , S. Wrobelskia, J. Ebnera, M. Kovacica, S. Grübera, "
                 "S. Lewandowskyd, J. Grüberb, P. Jankowskib, J. Franzd, S. Harmsend, S. Wisniewskid, C. Vogta, "
                 "T. Brandta, A. Kesslere, C. Brücknere, R. van Dalend, U. Hölzera, T. Novakowskia,c\n"
                 "aNaMLab gGMBH, bFraunhofer IPMS-CNT, cIHM TU-Dresden, dGLOBALFOUNDRIES, Dresden, Germany,\n"
                 "eHAW-München, München, Germany; fDEIB, Politecnico di Milano, Milan, Italy; Email: "
                 "stefan.wrobelski@example.org\n")
STABILIZATION = ("# Stabilization of ferroelectric properties in\n# Hafnia with doping\nRESEARCH INSTITUTION\n"
                 "Modeling and Simulation Lab, Department of Applied Sciences and Mechatronics, University of Applied "
                 "Sciences Munich\nPRINCIPAL INVESTIGATOR\nAlfred Kessler\nRESEARCHERS\nRobin Brandner, Christopher "
                 "Brückner\n")
AX = ("# Ax: A Platform for Adaptive Experimentation\nMiles Carter $^{1,*}$ Elizabeth Morandi $^{1,*}$ Louis C. Ferrand "
      "$^{1,*}$ Sait Yildiz $^{1,*}$\n$^{1}$ Meta\n")
BOTORCH = ("# BOTORCH: A Framework for Efficient Monte-Carlo Bayesian Optimization\n# Maximilian Brenner\nFacebook\n"
           "brenner@example.com\n# Brian Holloway\nFacebook\n")
BOOK = "Zhiwei Lan\n# Modern Computational Materials Science\nTheory, Methods, and Applications\n"
CHARACTERISTICS = ("# Characteristics of Low-Energy Phases of Hafnia and Zirconia from Density Functional Theory "
                   "Calculations\nLuis Moreira Santos, Richard Gassner, Christopher Brueckner, and Alfred Kessler\\*\n")


def test_glued_affiliation_letters():
    assert da.match_person(SINGLE_DOMAIN, BRUECKNER) == "possible_identity"      # the trailing e is an affiliation
                                                                                # letter
    assert da.match_person(SINGLE_DOMAIN, "Kessler") == "author"                 # "A. Kesslere"
    assert da.match_person(SINGLE_DOMAIN, "Stefan Wrobelski") == "possible_identity"


def test_names_on_their_own_line_after_a_label():
    assert da.match_person(STABILIZATION, "Alfred Kessler") == "author"
    assert da.match_person(STABILIZATION, BRUECKNER) == "author"


def test_names_separated_only_by_superscripts():
    assert da.match_person(AX, "Elizabeth Morandi") == "author"
    assert da.match_person(AX, "Louis Ferrand") == "author"


def test_names_as_headings_and_before_the_title():
    assert da.match_person(BOTORCH, "Maximilian Brenner") == "author"
    assert da.match_person(BOOK, "Zhiwei Lan") == "author"
    assert da.match_person(BOOK, "Theory Methods") == "no"


def test_and_before_the_last_name():
    assert da.match_person(CHARACTERISTICS, "Alfred Kessler") == "author"
    assert da.match_person(CHARACTERISTICS, BRUECKNER) == "author"


def test_classify_groups_the_documents():
    fronts = {"Characteristics": CHARACTERISTICS, "Evidence": SINGLE_DOMAIN, "Stabilization": STABILIZATION,
              "Ax": AX, "Running": "Polymer informatics notes prepared with Christopher Brueckner for the group\n"}
    got = da.classify(fronts, BRUECKNER)
    assert got == {"author": ["Characteristics", "Stabilization"], "possible_identity": ["Evidence"],
                   "possible_role": ["Running"]}
    assert da.documents_by(fronts, BRUECKNER) == ["Characteristics", "Stabilization"]


FIELD_CYCLING = ("# Impact of field cycling on $\\mathrm{HfO_2}$ based non-volatile memory devices\nU. Hoelzer "
                 "$^{a,\\dagger}$ , M. Kovačić $^{a}$ , E. D. Hartley $^{f}$ , S. Wisniewski $^{d}$ , R. van Dalen $^{d}$ , "
                 "A. Kessler $^{e}$ , C. Brückner $^{e}$ , T. Novakowski $^{a,c}$ $^{a}$ NaMLab gGmbH, 01197 Dresden, "
                 "Germany\n")


def test_an_author_line_that_runs_into_the_first_affiliation():
    # the names before the affiliation's mark are an author line; the affiliation itself is not
    assert da.match_person(FIELD_CYCLING, "Kessler") == "author"
    assert da.match_person(FIELD_CYCLING, "Alfred Kessler") == "possible_identity"
    assert da.match_person(FIELD_CYCLING, "Rob van Dalen") == "possible_identity"
    assert da.match_person(FIELD_CYCLING, "NaMLab") == "no"


# ── the answer ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_listing_shows_the_uncertain_documents_apart():
    import openwebui.functions.query_pipe as qp
    links = {t: f"/files/{t}" for t in ("A", "B", "C", "D")}
    en = library_listing.author_listing(BRUECKNER, ["B", "A"], links, de=False, missing=0, possible=["C"], role_unclear=["D"])
    assert en.startswith("**2 documents** in the library name Christopher Brueckner as an author:")
    assert "**Possibly the same person** (only an initial or a short form of the first name matches), 1 document:" in en
    assert "**Named at the start of the document, role unclear**, 1 document:" in en
    assert en.index("Possibly the same person") > en.index("[A]") and "[C]" in en and "[D]" in en
    de = library_listing.author_listing("Alfred Kessler", [], links, de=True, missing=0, possible=["C", "D"])
    assert "**Möglicherweise dieselbe Person** (nur eine Initiale oder eine Kurzform des Vornamens passt), 2 Dokumente:" in de
    assert library_listing.author_listing(BRUECKNER, ["A"], links, de=False, missing=0) == \
        library_listing.author_listing(BRUECKNER, ["A"], links, de=False, missing=0, possible=[], role_unclear=[])


def test_topic_note_names_the_documents_not_searched():
    import openwebui.functions.query_pipe as qp
    assert library_listing.author_possible_note(BRUECKNER, 2, de=False) == \
        f"_2 more documents may be by {BRUECKNER} (only an initial or a short form of the first name matches) and were not searched._\n\n"
    assert library_listing.author_possible_note("Alfred Kessler", 1, de=True) == \
        "_1 weiteres Dokument könnte von Alfred Kessler sein (nur eine Initiale oder eine Kurzform des Vornamens passt) und wurde nicht durchsucht._\n\n"
    assert library_listing.author_possible_note(BRUECKNER, 0, de=False) == ""
