# The sources list under an answer that cites one document names it once and lists page and section only. Presentation
# only: what the model gets, in this turn and as history in the next, stays exactly as before.
import re

import openwebui.functions.query_pipe as qp
from memento.query import presentation, request, texts
from memento.query.scope import conversation

DOC = "Polymer Engineering Sci - 2026 - Castro‐Landinez - Data‐Driven Design of Co‐Continuous Morphology.pdf"


def _hit(source, page, section=None):
    return {"source": source, "page": page, "page_end": None, "section_raw": section, "text": "x"}


HITS = [_hit(DOC, 0, "ABSTRACT"), _hit(DOC, 1, "Highlights"), _hit(DOC, 2, "2.3 Processing"),
        _hit(DOC, 2, "2.3 Processing"), _hit(DOC, 9, "Conclusion")]


def _lines(block):
    return [l for l in block.splitlines() if l.startswith("- (")]


def test_one_document_is_named_once_and_lines_show_page_and_section():
    block = presentation.sources_marker(HITS, {0, 1, 3})
    assert "Cited in this answer: 3 passages of this document" in block
    assert f"All from **{presentation.shown(DOC)}**." in block
    lines = _lines(block)
    assert [l.split(")")[0] for l in lines] == ["- (1", "- (2", "- (4"]              # every cited number, in order
    assert all(presentation.shown(DOC) not in l for l in lines)
    assert lines[0].startswith("- (1) p. 1")
    assert "Memento read" not in block                                              # no "of the 76 sources"


def test_one_passage():
    assert "Cited in this answer: 1 passage of this document" in presentation.sources_marker(HITS, {4})


def test_two_documents_keep_todays_list():
    hits = HITS[:2] + [_hit("Other paper", 3, "Results")]
    block = presentation.sources_marker(hits, {0, 2})
    assert "Cited in this answer: 2 of the 3 sources Memento read" in block and "All from" not in block
    assert _lines(block) == [f"- (1) {presentation.citation_label(presentation.source_name(hits[0]))}",
                             f"- (3) {presentation.citation_label(presentation.source_name(hits[2]))}"]


def test_the_rework_path_with_stored_names_and_a_name_with_its_own_brackets():
    names = ["BayBE: a Bayesian Back End (BayBE) (p. 5, §3.2)", "BayBE: a Bayesian Back End (BayBE) (p. 7, §4)"]
    block = presentation.names_marker(names, {0, 1}, docs=["baybe.pdf", "baybe.pdf"])
    assert "All from **BayBE: a Bayesian Back End (BayBE)**." in block
    assert _lines(block) == ["- (1) p. 5, §3.2", "- (2) p. 7, §4"]


def test_a_name_without_a_location_keeps_todays_list():
    block = presentation.names_marker(["Doc", "Doc (p. 2)"], {0, 1}, docs=["d", "d"])
    assert "All from" not in block and "Memento read" in block


def test_history_text_is_unchanged_by_the_new_layout():
    # the model's input in the next turn: the labels read back from the list are the same as with the old layout
    answer = "Aim of the study [1]. Methods [2][4]."
    new = answer + presentation.sources_marker(HITS, {0, 1, 3})
    old_lines = "\n".join(f"- ({i + 1}) {presentation.citation_label(presentation.source_name(HITS[i]))}" for i in (0, 1, 3))
    old = answer + ('\n\n<details type="memento_sources">\n<summary>Cited in this answer: 3 of the 5 sources Memento '
                    'read</summary>\n\n_The numbers match the links in the text._\n\n' + old_lines + "\n</details>\n")
    assert conversation.history_text(new) == conversation.history_text(old)
    assert presentation.shown(DOC) in conversation.history_text(new)


def test_german_forms():
    block = presentation.sources_marker(HITS, {0, 1})
    token = request.LANG.set("de")
    try:
        de = texts.tr(block)
    finally:
        request.LANG.reset(token)
    assert "In dieser Antwort zitiert: 2 Abschnitte dieses Dokuments" in de
    assert f"Alle aus **{presentation.shown(DOC)}**." in de
    assert "Die Nummern entsprechen den Links im Text." in de
    one = presentation.sources_marker(HITS, {0})
    token = request.LANG.set("de")
    try:
        assert "In dieser Antwort zitiert: 1 Abschnitt dieses Dokuments" in texts.tr(one)
    finally:
        request.LANG.reset(token)
    # and a German block read back as history gives the same labels
    assert conversation.history_text("A [1]." + de) == conversation.history_text("A [1]." + block)


def test_no_em_dash():
    assert "—" not in presentation.sources_marker(HITS, {0, 1})


# ── a reworked answer takes the document from the stored citations' identities ("X#N" / "X#gapN"), never from the
# display name; unknown identity keeps the full list ──
def _stored(ident, name, k):
    return {"document": ["x"], "source": {"name": name}, "metadata": [{"source": ident, "name": f"[{k}] {name}"}]}


def test_rework_same_name_different_files_keeps_the_full_list():
    srcs = [_stored("report-token1#0", "report.pdf (p. 1)", 1), _stored("report-token2#1", "report.pdf (p. 2)", 2)]
    block = presentation.names_marker([s["source"]["name"] for s in srcs], {0, 1}, docs=presentation.stored_doc_ids(srcs))
    assert "All from" not in block and "Memento read" in block


def test_rework_same_name_same_file_is_compact():
    srcs = [_stored("report-token1#0", "report.pdf (p. 1)", 1), _stored("report-token1#gap1", "report.pdf (p. 2)", 2)]
    block = presentation.names_marker([s["source"]["name"] for s in srcs], {0, 1}, docs=presentation.stored_doc_ids(srcs))
    assert "All from **report.pdf**." in block


def test_unknown_or_literature_identity_keeps_the_full_list():
    names = ["report.pdf (p. 1)", "report.pdf (p. 2)"]
    for srcs in ([{"source": {"name": n}} for n in names],                                     # no metadata
                 [_stored("paper#0", names[0], 1), _stored("paper#1", names[1], 2)],            # literature papers
                 [_stored("doc#0", names[0], 2), _stored("doc#1", names[1], 1)],                # numbers out of place
                 [{**_stored("doc#0", names[0], 1), "metadata": [{"source": "doc#0", "name": "[1] a"},
                                                                 {"source": "doc#5", "name": "[6] b"}]},
                  _stored("doc#1", names[1], 2)]):                                              # merged by OpenWebUI
        assert "All from" not in presentation.names_marker(names, {0, 1}, docs=presentation.stored_doc_ids(srcs))
    assert "All from" not in presentation.names_marker(names, {0, 1})                                    # no identities at all


def _events(hits):
    return [{"type": "citation", "data": _stored(f"{h['source']}#{i}", presentation.source_name(h), i + 1)}
            for i, h in enumerate(hits)]


def _old_block(hits, cited):
    lines = "\n".join(f"- ({i + 1}) {presentation.citation_label(presentation.source_name(hits[i]))}" for i in sorted(cited))
    return ('\n\n<details type="memento_sources">\n<summary>Cited in this answer: 3 of the 5 sources Memento read'
            '</summary>\n\n_The numbers match the links in the text._\n\n' + lines + "\n</details>\n")


def test_cached_summary_is_shown_in_the_new_layout():
    answer = "Aim [1]. Methods [2][4]." + _old_block(HITS, {0, 1, 3}) + "\n\n_Answered by X._\n"
    out = presentation.replay_sources(answer, _events(HITS))
    assert out == "Aim [1]. Methods [2][4]." + presentation.sources_marker(HITS, {0, 1, 3}) + "\n\n_Answered by X._\n"
    assert presentation.replay_sources(out, _events(HITS)) == out                                       # idempotent


def test_cached_german_summary_is_shown_in_the_new_layout():
    token = request.LANG.set("de")
    try:
        answer = "Ziel [1]." + texts.tr(_old_block(HITS, {0, 1}))
        out = presentation.replay_sources(answer, _events(HITS))
        assert "In dieser Antwort zitiert: 2 Abschnitte dieses Dokuments" in out
        assert f"Alle aus **{presentation.shown(DOC)}**." in out
    finally:
        request.LANG.reset(token)


def test_cached_answer_kept_when_documents_differ_or_are_unknown():
    hits = HITS[:2] + [_hit("Other paper", 3, "Results")]
    answer = "A [1][3]." + _old_block(hits, {0, 2})
    assert presentation.replay_sources(answer, _events(hits)) == answer
    assert presentation.replay_sources(answer, [{"type": "citation", "data": {"source": {"name": "x"}}}] * 3) == answer
    assert presentation.replay_sources("No block [1].", _events(HITS)) == "No block [1]."
    # a cited number without a stored citation
    assert presentation.replay_sources("A [9]." + _old_block(HITS + [HITS[0]] * 5, {8}), _events(HITS)) \
        == "A [9]." + _old_block(HITS + [HITS[0]] * 5, {8})
