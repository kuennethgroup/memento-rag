# List questions search their topic in the documents of the kind asked for: the topic alone, not the whole question
# ("dissertations … and the compounds they used"), with no fixed cap that cuts a matching document, and a document's
# best passage never a list-of-tables line. The answer ends on a positive, honest line, and a long list folds after 20
# like the library listing.
import asyncio
import json
import types

import pytest

import openwebui.functions.query_pipe as qp
from memento.query import presentation, request, router as qr
from memento.retrieval import search
from memento.query.scope import conversation, resolve
from memento.query.flows import answer as answer_flow, gap_answer, lists


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))])


def _route(reply):
    return asyncio.run(qr.route(_FakeLLM(json.dumps(reply)), "m", "q", []))


# ── the router: the topic and the kind of document ───────────────────────────────────────────────────────────────

def test_the_router_reads_the_topic_and_the_kind():
    r = _route({"intent": "set_query", "query": "dissertations about phosphorus-based flame retardants",
                "reasoning": "deep", "topic": "  phosphorus-based   flame retardants ", "doc_kind": "dissertation"})
    assert r.topic == "phosphorus-based flame retardants" and r.doc_kind == "dissertation"
    assert r.intent == "set_query" and r.query == "dissertations about phosphorus-based flame retardants"


@pytest.mark.parametrize("kind, want", [
    ("dissertation", "dissertation"), ("Dissertations", "dissertation"), ("PhD thesis", "dissertation"),
    ("doctoral thesis", "dissertation"), ("master's thesis", "master's thesis"), ("Masterarbeit", "master's thesis"),
    ("master thesis", "master's thesis"), ("bachelor's thesis", "bachelor's thesis"), ("Bachelorarbeit", "bachelor's thesis"),
    ("thesis", "thesis"), ("theses", "thesis"), ("Abschlussarbeiten", "thesis"), ("talk", "talk"), ("Vortrag", "talk"),
    ("presentation", "talk"), ("chapter", "chapter"), ("Kapitel", "chapter"), ("paper", ""), ("documents", ""),
    ("none", ""), (None, ""), (3, "")])
def test_the_kind_is_coerced_to_the_known_kinds(kind, want):
    reply = {"intent": "set_query", "query": "x", "reasoning": "deep"}
    if kind is not None:
        reply["doc_kind"] = kind
    assert _route(reply).doc_kind == want


@pytest.mark.parametrize("topic, want", [("none", ""), ("", ""), (None, ""), (7, ""), ("PEEK", "PEEK"),
                                         ("x" * 300, "x" * 200)])
def test_the_topic_is_cleaned(topic, want):
    reply = {"intent": "set_query", "query": "x", "reasoning": "deep"}
    if topic is not None:
        reply["topic"] = topic
    assert _route(reply).topic == want


def test_no_topic_without_the_fields_or_when_the_router_failed():
    r = _route({"intent": "ranked_retrieval", "query": "x", "reasoning": "light"})
    assert r.topic == "" and r.doc_kind == ""
    assert qr.RouterResponse(ok=False).topic == "" and qr.RouterResponse(ok=False).doc_kind == ""


def test_the_instructions_describe_topic_and_kind_and_keep_the_rest():
    p = qr.ROUTER_SYSTEM_PROMPT
    assert '"topic"' in p and '"doc_kind"' in p
    assert '"topic": "", "doc_kind": ""' in p                         # in the JSON shape the model is told to output
    assert "phosphorus-based flame retardants" in p and "PEEK" in p   # a topic and a material
    for kind in ("dissertation", "master's thesis", "bachelor's thesis", "thesis", "talk", "chapter"):
        assert f'"{kind}"' in p, kind
    for intent in qr._VALID_INTENTS:
        assert f'"{intent}"' in p, intent
    assert '"element": "none"' in p and '"scope": "none"' in p and '"author": ""' in p


# ── the documents of the kind asked for, from the library's titles ──────────────────────────────────────────────

DOCS = [("a", "Dissertation: Flammschutz für Epoxidharze"), ("b", "Dissertation (Weber)"),
        ("c", "Master's thesis: Epoxy powder composites"), ("d", "Talk (2014): Halogenfrei flammgeschützt"),
        ("e", "Chapter 4 (Ostertag): Ergebnisse"), ("f", "polyBERT: a chemical language model"),
        ("g", "Bachelor's thesis: Wirkungsweise verschiedener Flammschutzmittel"),
        ("h", "Dissertationsverteidigung Folien"), ("i", "Talk: Bio-basierte Flammschutzsysteme"), ("j", None)]


@pytest.mark.parametrize("kind, want", [
    ("dissertation", {"a", "b"}), ("master's thesis", {"c"}), ("bachelor's thesis", {"g"}),
    ("thesis", {"a", "b", "c", "g"}), ("talk", {"d", "i"}), ("chapter", {"e"})])
def test_the_kind_selects_documents_by_their_library_title(kind, want):
    assert set(lists._list_kind_sources(kind, DOCS)) == want


def test_no_kind_or_no_title_of_that_kind_means_no_restriction():
    assert lists._list_kind_sources("", DOCS) is None
    assert lists._list_kind_sources("dissertation", [("f", "polyBERT: a chemical language model"), ("x", "Optuna")]) is None


# ── a document's passages for the scan: no table of contents or list lines ──────────────────────────────────────

@pytest.mark.parametrize("section, text, toc", [
    ("Tabellenverzeichnis", "Tabelle 4.3 Daten verwendeter Flammschutzmittel [169-174]....43", True),
    ("Results and Discussion", "5.4 Flammschutz von phosphatierten Polyaromaten in DGEBA/AS . . . . . . 112", True),
    ("Inhaltsverzeichnis", "2 Stand der Technik", True), ("List of Figures", "Figure 2.1 Cone calorimeter", True),
    ("Table of Contents", "1 Introduction", True), ("Abbildungsverzeichnis", "Abbildung 3: REM-Aufnahme", True),
    ("References", "[12] A. Author, Polymer 2010", True), ("Literaturverzeichnis", "Grüber, J. (2004)", True),
    ("Background", "Phosphorhaltige Flammschutzmittel wirken hauptsächlich in der kondensierten Phase.", False),
    ("Materialien", "DOPO weist einen Phosphorgehalt von 14,3 ± 0,3 % auf.", False),
    ("Results", "Die Proben 1.1, 1.2 und 1.3 erreichten V-0.", False), (None, "PHRR 337 kW/m² at 1.9 wt.% P", False)])
def test_table_of_contents_and_list_lines_are_recognised(section, text, toc):
    assert search.toc_like({"section": section, "text": text}) is toc


class _Groups:
    """A Qdrant client whose grouped search returns the given documents' passages (best document first)."""
    def __init__(self, docs, limit_hit=False):
        self.docs, self.calls, self.limit_hit = docs, [], limit_hit

    def query_points_groups(self, **kw):
        self.calls.append(kw)
        groups = [types.SimpleNamespace(hits=[types.SimpleNamespace(payload={"source": src, **p}, score=0.9 - i * 0.01,
                                                                    id=f"{src}{i}") for i, p in enumerate(ps)])
                  for src, ps in self.docs]
        return types.SimpleNamespace(groups=groups[:kw["limit"]])


def test_list_heads_keeps_two_real_passages_per_document():
    toc = {"section": "Tabellenverzeichnis", "text": "Tabelle 4.3 Daten verwendeter Flammschutzmittel....43"}
    docs = [("K", [toc, {"section": "Background", "text": "K one"}, {"section": "Results", "text": "K two"},
                   {"section": "Results", "text": "K three"}]),
            ("T", [toc, dict(toc)]),                                   # only list lines: not a candidate
            ("R", [{"section": "Background", "text": "R one"}])]
    client = _Groups(docs)
    heads = search.list_heads(client, "c", [0.1], None, groups=60, per_doc=2)
    assert [(h["source"], h["text"]) for h in heads] == [("K", "K one"), ("K", "K two"), ("R", "R one")]
    kw = client.calls[0]
    assert kw["group_by"] == "source" and kw["group_size"] >= 4 and kw["limit"] == 60
    assert heads.capped is False and heads.found == 2


def test_list_heads_says_when_the_search_reached_its_limit():
    docs = [(f"D{i}", [{"section": "Results", "text": f"t{i}"}]) for i in range(5)]
    assert search.list_heads(_Groups(docs), "c", [0.1], None, groups=5, per_doc=2).capped is True


# ── which documents join the answer ──────────────────────────────────────────────────────────────────────────────

def _h(src, score, text=None):
    return {"source": src, "rerank_score": score, "type": "text", "text": text or f"passage of {src}", "page": 1,
            "img_path": None, "bbox": None}


def test_a_document_joins_with_its_passages_that_pass_the_mark():
    hits = [_h("A", 0.6)]
    heads = [_h("A", 0.6), _h("A", 0.2), _h("B", 0.5, "B1"), _h("B", 0.01, "B2"), _h("C", 0.3, "C1"),
             _h("C", 0.25, "C2"), _h("D", 0.02)]
    extra = lists.list_extra(hits, heads, ratio=0.25, cap=50)
    assert [h["text"] for h in extra] == ["B1", "C1", "C2"]        # A already in; B2 and D below 0.25 × 0.6
    assert extra.capped is False


def test_more_documents_than_the_cap_are_said():
    heads = [_h(f"D{i}", 0.9 - i * 0.005) for i in range(60)]
    extra = lists.list_extra([], heads, ratio=0.25, cap=50)
    assert len({h["source"] for h in extra}) == 50 and extra.capped is True
    few = lists.list_extra([], heads[:10], ratio=0.25, cap=50)
    assert len(few) == 10 and few.capped is False


def test_the_scan_valves():
    v = qp.Pipe().valves
    assert v.LIST_SCAN_MAX == 50 and v.LIST_SCAN_DOCS == 60 and v.LIST_SCAN_PER_DOC == 2 and v.LIST_SCAN_RATIO == 0.08


# ── the closing line ─────────────────────────────────────────────────────────────────────────────────────────────

def test_the_closing_line_is_positive_and_honest():
    add = lists.list_addendum("set_query", whole_doc=False)
    assert "I found N" in add and "I found 9 dissertations on this topic" in add and "I found 1 document" in add
    # honest about possible gaps, but positive
    assert "If one you expected is missing, ask about it by name and I'll check it." in add
    assert "never claim completeness" in add and "each document once" in add and "Also mentioned in" not in add
    assert "may also cover" not in add and "in the sources found" not in add


def test_a_capped_scan_says_the_best_matching_ones_are_shown():
    add = lists.list_addendum("set_query", whole_doc=False, capped=True)
    assert "best-matching" in add and "The library has more on this topic" in add
    assert "best-matching" not in lists.list_addendum("set_query", whole_doc=False)
    assert lists.list_addendum("ranked_retrieval", whole_doc=False, capped=True) == ""


# ── a long list folds after 20, like the library listing ────────────────────────────────────────────────────────

def _list(n, numbered=False, sub=True):
    out = []
    for i in range(1, n + 1):
        out.append(f"{i}. **Doc {i}**: uses DOPO [{i}]." if numbered else f"*   **Doc {i}**: uses DOPO [{i}].")
        if sub:
            out.append(f"    *   detail {i} [{i}]")
    return "\n".join(out)


def test_twenty_or_fewer_entries_stay_as_they_are():
    text = f"These dissertations use it:\n\n{_list(20)}\n\nI found 20 dissertations on this topic."
    assert presentation.fold_list(text) == text


def test_entries_after_twenty_fold_with_the_listing_labels():
    text = f"Intro:\n\n{_list(25)}\n\nI found 25 dissertations on this topic. To check a specific one, ask by name."
    out = presentation.fold_list(text)
    head, rest = out.split("<details>", 1)
    assert "**Doc 20**" in head and "detail 20" in head and "**Doc 21**" not in head
    assert rest.startswith("\n<summary>Show 5 more (21–25 of 25)</summary>\n\n*   **Doc 21**")
    assert "detail 25 [25]\n\n</details>" in rest
    assert out.rstrip().endswith("I found 25 dissertations on this topic. To check a specific one, ask by name.")
    assert out.index("</details>") < out.index("I found 25")


def test_numbered_and_german_lists_fold_too():
    text = f"Einleitung:\n\n{_list(23, numbered=True, sub=False)}\n\nIch habe 23 Dissertationen gefunden."
    out = presentation.fold_list(text, de=True)
    assert "<summary>3 weitere anzeigen (21–23 von 23)</summary>\n\n21. **Doc 21**" in out
    assert out.count("<details") == 1


def test_a_very_long_list_gets_steps_and_show_all():
    out = presentation.fold_list(_list(80, sub=False))
    assert "Show 50 more (21–70 of 80)" in out and "Show 10 more (71–80 of 80)" in out and "Show all 80" in out


def test_separate_short_lists_do_not_fold():
    text = f"Dissertations:\n\n{_list(12)}\n\nTalks:\n\n{_list(12)}\n\nI found 24 documents."
    assert presentation.fold_list(text) == text


@pytest.mark.parametrize("step", [1, 3, 7, 50, 10_000])
def test_the_streamed_fold_equals_the_whole_text_fold(step):
    text = f"Intro:\n\n{_list(26)}\n\nI found 26 dissertations on this topic.\n"
    f = presentation.ListFold()
    out = "".join(f.feed(text[i:i + step]) for i in range(0, len(text), step)) + f.close()
    assert out == presentation.fold_list(text)


def test_the_first_twenty_entries_stream_while_they_are_written():
    text = f"Intro:\n\n{_list(26)}\n\nI found 26."
    cut = text.index("*   **Doc 15**")
    f = presentation.ListFold()
    shown = f.feed(text[:cut + 5])
    assert "**Doc 14**" in shown and "detail 14" in shown          # nothing held back but the line being written


# ── Thorough check: the checked answer at once ──────────────────────────────────────────────────────────────────

def test_a_checked_answer_is_shown_at_once():
    p = qp.Pipe()
    assert presentation.typing_speed(p.valves, {}, at_once=True) == (0, 0, 0)
    assert presentation.typing_speed(p.valves, {}) == (p.valves.REPLAY_CHARS_PER_S, p.valves.REPLAY_MAX_S, p.valves.REPLAY_TYPED_CHARS)


def test_both_checked_answers_use_it():
    import inspect
    src = inspect.getsource(answer_flow.stream_answer)
    assert src.count("typing_speed(turn.valves, turn.metadata, at_once=True)") == 1     # the normal answer
    assert inspect.getsource(gap_answer.reply).count("typing_speed(turn.valves, turn.metadata, at_once=True)") == 1  # gaps


# ── the count sentence states the number of entries shown ──

@pytest.mark.parametrize("n, line, want", [
    (26, "I found 24 dissertations on this topic.", "I found 26 dissertations on this topic."),
    (26, "These are the 49 best-matching documents.", "These are the 26 best-matching documents."),
    (23, "Ich habe 21 Dissertationen zu diesem Thema gefunden.", "Ich habe 23 Dissertationen zu diesem Thema gefunden."),
    (23, "Das sind die 20 am besten passenden Dokumente.", "Das sind die 23 am besten passenden Dokumente."),
    (5, "I found 5 documents on this topic.", "I found 5 documents on this topic.")])
def test_the_count_sentence_gets_the_number_of_entries(n, line, want):
    text = f"Intro:\n\n{_list(n)}\n\n{line} If one you expected is missing, ask about it by name."
    for step in (3, 10_000):
        f = presentation.ListFold()
        out = "".join(f.feed(text[i:i + step]) for i in range(0, len(text), step)) + f.close()
        assert want in out and (line == want or line not in out)


def test_entries_of_several_lists_are_counted_together():
    text = f"Dissertations:\n\n{_list(3)}\n\nTalks:\n\n{_list(4)}\n\nI found 6 documents on this topic."
    assert presentation.fold_list(text).endswith("I found 7 documents on this topic.")


def test_a_number_without_a_list_or_in_other_sentences_stays():
    assert presentation.fold_list("I found 3 documents on this topic.") == "I found 3 documents on this topic."
    text = f"Intro:\n\n{_list(4)}\n\nTwo of them used 12 wt.% APP. I found 9 papers on it in total."
    assert presentation.fold_list(text) == text                                  # only a sentence that starts with the phrase


# ── "Which of them …?" after a list: the previous answer's documents ──

PREV = ["Diss_Garcia", "Dissertation_Hofbauer_final", "221004_Dissertation_final", "140626_Dissertation_compr"]


@pytest.mark.parametrize("q", ["Which of them reported the lowest peak heat release rate?",
                               "Which of these used DOPO?", "Do any of those mention APP?",
                               "Welche davon hatte die niedrigste Wärmefreisetzungsrate?",
                               "Welche dieser Arbeiten nutzte DOPO?"])
def test_a_plural_reference_means_the_previous_answers_documents(q):
    assert conversation.them_scope(q, PREV) == PREV


@pytest.mark.parametrize("q", ["Compare the thesis by Garcia with Hofbauer's dissertation",
                               'What does the document "Diss_Garcia" say about DOPO?', "What is DOPO?",
                               "Summarize this thesis"])
def test_other_questions_keep_their_own_scope(q):
    assert conversation.them_scope(q, PREV) is None


def test_no_previous_documents_no_scope():
    assert conversation.them_scope("Which of them reported the lowest pHRR?", None) is None
    assert conversation.them_scope("Which of them reported the lowest pHRR?", []) is None


def test_the_pipe_asks_for_them_before_trusting_names_in_the_rewrite():
    import inspect
    src = inspect.getsource(resolve.detect)
    i, j = src.index("conversation.them_scope("), src.index('logging.getLogger("memento.query").info("named documents: %s"')
    assert i < j


def test_each_language_gets_its_own_closing_sentence():
    # one sentence in the question's language (_LANG, set per question): with both in the instructions the answer can
    # end in the other language
    en = lists.list_addendum("set_query", whole_doc=False)
    assert "If one you expected is missing, ask about it by name and I'll check it." in en and "Wenn ein" not in en
    token = request.LANG.set("de")
    try:
        de = lists.list_addendum("set_query", whole_doc=False)
        de_more = lists.list_addendum("set_query", whole_doc=False, capped=True)
    finally:
        request.LANG.reset(token)
    assert "Wenn ein von Ihnen erwartetes Dokument fehlt, fragen Sie bitte nach dem Namen, und ich prüfe es." in de
    assert "Ich habe N Dokumente zu diesem Thema gefunden" in de and "If one you expected" not in de
    assert "Das sind die N am besten passenden Dokumente" in de_more


def test_them_is_one_search_within_those_documents_not_one_per_document():
    import inspect
    src = inspect.getsource(resolve.detect)
    assert "turn.paper, turn.named_docs, turn.author_docs = None, [], them" in src     # author_docs: one search


def test_an_english_list_question_is_answered_in_english():
    # many German documents in the context can turn an English answer German; a German question has DE_ANSWER_ADDENDUM
    # already
    en = lists.list_addendum("set_query", whole_doc=False)
    assert "The question is in English: write the answer in English" in en
    token = request.LANG.set("de")
    try:
        assert "The question is in English" not in lists.list_addendum("set_query", whole_doc=False)
    finally:
        request.LANG.reset(token)
