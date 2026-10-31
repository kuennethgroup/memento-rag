# Answer layout and routing: report headings, the Thorough check note, list wording and figures under list answers,
# think-further questions, help links, Document Chat hand-overs, nothing-found replies, the display cap of gap blocks.
import pytest

import openwebui.functions.query_pipe as qp
from memento.research_gaps import report as gr
from memento.research_gaps.passages import calm_gap_answer
from memento.query import cache, generation, presentation, prompts, request, search_step, service, texts
from memento.query.scope import conversation, resolve
from memento.query.flows import answer as answer_flow, gap_answer, general, lists
from memento.document_chat import chat


# the report heading keeps one question frame when the router drops the question word
@pytest.mark.parametrize("topic, shown", [
    ("open research questions in polymer informatics", "polymer informatics"),
    ("Open research questions in polymer informatics", "polymer informatics"),
    ("research gaps in hafnia ferroelectrics", "hafnia ferroelectrics"),
    ("potential research gaps across the library", None),
    ("open problems in ceramics", "ceramics"),
])
def test_heading_frame_removed_without_a_question_word(topic, shown):
    assert gr.display_topic(topic) == shown


@pytest.mark.parametrize("topic", ["research data management", "open-source polymer tools", "Gaussian processes"])
def test_topic_that_only_starts_like_a_frame_stays(topic):
    assert gr.display_topic(topic) == topic


# the Thorough check note stays after the collapsed blocks, not inside the last one
GAPS = """### Limitations
1. A limitation [1].

### Gaps in earlier work it addresses
1. Earlier work lacked X [2].
"""


@pytest.mark.parametrize("note", [texts.CHECKED_NOTE, texts.PARTLY_CHECKED_NOTE])
def test_check_note_after_the_blocks(note):
    out = calm_gap_answer(f"{GAPS}\n{note}")
    assert out.rstrip().endswith(note)
    assert note not in out[:out.rindex("</details>")]


def test_stored_answer_with_the_note_inside_the_last_block_is_moved_out():
    hidden = calm_gap_answer(GAPS).rstrip()
    hidden = hidden[:hidden.rindex("</details>")] + f"{texts.CHECKED_NOTE}\n\n</details>\n"    # an older stored form: the note
                                                                                            # inside the last block
    out = calm_gap_answer(hidden)
    assert out.rstrip().endswith(texts.CHECKED_NOTE)
    assert out.count(texts.CHECKED_NOTE) == 1
    assert out.count("</details>") == hidden.count("</details>")


def test_stored_answer_with_the_model_line_after_the_blocks():
    # as stored: the model line follows the last block
    hidden = calm_gap_answer(GAPS).rstrip()
    hidden = (hidden[:hidden.rindex("</details>")] + f"{texts.CHECKED_NOTE}\n\n</details>\n\n\n"
              "_Answered by Qwen 3.8 27B (Local)._")
    out = calm_gap_answer(hidden)
    assert out.index("</details>", out.rindex("<details")) < out.index(texts.CHECKED_NOTE)
    assert out.rstrip().endswith("_Answered by Qwen 3.8 27B (Local)._")


def test_answer_without_a_note_unchanged():
    assert calm_gap_answer(GAPS) == calm_gap_answer(calm_gap_answer(GAPS))
    assert not calm_gap_answer(GAPS).rstrip().endswith("_")


# Document Chat: no list wording about the attached file itself
def test_no_list_wording_in_document_chat():
    assert lists.list_addendum("set_query", whole_doc=False, dc=True) == ""
    assert lists.list_addendum("set_query", whole_doc=False, dc=False) == lists.LIST_ADDENDUM


# N6 (C2, C6, R-A4: 2–5 figures under list / count answers)
def test_no_figures_under_a_list_answer():
    assert prompts.figures_for(element="none", image_intent=False, query_intent="set_query") == (False, False)


def test_list_of_figures_still_shows_them():
    # a list of figures (set_query, element figure)
    assert prompts.figures_for(element="figure", image_intent=False, query_intent="set_query") == (True, True)
    assert prompts.figures_for(element="none", image_intent=True, query_intent="set_query") == (True, True)


def test_other_questions_keep_their_figures():
    assert prompts.figures_for(element="none", image_intent=False, query_intent="ranked_retrieval") == (True, False)
    assert prompts.figures_for(element="none", image_intent=False) == (True, False)


# list questions on the local model run without thinking: measured, no thinking found the same documents as thinking
# capped at 2,000 tokens in a fraction of the time, and the cap leaked its deliberation into an answer (vLLM ends the
# thinking at the cap, the model went on deliberating)
def test_list_questions_on_the_local_model_do_not_think():
    assert generation.list_thinks(True, "set_query", is_cloud=False, enabled=False) is False


def test_list_thinking_valve_and_cloud_models_keep_it():
    assert generation.list_thinks(True, "set_query", is_cloud=False, enabled=True) is True
    assert generation.list_thinks(True, "set_query", is_cloud=True, enabled=False) is True


def test_other_questions_keep_their_thinking():
    for intent in ("ranked_retrieval", "multi_hop", "summary"):
        assert generation.list_thinks(True, intent, is_cloud=False, enabled=False) is True
    assert generation.list_thinks(False, "set_query", is_cloud=True, enabled=True) is False     # never switched on


def test_pipe_applies_it_and_the_valve_defaults_off():
    import inspect
    assert qp.Pipe.Valves().LIST_THINKING is False
    assert "turn.think = generation.list_thinks(turn.think, turn.query_intent, turn.variant.is_cloud, turn.valves.LIST_THINKING)" in \
        inspect.getsource(service._route)


def test_list_wording_puts_the_number_after_the_list():
    # the number after the list: a number written before the list was often wrong
    add = lists.list_addendum("set_query", whole_doc=False)
    assert "write the list first and the number after it" in add
    assert "counting the documents you listed" in add


# think-further questions get a normal answer: the router LLM decides "suggest" (no keyword rules)
def test_router_sends_think_further_questions_to_a_normal_answer():
    # router only: think-further questions stay out of gap_query (a whole research-gap report); gap questions unchanged
    from memento.query import router as qr
    assert "Asking which technologies, applications or uses could follow from a document's findings" in \
        qr.ROUTER_SYSTEM_PROMPT
    assert "speculate or extrapolate" not in qr.ROUTER_SYSTEM_PROMPT       # the wider wording measured no difference


def test_suggest_is_logged():
    import inspect
    src = inspect.getsource(service._route)
    assert "suggest=%s" in src


def test_normal_answer_with_suggest_separates_the_ai_ideas():
    add = prompts.suggest_addendum(True, "ranked_retrieval")
    assert "Possible extensions (suggested by the AI)" in add
    assert "Mögliche Erweiterungen (von der KI vorgeschlagen)" in add
    assert "no [N] citations" in add
    assert prompts.suggest_addendum(True, "multi_hop") == add


def test_no_suggest_addendum_otherwise():
    assert prompts.suggest_addendum(False, "ranked_retrieval") == ""
    assert prompts.suggest_addendum(True, "set_query") == ""          # a list stays a list
    assert prompts.suggest_addendum(True, "gap_query") == ""          # gap answers have their own block


def test_pipe_adds_it_to_the_answer_prompt():
    import inspect
    assert "prompts.suggest_addendum(turn.suggest, turn.query_intent)" in inspect.getsource(answer_flow.build_prompt)


def test_list_wording_names_each_document_once():
    # "each document once": without it, a book was listed once per section
    assert "each document once (several passages from one document are one document)" in lists.LIST_ADDENDUM


# the AI's own ideas are kept out of the Thorough check (it judges them unsupported) and put back after it
ANSWER = ("Die Quellen nennen FeRAM und FeFET [1][2].\n\n### Mögliche Erweiterungen (von der KI vorgeschlagen)\n"
          "Daraus ließen sich neuromorphe Bausteine ableiten.")


def test_extensions_split_off_before_the_check():
    # a list of sections, each with the heading it stood before (tests/verification/test_check_outcome.py)
    body, exts = prompts.split_extensions(ANSWER)
    assert body.rstrip() == "Die Quellen nennen FeRAM und FeFET [1][2]."
    assert len(exts) == 1 and exts[0][0] is None
    assert exts[0][1].startswith("### Mögliche Erweiterungen") and "neuromorphe" in exts[0][1]
    assert prompts.split_extensions("No section [1].") == ("No section [1].", [])


def test_extensions_put_back_after_the_check_note():
    # after the note, labelled as not checked
    out = prompts.with_extensions(f"Checked body [1].\n\n{texts.CHECKED_NOTE}",
                              [(None, "### Possible extensions (suggested by the AI)\nIdea.")])
    assert out.index(texts.CHECKED_NOTE) < out.index("### Possible extensions")
    assert texts.EXTENSIONS_UNCHECKED_NOTE in out and out.rstrip().endswith("Idea.")
    assert prompts.with_extensions("Body [1].", []) == "Body [1]."


def test_check_gets_the_answer_without_the_section():
    import inspect
    src = inspect.getsource(answer_flow.stream_answer)
    assert "checked_part, extensions = prompts.split_extensions(\"\".join(answer_text))" in src
    assert "final = prompts.with_extensions(final, extensions)" in src


def test_no_suggest_addendum_for_summaries():
    assert prompts.suggest_addendum(True, "summary") == ""


# the first mention of each way to use Memento becomes the link the prompt gives
def test_first_mention_of_each_way_becomes_its_link():
    out = presentation.link_app_names("Pick **Library Upload** in the model menu. Or use **Document Chat**; Document Chat "
                             "keeps it private.")
    assert "**[Library Upload](/?model=memento-ingestion-prod)**" in out
    assert "**[Document Chat](/?model=memento-document-chat)**" in out
    assert out.count("](/?model=memento-document-chat)") == 1                 # only the first mention


def test_german_quotes_and_existing_links_are_kept():
    out = presentation.link_app_names("Wähle „Library Upload“ aus.")
    assert "„[Library Upload](/?model=memento-ingestion-prod)“" in out
    linked = "Use [Document Chat](/?model=memento-document-chat) for your own file."
    assert presentation.link_app_names(linked) == linked
    assert presentation.link_app_names("Nothing about the app here.") == "Nothing about the app here."


def test_general_answers_get_the_links():
    import inspect
    assert "presentation.link_app_names(re.sub(" in inspect.getsource(general.reply)


def test_what_can_you_do_names_the_buttons_and_both_ways():
    # help answers name the start-screen buttons and Document Chat
    assert ("When asked what you can do or how to use Memento, name the start-screen buttons and both ways to use "
            "your own documents") in general.GENERAL_SYSTEM_PROMPT


# Document Chat without a file, a library question: pointed to Library Chat
def test_document_chat_question_without_a_file_offers_library_chat():
    out = chat.doc_chat_no_file("Can you list some databanks for polymers")
    assert out.startswith(texts.DOC_CHAT_NO_FILE_MSG)
    assert "[ask it in Library Chat](/?model=memento-query&q=Can%20you%20list%20some%20databanks%20for%20polymers)" in out
    assert chat.doc_chat_no_file("") == texts.DOC_CHAT_NO_FILE_MSG


def test_pipe_uses_it_for_a_question_without_a_file():
    import inspect
    assert "yield chat.doc_chat_no_file(turn.original_query)" in inspect.getsource(service.run)


# "this paper" in Library Chat with nothing named, picked or attached: a short hand-over (which document, or Document
# Chat), not a summary across the library
@pytest.mark.parametrize("q", ["Summarize me this paper and make sure the highlight leans towards method",
                               "What are the limitations of this study?", "summarise the attached file",
                               "Fasse dieses Paper zusammen", "Was sind die Ergebnisse dieser Arbeit?"])
def test_this_paper_with_nothing_named_gets_the_hand_over(q):
    assert conversation.unnamed_document(q, dc=False, paper=None, named_docs=[], history=[])


def test_no_hand_over_when_something_identifies_the_document():
    q = "Summarize this paper"
    assert not conversation.unnamed_document(q, dc=True, paper=None, named_docs=[], history=[])          # Document Chat
    assert not conversation.unnamed_document(q, dc=False, paper="PolyBERT", named_docs=[], history=[])  # resolved
    assert not conversation.unnamed_document(q, dc=False, paper=None, named_docs=[],
                                    history=[{"role": "user", "content": "x"}])                # a follow-up
    assert not conversation.unnamed_document("Summarize the polyBERT paper", dc=False, paper=None, named_docs=[], history=[])
    assert not conversation.unnamed_document("What papers are about hafnia?", dc=False, paper=None, named_docs=[], history=[])


def test_hand_over_text_in_both_languages():
    en, de = conversation.unnamed_document_msg(False), conversation.unnamed_document_msg(True)
    assert "Could you tell me which document you mean?" in en and "[Document Chat](/?model=memento-document-chat)" in en
    assert "Summarize a document" in en
    assert "Können Sie mir sagen, welches Dokument Sie meinen?" in de and "[Document Chat](/?model=memento-document-chat)" in de
    assert "—" not in en + de


def test_pipe_replies_with_the_hand_over():
    import inspect
    assert "conversation.unnamed_document(turn.original_query, dc=turn.doc_chat is not None" in inspect.getsource(resolve.decide)


# nothing relevant: a short model reply names a related document only when one is useful; the fixed text stays as the
# fallback
HEADS = [{"source": f"doc{i}", "text": f"passage {i} " + "x" * 900, "rerank_score": 0.05 - i / 100} for i in range(7)]


def test_nothing_found_messages_carry_the_prompt_and_the_closest_passages():
    msgs = answer_flow.nothing_found_messages("rubber compounding?", HEADS, {"doc0": "Polymer informatics review"})
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == answer_flow.NOTHING_FOUND_PROMPT
    user = msgs[1]["content"]
    assert 'Document "Polymer informatics review":' in user and 'Document "doc1":' in user
    assert 'Document "doc5"' not in user                                  # the 5 closest only
    assert "x" * 601 not in user                                          # passages cut to 600 characters
    assert user.rstrip().endswith("Question: rubber compounding?")


def test_nothing_found_prompt_never_lets_a_document_cover_more_than_it_shows():
    p = answer_flow.NOTHING_FOUND_PROMPT
    assert "only if a document below is genuinely useful" in p and "Never say a document covers" in p
    assert "Library Upload" in p and "No citations" in p


def test_pipe_writes_the_reply_and_falls_back_to_the_fixed_text():
    import inspect
    src = inspect.getsource(service.run)
    assert "await answer_flow.nothing_found_reply(" in src
    assert "NOTHING_IN_LIBRARY_MSG" in inspect.getsource(answer_flow.nothing_found_reply)
    assert qp.Pipe.Valves().NOTHING_FOUND_REPLY is True


# for a figure request only "cites nothing" means nothing was found (figures score low against a text question)
def test_figure_request_that_cites_figures_found_something():
    assert not presentation._found_nothing(cited={1, 3}, top_score=0.04, low=0.1, figure_asked=True)
    assert presentation._found_nothing(cited=set(), top_score=0.04, low=0.1, figure_asked=True)


def test_other_questions_keep_the_score_test():
    assert presentation._found_nothing(cited={1}, top_score=0.04, low=0.1, figure_asked=False)
    assert not presentation._found_nothing(cited={1}, top_score=0.5, low=0.1, figure_asked=False)


def test_the_block_knows_a_figure_request():
    import inspect
    assert 'figure_asked=turn.image_intent or turn.element == "figure"' in inspect.getsource(answer_flow.stream_answer)


# the model reads the title (with the file name when it differs), not only the file name; citations keep the file names
def test_context_names_use_the_title_when_known():
    names = {"978-3-032-14541-3": "Modern Computational Materials Science (978-3-032-14541-3)"}
    assert presentation.context_name("978-3-032-14541-3", names) == "Modern Computational Materials Science (978-3-032-14541-3)"
    assert presentation.context_name("PolyBERT", names) == "PolyBERT"          # not in the map: as before


def test_library_chat_context_gets_the_titles():
    import inspect
    src = inspect.getsource(answer_flow.build_prompt)
    assert "context_name(h['source'], ctx_names)" in inspect.getsource(presentation.chunk_label)
    assert "turn.ctx_names.update(await asyncio.to_thread(documents.display_titles" in src


# asked about extending (no one gap kind asked), "Possible extensions" is shown unfolded, as a section
def test_unfold_suggestions_shows_the_block_as_a_section():
    from memento.research_gaps.passages import suggestions_block, unfold_suggestions
    block = suggestions_block(["Extend to copolymers [2]."], german=False)
    out = unfold_suggestions(block)
    assert "<details" not in out and "### Possible extensions (suggested by the AI)" in out
    assert "_Suggested by the AI, not stated by the authors. Please verify._" in out and "1. Extend to copolymers [2]." in out
    de = unfold_suggestions(suggestions_block(["Copolymere [1]."], german=True))
    assert "### Mögliche Erweiterungen (von der KI vorgeschlagen)" in de
    assert unfold_suggestions("") == "" and unfold_suggestions("_note_") == "_note_"


def test_gap_route_unfolds_when_no_one_kind_is_asked():
    import inspect
    assert inspect.getsource(gap_answer.reply).count("block = block if focus else unfold_suggestions(block)") == 2


def test_unfolded_suggestions_are_never_stored():
    from memento.research_gaps.passages import suggestions_block, unfold_suggestions
    answer = "### Limitations\n1. A point [1].\n"
    shown = answer + unfold_suggestions(suggestions_block(["Idea [1]."], german=False)) + "\n_Answered by Qwen (Local)._"
    stored = conversation.without_suggestions(shown)
    assert "Possible extensions" not in stored and "Idea [1]." not in stored
    assert "1. A point [1]." in stored and "_Answered by Qwen (Local)._" in stored


def test_search_status_names_what_is_searched():
    # Document Chat's status names the file it reads, not the knowledge base
    import inspect
    src = inspect.getsource(search_step.run)
    assert 'await turn.status("Searching your files..." if turn.doc_chat is not None else "Searching the library...")' in src
    assert "Searching the knowledge base" not in src


# leaked thinking in a "###" line: only the known headings stay
LEAKED = ('### Open problems in the field" only. But the layout says group gaps under headings. If only open '
          'problems, fine.\n1. Draft point [1]\n2. Another draft [2]\n\n### Open problems in the field\n'
          '1. Real point one [3]\n2. Real point two [4]\n')


def test_unknown_heading_and_its_points_are_dropped():
    from memento.research_gaps.passages import calm_gap_answer, strip_leaked
    out = strip_leaked(LEAKED)
    assert "But the layout says" not in out and "Draft point" not in out
    assert out.startswith("### Open problems in the field\n") and "Real point two [4]" in out
    shown = calm_gap_answer(LEAKED)
    assert "But the layout says" not in shown and "| Open problems in the field | 2 |" in shown


def test_known_headings_in_both_languages_stay():
    from memento.research_gaps.passages import strip_leaked
    ok = "### Limitations\n1. A [1]\n\n### Zukünftige Arbeiten\n2. B [2]\n"
    assert strip_leaked(ok) == ok


# a long block shows the first GAP_POINTS_SHOWN points, the rest in a folded block after it; the table keeps the full
# count
def _gaps(n, heading="Open problems in the field"):
    return f"### {heading}\n" + "\n".join(f"{k}. Point {k} [{k}].\n   continued line {k}" for k in range(1, n + 1)) + "\n"


def test_a_long_block_shows_ten_points_and_folds_the_rest():
    from memento.research_gaps.passages import calm_gap_answer
    out = calm_gap_answer(_gaps(53))
    assert "| Open problems in the field | 53 |" in out
    first = out[out.index("<summary>Open problems in the field (53)</summary>"):]
    first = first[:first.index("</details>")]
    assert "10. Point 10 [10]." in first and "continued line 10" in first and "11. Point 11" not in first
    assert "<summary>Open problems in the field: 43 more</summary>" in out
    rest = out[out.index("43 more</summary>"):]
    assert "11. Point 11 [11]." in rest and "53. Point 53 [53]." in rest


def test_short_blocks_and_german_headings():
    from memento.research_gaps.passages import calm_gap_answer
    assert "more</summary>" not in calm_gap_answer(_gaps(10))
    de = calm_gap_answer(_gaps(12, "Offene Probleme des Fachgebiets"))
    assert "<summary>Offene Probleme des Fachgebiets: 2 weitere</summary>" in de


def test_the_open_focus_block_shows_ten_too():
    from memento.research_gaps.passages import calm_gap_answer
    out = calm_gap_answer(_gaps(15, "Limitations"), focus="Limitations")
    head = out[:out.index("<details")]                       # what is shown open, before the folded rest
    assert "**Limitations (15)**" in head and "10. Point 10" in head and "11. Point 11" not in head
    assert "<summary>Limitations: 5 more</summary>" in out


def test_stored_answers_in_the_block_layout_are_capped_on_replay():
    from memento.research_gaps.passages import GAP_POINTS_SHOWN, calm_gap_answer
    body = "\n".join(f"{k}. Point {k} [{k}]." for k in range(1, 54))
    stored = ("| What the authors state | Points |\n|---|---|\n| Open problems in the field | 53 |\n\n\n"
              f'<details type="memento_gaps">\n<summary>Open problems in the field (53)</summary>\n\n{body}\n\n'
              "</details>\n")
    out = calm_gap_answer(stored)
    assert "<summary>Open problems in the field: 43 more</summary>" in out
    first = out[out.index("(53)</summary>"):out.index("</details>")]
    assert f"{GAP_POINTS_SHOWN}. Point {GAP_POINTS_SHOWN}" in first and "11. Point 11" not in first
    assert calm_gap_answer(out) == out                                         # once only


# the general prompt says who built Memento and names the model
def test_general_prompt_says_who_built_memento_and_names_the_model(monkeypatch):
    monkeypatch.delenv("MEMENTO_BUILT_BY", raising=False)
    p = general.general_prompt("Saturday", "Qwen 3.8 27B (Local)")
    assert "Memento was built as part of a Master's thesis" in p
    assert "this answer is written by Qwen 3.8 27B (Local)" in p
    monkeypatch.setenv("MEMENTO_BUILT_BY", "the Example Lab")
    assert "Memento was built at the Example Lab as part of a Master's thesis" in general.general_prompt("Saturday", "M")


def test_general_route_passes_the_model_name():
    import inspect
    assert "general_prompt(now, turn.variant.label)" in inspect.getsource(general.reply)


def test_follow_ups_about_the_project_get_no_invented_details():
    # follow-up questions about the project get no invented details
    p = general.GENERAL_SYSTEM_PROMPT
    assert "Give only these facts" in p and "the administrator can help" in p
    assert "finds the passages in the library's documents closest to the question" in p


def test_an_answer_for_a_removed_document_is_not_stored():
    # a background answer for a removed document is stored only while the document is still in the library list (a
    # database error keeps the old behaviour: stored)
    from sqlalchemy import create_engine
    from memento.library import documents as ldb
    engine = create_engine("sqlite://")
    ldb.ensure_tables(engine)
    ldb.add_document(engine, collection="c", entry={"source": "Kept", "title": "Kept", "parsed_title": None,
                                                    "pdf_title": None}, content_hash="h", uploaded_by=None,
                     output_root=None)
    assert cache.still_in_library(engine, "c", "Kept") is True
    assert cache.still_in_library(engine, "c", "Removed") is False
    assert cache.still_in_library(None, "c", "Kept") is True
    import inspect
    assert "cache.still_in_library," in inspect.getsource(service.handle)     # checked before ac.put


def test_document_chat_asked_about_the_library_gets_only_the_pointer():
    # Document Chat asked about the library: only the pointer to Library Chat, no model answer after it
    import inspect
    src = inspect.getsource(service.run)
    i = src.index("lib_note = chat.doc_chat_library_note(")
    after = src[i:i + 400]
    assert "yield lib_note" in after and "return" in after
    assert not hasattr(qp, "_doc_chat_scope_addendum")
    token = request.LANG.set("de")
    try:
        de = texts.tr(chat.doc_chat_library_note(True, "library", "Gibt es dazu etwas in der Bibliothek?"))
    finally:
        request.LANG.reset(token)
    assert de.startswith("Document Chat beantwortet Fragen zu Ihren angehängten Dateien und durchsucht die Bibliothek nicht. Für die Bibliothek [nutzen Sie bitte Library Chat](")


def test_a_count_of_one_document_is_singular():
    # a count of one document is singular
    assert "I found 1 document on this topic" in lists.LIST_ADDENDUM


def test_german_answers_say_bibliothek():
    # German answers say "Bibliothek"
    token = request.LANG.set("de")
    try:
        assert "Bibliothek" in texts.answer_language_addendum()
    finally:
        request.LANG.reset(token)
    assert texts.answer_language_addendum() == ""
    import inspect
    assert "texts.answer_language_addendum(" in inspect.getsource(answer_flow.build_prompt)     # with the passages


def test_a_citation_event_names_its_passage_uniquely():
    h = {"source": "Doc_A", "page": 2, "text": "t", "section": "Results", "section_raw": "4.2 Fatigue", "bbox": [1, 2, 3, 4]}
    ev = presentation.citation_event(0, h, "")
    assert ev["type"] == "citation" and ev["data"]["document"] == ["t"]
    assert ev["data"]["metadata"][0]["source"] == "Doc_A#0" and ev["data"]["metadata"][0]["name"].startswith("[1] ")
    assert presentation.chunk_label(h, False, {}) == "Doc_A, page 3, Section: Results › 4.2 Fatigue"
    assert presentation.chunk_label(h, True, {}) == "page 3, Section: Results › 4.2 Fatigue"     # a summary's sources
