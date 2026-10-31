# Document Chat talks about the reader's own file: it has its own system prompt (not the library's), and a topic
# question about the file is answered, although the router may label it scope "library".
import inspect

import openwebui.functions.query_pipe as qp
from memento.document_chat import chat
from memento.query.flows import answer as answer_flow

FIGURE_Q = "Which figure shows the energy consumption of additive manufacturing processes, and what does it show?"


def test_a_topic_question_without_the_library_is_answered_from_the_file():
    assert chat.doc_chat_library_note(True, "library", FIGURE_Q) == ""
    assert chat.doc_chat_library_note(True, "library", "Welche Abbildung zeigt den Energieverbrauch?") == ""


def test_a_question_that_names_the_library_or_other_documents_still_gets_the_pointer():
    for q in ("Is this related to anything in the group's library?", "Gibt es dazu etwas in der Bibliothek?",
              "Which other papers use polyBERT?", "Gibt es andere Dokumente zu diesem Thema?",
              "Is this in the collection?"):
        assert chat.doc_chat_library_note(True, "library", q).startswith("Document Chat answers"), q


def test_document_chat_answers_call_the_sources_the_users_files():
    add = chat.doc_chat_addendum(True)
    assert "attached" in add and "never" in add and "library" in add
    assert chat.doc_chat_addendum(False) == ""


def test_something_the_file_does_not_cover_is_said_about_the_file():
    # the domain prompt's example "The papers I found don't mention GANs." was copied into Document Chat
    add = chat.doc_chat_addendum(True)
    assert "Your document does not mention" in add and "Ihr Dokument" in add and "papers I found" in add
    assert "Your documents do not mention" in add and "Ihre Dokumente" in add          # several files


def test_the_addendum_is_part_of_the_answer_prompt():
    src = inspect.getsource(answer_flow.build_prompt)
    i = src.index("list_add = lists.list_addendum(")        # the list line kept for the language line
    assert "chat.doc_chat_addendum(turn.doc_chat is not None)" in src[i:i + 300]
