# Attach vs upload: "How do I add / attach a document?" names both Library Upload and Document Chat; Document Chat asked
# about the library points to Library Chat.
import openwebui.functions.query_pipe as qp
from memento.query import texts
from memento.query.flows import general
from memento.document_chat import chat


def test_app_help_names_both_ways_with_links():
    p = general.GENERAL_SYSTEM_PROMPT
    assert "Document Chat" in p and "Library Upload" in p
    assert "(/?model=memento-document-chat)" in p and "(/?model=memento-ingestion-prod)" in p
    assert "name both" in p                                      # asked how to add / attach / upload: both options
    assert "private" in p and "shared" in p
    for button in ("Summarize a document", "Limitations & future work", "Draft a proposal"):
        assert button in p, button


def test_a_file_attached_in_library_chat_points_to_both():
    assert "Document Chat" in texts.FILE_NOTE and "Library Upload" in texts.FILE_NOTE
    assert "can't read attachments directly" in texts.IMAGE_MSG and "Library Upload" in texts.IMAGE_MSG


def test_document_chat_says_it_cannot_search_the_library():
    note = chat.doc_chat_library_note(True, "library", "Is this related to anything in the group's library?")
    # the whole reply, no answer from the file after it
    assert note.startswith("Document Chat answers questions about your attached files and does not search the library. For the library, please [use Library Chat](")
    assert "(/?model=memento-query&q=Is%20this%20related%20to%20anything%20in%20the%20group%27s%20library%3F)" in note
    assert chat.doc_chat_library_note(True, "document", "What method does this paper use?") == ""
    assert chat.doc_chat_library_note(True, "none", "What method does this paper use?") == ""
    assert chat.doc_chat_library_note(False, "library", "Which papers use polyBERT?") == ""     # Library Chat: no note
