# A requested document that matches nothing never becomes a silent search of the whole library: a named document that
# matches nothing gets a polite ask with the offer to search the whole library ("yes" next turn does), and "my uploads"
# with none says so; neither searches.
import json

from test_scope_kept import _run, allows, PAYLOADS

NAMED = "I could not find that document in the library."
OLD_NAMED = "I could not find the document you named in the library."      # an older wording


def test_named_document_that_matches_nothing_asks_and_does_not_search(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What does the missing paper report about hardness?",
                                 {"intent": "ranked_retrieval", "query": "missing paper hardness", "scope": "document"})
    assert rec["searches"] == [] and model_input == ""
    assert out["answer"].startswith(NAMED) and "reply **yes**" in out["answer"]
    assert "Could you check the title?" in out["answer"]


def test_two_named_documents_that_match_nothing_ask_too(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "Compare the foo paper with the bar paper on hardness",
                                 {"intent": "ranked_retrieval", "query": "compare foo and bar", "scope": "documents"})
    assert rec["searches"] == [] and out["answer"].startswith(NAMED)


def test_yes_searches_the_whole_library_for_the_previous_question(monkeypatch):
    history = [{"role": "user", "content": "What does the missing paper report about hardness?"},
               {"role": "assistant", "content": OLD_NAMED + " Please check its name, or reply **yes** if I should search "
                                                "the whole library instead."}]
    out, rec, model_input = _run(monkeypatch, "yes please",
                                 {"intent": "ranked_retrieval", "query": "missing paper hardness", "scope": "document"},
                                 history=history)
    assert rec["searches"] and all(allows(f, d) for f in rec["searches"] for d in PAYLOADS if d["type"] == "text")
    assert "Paper B" in model_input and "Paper C" in model_input
    from test_scope_kept import _LLM
    router = next(c for c in _LLM.calls if not c.get("stream"))
    assert "missing paper report about hardness" in json.dumps(router["messages"])     # the previous question


def test_yes_without_the_offer_is_an_ordinary_message(monkeypatch):
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}]
    out, rec, _ = _run(monkeypatch, "yes", {"intent": "chitchat", "query": "yes"}, history=history)
    assert NAMED not in out["answer"]


def test_the_previous_answers_document_is_used_before_asking(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What are its limitations regarding hardness?",
                                 {"intent": "ranked_retrieval", "query": "limitations hardness", "scope": "document"},
                                 history=[{"role": "user", "content": "Tell me about paper A"},
                                          {"role": "assistant", "content": "Paper A studies hardness [1]."}],
                                 target="A")
    assert rec["searches"] and all({d["source"] for d in PAYLOADS if allows(f, d)} <= {"A"} for f in rec["searches"])


def test_my_uploads_with_none_says_so_and_does_not_search(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What do my latest uploads report about hardness?",
                                 {"intent": "ranked_retrieval", "query": "hardness in my uploads", "scope": "documents"},
                                 uploads=[])
    assert rec["searches"] == [] and model_input == ""
    assert out["answer"].startswith("I could not find any documents you uploaded to the library.")
    assert "(/?model=memento-ingestion-prod)" in out["answer"]


def test_a_library_question_still_searches_the_library(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What is known about hardness?",
                                 {"intent": "ranked_retrieval", "query": "hardness", "scope": "library"})
    assert rec["searches"] and "Paper B" in model_input


def test_german_forms(monkeypatch):
    out, rec, _ = _run(monkeypatch, "Was berichtet das fehlende Paper über die Härte?",
                       {"intent": "ranked_retrieval", "query": "fehlendes Paper Härte", "scope": "document"})
    assert rec["searches"] == []
    assert out["answer"].startswith("Ich konnte dieses Dokument nicht in der Bibliothek finden.")
    assert "**ja**" in out["answer"]


# ── a newly named document is never swapped for the previous one; names in a "my uploads" question resolve among the
# asker's own uploads only ──

HISTORY_A = [{"role": "user", "content": "Tell me about paper A"},
             {"role": "assistant", "content": "Paper A studies hardness [1]."}]


def test_two_newly_named_documents_are_not_swapped_for_the_previous_one(monkeypatch):
    out, rec, _ = _run(monkeypatch, "Compare the Zebra paper with the Yak paper on hardness",
                       {"intent": "ranked_retrieval", "query": "compare Zebra and Yak", "scope": "documents"},
                       history=HISTORY_A, target="A")
    assert rec["searches"] == [] and out["answer"].startswith(NAMED)


def test_a_name_in_a_my_uploads_question_resolves_among_the_own_uploads(monkeypatch):
    out, rec, model_input = _run(monkeypatch, "What does my latest upload of paper C report about hardness?",
                                 {"intent": "ranked_retrieval", "query": "paper C hardness", "scope": "document"},
                                 uploads=["A", "C"], resolve="C")
    assert rec["searches"] and all({d["source"] for d in PAYLOADS if allows(f, d)} <= {"C"} for f in rec["searches"])


def test_a_my_uploads_question_naming_none_keeps_the_whole_batch(monkeypatch):
    # names resolve within the own uploads, a set of two before a single match; a question that names none of them keeps
    # them all
    out, rec, _ = _run(monkeypatch, "What do my latest uploads report about hardness?",
                       {"intent": "ranked_retrieval", "query": "hardness in my uploads", "scope": "documents"},
                       uploads=["A", "C"])
    covered = {d["source"] for f in rec["searches"] for d in PAYLOADS if allows(f, d) and d["type"] == "text"}
    assert rec["searches"] and covered == {"A", "C"}          # each searched with its own share, B never


def test_yes_after_the_new_wording_searches_the_whole_library(monkeypatch):
    history = [{"role": "user", "content": "What does the missing paper report about hardness?"},
               {"role": "assistant", "content": NAMED + " Could you check the title? If you like, reply **yes** and I "
                                                "will search the whole library instead."}]
    out, rec, model_input = _run(monkeypatch, "yes",
                                 {"intent": "ranked_retrieval", "query": "missing paper hardness", "scope": "document"},
                                 history=history)
    assert rec["searches"] and "Paper B" in model_input and "Paper C" in model_input
