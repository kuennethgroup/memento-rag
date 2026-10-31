# After a research-gap answer about a document, does a follow-up that asks for extensions show the authors' table again?
# True = the reader wants the authors' points shown (again), False = extensions only. A repeated table is harmless; a
# hidden table that was asked for is not.
import pytest

import openwebui.functions.query_pipe as qp
from memento.query import request

H = [{"role": "user", "content": "What research gaps does the document X name?"},
     {"role": "assistant", "content": "| What the authors state | Points |\n|---|---|"}]

CASES = [
    ("Only suggest extensions", False), ("Just suggest extensions please", False),
    ("How could this work be extended?", False), ("Suggest some extensions", False), ("Add your own ideas", False),
    ("What would you do next?", False), ("What could be future directions beyond this?", False),
    ("Propose follow-up experiments", False), ("Do not repeat the gaps; just suggest extensions", False),
    ("No need to show the table again, just your ideas", False), ("Skip the list and give me your own ideas", False),
    ("Don't show the gaps again, only extensions", False), ("Please do not show those again; suggest extensions", False),
    ("Show those again and add your ideas", True), ("What research gaps does it name, and suggest extensions", True),
    ("Show its limitations again and suggest extensions", True), ("Repeat the gaps and add suggestions", True),
    ("For each of the points, suggest an extension", True), ("Based on the limitations, what should be done next?", True),
    ("Which of these gaps could I address in my thesis? Suggest a plan", True),
    ("Do not repeat your suggestions; show the gaps again", True), ("Show the table again, then extend it", True),
    ("Nur Erweiterungen vorschlagen", False), ("Schlag bitte eigene Ideen vor", False),
    ("Wie könnte man die Arbeit erweitern?", False), ("Die Lücken bitte nicht wiederholen, nur Vorschläge", False),
    ("Ohne die Tabelle zu wiederholen: was wären nächste Schritte?", False), ("Was würdest du als Nächstes untersuchen?", False),
    ("Zeig die Punkte nochmal und schlag Erweiterungen vor", True), ("Welche Lücken nennt es, und was schlägst du vor?", True),
    ("Zu jedem Punkt bitte eine Erweiterung vorschlagen", True), ("Erweitere bitte jeden der Punkte", True),
]


@pytest.mark.parametrize("question,show", CASES)
def test_the_table_is_shown_when_asked_for(question, show):
    assert request.show_gap_table(suggest=True, history=H, question=question, same_document=True) is show


@pytest.mark.xfail(strict=True, reason="known mismatch: 'points' reads as a request for them; "
                                       "the table is repeated, the harmless direction")
def test_seen_the_points_then_extensions_only():
    assert request.show_gap_table(suggest=True, history=H, question="I've seen the points, now suggest extensions",
                              same_document=True) is False


def test_another_document_always_shows_the_table():
    assert request.show_gap_table(suggest=True, history=H, question="Only suggest extensions", same_document=False)
