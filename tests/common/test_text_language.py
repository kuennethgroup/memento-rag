# Which language an answer or a question is written in.
import sys
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.common.text_language import guess_language, question_language  # noqa: E402

DE = ("### Einschränkungen\n1. Die Sauerstoffpositionen, die für die Unterscheidung polarer und nichtpolarer "
      "orthorhombischer Gitter entscheidend sind, lassen sich mit Röntgenbeugung nicht bestimmen [1].\n"
      "2. Die Proben wurden nur bei Raumtemperatur gemessen, und der Einfluss der Dotierung ist noch offen [2].")
EN = ("### Limitations\n1. The models assume all copolymers are random because copolymer-type information was not "
      "uniformly available [3].\n### Future work\n2. Prior polymer datasets should be extended to the other classes [4].")


def test_answers():
    assert guess_language(DE) == "de"
    assert guess_language(EN) == "en"


def test_too_little_text_is_unsure():
    assert guess_language("[1][2]") is None
    assert guess_language("HfO2, ZrO2, Si:HfO2") is None
    assert guess_language("") is None


def test_questions():
    assert question_language('What research gaps does the document "Impact of field cycling" name?') == "en"
    assert question_language("Welche Forschungslücken nennt das Dokument über Hafnia?") == "de"
    # a German title inside an English picker question stays an English question
    assert question_language('Summarize the document "Einfluss der Temperatur auf die Festigkeit von Keramik"') == "en"
    assert question_language('Fasse das Dokument "Impact of field cycling on HfO2" zusammen') == "de"


def test_short_german_questions_are_german():
    # short German questions without an umlaut are German
    for q in ("Was könnte man darauf aufbauend untersuchen?", "Was könnte man tun?", "Warum ist das so?",
              "Welche Einschränkungen nennen die Autoren?", "Gibt es dazu Daten?", "Wie lässt sich das erweitern?"):
        assert question_language(q) == "de", q
    for q in ("What was the result?", "Can you suggest future work for this study?", "Is there a figure of the pipeline?",
              'Summarize the document "Einfluss der Temperatur auf die Festigkeit von Keramik"'):
        assert question_language(q) == "en", q
