"""memento/common/text_language.py: is a text English or German?

Answer prompts name the question's language, and the answer store checks an answer's language before keeping it.
Counting common function words is enough for the two languages Memento serves: they barely overlap, and a scientific
answer has plenty of them. Too few to decide (formulas, citations only) gives None, "unsure", which callers treat as no
objection."""
import re

_EN = frozenset("""the and of to in is are was were be been this that these those with for from by on at as it its not
only which while but or than their there they we our has have had can may should would could will also into between
because such each more most other no any all both after before when where how what why""".split())
_DE = frozenset("""der die das den dem des und oder nicht ist sind war waren wird werden wurde wurden ein eine einer
eines einem einen mit für von zu zum zur auf bei aus nach über unter auch nur noch als wie wenn dass daß sich sie wir
ihre ihr sein seine kann können sollte sollten wäre durch zwischen oder aber jedoch welche welcher welches noch bisher
bleibt bleiben""".split())
_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_QUOTED = re.compile(r"[\"“„«»][^\"“”„«»]*[\"”“«»]")
_MIN_HITS = 3


def guess_language(text: str) -> str | None:
    """"en", "de", or None when the text has too few common words to tell (or both count the same)."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words)
    if max(en, de) < _MIN_HITS or en == de:
        return None
    return "en" if en > de else "de"


def question_language(question: str) -> str:
    """The language a question is asked in: "de" or "en" (the default). A quoted document title does not count
    ('Summarize the document "Einfluss der Temperatur …"' is an English question)."""
    q = _QUOTED.sub(" ", question or "")
    words = [w.lower() for w in _WORD.findall(q)]
    # German question words and umlauts count too
    de = sum(w in _DE for w in words) + sum(w in {"fasse", "zusammen", "welche", "forschungslücken", "dokument",
                                                  "nennt", "zeig", "zeige", "gib", "man", "warum", "wieso", "weshalb",
                                                  "gibt", "es", "lässt", "darauf", "dazu", "davon", "dafür", "wo",
                                                  "wer", "ich", "mir", "mich", "uns", "im", "vom", "beim",
                                                  "nennen",
                                                  # common German words that also decide short questions
                                                  "diese", "diesen", "dieser", "dieses", "diesem", "kurz", "steht",
                                                  "stehen", "hier", "bitte", "kannst", "du", "dir", "alle", "mal",
                                                  "liste", "erkläre", "erklär", "beschreibe", "vergleiche", "erzähl",
                                                  "erzähle", "welches", "welchen", "hat", "habe", "haben"}
                  for w in words) \
        + sum(1 for w in words if re.search(r"[äöüß]", w))
    # "was" and "in" are as common in German questions: not counted as English
    en = (sum(w in _EN and w not in ("was", "in") for w in words)
          + sum(w in {"summarize", "summarise", "research", "gaps", "document", "name", "show", "give", "list"}
                for w in words))
    return "de" if de > en else "en"
