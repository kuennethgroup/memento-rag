# A heading is not a statement: a statement whose only sentences are headings is not checked, so a ⚠ can only follow a
# real sentence.
import json
import types

from memento.agentic_validation.claims import segment, split_claims


class Fake:
    def __init__(self, outs):
        self.outs, self.requests = list(outs), []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.requests.append(kw)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            content=self.outs.pop(0)))])


ANS = ("Phosphorus systems work at low loadings [2].\n\n**Mechanisms that produce the low pHRR**\n"
       "- In the gas phase, phosphorus traps radicals [19].\n\n## Impact\nA lower pHRR means less flame spread [12].")


def test_the_sentences_are_as_expected():
    assert [ANS[a:b] for a, b in segment(ANS)] == [
        "Phosphorus systems work at low loadings [2].", "**Mechanisms that produce the low pHRR**",
        "- In the gas phase, phosphorus traps radicals [19].", "## Impact", "A lower pHRR means less flame spread [12]."]


def test_a_statement_on_a_heading_alone_is_not_checked():
    out = json.dumps([
        {"text": "Phosphorus systems work at low loadings", "cited": [2], "s": [1]},
        {"text": "Mechanisms produce the low pHRR", "cited": [], "s": [2]},
        {"text": "In the gas phase phosphorus traps radicals", "cited": [19], "s": [3]},
        {"text": "The impact is discussed", "cited": [], "s": [4]},
        {"text": "A lower pHRR means less flame spread", "cited": [12], "s": [5]}])
    cl = split_claims(Fake([out]), "m", ANS)
    assert [c.text for c in cl] == ["Phosphorus systems work at low loadings",
                                    "In the gas phase phosphorus traps radicals", "A lower pHRR means less flame spread"]
    assert [c.id for c in cl] == ["c1", "c2", "c3"]
    assert all(not ANS[c.span[0]:c.span[1]].startswith(("**Mech", "## ")) for c in cl)


def test_a_statement_from_a_heading_and_its_sentence_stays():
    out = json.dumps([{"text": "Gas-phase radical trapping lowers the pHRR", "cited": [19], "s": [2, 3]}])
    cl = split_claims(Fake([out]), "m", ANS)
    assert len(cl) == 1 and ANS[cl[0].span[0]:cl[0].span[1]].endswith("traps radicals [19].")


def test_a_bold_lead_in_with_text_is_a_sentence_not_a_heading():
    ans = "**DOPO** reduced the pHRR by 70 % [3].\n**Note:** the smoke rose by 50 % [3]."
    out = json.dumps([{"text": "DOPO reduced the pHRR by 70 %", "cited": [3], "s": [1]},
                      {"text": "The smoke rose by 50 %", "cited": [3], "s": [2]}])
    assert len(split_claims(Fake([out]), "m", ans)) == 2
