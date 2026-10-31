"""RQ3 task 2: claim splitter (memento/agentic_validation/claims.py)."""
import json
import types

import pytest

from memento.agentic_validation.claims import (MAX_BLOCK_CHARS, ClaimSplitError, clean_answer, is_aggregate, markers, segment,
                                       split_claims, strip_markers)


class Fake:
    """OpenAI-shaped sync client returning queued outputs; records every request."""
    def __init__(self, outs):
        self.outs, self.requests = list(outs), []
        self.chat = self; self.completions = self

    def create(self, **kw):
        self.requests.append(kw)
        msg = types.SimpleNamespace(content=self.outs.pop(0))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


ANS = "Alumina reached 99 % density at 1600 °C [2]. It was then polished.\n\n## Summary\nSPS was faster [1][3]."
# segment(ANS): S1 "Alumina … [2].", S2 "It was then polished.", S3 "## Summary", S4 "SPS was faster [1][3]."


def test_markers_and_strip():
    assert markers("a [2][5] b [2, 7] c [3–5] d [1-2]") == (2, 5, 7, 3, 4, 1)
    assert strip_markers("SPS was faster [1][3].") == "SPS was faster."


@pytest.mark.parametrize("raw,clean", [
    ("In [1] wird der Aushärtegrad mittels Infrarotspektroskopie bestimmt.",
     "In der Quelle wird der Aushärtegrad mittels Infrarotspektroskopie bestimmt."),
    ("In [10] und [11] wird der Umsatz der Isocyanatkomponente bestimmt.",
     "In den Quellen wird der Umsatz der Isocyanatkomponente bestimmt."),
    ("Die in [8] tabellarisch aufgeführten Werte sind der Umsatz an Epoxidgruppen.",
     "Die in der Quelle tabellarisch aufgeführten Werte sind der Umsatz an Epoxidgruppen."),
    ("In [4], entropy design stabilizes solid solutions.", "In the source, entropy design stabilizes solid solutions."),
    ("Source [16] contains no such statement.", "The source contains no such statement."),
    ("According to [2] and [3], SPS was faster [2].", "According to the sources, SPS was faster."),
    ("Nach 10 min [1] war die Probe ausgehärtet.", "Nach 10 min war die Probe ausgehärtet."),
    ("In [6] wird der Vernetzungsgrad based on the analysis of the epoxy resins using FT-IR berechnet.",
     "In der Quelle wird der Vernetzungsgrad based on the analysis of the epoxy resins using FT-IR berechnet."),
])
def test_citation_used_as_a_noun_becomes_a_neutral_noun(raw, clean):
    # qwen keeps "In [1] wird …" despite the prompt; removing the frame would break German word order
    assert strip_markers(raw) == clean


def test_clean_answer_drops_think_sources_block_and_images():
    raw = ("<think>plan</think>Answer [1].\n\n![P p.2](data:image/png;base64,AAA)\n\n"
           '<details type="memento_sources"><summary>Sources</summary>1=P</details>')
    assert clean_answer(raw) == "Answer [1]."


def test_segment_handles_abbreviations_decimals_markers_and_lists():
    t = ("Die Härte betrug ca. 1,5 GPa bei 10 Gew.-% SiO2 [1]. Z. B. zeigte Probe A 2.3 µm Körner [2][3]. "
         "Nach Smith et al. war das so [4]. Fig. 3 shows grain growth.\n- Second item: 40 % [9]!\n"
         "The qualifying documents are:\n1. Item one [5]. Item one b.\nWhat about e.g. yttria? It works [6].")
    assert [t[a:b] for a, b in segment(t)] == [
        "Die Härte betrug ca. 1,5 GPa bei 10 Gew.-% SiO2 [1].", "Z. B. zeigte Probe A 2.3 µm Körner [2][3].",
        "Nach Smith et al. war das so [4].", "Fig. 3 shows grain growth.", "- Second item: 40 % [9]!",
        "The qualifying documents are:", "1. Item one [5].", "Item one b.", "What about e.g. yttria?", "It works [6]."]


def test_split_uses_numbered_sentences_our_ids_and_exact_spans():
    out = json.dumps([
        {"id": "x9", "text": "Alumina reached 99 % density at 1600 °C [2]", "cited": [2], "s": [1]},
        {"id": "x1", "text": "SPS was faster", "cited": [1, 3], "s": [4]}])
    llm = Fake([out])
    cl = split_claims(llm, "m", ANS)
    assert [c.id for c in cl] == ["c1", "c2"] and cl[1].cited == (1, 3)
    assert cl[0].text == "Alumina reached 99 % density at 1600 °C"          # markers stripped
    assert ANS[cl[0].span[0]:cl[0].span[1]] == "Alumina reached 99 % density at 1600 °C [2]."
    assert ANS[cl[1].span[0]:cl[1].span[1]] == "SPS was faster [1][3]."
    kw = llm.requests[0]
    assert kw["messages"][1]["content"].startswith("S1: Alumina reached") and "S4: SPS was faster" in kw["messages"][1]["content"]
    assert kw["temperature"] == 0.0 and kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_citations_checked_against_the_claims_own_sentences():
    out = json.dumps([
        {"text": "SPS was faster", "cited": [9], "s": [4]},          # 9 is not in S4 → dropped → inherits (1, 3)
        {"text": "SPS was faster than HP", "cited": [2], "s": [4]},  # 2 exists in the answer but not in S4 → dropped
        {"text": "Alumina reached 99 % density", "cited": [], "s": [1]},
        {"text": "It was then polished", "cited": [], "s": [2]},     # truly uncited
        {"text": "orphan", "cited": [1], "s": [99]}])               # unknown sentence → no span, no citation
    cl = split_claims(Fake([out]), "m", ANS)
    assert [c.cited for c in cl] == [(1, 3), (1, 3), (2,), (), ()]
    assert cl[4].span == (-1, -1)


def test_fenced_json_and_one_retry():
    good = "```json\n" + json.dumps([{"text": "SPS was faster", "cited": [1], "s": [4]}]) + "\n```"
    llm = Fake(["Sure! Here are the claims:", good])
    assert len(split_claims(llm, "m", ANS)) == 1 and len(llm.requests) == 2
    assert llm.requests[1]["messages"][-1]["content"].startswith("Output ONLY")


def test_raises_after_two_bad_outputs():
    with pytest.raises(ClaimSplitError):
        split_claims(Fake(["not json", "[{\"no_text\": 1}]"]), "m", ANS)


def test_long_answers_are_split_into_blocks_with_global_spans():
    para = ("Grain size grew with temperature [1]. " * 40).strip()
    answer = "\n\n".join([para] * 4)
    assert len(answer) > MAX_BLOCK_CHARS
    one = json.dumps([{"text": "Grain size grew with temperature", "cited": [1], "s": [2]}])
    llm = Fake([one] * 10)
    cl = split_claims(llm, "m", answer)
    assert len(llm.requests) >= 2
    assert [c.id for c in cl] == [f"c{i}" for i in range(1, len(cl) + 1)]
    assert all(answer[c.span[0]:c.span[1]] == "Grain size grew with temperature [1]." for c in cl)
    assert len({c.span for c in cl}) == len(cl)                 # spans are global, not block-relative


def test_empty_answer_needs_no_call():
    llm = Fake([])
    assert split_claims(llm, "m", "<think>x</think>  ") == [] and llm.requests == []


def test_latex_backslashes_in_model_json_are_tolerated():
    # models copy LaTeX into JSON strings: a lone backslash must not fail the parse
    out = '[{"text": "Sintered at $1600^{\\circ}\\mathrm{C}$", "cited": [1], "s": [1]}]'
    cl = split_claims(Fake([out]), "m", "Sintered at 1600 °C [1].")
    assert len(cl) == 1 and "circ" in cl[0].text


@pytest.mark.parametrize("text,agg", [
    ("Sechs der vorliegenden Arbeiten beschreiben die Bildung von Ester-Bindungen [1][2][5].", True),
    ("CSM is used as a measurement technique in four works [2][5][6][9].", True),
    ("Therefore, there are more works using CSM as a module than as a technique [1][2].", True),
    ("11 documents report XRD data [1].", True),
    ("Die meisten Arbeiten verwenden PP [2].", True),
    ("In one study [3], the hardness was 5 GPa.", False),
    ("Both works [1][2] report sintering at 1500 °C.", False),
    ("Figure 3 of the paper shows the grain size [2].", False),
    ("Table 2 in the work lists 12 samples [1].", False),
    ("Eine Arbeit [3] zeigt, dass die Dichte 98 % beträgt.", False),
    ("The study reports six samples with a density of 3.2 g/cm³ [4].", False),
    ("Abbildung 5 der Arbeit zeigt REM-Aufnahmen [2].", False),
])
def test_is_aggregate(text, agg):
    assert is_aggregate(text) is agg


def test_loads_lenient_repairs_unescaped_inner_quotes():
    """A quotation copied into a claim without escaping it must not leave the whole answer unjudged."""
    from memento.agentic_validation.jsonfix import loads_lenient
    raw = ('[{"text": "The images label "agglomerates of silica nanoparticles"", "cited": [3, 8], "s": [2]},\n'
           ' {"text": "The authors call them "unexpected" here", "cited": [13], "s": [4]},\n'
           ' {"text": "LaTeX \\mathrm{C} stays", "cited": [], "s": [5]}]')
    out = loads_lenient(raw)
    assert [d["text"] for d in out] == ['The images label "agglomerates of silica nanoparticles"',
                                        'The authors call them "unexpected" here', "LaTeX \\mathrm{C} stays"]
    assert out[0]["cited"] == [3, 8]
    assert loads_lenient('{"a": "x", "b": [1, 2]}') == {"a": "x", "b": [1, 2]}     # valid JSON untouched


def test_loads_lenient_mixed_latex_escapes():
    """One claim may write a single backslash and the next an escaped one: both parse; rho, frac and beta look like
    valid JSON escapes (r, f, b) and are kept as LaTeX."""
    from memento.agentic_validation.jsonfix import loads_lenient
    raw = (r'[{"text": "diffusivity ($\alpha$), ($\lambda$ or $\kappa$), density ($\rho$), $\frac{1}{2}$, \beta", "s": [1]},'
           r' {"text": "$a = \\frac{\\lambda}{\\rho \\cdot c_p}$", "s": [3]},'
           r' {"text": "line1\nLine2 \t \"q\" ä \\\\ ok", "s": [4]}]')
    out = [d["text"] for d in loads_lenient(raw)]
    assert out[0] == r"diffusivity ($\alpha$), ($\lambda$ or $\kappa$), density ($\rho$), $\frac{1}{2}$, \beta"
    assert out[1] == r"$a = \frac{\lambda}{\rho \cdot c_p}$"
    assert out[2] == 'line1\nLine2 \t "q" ä \\\\ ok'


# ── re-split after a revision: only changed sentences, whole block as context ──
def test_resplit_after_revision_only_splits_changed_sentences():
    import json as _json
    import re
    import types as _types

    class Rec:
        def __init__(self):
            self.contents = []
            self.chat = self; self.completions = self

        def create(self, **kw):
            content = kw["messages"][1]["content"]
            self.contents.append(content)
            only = re.search(r"Split ONLY these sentences: ([S\d, ]+)\.", content)
            only = {int(x) for x in re.findall(r"S(\d+)", only.group(1))} if only else None
            out = []
            for line in content.splitlines():
                m = re.match(r"S(\d+): (.*)", line)
                if m and (only is None or int(m.group(1)) in only):
                    out.append({"text": m.group(2), "cited": list(markers(m.group(2))), "s": [int(m.group(1))]})
            return _types.SimpleNamespace(choices=[_types.SimpleNamespace(
                message=_types.SimpleNamespace(content=_json.dumps(out)))])

    v1 = "Alumina was sintered at 1600 °C [1]. The density reached 3.9 g/cm3 [2]. Hardness was 18 GPa [3]."
    v2 = "Alumina was sintered at 1600 °C [1]. The density reached 3.95 g/cm3 [2]. Hardness was 18 GPa [3]."
    llm = Rec()
    first = split_claims(llm, "m", v1)
    again = split_claims(llm, "m", v2, previous=(v1, first))
    assert "Split ONLY these sentences: S2." in llm.contents[1]                 # only the changed sentence
    assert "S1: Alumina" in llm.contents[1] and "S3: Hardness" in llm.contents[1]   # the rest shown as context
    assert [c.id for c in again] == ["c1", "c2", "c3"]
    assert [c.text for c in again][0] == first[0].text and "3.95" in again[1].text and again[2].text == first[2].text
    assert all(v2[c.span[0]:c.span[1]].startswith(c.text.split()[0]) for c in again)   # spans point into v2
    assert split_claims(llm, "m", v2, previous=(v2, again)) and len(llm.contents) == 2   # nothing changed: no call
