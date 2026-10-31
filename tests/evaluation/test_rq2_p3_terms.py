# tests/evaluation/test_rq2_p3_terms.py
from evaluation.RQ2.questions.terms import candidate_terms, hub_cap, term_doc_freq, term_key


def test_candidate_terms_finds_formulas_acronyms_and_codes():
    t = candidate_terms(r"Samples of $ZrB_{2}$ and TGMDA were cured with DDS; the PA66-GF30 compound (see Fig. 3).")
    assert {"ZrB2", "TGMDA", "DDS", "PA66-GF30"} <= t
    assert not ({"Fig", "FIG", "see"} & t)


def test_term_doc_freq_counts_documents_not_chunks():
    chunks = [{"source": "d1", "text": "TGMDA cured"}, {"source": "d1", "text": "more TGMDA"},
              {"source": "d2", "text": "TGMDA again"}]
    assert term_doc_freq(chunks)["TGMDA"] == {"d1", "d2"}


def test_term_key_folds_variants():
    assert term_key("Carbon-Fibre") == term_key("carbon fibre") == term_key("carbon fibres")
    assert term_key("ZrB2") != term_key("ZrB")


def test_hub_cap():
    assert hub_cap(405) == 8 and hub_cap(30) == 2


class _GateLLM:
    """Keeps the terms it is told to keep; records how many terms each call saw."""
    def __init__(self, keep, garbage_on=()):
        self.keep, self.garbage_on, self.sizes = set(keep), set(garbage_on), []
        self.chat = self.completions = self

    def create(self, **kw):
        import json as _j
        user = kw["messages"][1]["content"]
        terms = [l[2:] for l in user.splitlines() if l.startswith("- ")]
        self.sizes.append(len(terms))
        body = "not json" if self.garbage_on & set(terms) else _j.dumps(
            {"keep": [t for t in terms if t in self.keep] + ["INVENTED"]})
        msg = type("M", (), {"content": body})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_term_gate_keeps_only_scientific_terms_it_was_asked_about():
    # no organisations, adjectives or function phrases as bridges/seeds
    from evaluation.RQ2.questions.terms import gate_terms
    terms = ["ZrB2", "NIST", "non-reactive", "TGMDA", "above-mentioned"] * 1 + [f"T{i}X" for i in range(50)]
    llm = _GateLLM(keep={"ZrB2", "TGMDA", "T3X"})
    kept, unparsed = gate_terms(llm, "m", terms, batch=40)
    assert kept == {"ZrB2", "TGMDA", "T3X"} and unparsed == 0          # never an invented term
    assert llm.sizes == [40, 15]


def test_term_gate_drops_a_batch_it_cannot_parse():
    from evaluation.RQ2.questions.terms import gate_terms
    kept, unparsed = gate_terms(_GateLLM(keep={"ZrB2"}, garbage_on={"ZrB2"}), "m", ["ZrB2", "NIST"], batch=40)
    assert kept == set() and unparsed == 2
