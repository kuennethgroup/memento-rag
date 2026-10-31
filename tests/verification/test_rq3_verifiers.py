"""RQ3 task 3: verifiers (memento/agentic_validation/verifiers.py)."""
import json
import re
import threading
import types

import pytest
from PIL import Image

from memento.agentic_validation.claims import Claim
from memento.agentic_validation.verifiers import (LLMJudgeVerifier, VerdictCache, Verdict, VLMFigureVerifier, combine,
                                          verify_claims, windows)


class ScriptedLLM:
    """Answers every call with a function of the request; records requests (thread-safe)."""
    def __init__(self, answer):
        self.answer, self.requests, self._lock = answer, [], threading.Lock()
        self.chat = self; self.completions = self

    def create(self, **kw):
        with self._lock:
            self.requests.append(kw)
        out = self.answer(kw)
        if isinstance(out, Exception):
            raise out
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=out))])


def _local_ids(kw):
    msg = kw["messages"][-1]["content"]
    text = msg if isinstance(msg, str) else msg[0]["text"]
    return re.findall(r"^(c\d+): (.*)$", text.split("Claims:\n", 1)[1], re.M)


def label_by_text(rules, default="supported"):
    """Label each claim in the request by the first rule (substring → label) that matches its text."""
    def answer(kw):
        out = []
        for lid, text in _local_ids(kw):
            label = next((lab for sub, lab in rules.items() if sub in text), default)
            out.append({"id": lid, "label": label, "rationale": "r"})
        return json.dumps(out)
    return answer


H = lambda text, typ="text", img=None: {"text": text, "type": typ, "img_path": img, "source": "P", "page": 0}


def test_missing_and_unknown_verdicts_are_unjudged_never_supported():
    v = LLMJudgeVerifier(ScriptedLLM(lambda kw: json.dumps([{"id": "c1", "label": "supported"},
                                                             {"id": "c2", "label": "probably"},
                                                             {"id": "zzz", "label": "supported"}])), "m")
    out = v.verify([("a", "A", "src"), ("b", "B", "src"), ("c", "C", "src")])
    assert (out["a"].label, out["b"].label, out["c"].label) == ("supported", "unjudged", "unjudged")


def test_errors_are_unjudged():
    v = LLMJudgeVerifier(ScriptedLLM(lambda kw: RuntimeError("server down")), "m")
    out = v.verify([("a", "A", "src")])
    assert out["a"].label == "unjudged" and "RuntimeError" in out["a"].rationale
    v2 = LLMJudgeVerifier(ScriptedLLM(lambda kw: "no json here"), "m")
    assert v2.verify([("a", "A", "src")])["a"].label == "unjudged"


def test_claims_sharing_a_source_are_batched_and_mapped_back_by_id():
    llm = ScriptedLLM(label_by_text({"bad": "contradicted"}))
    v = LLMJudgeVerifier(llm, "m", batch=12)
    out = v.verify([("x|1", "good one", "S1"), ("y|2", "bad one", "S1"), ("z|3", "good two", "S2")])
    assert len(llm.requests) == 2                           # one call per distinct source
    assert (out["x|1"].label, out["y|2"].label, out["z|3"].label) == ("supported", "contradicted", "supported")
    assert out["y|2"].claim_id == "y|2"


def test_windows_and_combine():
    w = windows("x" * 15000, 6000, overlap=400)
    assert len(w) == 3 and sum(len(p) for p in w) >= 15000
    V = lambda lab: Verdict("c", lab)
    assert combine([V("unsupported"), V("supported")], "c").label == "supported"
    assert combine([V("unsupported"), V("contradicted")], "c").label == "contradicted"
    assert combine([V("unjudged"), V("unsupported")], "c").label == "unsupported"
    assert combine([V("unjudged")], "c").label == "unjudged" and combine([], "c").label == "unjudged"


def test_long_source_supported_if_any_window_supports():
    def answer(kw):
        src = kw["messages"][-1]["content"]
        return json.dumps([{"id": lid, "label": "supported" if "NEEDLE" in src else "unsupported"}
                           for lid, _ in _local_ids(kw)])
    v = LLMJudgeVerifier(ScriptedLLM(answer), "m", window_chars=6000)
    src = "filler " * 1500 + "NEEDLE fact" + " filler" * 100
    assert v.verify([("a", "fact", src)])["a"].label == "supported"


def _png(p):
    Image.new("RGB", (30, 20), "red").save(p)
    return str(p)


def test_routing_uncited_invalid_text_table_figure_and_caption(tmp_path):
    hits = [H("T1"), H("<table><tr><td>5 GPa</td></tr></table>", "table"),
            H("Fig caption", "image", _png(tmp_path / "a.png")), H("Chart caption", "chart", str(tmp_path / "missing.png"))]
    claims = [Claim("c1", "uncited claim", (), (0, 1)), Claim("c2", "bad number", (9,), (0, 1)),
              Claim("c3", "text claim", (1,), (0, 1)), Claim("c4", "table claim", (2,), (0, 1)),
              Claim("c5", "figure claim", (3,), (0, 1)), Claim("c6", "caption claim", (4,), (0, 1))]
    text_llm, fig_llm = ScriptedLLM(label_by_text({})), ScriptedLLM(label_by_text({}))
    out = verify_claims(claims, hits, text_verifier=LLMJudgeVerifier(text_llm, "m"),
                        figure_verifier=VLMFigureVerifier(fig_llm, "m"))
    assert (out["c1"].label, out["c1"].rationale) == ("unsupported", "uncited")
    assert (out["c2"].label, out["c2"].rationale) == ("unsupported", "invalid citation")
    assert [out[c].via for c in ("c3", "c4", "c5", "c6")] == ["text", "table", "vlm", "caption"]
    assert all(out[c].label == "supported" for c in ("c3", "c4", "c5", "c6"))
    img_req = fig_llm.requests[0]["messages"][-1]["content"]
    assert img_req[1]["type"] == "image_url" and "Fig caption" in img_req[0]["text"]


def test_claim_citing_text_and_figure_is_supported_if_either_supports(tmp_path):
    hits = [H("unrelated text"), H("cap", "chart", _png(tmp_path / "c.png"))]
    out = verify_claims([Claim("c1", "value read from chart", (1, 2), (0, 1))], hits,
                        text_verifier=LLMJudgeVerifier(ScriptedLLM(label_by_text({}, "unsupported")), "m"),
                        figure_verifier=VLMFigureVerifier(ScriptedLLM(label_by_text({}, "supported")), "m"))
    assert out["c1"].label == "supported" and out["c1"].via == "vlm"


def test_cache_avoids_repeat_calls():
    llm = ScriptedLLM(label_by_text({}))
    v, cache = LLMJudgeVerifier(llm, "m"), VerdictCache()
    hits, cl = [H("S")], [Claim("c1", "A", (1,), (0, 1))]
    verify_claims(cl, hits, text_verifier=v, cache=cache)
    verify_claims(cl, hits, text_verifier=v, cache=cache)
    assert len(llm.requests) == 1


def test_evidence_gap_only_when_no_source_supports():
    def answer(kw):
        src = kw["messages"][-1]["content"]
        return json.dumps([{"id": lid, "label": "supported" if ("zirconia" in text and "yttria" in src) else "unsupported"}
                           for lid, text in _local_ids(kw)])
    hits = [H("alumina only"), H("zirconia toughened by yttria")]
    claims = [Claim("c1", "zirconia toughened", (1,), (0, 1)), Claim("c2", "silicon carbide sinters", (1,), (0, 1))]
    out = verify_claims(claims, hits, text_verifier=LLMJudgeVerifier(ScriptedLLM(answer), "m"), check_gap=True)
    assert out["c1"].label == "unsupported" and out["c1"].evidence_gap is False   # another hit supports it
    assert out["c2"].evidence_gap is True


def test_aggregate_claims_are_unjudged_and_never_sent_to_the_verifier():
    """A count statement over several citations is judged against the cited excerpts together, not one at a time (else a
    correct count answer is deleted or rewritten to "not stated")."""
    hits = [H("Ester bonds form."), H("Esterification occurs.")]
    claims = [Claim("c1", "Sechs der vorliegenden Arbeiten beschreiben die Bildung von Ester-Bindungen", (1, 2), (0, 1)),
              Claim("c2", "More works use CSM as a module than as a technique", (1,), (0, 1)),
              Claim("c3", "In one study, ester bonds form", (1,), (0, 1)),
              Claim("c4", "Six works describe it", (), (0, 1))]
    llm = ScriptedLLM(label_by_text({}))
    out = verify_claims(claims, hits, text_verifier=LLMJudgeVerifier(llm, "m"))
    assert [(out[c].label, out[c].rationale) for c in ("c1", "c2")] == [("unjudged", "aggregate claim")] * 2
    assert out["c3"].label == "supported"
    assert (out["c4"].label, out["c4"].rationale) == ("unsupported", "uncited")    # the citation rule still applies
    sent = [t for kw in llm.requests for _, t in _local_ids(kw)]
    assert sent == ["In one study, ester bonds form"]


def test_verifier_sees_the_source_label_the_answer_model_saw():
    """The verifier gets each excerpt's document name, so a statement about which document a citation comes from can be
    checked."""
    hits = [{"text": "Excerpt.", "type": "text", "source": "02_SdT_komplett", "page": 4, "section": "Rheologie"}]
    llm = ScriptedLLM(label_by_text({}))
    verify_claims([Claim("c1", "Source 1 comes from 02_SdT_komplett", (1,), (0, 1))], hits,
                  text_verifier=LLMJudgeVerifier(llm, "m"))
    sent = llm.requests[0]["messages"][-1]["content"]
    sent = sent if isinstance(sent, str) else sent[0]["text"]
    assert "02_SdT_komplett, page 5, Section: Rheologie\nExcerpt." in sent
