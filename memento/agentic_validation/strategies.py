"""The RQ3 arms.

P0: production prompt. P1: P0 + strict citation instruction (baseline).
C: P1 answer → claims → verifier + scientific checks → sentences with a failing claim removed.
A: P1 answer → the same checks → the same generator rewrites only the failing sentences (verdicts as feedback); on an
evidence gap one targeted re-retrieval adds sources; up to max_rounds, then a final check of the result.
A and C start from the same P1 answer, so the comparison isolates the loop.

Failing claim = verdict "unsupported" / "contradicted" or any CheckIssue. "unjudged" claims are neither removed nor
rewritten (a verifier outage must not delete content) and are reported.

The notes under a checked answer, and the one finalize_for_display writes when the check fails, are defined here;
every place that recognises them uses these definitions (the query pipe keeps its own copies of the texts, pinned
to these by tests/verification/test_check_notes.py).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from memento.agentic_validation.claims import Claim, clean_answer, is_aggregate, split_claims
from memento.agentic_validation.scientific_checks import CLAIM_RULES, CheckIssue, run_checks
from memento.agentic_validation.verifiers import VerdictCache, Verdict, verify_claims
from memento.retrieval.answer_pipeline import build_messages, generate_answer


P1_ADDENDUM = (
    "\nStrict citation rule: every factual statement must end with the [N] of the source that states it. Do not use "
    "outside knowledge. If the sources do not cover part of the question, say so explicitly instead of filling the gap.")

REWRITE_INSTRUCTION = """\
Below is your previous answer and a list of its statements that the sources do not support, with the reason. \
Revise the answer:
- Change ONLY the listed statements. Keep every other sentence exactly as it is, including its [N] citations.
- For each listed statement: correct it so it says exactly what the cited or another numbered source states, with \
the right [N]; or delete it if no source supports it.
- Do not add information that is not in the sources. Keep the language, structure and formatting of the answer.
Output ONLY the revised answer."""

SUBQUERY_PROMPT = """\
The following statements could not be verified in the retrieved sources. Write ONE short search query (keywords, same \
language as the statements) that would find a source for them. Output only the query."""

NO_ANSWER = {"de": "Die Quellen stützen keine verlässliche Antwort auf diese Frage.",
             "en": "The sources do not support a reliable answer to this question."}
_DE_WORDS = re.compile(r"\b(der|die|das|und|welche|wie|wird|werden|ist|sind|mit|von|für|bei|nicht)\b", re.I)


def _lang(text: str) -> str:
    return "de" if len(_DE_WORDS.findall(text or "")) >= 2 else "en"


class _Counting:
    """Wraps an OpenAI-shaped client and counts chat calls."""

    def __init__(self, llm):
        self._llm, self.calls = llm, 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        return self._llm.chat.completions.create(**kw)


@dataclass
class ArmContext:
    llm: object
    model: str
    system_prompt: str                      # the domain's production prompt (P1 adds P1_ADDENDUM)
    question: str                           # the question's original wording
    hits: list[dict]                        # frozen retrieval, numbered 1..n
    text_verifier: object
    figure_verifier: object | None = None
    cache: VerdictCache = field(default_factory=VerdictCache)
    retrieve_fn: Callable[[str], list[dict]] | None = None   # arm A re-retrieval
    enable_thinking: bool = True            # the production thinking decision for this question
    # A's revision calls: "escalate" = round 1 without thinking, round 2 with the answer's thinking setting (no-thinking
    # revisions halve the latency at equal quality; thinking fixes more hard leftovers); True/False = every round; None
    # = same as enable_thinking
    revise_thinking: bool | str | None = "escalate"
    # the revision's thinking cap (vLLM thinking_token_budget): 12,288 as evaluated in RQ3; the live Thorough check
    # passes its own, THOROUGH_THINKING_BUDGET
    thinking_budget: int = 12288
    figure_mode: str = "skip"
    claim_rules: tuple = CLAIM_RULES        # the library's model-free checks (domain profile `claim_rules`)
    max_new_hits: int = 5
    timings: list = field(default_factory=list)   # (step, seconds) per step, for the latency log


@dataclass
class ArmResult:
    arm: str
    answer: str
    claims: list[Claim] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    issues: dict[str, list[CheckIssue]] = field(default_factory=dict)
    rounds: int = 0
    llm_calls: int = 0                  # all model calls: generation + splitting + verification
    verifier_calls: int = 0             # of which verification (text + figure)
    latency_s: float = 0.0
    unjudged: int = 0
    removed: list[str] = field(default_factory=list)       # C: failing claim ids removed
    collateral: list[str] = field(default_factory=list)    # C: passing claim ids removed with a failing sentence
    hits: list[dict] = field(default_factory=list)         # final numbered sources (A may add some)
    subqueries: list[str] = field(default_factory=list)    # A: re-retrieval queries
    thinking: str | None = None
    trace: list[dict] = field(default_factory=list)        # per round: n claims, failing ids, reasons
    timings: list = field(default_factory=list)            # (step, seconds): split / verify / re-search / revise


def _verifier_calls(ctx: ArmContext) -> int:
    return sum(getattr(v, "calls", 0) for v in (ctx.text_verifier, ctx.figure_verifier) if v is not None)


def _failing(claims, verdicts, issues) -> list[Claim]:
    return [c for c in claims
            if verdicts[c.id].label in ("unsupported", "contradicted") or issues.get(c.id)]


def _reason(c: Claim, v: Verdict, iss: list[CheckIssue]) -> str:
    parts = []
    if v.label in ("unsupported", "contradicted"):
        parts.append(v.label + (f" ({v.rationale})" if v.rationale else ""))
    parts += [f"{i.kind}: {i.detail}" for i in iss]
    return "; ".join(parts)


def check_answer(ctx: ArmContext, llm, answer: str, hits: list[dict], *, check_gap: bool = False, previous=None):
    """Claims of `answer` + verdicts + scientific-check issues against `hits`. `previous` = (answer, claims) of the
    version before a revision: only changed sentences are split again (split_claims)."""
    t = time.perf_counter()
    claims = split_claims(llm, ctx.model, answer, previous=previous)
    ctx.timings.append(("split", round(time.perf_counter() - t, 1)))
    t = time.perf_counter()
    verdicts = verify_claims(claims, hits, text_verifier=ctx.text_verifier, figure_verifier=ctx.figure_verifier,
                             cache=ctx.cache, check_gap=check_gap)
    ctx.timings.append(("verify", round(time.perf_counter() - t, 1)))
    issues = {}
    for c in claims:
        cited = [hits[n - 1] for n in c.cited if 1 <= n <= len(hits)]
        iss = (run_checks(c.id, c.text, cited, figure_mode=ctx.figure_mode, rules=ctx.claim_rules)
               if cited and not is_aggregate(c.text) else [])
        if iss:
            issues[c.id] = iss
    return claims, verdicts, issues


# ── P0 / P1 ───────────────────────────────────────────────────────────────────
def run_prompt_arm(ctx: ArmContext, arm: str) -> ArmResult:
    t0, llm = time.perf_counter(), _Counting(ctx.llm)
    system = ctx.system_prompt + (P1_ADDENDUM if arm == "P1" else "")
    out = generate_answer(llm, ctx.model, build_messages(system, ctx.hits, ctx.question),
                          enable_thinking=ctx.enable_thinking)
    return ArmResult(arm, out["answer"], llm_calls=llm.calls, latency_s=round(time.perf_counter() - t0, 2),
                     hits=list(ctx.hits), thinking=out["thinking"])


# ── C: remove failing sentences ───────────────────────────────────────────────
def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    for a, b in sorted(_merge(spans), reverse=True):
        text = text[:a] + text[b:]
    lines = []
    for ln in text.split("\n"):
        stripped = ln.strip()
        if re.fullmatch(r"(?:[-*•]|\d+[.)])\s*(?:\*\*[^*]*\*\*:?)?\s*", stripped):   # emptied list item
            continue
        lines.append(re.sub(r"[ \t]{2,}", " ", ln).rstrip())
    text = "\n".join(lines)
    # a heading whose section became empty
    text = re.sub(r"(?m)^#{1,6} [^\n]*\n+(?=#{1,6} |\Z)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _merge(spans):
    out = []
    for a, b in sorted(s for s in spans if s[0] >= 0):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def run_arm_c(ctx: ArmContext, p1_answer: str) -> ArmResult:
    t0, llm, v0 = time.perf_counter(), _Counting(ctx.llm), _verifier_calls(ctx)
    text = clean_answer(p1_answer)
    claims, verdicts, issues = check_answer(ctx, llm, text, ctx.hits)
    failing = _failing(claims, verdicts, issues)
    spans = [c.span for c in failing if c.span[0] >= 0]
    removed_set = _merge(spans)
    inside = lambda c: c.span[0] >= 0 and any(a <= c.span[0] and c.span[1] <= b for a, b in removed_set)
    collateral = [c.id for c in claims if c not in failing and inside(c)]
    answer = _remove_spans(text, spans)
    if not re.search(r"\w", re.sub(r"\[\d+\]", "", answer)):
        answer = NO_ANSWER[_lang(ctx.question)]
    vc = _verifier_calls(ctx) - v0
    return ArmResult("C", answer, claims, verdicts, issues, rounds=1, llm_calls=llm.calls + vc, verifier_calls=vc,
                     latency_s=round(time.perf_counter() - t0, 2),
                     unjudged=sum(v.label == "unjudged" for v in verdicts.values()),
                     removed=[c.id for c in failing if c.span[0] >= 0], collateral=collateral, hits=list(ctx.hits),
                     trace=[{"round": 1, "claims": len(claims), "failing": {c.id: _reason(c, verdicts[c.id], issues.get(c.id, []))
                                                                         for c in failing}}])


# ── A: verify → revise (+ re-retrieval) ───────────────────────────────────────
def _key(h: dict) -> tuple:
    return (h.get("source"), h.get("page"), (h.get("text") or "")[:200])


def run_arm_a(ctx: ArmContext, p1_answer: str, *, max_rounds: int = 2) -> ArmResult:
    t0, llm, v0 = time.perf_counter(), _Counting(ctx.llm), _verifier_calls(ctx)
    answer, hits = clean_answer(p1_answer), list(ctx.hits)
    subqueries, trace, rounds = [], [], 0
    claims, verdicts, issues = check_answer(ctx, llm, answer, hits, check_gap=ctx.retrieve_fn is not None)
    while True:
        failing = _failing(claims, verdicts, issues)
        trace.append({"round": rounds, "claims": len(claims),
                      "failing": {c.id: _reason(c, verdicts[c.id], issues.get(c.id, [])) for c in failing},
                      "answer": answer, "t_s": round(time.perf_counter() - t0, 2)})   # state after this round (logging)
        if not failing or rounds >= max_rounds:
            break
        gap = [c for c in failing if verdicts[c.id].evidence_gap]
        t = time.perf_counter()
        if gap and ctx.retrieve_fn is not None:
            q = llm.chat.completions.create(
                model=ctx.model, temperature=0.0, max_tokens=64,
                messages=[{"role": "system", "content": SUBQUERY_PROMPT},
                          {"role": "user", "content": "\n".join(f"- {c.text}" for c in gap)}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}).choices[0].message.content
            q = (q or "").strip().strip('"').splitlines()[0][:200] if q else ""
            if q:
                subqueries.append(q)
                seen = {_key(h) for h in hits}
                new = [h for h in (ctx.retrieve_fn(q) or []) if _key(h) not in seen][:ctx.max_new_hits]
                hits += new
            ctx.timings.append(("re-search", round(time.perf_counter() - t, 1)))
        problems = "\n".join(f"- \"{c.text}\" → {_reason(c, verdicts[c.id], issues.get(c.id, []))}" for c in failing)
        messages = build_messages(ctx.system_prompt + P1_ADDENDUM, hits, ctx.question)
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"{REWRITE_INSTRUCTION}\n\nStatements to fix:\n{problems}"})
        if ctx.revise_thinking == "escalate":
            think = ctx.enable_thinking and rounds > 0
        else:
            think = ctx.enable_thinking if ctx.revise_thinking is None else ctx.revise_thinking
        t = time.perf_counter()
        revised = generate_answer(llm, ctx.model, messages, enable_thinking=think,
                                  thinking_budget=ctx.thinking_budget)["answer"]
        ctx.timings.append(("revise", round(time.perf_counter() - t, 1)))
        if not revised:                                   # no revision came back: keep the answer, record it
            trace[-1]["revision_empty"] = True
        previous = (answer, claims)
        answer = revised or answer
        rounds += 1
        claims, verdicts, issues = check_answer(ctx, llm, answer, hits, check_gap=ctx.retrieve_fn is not None,
                                                previous=previous)
    vc = _verifier_calls(ctx) - v0
    return ArmResult("A", answer, claims, verdicts, issues, rounds=rounds, llm_calls=llm.calls + vc, verifier_calls=vc,
                     latency_s=round(time.perf_counter() - t0, 2),
                     unjudged=sum(v.label == "unjudged" for v in verdicts.values()),
                     hits=hits, subqueries=subqueries, trace=trace, timings=list(ctx.timings))


# ── dispatch + display ────────────────────────────────────────────────────────
def run_strategy(name: str, ctx: ArmContext, *, p1_answer: str | None = None, max_rounds: int = 2) -> ArmResult:
    if name in ("P0", "P1"):
        return run_prompt_arm(ctx, name)
    if p1_answer is None:
        raise ValueError(f"arm {name} starts from a P1 answer")
    if name == "C":
        return run_arm_c(ctx, p1_answer)
    if name == "A":
        return run_arm_a(ctx, p1_answer, max_rounds=max_rounds)
    raise ValueError(f"unknown arm {name!r}")


def remaining_failures(result: ArmResult | None) -> list[Claim]:
    """The statements of the displayed answer that still fail after the check: a verdict of unsupported / contradicted,
    or a scientific-check issue. A: the final round's claims (they describe result.answer); C: the failing claims it
    could not remove (no sentence for them)."""
    claims = getattr(result, "claims", None) or []
    if not claims:
        return []
    verdicts, issues = getattr(result, "verdicts", None) or {}, getattr(result, "issues", None) or {}
    if not verdicts and not issues:
        return []
    # a claim without a verdict is not judged, so not failing (as "unjudged")
    failing = [c for c in claims if (c.id in verdicts and verdicts[c.id].label in ("unsupported", "contradicted"))
               or issues.get(c.id)]
    if getattr(result, "arm", "A") == "C":
        removed = set(getattr(result, "removed", None) or [])
        return [c for c in failing if c.id not in removed]
    return failing


def mark_unconfirmed(answer: str, claims: list[Claim]) -> tuple[str, bool]:
    """(the answer with " ⚠" after each sentence holding one of `claims`, whether every claim got its mark). Marks only
    when every claim has a sentence and the spans fit the shown text (they refer to clean_answer(answer)); otherwise
    the text is returned unchanged, never with stale offsets."""
    if not claims:
        return answer, True
    text = (answer or "").strip()
    if clean_answer(answer) != text or any(c.span[0] < 0 or c.span[1] > len(text) or c.span[1] <= c.span[0]
                                           for c in claims):
        return answer, False
    ends = sorted({c.span[1] for c in claims}, reverse=True)
    for end in ends:
        text = text[:end] + " ⚠" + text[end:]
    return text, True


def check_state(result: ArmResult | None, error: Exception | None, late: bool = False) -> str:
    """How complete a check was: "complete" (every claim judged), "no_claims" (no claim was extracted: the answer may
    hold nothing to check, or the extraction missed it, so this certifies nothing), "partial" (some claims unjudged: a
    judge call failed for them, and they stay in the text), "unavailable" (an error, out of time, no result, or claims
    of which none was judged)."""
    if late or error is not None or result is None:
        return "unavailable"
    n, unjudged = len(getattr(result, "claims", None) or []), int(getattr(result, "unjudged", 0) or 0)
    if not n:
        return "no_claims"
    if unjudged >= n:
        return "unavailable"
    return "partial" if unjudged else "complete"


def finalize_for_display(answer: str, result: ArmResult | None, error: Exception | None) -> str:
    """What the live chat shows: the verified answer, or — if verification failed — the unverified answer with a
    visible note (never a silent swallow)."""
    if result is not None and error is None:
        return result.answer
    return f"{answer}\n\n_⚠ Verification unavailable for this answer ({type(error).__name__ if error else 'no result'})._"


# ── Notes under a checked answer ──────────────────────────────────────────────
CHECKED = "_Checked with Thorough check._"
PARTLY = "_Partly checked with Thorough check: some statements could not be checked._"
NO_CLAIMS = "_Not checked with Thorough check: it found no statements to check._"
# statements still failing after the check: marked ⚠ where their sentence is known
CHECKED_UNCONFIRMED = "_Checked with Thorough check: some statements could not be confirmed and are marked ⚠._"
CHECKED_UNCONFIRMED_UNMARKED = "_Checked with Thorough check: some statements could not be confirmed._"
PARTLY_UNCONFIRMED = ("_Partly checked with Thorough check: some statements could not be checked, and some could not "
                      "be confirmed and are marked ⚠._")
PARTLY_UNCONFIRMED_UNMARKED = ("_Partly checked with Thorough check: some statements could not be checked, and some "
                               "could not be confirmed._")
# the check-status notes: removed when a stored answer replays that was not checked under the current contract
STATUS = (CHECKED, PARTLY, NO_CLAIMS, CHECKED_UNCONFIRMED, CHECKED_UNCONFIRMED_UNMARKED, PARTLY_UNCONFIRMED,
          PARTLY_UNCONFIRMED_UNMARKED)
# under the AI's own extensions section of a checked answer: the check left it out
EXTENSIONS_UNCHECKED = "_Not checked with Thorough check._"
TIMEOUT = "_⚠ The thorough check could not finish in time, so this answer was not checked._"
# finalize_for_display: "_⚠ Verification unavailable for this answer (<error type>)._"
UNAVAILABLE_PREFIX = "_⚠ Verification unavailable for this answer"
UNAVAILABLE_RE = r"_⚠ Verification unavailable for this answer \([^)\n]*\)\._"

DE = {
    CHECKED: "_Geprüft mit Thorough check._",
    PARTLY: "_Teilweise geprüft mit Thorough check: einige Aussagen konnten nicht geprüft werden._",
    NO_CLAIMS: "_Nicht mit Thorough check geprüft: Es wurden keine prüfbaren Aussagen gefunden._",
    TIMEOUT: "_⚠ Thorough check konnte nicht rechtzeitig fertig werden, daher wurde diese Antwort nicht geprüft._",
    CHECKED_UNCONFIRMED: ("_Geprüft mit Thorough check: einige Aussagen konnten nicht bestätigt werden und sind mit ⚠ "
                          "markiert._"),
    CHECKED_UNCONFIRMED_UNMARKED: "_Geprüft mit Thorough check: einige Aussagen konnten nicht bestätigt werden._",
    PARTLY_UNCONFIRMED: ("_Teilweise geprüft mit Thorough check: einige Aussagen konnten nicht geprüft werden, und "
                         "einige konnten nicht bestätigt werden und sind mit ⚠ markiert._"),
    PARTLY_UNCONFIRMED_UNMARKED: ("_Teilweise geprüft mit Thorough check: einige Aussagen konnten nicht geprüft werden, "
                                  "und einige konnten nicht bestätigt werden._"),
    EXTENSIONS_UNCHECKED: "_Nicht mit Thorough check geprüft._",
}
UNAVAILABLE_DE_RE = r"_⚠ Prüfung für diese Antwort nicht verfügbar \(([^)\n]*)\)\._"
UNAVAILABLE_DE = "_⚠ Prüfung für diese Antwort nicht verfügbar ({reason})._"

# any of the notes, in English (German notes are turned back with the pipe's texts.untr before they are recognised)
NOTE_RE = (r"_(?:Checked with Thorough check(?:: [^\n]*)?\.|Partly checked with Thorough check:[^\n]*|"
           r"Not checked with Thorough check: [^\n]*\.|"
           r"⚠ The thorough check could not finish in time[^\n]*|"
           r"⚠ Verification unavailable for this answer \([^)\n]*\)\.)_")

# a whole line that is one of the pipe's own italic notes: the check notes in English and German, the "Answered by"
# line; any other "_…_" line is the model's text and is judged as such
import re as _re
PIPE_LINE_RE = _re.compile(
    "^(?:" + NOTE_RE + "|" + "|".join(_re.escape(v) for v in DE.values()) + "|" + UNAVAILABLE_DE_RE + "|"
    + _re.escape(EXTENSIONS_UNCHECKED) + r"|_(?:Answered by|Beantwortet von) [^\n]+_)$")
