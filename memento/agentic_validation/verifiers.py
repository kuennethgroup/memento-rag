"""Check claims against their cited sources. The verifier is the local model: LLMJudgeVerifier for text, tables and
captions, VLMFigureVerifier for claims citing a figure (the model sees the image); verify_claims routes each claim.

Rules:
- every claim gets a verdict; a claim the model does not answer for, an unknown label or any error → "unjudged", never
"supported"; verdicts are matched by claim id, never by position;
- a claim without a citation → "unsupported" (uncited); a citation number with no source → "unsupported" (invalid);
- long sources are checked in overlapping windows: supported if any window supports, contradicted if one contradicts and
none supports;
- `check_gap=True` re-checks failing claims against all retrieved text: `evidence_gap` = nothing supports it (arm A then
re-retrieves);
- verdicts are cached per (claim, source, verifier), so revision rounds re-check only what changed."""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from memento.agentic_validation.claims import is_aggregate
from memento.agentic_validation.jsonfix import loads_lenient
from memento.retrieval.answer_pipeline import chunk_label, source_text

LABELS = ("supported", "unsupported", "contradicted", "unjudged")
FIGURE_TYPES = ("image", "chart")


@dataclass
class Verdict:
    claim_id: str
    label: str                  # one of LABELS
    score: float | None = None  # P(supported) where available (the LLM judge gives none)
    via: str = "text"           # "text" | "table" | "vlm" | "caption" | "none"
    rationale: str = ""
    evidence_gap: bool = False  # unsupported by ALL retrieved sources (set by check_gap)


class Verifier(Protocol):
    name: str

    def verify(self, pairs: list[tuple[str, str, str]]) -> dict[str, Verdict]:
        """pairs = (claim_id, claim_text, source_text); returns a Verdict for EVERY claim_id."""


VERIFY_SYSTEM_PROMPT = """\
You check claims against a source excerpt from a scientific document. For each claim decide:
- "supported": the excerpt states it or it follows directly from the excerpt.
- "contradicted": the excerpt states something incompatible (a different value, unit, material, direction, condition).
- "unsupported": the excerpt does not contain the information (neither states nor contradicts it).
Check EVERY number, unit, material and condition in the claim (including conditions such as temperatures, times and \
compositions, not only the main value): all must match. Equivalent notations count as a match \
(1,5 = 1.5; N/mm² = MPa; Gew.-% = wt%). The claim and the excerpt may be in different languages. Judge only against \
the excerpt, not your own knowledge.
Output ONLY a JSON list with one entry per claim id: [{"id": "c1", "label": "supported", "rationale": "short reason"}]"""

FIGURE_SYSTEM_PROMPT = """\
You check claims against a figure from a scientific document (the image and its caption). For each claim decide:
- "supported": the figure (plot values, labels, legend, visible content) or its caption shows it.
- "contradicted": the figure or caption shows something incompatible (a different value, trend, label or unit).
- "unsupported": the figure and caption do not show it.
Check EVERY number, unit, material and condition in the claim — including the conditions a curve or panel belongs to \
(temperatures, times, compositions, sample names in the legend or caption), not only the plotted value. If any of \
them is not shown in the figure or caption, the claim is "unsupported"; if one differs, it is "contradicted". Read \
values from axes and legends carefully; approximate readings within the plot's resolution count as a match. Judge \
only what the figure and caption show.
Output ONLY a JSON list with one entry per claim id: [{"id": "c1", "label": "supported", "rationale": "short reason"}]"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _parse_verdicts(raw: str) -> list[dict]:
    raw = (raw or "").strip()
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        raise ValueError("no JSON list")
    data = loads_lenient(raw[i:j + 1])
    if not isinstance(data, list):
        raise ValueError("not a list")
    return [d for d in data if isinstance(d, dict)]


def _to_verdicts(items: list[dict], ids: list[str], via: str) -> dict[str, Verdict]:
    """Known ids with a valid label get it; everything else is unjudged."""
    out = {cid: Verdict(cid, "unjudged", None, via, "no verdict returned") for cid in ids}
    for d in items:
        cid, label = str(d.get("id", "")), str(d.get("label", "")).strip().lower()
        if cid in out and label in LABELS[:3]:
            out[cid] = Verdict(cid, label, None, via, str(d.get("rationale", ""))[:300])
    return out


def windows(text: str, size: int, overlap: int = 400) -> list[str]:
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def combine(verdicts: list[Verdict], claim_id: str) -> Verdict:
    """Several checks of one claim (windows, or several cited sources): supported if any supports; else contradicted
    if any contradicts; else unsupported if any judged; else unjudged."""
    for label in ("supported", "contradicted", "unsupported"):
        hit = next((v for v in verdicts if v.label == label), None)
        if hit:
            return replace(hit, claim_id=claim_id)
    base = verdicts[0] if verdicts else Verdict(claim_id, "unjudged", None, "text", "no check")
    return replace(base, claim_id=claim_id, label="unjudged")


class VerdictCache:
    def __init__(self):
        self._d: dict[str, Verdict] = {}

    @staticmethod
    def key(claim: str, source: str, verifier: str) -> str:
        return hashlib.sha1(f"{claim}\x00{source}\x00{verifier}".encode()).hexdigest()

    def get(self, key: str) -> Verdict | None:
        return self._d.get(key)

    def put(self, key: str, v: Verdict) -> None:
        self._d[key] = v


def _chat(llm, model: str, messages: list[dict], enable_thinking: bool) -> str:
    resp = llm.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=4096,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}})
    return resp.choices[0].message.content or ""


class LLMJudgeVerifier:
    """An OpenAI-compatible chat model judging claims against source text (qwen3.8 in RQ3)."""

    def __init__(self, llm, model: str, *, batch: int = 12, window_chars: int = 6000, workers: int = 4,
                 enable_thinking: bool = False, via: str = "text"):
        self.llm, self.model, self.batch, self.window_chars = llm, model, batch, window_chars
        self.workers, self.enable_thinking, self.via = workers, enable_thinking, via
        self.name = f"llm:{model}:think={enable_thinking}"
        self.calls = 0

    def _one(self, source: str, claims: list[tuple[str, str]]) -> dict[str, Verdict]:
        # short local ids in the prompt (models mangle long ones), mapped back afterwards
        local = {f"c{k + 1}": cid for k, (cid, _) in enumerate(claims)}
        listing = "\n".join(f"c{k + 1}: {text}" for k, (_, text) in enumerate(claims))
        msgs = [{"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Source excerpt:\n{source}\n\nClaims:\n{listing}"}]
        try:
            self.calls += 1
            got = _to_verdicts(_parse_verdicts(_chat(self.llm, self.model, msgs, self.enable_thinking)),
                               list(local), self.via)
        except Exception as e:                      # model/server/parse failure → unjudged, never supported
            got = {k: Verdict(k, "unjudged", None, self.via, f"error: {type(e).__name__}") for k in local}
        return {local[k]: replace(v, claim_id=local[k]) for k, v in got.items()}

    def verify(self, pairs: list[tuple[str, str, str]]) -> dict[str, Verdict]:
        by_source: dict[str, list[tuple[str, str]]] = {}
        for cid, claim, source in pairs:
            by_source.setdefault(source, []).append((cid, claim))
        jobs = []                                   # (claim ids, source window, claims)
        for source, claims in by_source.items():
            for w in windows(source, self.window_chars):
                for k in range(0, len(claims), self.batch):
                    jobs.append((w, claims[k:k + self.batch]))
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            results = list(pool.map(lambda j: self._one(*j), jobs))
        per_claim: dict[str, list[Verdict]] = {}
        for res in results:
            for cid, v in res.items():
                per_claim.setdefault(cid, []).append(v)
        return {cid: combine(per_claim.get(cid, []), cid) for cid, _, _ in pairs}


class VLMFigureVerifier:
    """qwen3.8 with the figure image: all claims about one figure in one call."""

    def __init__(self, llm, model: str, *, enable_thinking: bool = False):
        self.llm, self.model, self.enable_thinking = llm, model, enable_thinking
        self.name = f"vlm:{model}:think={enable_thinking}"
        self.calls = 0

    def verify_figure(self, claims: list[tuple[str, str]], caption: str, img_path: str | None) -> dict[str, Verdict] | None:
        """Verdicts for (claim_id, text) pairs, or None if the image cannot be read (caller falls back to caption)."""
        from memento.retrieval.answer_pipeline import image_data_url
        url = image_data_url(img_path) if img_path else None
        if not url:
            return None
        local = {f"c{k + 1}": cid for k, (cid, _) in enumerate(claims)}
        listing = "\n".join(f"c{k + 1}: {text}" for k, (_, text) in enumerate(claims))
        msgs = [{"role": "system", "content": FIGURE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Caption / text of the figure:\n{caption}\n\nClaims:\n{listing}"},
                    {"type": "image_url", "image_url": {"url": url}}]}]
        try:
            self.calls += 1
            got = _to_verdicts(_parse_verdicts(_chat(self.llm, self.model, msgs, self.enable_thinking)), list(local), "vlm")
        except Exception as e:
            got = {k: Verdict(k, "unjudged", None, "vlm", f"error: {type(e).__name__}") for k in local}
        return {local[k]: replace(v, claim_id=local[k]) for k, v in got.items()}


def _label(h: dict) -> str:
    """The source line the answer model saw above this excerpt (answer_pipeline.chunk_label), so claims about which
    document/section a statement comes from are checkable; nothing for a hit without source/page."""
    return chunk_label(h) + "\n" if h.get("source") and isinstance(h.get("page"), int) else ""


def _source_text(hits: list[dict], nums: list[int]) -> str:
    """The cited sources as the answer model read them (with the authors' text under a table or an equation)."""
    return "\n\n".join(f"[{n}] {_label(hits[n - 1])}{source_text(hits[n - 1])}" for n in nums)


def verify_claims(claims, hits: list[dict], *, text_verifier: Verifier, figure_verifier: VLMFigureVerifier | None = None,
                  cache: VerdictCache | None = None, check_gap: bool = False) -> dict[str, Verdict]:
    """Route and check every claim against the hits it cites (1-based numbers into `hits`).
    - no citation → unsupported "uncited"; no valid citation number → unsupported "invalid citation";
    - a count/comparison across works (claims.is_aggregate) → unjudged "aggregate claim";
    - cited text/table hits are checked together as one source (via "table" if all are tables);
    - each cited figure is checked by the VLM with its image, or on its caption text if the image is unreadable
      (via "caption"); a claim's checks are combined (supported if any cited source supports it);
    - check_gap: failing claims are re-checked against all hits' text → evidence_gap."""
    cache = cache or VerdictCache()
    out: dict[str, Verdict] = {}
    text_jobs: dict[str, tuple[str, str, str]] = {}      # key → (claim_id, claim, source) for the text verifier
    parts: dict[str, list] = {}                           # claim_id → list of Verdict | cache key
    fig_jobs: dict[int, list[tuple[str, str]]] = {}       # figure hit number → [(claim_id, claim)]

    for c in claims:
        if not c.cited:
            out[c.id] = Verdict(c.id, "unsupported", None, "none", "uncited")
            continue
        nums = [n for n in c.cited if 1 <= n <= len(hits)]
        if not nums:
            out[c.id] = Verdict(c.id, "unsupported", None, "none", "invalid citation")
            continue
        if is_aggregate(c.text):                          # a count/comparison across works: no excerpt can state it
            out[c.id] = Verdict(c.id, "unjudged", None, "none", "aggregate claim")
            continue
        parts[c.id] = []
        figs = [n for n in nums if hits[n - 1].get("type") in FIGURE_TYPES]
        texts = [n for n in nums if n not in figs]
        if texts:
            via = "table" if all(hits[n - 1].get("type") == "table" for n in texts) else "text"
            src = _source_text(hits, texts)
            key = VerdictCache.key(c.text, src, text_verifier.name) + via
            if cache.get(key):
                parts[c.id].append(cache.get(key))
            else:
                text_jobs[key] = (f"{c.id}|{key[:8]}", c.text, src)
                parts[c.id].append(key)
        for n in figs:
            fig_jobs.setdefault(n, []).append((c.id, c.text))

    # figures: one VLM call per figure; an unreadable image's caption goes to the text verifier. The figure calls run in
    # parallel with each other and with the text checks (the model server queues beyond its slots)
    def _figure(n: int, fclaims: list):
        h = hits[n - 1]
        fkey = {cid: VerdictCache.key(text, f"fig:{h.get('img_path')}", figure_verifier.name) for cid, text in fclaims}
        todo = [(cid, text) for cid, text in fclaims if not cache.get(fkey[cid])]
        res = figure_verifier.verify_figure(todo, h.get("text", ""), h.get("img_path")) if todo else {}
        return n, fkey, res

    def _run_text(jobs: dict) -> None:
        if not jobs:
            return
        res = text_verifier.verify(list(jobs.values()))
        for key, (jid, _, _) in jobs.items():
            v = res.get(jid) or Verdict(jid, "unjudged", None, "text", "no verdict returned")
            if key.endswith("caption"):
                v = replace(v, via="caption")
            elif key.endswith("table"):
                v = replace(v, via="table")
            cache.put(key, v)

    fig_results = []
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(fig_jobs)) + 1)) as pool:
        text_done = pool.submit(_run_text, dict(text_jobs))
        futures = ([pool.submit(_figure, n, fclaims) for n, fclaims in fig_jobs.items()]
                   if figure_verifier is not None else [])
        fig_results = [f.result() for f in futures]
        text_done.result()

    captions: dict[str, tuple[str, str, str]] = {}
    got = {n: (fkey, res) for n, fkey, res in fig_results}
    for n, fclaims in fig_jobs.items():
        fkey, res = got.get(n, (None, None))
        if res is not None:
            for cid, text in fclaims:
                v = cache.get(fkey[cid]) or res[cid]
                cache.put(fkey[cid], v)
                parts[cid].append(v)
            continue
        src = _source_text(hits, [n])                     # caption fallback (image unreadable / no figure verifier)
        for cid, text in fclaims:
            key = VerdictCache.key(text, src, text_verifier.name) + "caption"
            if cache.get(key):
                parts[cid].append(cache.get(key))
            else:
                captions[key] = (f"{cid}|{key[:8]}", text, src)
                parts[cid].append(key)
    _run_text(captions)
    for cid, ps in parts.items():
        vs = [cache.get(p) if isinstance(p, str) else p for p in ps]
        out[cid] = combine([v for v in vs if v], cid)

    if check_gap:
        failing = [c for c in claims if c.id in out and out[c.id].label != "supported"]
        if failing:
            everything = _source_text(hits, [n for n in range(1, len(hits) + 1)])
            res = text_verifier.verify([(c.id, c.text, everything) for c in failing])
            for c in failing:
                gap = res.get(c.id) is None or res[c.id].label != "supported"
                out[c.id] = replace(out[c.id], evidence_gap=gap)
    return {c.id: out[c.id] for c in claims}
