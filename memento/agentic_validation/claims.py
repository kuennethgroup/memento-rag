"""Split an answer into atomic, cited claims for the RQ3 check. The answer is cut into numbered sentences here
(deterministic; list items, German and English abbreviations, decimals and citation markers handled). One no-thinking
LLM call per block returns claims that point to those sentence numbers, so the model never copies sentences back. This
module makes the output safe:
- claim ids are ours (c1, c2, … in answer order), never the model's;
- citation numbers are checked against the claim's own sentences: a number not written there is dropped, and a claim
with no valid number inherits its sentences' markers, so a slip never makes arm C delete a cited claim;
- `span` = the claim's sentences in the answer, so arm C removes exactly those;
- thinking blocks, the pipe's sources block and inline images are stripped before splitting;
- unparseable output is retried once, then ClaimSplitError, never a silent empty list."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from memento.agentic_validation.jsonfix import loads_lenient



@dataclass(frozen=True)
class Claim:
    id: str                    # "c1", "c2", … stable within an answer
    text: str                  # atomic claim, citation markers removed
    cited: tuple[int, ...]     # source numbers (1-based) supporting it; empty = uncited
    span: tuple[int, int]      # char offsets in the (cleaned) answer of the claim's sentence(s); (-1,-1) = none given


class ClaimSplitError(RuntimeError):
    pass


SPLIT_SYSTEM_PROMPT = """\
You split a scientific answer into atomic factual claims. The answer is given as numbered sentences "S1: …".
Rules:
- One claim = one checkable statement about the subject matter (a value, a relation, a finding, a method). Split \
sentences that state several facts; keep each claim self-contained (replace "it"/"they"/"this material" by what \
they refer to).
- State the fact itself. When a sentence uses a source as its subject ("In [1] the curing degree was determined by \
FT-IR", "Laut [2] …"), the claim is the fact without the source: "The curing degree was determined by FT-IR".
- Copy numbers, units, formulas and material names EXACTLY as written; never round or convert.
- "cited": the citation numbers [N] that support this claim, as integers.
- "s": the sentence numbers the claim comes from (usually one).
- Skip headings, questions, transitions, sentences that only introduce a list ("The following documents …:"), \
summaries of the other claims ("all sources agree …"), and statements that the sources do not cover something.
- Keep the answer's language.
Output ONLY a JSON list: [{"text": "...", "cited": [1, 2], "s": [3]}]"""

_MARKER = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_DETAILS = re.compile(r"<details\b.*?</details>", re.S | re.I)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
MAX_BLOCK_CHARS = 6000
# a long answer is split in up to SPLIT_PARALLEL paragraph groups at once, each request with the whole answer as context
SPLIT_PARALLEL = 4
MIN_BLOCK_CHARS = 1200
# words after which a period does not end a sentence (German + English, lower-cased, without the period)
_ABBREV = {"z", "b", "bzw", "ca", "vgl", "u", "a", "d", "h", "o", "s", "ggf", "usw", "etc", "evtl", "inkl", "max", "min",
           "nr", "abb", "tab", "kap", "gl", "bsp", "fig", "figs", "eq", "eqs", "ref", "refs", "e", "g", "i", "al", "et",
           "vs", "approx", "no", "sec", "resp", "dr", "prof", "st", "mio", "mrd", "jh", "sog", "zzgl"}
_END = re.compile(r"(?<=[.!?])(?:\s*\[\d[\d,\s–-]*\])*\s+(?=[\"„“(\[*]?[A-ZÄÖÜ0-9])")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]\s*)?(?:\d+|[a-zA-Z]|[ivx]+)[.)]$")


def clean_answer(answer: str) -> str:
    """The answer text a reader sees: no thinking block, no sources block, no inline images."""
    return _IMAGE.sub("", _DETAILS.sub("", _THINK.sub("", answer or ""))).strip()


def markers(text: str) -> tuple[int, ...]:
    """Citation numbers written in `text`: [3], [2][5], [2, 5], [2–4] (ranges expanded), in order, unique."""
    out: list[int] = []
    for m in _MARKER.finditer(text or ""):
        for part in re.split(r"\s*,\s*", m.group(1)):
            rng = re.split(r"\s*[–-]\s*", part)
            nums = range(int(rng[0]), int(rng[-1]) + 1) if len(rng) == 2 and int(rng[-1]) >= int(rng[0]) else [int(rng[0])]
            out += [n for n in nums if n not in out]
    return tuple(out)


_M = r"\[\d[\d,\s–-]*\]"
_MRUN = re.compile(rf"{_M}(?:\s*(?:,|und|and|sowie|&)?\s*{_M})*")
_DE = re.compile(r"\b(der|die|das|und|wird|werden|ist|sind|mit|von|für|eine?[mnrs]?|nicht|bei)\b", re.I)
_EN = re.compile(r"\b(the|and|is|are|was|were|with|of|for|an?|not|at)\b", re.I)
# a citation used as a noun ("In [1] wird …", "according to [2] and [3]", "Source [16] contains …") becomes a neutral
# noun in the claim's language; plain trailing markers are removed (qwen keeps this framing despite the prompt)
_NOUN_BEFORE = re.compile(r"\b(in|laut|gemäß|nach|according to|as reported in|reported in|from|aus|source|document|"
                          r"study|paper|quelle|dokument|studie|arbeit)(\s+)(?=\[\d)", re.I)


def strip_markers(text: str) -> str:
    """Claim text without citation markers; a marker used as a noun is replaced by "the source(s)" / "der Quelle"
    (so "In [1] wird X bestimmt" → "In der Quelle wird X bestimmt", grammatical in either language)."""
    def is_german(m: re.Match, run) -> bool:
        # the words right around the citation decide (a German claim may quote an English phrase); whole text as tie-break
        near = " ".join(text[max(0, m.start() - 40):m.start()].split()[-3:]
                        + text[run.end() if run else m.end():].split()[:5])
        d, e = len(_DE.findall(near)), len(_EN.findall(near))
        return d > e if d != e else len(_DE.findall(text)) > len(_EN.findall(text))

    def noun(m: re.Match) -> str:
        word, gap = m.group(1), m.group(2)
        run = _MRUN.match(text, m.end())
        german = is_german(m, run)
        plural = len(markers(run.group())) > 1 if run else False
        w = word.lower()
        if w in ("source", "document", "study", "paper", "quelle", "dokument", "studie", "arbeit"):
            return ("Die Quellen" if plural else "Die Quelle") if german else ("The sources" if plural else "The source")
        if german:
            obj = {"in": "den Quellen" if plural else "der Quelle", "laut": "den Quellen" if plural else "der Quelle",
                   "gemäß": "den Quellen" if plural else "der Quelle", "nach": "den Quellen" if plural else "der Quelle",
                   "aus": "den Quellen" if plural else "der Quelle"}.get(w, "den Quellen" if plural else "der Quelle")
        else:
            obj = "the sources" if plural else "the source"
        return f"{word}{gap}{obj}\x00"

    t = _NOUN_BEFORE.sub(noun, text)
    t = re.sub(r"\x00\s*" + _MRUN.pattern, "", t)            # drop the marker run that became the noun
    t = t.replace("\x00", "")
    t = re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub("", t))
    return re.sub(r"\s{2,}", " ", t).strip()


# ── aggregate claims ──────────────────────────────────────────────────────────
# "Six works describe X [1][2][5]…", "more works use A than B": a count or comparison across the collection. No single
# excerpt states it, so an excerpt check cannot judge it; such claims are left unjudged (list questions are graded on
# the cited documents and the stated count instead). "one study [3] found X" / "both works [1][2]" stay ordinary claims.
NUMBER_WORDS = {w: i for i, ws in enumerate((
    (), (), ("two", "zwei"), ("three", "drei"), ("four", "vier"), ("five", "fünf"), ("six", "sechs"), ("seven", "sieben"), ("eight", "acht"),
    ("nine", "neun"), ("ten", "zehn"), ("eleven", "elf"), ("twelve", "zwölf"), ("thirteen", "dreizehn"),
    ("fourteen", "vierzehn"), ("fifteen", "fünfzehn"), ("sixteen", "sechzehn"), ("seventeen", "siebzehn"),
    ("eighteen", "achtzehn"), ("nineteen", "neunzehn"), ("twenty", "zwanzig"))) for w in ws}
_WORKS = (r"(?:works?|papers?|stud(?:y|ies)|documents?|publications?|articles?|sources?|theses|thesis|dissertations?|"
          r"reports?|Arbeiten|Arbeit|Dokumente[n]?|Dokument|Studien|Studie|Publikationen|Veröffentlichungen|Quellen|"
          r"Dissertationen|Beiträge[n]?)")
_BETWEEN = (r"(?:of|the|these|those|all|present|listed|retrieved|cited|analy[sz]ed|different|further|other|separate|"
            r"distinct|der|die|den|dieser|vorliegenden|genannten|zitierten|angegebenen|verschiedenen|unterschiedlichen|"
            r"weiteren|anderen|von|aus|insgesamt|in\s+total)")
_COUNT = re.compile(rf"(?<!Figure )(?<!Fig\. )(?<!Table )(?<!Tab\. )(?<!Abb\. )(?<!Abbildung )(?<!Tabelle )"
                    rf"\b(?:\d{{1,3}}|{'|'.join(sorted(NUMBER_WORDS, key=len, reverse=True))})"
                    rf"(?:\s+{_BETWEEN})*\s+{_WORKS}\b", re.I)
_COMPARE = re.compile(rf"\b(?:more|fewer|most|majority|mehr|weniger|meisten|Mehrheit|Mehrzahl|überwieg\w*|predominate\w*)"
                      rf"(?:\s+(?:deutlich|clearly|{_BETWEEN}))*\s+{_WORKS}\b", re.I)


def is_aggregate(text: str) -> bool:
    t = _MARKER.sub("", text or "")
    return bool(_COUNT.search(t) or _COMPARE.search(t))


# a line that is only a heading ("## Impact", "**Mechanisms …**"): never a statement, so no ⚠ stands after a heading
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S.*|\*\*[^*\n]+\*\*\s*:?|__[^_\n]+__\s*:?)\s*$")


def _heading(sentence: str) -> bool:
    return bool(_HEADING.match(sentence or ""))


def segment(text: str) -> list[tuple[int, int]]:
    """Sentence spans (start, end) in `text`: each non-empty line (paragraph, list item, table row) is split further
    at sentence ends — '.', '!' or '?' (+ trailing citation markers) followed by whitespace and an upper-case letter,
    digit or opening quote — unless the period ends an abbreviation (z. B., ca., et al., Fig.), a single letter or a
    list number ("1."). Decimals ("1.5", "1,5") never split (no whitespace after the dot)."""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        if not line.group().strip():
            continue
        start = line.start()
        for m in _END.finditer(line.group()):
            cut = line.start() + m.start()
            prev = re.search(r"([A-Za-zÄÖÜäöüß]+)\.\s*(?:\[[^\]]*\])*$", text[start:cut + 1].rstrip())
            if prev and (prev.group(1).lower() in _ABBREV or len(prev.group(1)) == 1):
                continue
            if _LIST_PREFIX.match(text[start:cut]):          # "1." / "a)" of a numbered list item
                continue
            seg = text[start:cut].strip()
            if seg:
                a = text.index(seg, start)
                spans.append((a, a + len(seg)))
            start = line.start() + m.end()
        seg = text[start:line.end()].strip()
        if seg:
            a = text.index(seg, start)
            spans.append((a, a + len(seg)))
    return spans


def _blocks(text: str, max_chars: int = MAX_BLOCK_CHARS) -> list[tuple[int, str]]:
    """Paragraph groups of at most `max_chars`, with their offsets in `text`."""
    paras = [(m.start(), m.group()) for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]*)*", text)]
    out, cur, start = [], "", None
    for off, p in paras:
        if cur and len(cur) + len(p) > max_chars:
            out.append((start, cur)); cur, start = "", None
        if start is None:
            start = off
        cur = text[start:off + len(p)]
    if cur:
        out.append((start, cur))
    return out


def _parse(raw: str) -> list[dict]:
    raw = (raw or "").strip()
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1).strip()
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        raise ValueError("no JSON list")
    data = loads_lenient(raw[i:j + 1])
    if not isinstance(data, list) or not all(isinstance(d, dict) and str(d.get("text", "")).strip() for d in data):
        raise ValueError("not a list of claims")
    return data


def _call(llm, model: str, numbered: str, reminder: bool, only: list[int] | None = None,
          context: str | None = None) -> str:
    content = numbered
    if context:   # one of several paragraph groups: the whole answer resolves "it" / "this material" across groups
        content = (f"The whole answer, for context only (do not split it):\n{context}\n\n"
                   f"Numbered sentences to split:\n{numbered}")
    if only:      # re-split after a revision: the whole block stays visible as context, claims only for these
        content += (f"\n\nSplit ONLY these sentences: {', '.join(f'S{k}' for k in only)}. The other sentences are "
                    f"context only: produce no claims for them.")
    msgs = [{"role": "system", "content": SPLIT_SYSTEM_PROMPT}, {"role": "user", "content": content}]
    if reminder:
        msgs.append({"role": "user", "content": "Output ONLY the JSON list, nothing else."})
    resp = llm.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=8192,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return resp.choices[0].message.content or ""


def _reusable(previous, text: str) -> tuple[dict[str, list[tuple[str, tuple, list[str]]]], bool]:
    """From a previous (answer, claims) split: for each sentence text, the claims that came only from unchanged
    sentences, as (claim text, cited, [sentence texts]), so a revision re-splits only what changed."""
    if not previous:
        return {}, False
    old_text, old_claims = clean_answer(previous[0]), previous[1]
    old_sents = [(a, b) for off, block in _blocks(old_text) for a, b in [(off + x, off + y) for x, y in segment(block)]]
    new_set = {text[a:b] for off, block in _blocks(text) for a, b in [(off + x, off + y) for x, y in segment(block)]}
    by_first: dict[str, list] = {}
    for c in old_claims:
        if c.span[0] < 0:
            continue
        sents = [old_text[a:b] for a, b in old_sents if a >= c.span[0] and b <= c.span[1]]
        if sents and all(s in new_set for s in sents):
            by_first.setdefault(sents[0], []).append((c.text, c.cited, sents))
    return by_first, True


def split_claims(llm, model: str, answer: str, previous: tuple[str, list[Claim]] | None = None) -> list[Claim]:
    """Atomic claims of `answer` (see module docstring); spans refer to clean_answer(answer). `previous` = (answer,
    claims) of an earlier split of a revised version: claims of unchanged sentences are kept and only changed sentences
    are split again, with the whole block still shown as context. Raises ClaimSplitError if a block's output is
    unparseable twice."""
    text = clean_answer(answer)
    if not text:
        return []
    reuse, incremental = _reusable(previous, text)
    claims: list[Claim] = []
    blocks = _blocks(text, max(MIN_BLOCK_CHARS, min(MAX_BLOCK_CHARS, -(-len(text) // SPLIT_PARALLEL))))
    context = text if len(blocks) > 1 else None
    jobs = []           # per block: (off, sents, numbered, kept, changed)
    for off, block in blocks:
        sents = [(off + a, off + b) for a, b in segment(block)]
        if not sents:
            continue
        numbered = "\n".join(f"S{k + 1}: {text[a:b]}" for k, (a, b) in enumerate(sents))
        kept, covered = [], set()
        if incremental:
            for k, (a, b) in enumerate(sents):
                for ctext, cited, ss in reuse.get(text[a:b], []):
                    ks = list(range(k + 1, k + 1 + len(ss)))
                    if ks[-1] <= len(sents) and [text[x:y] for x, y in sents[k:k + len(ss)]] == ss:
                        kept.append((sents[k][0], ctext, cited, (sents[k][0], sents[ks[-1] - 1][1])))
                        covered.update(ks)
            changed = [k for k in range(1, len(sents) + 1) if k not in covered]
        else:
            changed = None
        jobs.append((off, sents, numbered, kept, changed))

    def _split(job):
        off, _sents, numbered, _kept, changed = job
        if changed is not None and not changed:
            return []
        for attempt in (0, 1):
            try:
                return _parse(_call(llm, model, numbered, reminder=attempt == 1, only=changed, context=context))
            except (ValueError, json.JSONDecodeError):
                continue
        raise ClaimSplitError(f"unparseable claim-split output for the block at offset {off}")

    if len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=SPLIT_PARALLEL) as pool:
            results = list(pool.map(_split, jobs))
    else:
        results = [_split(j) for j in jobs]

    for (off, sents, numbered, kept, changed), items in zip(jobs, results):
        block_claims = [(start, Claim("", ctext, tuple(cited), span)) for start, ctext, cited, span in kept]
        for it in items:
            try:
                ks = sorted({int(k) for k in (it.get("s") or []) if 1 <= int(k) <= len(sents)})
            except (TypeError, ValueError):
                ks = []
            span = (sents[ks[0] - 1][0], sents[ks[-1] - 1][1]) if ks else (-1, -1)
            own = " ".join(text[sents[k - 1][0]:sents[k - 1][1]] for k in ks)
            allowed = set(markers(own))
            try:
                given = tuple(int(n) for n in (it.get("cited") or []))
            except (TypeError, ValueError):
                given = ()
            cited = tuple(dict.fromkeys(n for n in given if n in allowed)) or markers(own)
            if changed and ks and not set(ks) & set(changed):      # the model split a context-only sentence
                continue
            if ks and all(_heading(text[sents[k - 1][0]:sents[k - 1][1]]) for k in ks):
                continue                                            # a heading is not a statement
            block_claims.append((span[0] if span[0] >= 0 else 10 ** 9,
                                 Claim("", strip_markers(str(it["text"])), cited, span)))
        if incremental:                                             # kept + re-split claims: back into answer order
            block_claims = sorted(block_claims, key=lambda x: x[0])
        for _, c in block_claims:                                   # a first split keeps the model's order, as before
            claims.append(Claim(f"c{len(claims) + 1}", c.text, c.cited, c.span))
    return claims
