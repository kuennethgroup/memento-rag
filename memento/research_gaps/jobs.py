# memento/research_gaps/jobs.py: building the gap ledger, step 1
"""Per document: its gap-tagged passages → (1) extract: open research questions of the field vs this study's own scope
(model, no thinking); (2) ground: every question's claim is judged against its own passage by the RQ3 judge (only
"supported" enters the ledger; the rest is recorded with the reason); (3) merge: a question the ledger already holds
(same open question, judged by the model among the nearest by embedding) becomes a further mention, else a new gap. All
model and store access is passed in, so the logic is tested without a model. Runs at build time only (the background),
never while a reader waits."""
from __future__ import annotations

import json
import re

KINDS = ("limitation", "future_work", "untested_condition", "uncertainty")

EXTRACT_SYSTEM = """\
You read passages from ONE scientific document. Each passage was tagged at upload as possibly naming a research gap \
(limitation, future work, untested condition or uncertainty), with a one-line note. Decide for each passage:
- "field": it names an OPEN QUESTION FOR THE RESEARCH FIELD that another study could take up: something not yet \
known, not yet tested, not yet possible or not yet resolved ("X has not been tested", "future work should examine \
X", "it remains unclear whether X", "X is still limited by Y").
- "scope": it only describes THIS study's own choices or boundaries (what the authors did not do in this paper, \
the sample, the chosen method, simplifying assumptions), without saying the question is open for the field \
("we only tested 10 nm films" is scope; "thicker films have not been studied" is field). "X is beyond the scope of \
this study" followed by what future studies could do with it ("the data set can be used in future studies to …, but \
this is beyond the scope of this study") is field (future_work): the work is left open for others.
- "solved": it names a limitation of EARLIER work that THIS document itself sets out to solve or solves ("earlier \
models cannot do X; we show how X can be done", "here we propose a method that overcomes X", "in this work we \
extend the model to X"). It is not open: this document addresses it.
- "none": it names no gap at all (it reports a result or a method). A statement of capability or expectation \
("the models are scalable to more properties when data become available", "charge compensation is expected") is \
"none" unless the passage says it is untested or unresolved.
For "field" write:
- "question": the open question as ONE natural English question about the science (how / whether / why / what \
happens / which), not about a study ("What is the systematic study of X?" is wrong; "How do dopants stabilize the \
ferroelectric phase of X?" is right), in general form (no "this paper", no "the authors"), keeping the specific \
material, property, condition and range the passage names, and saying specifically what is unknown (not vague: \
"What are the comprehensive material properties of X?" is wrong; "How does Si content affect the coercive field of \
X?" is right, if the passage says so).
- "claim": ONE English sentence saying what the passage states is open or limited, close to its wording, with \
the same material, property, condition and range as the question.
- "kind": limitation, future_work, untested_condition or uncertainty.
For "scope" and "solved" write "claim" (what the passage says) and "reason" (why it is only the study's scope, or \
where this document addresses it).
Use only the passages, never your own knowledge. Each question asks one thing: a passage that names two different \
open points gives two items with the same "i" (never "…, and how …?" in one question). Answer every passage.
Output ONLY a JSON list, e.g. [{"i": 1, "type": "field", "question": "…?", "claim": "….", "kind": "limitation"}, \
{"i": 2, "type": "scope", "claim": "….", "reason": "…"}, {"i": 3, "type": "solved", "claim": "….", "reason": "…"}, \
{"i": 4, "type": "none"}]"""

RELATION_SYSTEM = """\
Compare a NEW research question with each EXISTING one and decide:
- SAME: a study that answers one would necessarily answer the other (the same input, method, material or material \
class, property and condition; only the wording differs).
- RELATED: a different question about the same material system AND the same property or phenomenon (e.g. two \
questions about what stabilizes the ferroelectric phase of hafnia, or about the cycling stability of hafnia devices); \
an answer to one would inform the other but not answer it. The same broad field or material class with a different \
property or aspect is NOT related (e.g. "thermal stability of polymer dielectrics" and "green solvents for making \
polymer dielectrics").
- otherwise: different.
A different input, method, material, property or condition, or a clearly broader or narrower question, is NOT the \
same (it may be related). Example: "How can polymer NAMES be converted to SMILES?" and "How can IMAGES of polymer \
structures be converted to SMILES?" are related, not the same (different input). When in doubt between same and \
related, choose related; between related and different, choose different.
Output ONLY JSON: {"same": <number of the existing question that is the same, or 0>, "related": [<numbers of the \
related ones>]}"""


def _label(p: dict) -> str:
    sig = (p.get("gap_signal") or "").strip()
    return f"page {int(p.get('page') or 0) + 1} · tag: {p.get('gap_type') or '?'}" + (f" · {sig}" if sig else "")


def extract_messages(title: str, passages: list[dict]) -> list[dict]:
    listing = "\n\n".join(f"[{k + 1}] {_label(p)}\n{' '.join(str(p.get('text') or '').split())}"
                          for k, p in enumerate(passages))
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Document: {title}\n\nPassages:\n{listing}"}]


def _json(raw: str):
    from memento.agentic_validation.jsonfix import loads_lenient
    raw = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    return loads_lenient(raw)


def parse_extract(raw: str, passages: list[dict]) -> list[dict]:
    """Validated items with 0-based passage index `i`; invalid entries are skipped. Raises ValueError when the output
    is no JSON list at all."""
    try:
        data = _json(raw[raw.find("["):raw.rfind("]") + 1] if "[" in (raw or "") else raw)
    except Exception as e:
        raise ValueError(f"unreadable extraction output: {e}") from e
    if not isinstance(data, list):
        raise ValueError("extraction output is not a list")
    out = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            i = int(d.get("i")) - 1
        except (TypeError, ValueError):
            continue
        typ = str(d.get("type") or "").strip().lower()
        if not 0 <= i < len(passages) or typ not in ("field", "scope", "solved", "none"):
            continue
        if typ == "field":
            q, c = str(d.get("question") or "").strip(), str(d.get("claim") or "").strip()
            if not q or not c:
                continue
            kind = str(d.get("kind") or "").strip().lower()
            if kind not in KINDS:
                tag = str(passages[i].get("gap_type") or "").strip().lower()
                kind = tag if tag in KINDS else "uncertainty"
            out.append({"i": i, "type": "field", "question": q, "claim": c, "kind": kind})
        elif typ in ("scope", "solved"):
            out.append({"i": i, "type": typ, "claim": str(d.get("claim") or "").strip(),
                        "reason": str(d.get("reason") or "").strip()})
        else:
            out.append({"i": i, "type": "none"})
    return out


def extract_document(llm, title: str, passages: list[dict], batch: int = 25) -> list[dict]:
    """All passages of one document in batches; passages the model left out are asked again once. `llm(messages)`
    returns the model's text. Indices in the result refer to `passages`."""
    out = []
    for start in range(0, len(passages), batch):
        part = passages[start:start + batch]
        items = parse_extract(llm(extract_messages(title, part)), part)
        missing = [k for k in range(len(part)) if k not in {x["i"] for x in items}]
        if missing:
            sub = [part[k] for k in missing]
            try:
                again = parse_extract(llm(extract_messages(title, sub)), sub)
            except ValueError:
                again = []
            items += [{**x, "i": missing[x["i"]]} for x in again]
        out += [{**x, "i": x["i"] + start} for x in items]
    return out


def left_out(items: list[dict]) -> list[tuple[int, str, str]]:
    """Statements that are not open questions, with the reason recorded for review: (passage index, claim, reason)."""
    label = {"scope": "study scope only", "solved": "solved in this document"}
    return [(x["i"], x["claim"], f"{label[x['type']]}: {x['reason']}") for x in items if x["type"] in label]


def ground(verifier, items: list[dict], passages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Judge each field question's claim against its own passage; (kept, dropped). Only "supported" is kept; the
    judge's "unjudged" (model or parse failure) is never treated as supported."""
    fields = [x for x in items if x["type"] == "field"]
    pairs = [(f"g{k}", x["claim"], str(passages[x["i"]].get("text") or "")) for k, x in enumerate(fields)]
    verdicts = verifier.verify(pairs) if pairs else {}
    kept, dropped = [], []
    for k, x in enumerate(fields):
        v = verdicts.get(f"g{k}")
        label = getattr(v, "label", "unjudged") if v is not None else "unjudged"
        reason = getattr(v, "rationale", "") if v is not None else "no verdict"
        if label == "supported":
            kept.append({**x, "judge": label, "judge_reason": reason or ""})
        else:
            dropped.append({**x, "reason": f"{label}: {reason or ''}".strip()})
    return kept, dropped


def relation_messages(question: str, near: list[tuple[int, str]]) -> list[dict]:
    listing = "\n".join(f"{k + 1}. {s}" for k, (_, s) in enumerate(near))
    return [{"role": "system", "content": RELATION_SYSTEM},
            {"role": "user", "content": f"NEW: {question}\n\nEXISTING:\n{listing}"}]


def parse_relation(raw: str, ids: list[int]) -> tuple[int | None, list[int]]:
    """(the gap id the new question is the same as, or None; the related gap ids). Anything unreadable or out of
    range is dropped: never a wrong merge or link (a missed merge only splits a count)."""
    try:
        d = _json(raw[raw.find("{"):raw.rfind("}") + 1] if "{" in (raw or "") else raw)
    except Exception:
        return None, []
    try:
        n = int(d.get("same") or 0)
    except (TypeError, ValueError):
        n = 0
    same = ids[n - 1] if 1 <= n <= len(ids) else None
    rel = []
    for r in d.get("related") or []:
        try:
            k = int(r)
        except (TypeError, ValueError):
            continue
        if 1 <= k <= len(ids) and ids[k - 1] != same and ids[k - 1] not in rel:
            rel.append(ids[k - 1])
    return same, rel


def merge(item: dict, *, embed, search, relate, add_gap, add_mention, upsert, add_relation=None, floor: float = 0.6,
          k: int = 5) -> tuple[int, bool]:
    """Add one checked question to the ledger: (gap id, created). The nearest existing questions by embedding (score
    ≥ `floor`, at most `k`) go to the relation judge: the SAME one gets a further mention, otherwise a new gap with its
    vector; RELATED ones are linked to it (a theme, never claimed identical)."""
    vec = embed(item["question"])
    near = [(gid, st) for gid, st, sc in search(vec, k) if sc >= floor]
    gid, rel = relate(item["question"], near) if near else (None, [])
    created = gid is None
    if created:
        gid = add_gap(item["question"], item["kind"])
        upsert(gid, vec, item["question"])
    add_mention(gid, item)
    if add_relation is not None:
        for r in rel:
            if r != gid:
                add_relation(gid, r)
    return gid, created


# ── step 2: the library check ──
# Per gap: passages of other documents (normal search + the best passage of every document), each labelled against the
# question by the model; the status follows fixed rules from the labels (traceable to labelled passages), never the
# model's overall impression. Open gaps keep their closest near-misses, so "open" can be checked by a reader.
RELATIONS = ("addresses", "partly", "contradicts", "not_concern")

CHECK_SYSTEM = """\
You check whether OTHER documents of a research library answer an open research question. The question was raised by \
one document; its own statement is given as the claim. For EACH numbered passage decide:
- "addresses": the passage answers the question as stated: it reports the result (a measurement, calculation or \
clear finding) for the same material, property and condition. Merely mentioning the topic, stating the same gap or \
calling for the same work is NOT addressing it.
- "partly": it gives part of the answer (a finding): the same question answered for another material, condition or \
range, or the answer only qualitatively or indirectly. A passage that only mentions or discusses the topic, restates \
the problem, or says it is difficult, without giving any part of the answer, is "not_concern".
- "contradicts": it reports findings that conflict with the claim (an opposite effect, incompatible values or \
conclusions on the same material and condition).
- "not_concern": anything else.
Judge only from the passage, never from your own knowledge. Be strict: between addresses and partly choose partly; \
between partly and not_concern choose not_concern.
Output ONLY a JSON list covering every passage: [{"i": 1, "relation": "partly", "reason": "one short sentence"}]"""


def check_candidates(hits: list[dict], heads: list[dict], *, stating: set, cap: int = 15) -> list[dict]:
    """The passages to judge: both searches merged (best score per passage), the stating documents left out, best first."""
    best: dict[str, dict] = {}
    for h in list(hits) + list(heads):
        if h["source"] in stating:
            continue
        pid = str(h["point_id"])
        if pid not in best or (h.get("rerank_score") or 0) > (best[pid].get("rerank_score") or 0):
            best[pid] = h
    return sorted(best.values(), key=lambda h: -(h.get("rerank_score") or 0))[:cap]


def check_messages(question: str, claim: str, cands: list[dict]) -> list[dict]:
    listing = "\n\n".join(f"[{k + 1}] {h['source']}, page {int(h.get('page') or 0) + 1}\n"
                          f"{' '.join(str(h.get('text') or '').split())[:1500]}" for k, h in enumerate(cands))
    return [{"role": "system", "content": CHECK_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nClaim of the document that raises it: {claim}\n\n"
                                        f"Passages:\n{listing}"}]


UNKNOWN = "unknown"          # a passage the model did not judge (missing, invalid index or relation)


def parse_check(raw: str, n: int) -> list[dict]:
    """One {"relation", "reason"} per passage; invalid or missing entries become "unknown" (never "not_concern", which
    would turn an unfinished check into a false "open"). Raises ValueError when the output is unreadable: then no status
    is set at all."""
    try:
        data = _json(raw[raw.find("["):raw.rfind("]") + 1] if "[" in (raw or "") else raw)
    except Exception as e:
        raise ValueError(f"unreadable check output: {e}") from e
    if not isinstance(data, list):
        raise ValueError("check output is not a list")
    out = [{"relation": UNKNOWN, "reason": ""} for _ in range(n)]
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            i = int(d.get("i")) - 1
        except (TypeError, ValueError):
            continue
        rel = str(d.get("relation") or "").strip().lower()
        if 0 <= i < n and rel in RELATIONS:
            out[i] = {"relation": rel, "reason": str(d.get("reason") or "").strip()}
    return out


def complete(labels: list[dict]) -> bool:
    """Every passage judged: only then may a status be set from the labels."""
    return all(lb.get("relation") != UNKNOWN for lb in labels)


def judge_labels(llm, build, cands: list) -> list[dict]:
    """The check of `cands` (one {"relation", "reason"} each): one call (`llm(messages) -> text`, the messages from
    `build(passages)`); unreadable output gets one more call asking for the list only; passages left unknown are asked
    once more on their own and their answers merged in; still unknown → stays unknown (the caller then sets no status)."""
    def ask(sub):
        msgs = build(sub)
        try:
            return parse_check(llm(msgs), len(sub))
        except ValueError:
            return parse_check(llm(msgs + [{"role": "user", "content": "Output ONLY the JSON list."}]), len(sub))
    labels = ask(cands)
    missing = [i for i, lb in enumerate(labels) if lb["relation"] == UNKNOWN]
    if missing and len(missing) <= len(cands):
        try:
            again = ask([cands[i] for i in missing])
        except ValueError:
            again = []
        for i, lb in zip(missing, again):
            if lb["relation"] != UNKNOWN:
                labels[i] = lb
    return labels


def library_status(labels: list[dict]) -> str:
    rels = {lb["relation"] for lb in labels}
    if "contradicts" in rels:
        return "contested"
    if "addresses" in rels:
        return "addressed"
    if "partly" in rels:
        return "partly"
    return "open"


def near_misses(cands: list[dict], labels: list[dict], n: int = 3) -> list[tuple[dict, dict]]:
    """The closest passages judged not to concern the question, one per document, best first."""
    out, seen = [], set()
    for h, lb in zip(cands, labels):
        if lb["relation"] == "not_concern" and h["source"] not in seen:
            out.append((h, lb))
            seen.add(h["source"])
        if len(out) >= n:
            break
    return out


# ── step 3: the literature check ──
# Only the search phrase leaves the server. The papers' abstracts are judged with the library check's relations and
# rules; an abstract shows only part of a paper, so the strongest status is "possibly addressed".

QUERY_SYSTEM = """\
Turn the research question into a search query for a scholarly search engine (Semantic Scholar, OpenAlex): 4 to 10 \
key terms naming the specific materials, properties, methods and conditions of the question, in English. No question \
words, no quotes, no AND/OR, no explanation. Output only the query on one line."""

WEB_CHECK_SYSTEM = """\
You check whether PUBLISHED PAPERS answer an open research question, judging each paper only from its title and \
abstract. The question was raised by a document of a research library; its own statement is given as the claim. For \
EACH numbered paper decide:
- "addresses": the abstract reports a result (a measurement, calculation or clear finding) that answers the question \
as stated, for the same material, property and condition. Merely studying the topic, stating the same gap or calling \
for the same work is NOT addressing it.
- "partly": the abstract gives part of the answer (a finding): the same question answered for another material, \
condition or range, or only qualitatively or indirectly. A paper that only works on the topic, without a finding \
that answers part of the question, is "not_concern".
- "contradicts": the abstract reports findings that conflict with the claim (an opposite effect, incompatible values \
or conclusions on the same material and condition).
- "not_concern": anything else, and every paper without an abstract unless its title alone states the answer.
Judge only from the title and abstract, never from your own knowledge. Be strict: between addresses and partly choose \
partly; between partly and not_concern choose not_concern.
Output ONLY a JSON list covering every paper: [{"i": 1, "relation": "partly", "reason": "one short sentence"}]"""

WEB_CHECK_EXCERPTS_RULE = """
Some papers also show excerpts from their full text. For those, judge from the title, the abstract and the excerpts, \
and read "abstract" above as all three. The excerpts are only parts of the paper: something missing from them is no \
evidence either way."""


def query_messages(statement: str) -> list[dict]:
    return [{"role": "system", "content": QUERY_SYSTEM}, {"role": "user", "content": statement}]


def parse_query(raw: str, max_words: int = 12) -> str:
    """The model's search phrase as sent: one line, without a "Query:" label, quotes or boolean words, at most
    `max_words` words. Raises ValueError when empty."""
    line = next((x.strip() for x in (raw or "").splitlines() if x.strip()), "")
    line = re.sub(r"^(search\s+)?query\s*:\s*", "", line, flags=re.I)
    line = line.replace('"', " ").replace("“", " ").replace("”", " ")
    words = [w for w in line.split() if w not in ("AND", "OR", "NOT")]
    if not words:
        raise ValueError("empty search phrase")
    return " ".join(words[:max_words])


def web_check_messages(question: str, claim: str, works: list, abstract_chars: int = 1500) -> list[dict]:
    def entry(k, w):
        year = w.year if w.year else "year unknown"
        body = " ".join((w.abstract or "").split())[:abstract_chars] or \
            "(no abstract: judge from the title only, normally not_concern)"
        excerpts = getattr(w, "excerpts", None)
        extra = f"\n{ft.MARK}\n" + "\n\n".join(excerpts) if excerpts else ""
        return f"[{k + 1}] {w.title} ({year})\n{body}{extra}"
    from memento.research_gaps import fulltext as ft
    listing = "\n\n".join(entry(k, w) for k, w in enumerate(works))
    rule = WEB_CHECK_EXCERPTS_RULE if any(getattr(w, "excerpts", None) for w in works) else ""
    return [{"role": "system", "content": WEB_CHECK_SYSTEM + rule},
            {"role": "user", "content": f"Question: {question}\nClaim of the document that raises it: {claim}\n\n"
                                        f"Papers:\n{listing}"}]


def shorter_query(query: str, words: int = 5) -> str | None:
    """A second, shorter phrase when the first found few papers (search engines match every term): its first words,
    or None when it is already that short."""
    parts = query.split()
    return " ".join(parts[:words]) if len(parts) > words else None


def web_status(labels: list[dict]) -> str:
    """The library check's rules on abstracts: contested > possibly_addressed > partly > nothing_found; no paper read
    at all is "no_results" (never reported as "nothing found in the literature")."""
    if not labels:
        return "no_results"
    return {"contested": "contested", "addressed": "possibly_addressed", "partly": "partly",
            "open": "nothing_found"}[library_status(labels)]


def web_near_misses(works: list, labels: list[dict], n: int = 3) -> list[tuple]:
    """The best-ranked papers judged "not concerned": kept for "nothing found", so the reader sees what was read."""
    return [(w, lb) for w, lb in zip(works, labels) if lb["relation"] == "not_concern"][:n]


# ── step 5: a new document checked against the existing questions ──
# Only passages of the new document that the reranker scores at least NEW_DOC_FLOOR reach the model (calibrated on the
# library check: answering passages score above it, near-misses mostly below).
NEW_DOC_FLOOR = 0.04


def worth_checking(hits: list[dict], floor: float = NEW_DOC_FLOOR) -> list[dict]:
    return [h for h in hits if (h.get("rerank_score") or 0) >= floor]


def combined_status(old_relations: list[str], new_labels: list[dict]) -> str:
    """A question's library status from its stored evidence relations plus the new document's labels (same rules)."""
    return library_status([{"relation": r} for r in old_relations if r in RELATIONS] + list(new_labels))
