"""Per-document fact sheets (the RQ2 fact-sheets arm, "structure then query"). One structured record per document,
extracted by the LLM from the document's full text (read in parts; the parts' lists are merged by de-duplicated union).
The JSON file next to the parsed output (`<doc>_fact_sheet.json`) is the extraction artifact; Qdrant holds the index
that queries read: one payload-only point per document in `<corpus prefix>-sheets`, with registry-canonical ids (keyword
index) next to the raw names (full-text index), so filters and counts stay fast at scale.

Fields are the same for every domain: the five graph entity types plus property values, document type, year and a
one-line topic. Names are written in English, so English questions match German documents."""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from typing import Callable, Optional

LIST_FIELDS = ("materials", "processes", "characterization", "conditions")
DOC_TYPES = ("journal article", "dissertation", "thesis chapter", "presentation", "report", "datasheet",
             "patent", "other")
PART_CHARS = 48_000

SYSTEM = ("You build a structured fact sheet of a scientific or technical document for a search index. "
          "Extract only what the text explicitly states. Output one JSON object and nothing else.")

FIELDS_TEXT = f"""Return a JSON object with these fields:
  "topic": one sentence (at most 30 words) saying what this text is about
  "doc_type": one of {", ".join(DOC_TYPES)}
  "year": the publication year if the text states it, else null
  "materials": specific materials, compounds, alloys, polymers or composites studied or used
  "processes": processing, synthesis or manufacturing processes applied
  "properties": up to 30 reported property values, each {{"name": str, "value": str, "unit": str, "material": str}}
  "characterization": characterization or test techniques used
  "conditions": processing or test conditions with their values (e.g. "sintered at 1450 °C for 2 h")
Write every name in English (translate German terms). List specific scientific entities only: no person
names, organisations, funding or grant numbers, software or instrument brand names, or generic words
(e.g. "sample", "results"). Empty lists are fine."""


def split_parts(texts: list[str], part_chars: int = PART_CHARS) -> list[str]:
    """Consecutive chunk texts grouped into parts of at most ~part_chars (a longer single chunk is cut)."""
    parts, cur, n = [], [], 0
    for t in texts:
        t = (t or "").strip()
        while len(t) > part_chars:
            if cur:
                parts.append("\n\n".join(cur)); cur, n = [], 0
            parts.append(t[:part_chars]); t = t[part_chars:]
        if not t:
            continue
        if cur and n + len(t) > part_chars:
            parts.append("\n\n".join(cur)); cur, n = [], 0
        cur.append(t); n += len(t) + 2
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def user_prompt(doc_id: str, part: str, i: int, n: int) -> str:
    return f"<document id=\"{doc_id}\" part=\"{i + 1} of {n}\">\n{part}\n</document>\n\n{FIELDS_TEXT}"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_part(raw: str) -> Optional[dict]:
    """The part's JSON object, or None when the response is not usable."""
    s = _THINK.sub("", raw or "").strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1)
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def _names(v) -> list[str]:
    out = []
    for x in v if isinstance(v, list) else []:
        s = " ".join(str(x).split()) if isinstance(x, (str, int, float)) else ""
        if s:
            out.append(s)
    return out


def merge_parts(parts: list[dict]) -> dict:
    """Union of the parts' lists (case-insensitive, first spelling kept); first topic; most common
    doc type and year."""
    sheet = {"topic": "", "part_topics": [], "doc_type": None, "year": None, "properties": []}
    for f in LIST_FIELDS:
        sheet[f] = []
    seen = {f: set() for f in LIST_FIELDS}
    seen_p, types, years = set(), Counter(), Counter()
    for p in parts:
        t = " ".join(str(p.get("topic") or "").split())
        if t:
            sheet["part_topics"].append(t)
        if p.get("doc_type") in DOC_TYPES:
            types[p["doc_type"]] += 1
        y = p.get("year")
        if isinstance(y, (int, str)) and re.fullmatch(r"(19|20)\d\d", str(y).strip()):
            years[int(str(y).strip())] += 1
        for f in LIST_FIELDS:
            for s in _names(p.get(f)):
                if s.lower() not in seen[f]:
                    seen[f].add(s.lower()); sheet[f].append(s)
        for pr in p.get("properties") or []:
            if not isinstance(pr, dict) or not str(pr.get("name") or "").strip():
                continue
            rec = {k: " ".join(str(pr.get(k) or "").split()) for k in ("name", "value", "unit", "material")}
            key = tuple(v.lower() for v in rec.values())
            if key not in seen_p:
                seen_p.add(key); sheet["properties"].append(rec)
    sheet["topic"] = sheet["part_topics"][0] if sheet["part_topics"] else ""
    sheet["doc_type"] = types.most_common(1)[0][0] if types else None
    sheet["year"] = years.most_common(1)[0][0] if years else None
    return sheet


def _halve(part: str) -> tuple[str, str]:
    """Split a part near its middle, at a paragraph break when there is one."""
    mid = len(part) // 2
    cands = [c for c in (part.rfind("\n\n", 0, mid), part.find("\n\n", mid)) if c > 0]
    cut = min(cands, key=lambda c: abs(c - mid)) if cands else mid
    if not len(part) // 4 <= cut <= 3 * len(part) // 4:      # no break near the middle: cut mid-text
        cut = mid
    return part[:cut].strip(), part[cut:].strip()


def _too_long(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "maximum context length" in msg or "context length" in msg and "token" in msg


def extract_sheet(llm, model: str, doc_id: str, texts: list[str], cache=None,
                  part_chars: int = PART_CHARS, max_tokens: int = 4096) -> dict:
    """One call per part (cached by prompt, cache saved after every part), merged. A part the server
    rejects as longer than its context (dense tables/numbers tokenize at < 2 chars per token) is split
    in half and retried; normal parts keep their exact prompt. A part whose response does not parse
    after one retry is counted in `n_failed_parts` and left out."""
    from memento.graph.build.extraction_cache import RawResponseCache
    parts = split_parts(texts, part_chars)
    n = len(parts)
    queue = [(f"{i + 1} of {n}", part) for i, part in enumerate(parts)]
    parsed, failed, n_split = [], 0, 0
    usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
    while queue:
        label, part = queue.pop(0)
        up = f"<document id=\"{doc_id}\" part=\"{label}\">\n{part}\n</document>\n\n{FIELDS_TEXT}"
        key = RawResponseCache.key(model, SYSTEM, up)
        d, split = None, False
        for attempt in range(2):
            raw = cache.get(key) if (cache and attempt == 0) else None
            if raw is not None:
                usage["cache_hits"] += 1
            else:
                try:
                    r = llm.chat.completions.create(
                        model=model, temperature=0, max_tokens=max_tokens,
                        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": up}],
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                except Exception as exc:
                    if _too_long(exc) and len(part) > 2000:
                        a, b = _halve(part)
                        queue[:0] = [(f"{label} (a)", a), (f"{label} (b)", b)]
                        n_split += 1
                        split = True
                        break
                    raise
                raw = r.choices[0].message.content or ""
                usage["calls"] += 1
                u = getattr(r, "usage", None)
                usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            d = parse_part(raw)
            if d is not None:
                if cache:
                    cache.put(key, raw)
                    cache.save()
                break
        if split:
            continue
        if d is None:
            failed += 1
        else:
            parsed.append(d)
    sheet = merge_parts(parsed)
    sheet.update({"source": doc_id, "n_parts": n, "n_split_parts": n_split, "n_failed_parts": failed,
                  "usage": usage})
    return sheet


# ── Qdrant index ─────────────────────────────────────────────────────────────
CANON_FIELDS = ("materials", "processes", "characterization", "conditions", "properties")


def sheets_collection(chunk_collection: str) -> str:
    """memento-ceramics-rag -> memento-ceramics-sheets"""
    base = chunk_collection[:-4] if chunk_collection.endswith("-rag") else chunk_collection
    return f"{base}-sheets"


def payload(sheet: dict, canonicalize: Callable[[str], Optional[object]]) -> dict:
    """Index payload: raw names (full-text) and registry-canonical ids or lower-cased names (keyword)."""
    p = {"source": sheet["source"], "topic": sheet.get("topic", ""), "doc_type": sheet.get("doc_type") or "",
         "year": sheet.get("year"), "properties": sheet.get("properties", [])}
    for f in CANON_FIELDS:
        names = ([pr["name"] for pr in sheet.get("properties", [])] if f == "properties"
                 else sheet.get(f, []))
        p[f"{f}_raw"] = " | ".join(names)
        canon = []
        for n in names:
            e = canonicalize(n)
            canon.append(e.canonical_id if e is not None else n.lower())
        p[f"{f}_canon"] = sorted(set(canon))
    return p


def point_id(source: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fact-sheet:{source}"))


def index_sheets(client, collection: str, payloads: list[dict]) -> int:
    """(Re)create the payload-only sheets collection and its payload indexes; upsert one point per document."""
    from qdrant_client.models import PayloadSchemaType, PointStruct, TextIndexParams, TokenizerType
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config={})
    client.create_payload_index(collection, "source", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "doc_type", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "year", PayloadSchemaType.INTEGER)
    for f in CANON_FIELDS:
        client.create_payload_index(collection, f"{f}_canon", PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection, f"{f}_raw", TextIndexParams(
            type="text", tokenizer=TokenizerType.WORD, lowercase=True, min_token_len=1))
    pts = [PointStruct(id=point_id(p["source"]), vector={}, payload=p) for p in payloads]
    for i in range(0, len(pts), 256):
        client.upsert(collection, points=pts[i:i + 256])
    return client.count(collection, exact=True).count


def match_filter(variants: list[str], canonicalize: Callable[[str], Optional[object]]):
    """A document matches a term when any sheet field holds one of its variants: canonical id / exact
    lower-cased name (keyword) or the variant's words in the raw names (full text)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText
    keys = set()
    for v in variants:
        e = canonicalize(v)
        keys.add(e.canonical_id if e is not None else v.lower())
    should = [FieldCondition(key=f"{f}_canon", match=MatchAny(any=sorted(keys))) for f in CANON_FIELDS]
    should += [FieldCondition(key=f"{f}_raw", match=MatchText(text=v)) for f in CANON_FIELDS for v in variants
               if v.strip()]
    return Filter(should=should)


def match_counts(client, collection: str, variants: list[str], canonicalize) -> dict[str, int]:
    """{source: number of sheet names that match a variant} over the documents the filter selects; the
    count ranks documents the way LX ranks by matching chunks."""
    keys = set()
    for v in variants:
        e = canonicalize(v)
        keys.add(e.canonical_id if e is not None else v.lower())
    words = [set(re.findall(r"\w+", v.lower())) for v in variants if v.strip()]
    out, off = {}, None
    flt = match_filter(variants, canonicalize)
    while True:
        pts, off = client.scroll(collection, scroll_filter=flt, limit=1000, offset=off, with_payload=True,
                                 with_vectors=False)
        for p in pts:
            n = 0
            for f in CANON_FIELDS:
                for name in (p.payload.get(f"{f}_raw") or "").split(" | "):
                    if not name:
                        continue
                    e = canonicalize(name)
                    key = e.canonical_id if e is not None else name.lower()
                    toks = set(re.findall(r"\w+", name.lower()))
                    if key in keys or any(w and w <= toks for w in words):
                        n += 1
            if n:        # the full-text filter matches words anywhere in a field; keep only real name matches
                out[p.payload["source"]] = n
        if off is None:
            return out


def matching_sources(client, collection: str, variants: list[str], canonicalize) -> set[str]:
    return set(match_counts(client, collection, variants, canonicalize))
