"""List and count questions: the documents scanned for the list's kind and topic, the instruction to list them (with
the ones beyond the first), and a list with nothing relevant.
"""
import asyncio
import re
import time

from memento.query import library_state, presentation, request
from memento.query.flows import library_listing


# list and count questions: the answer sees the passages found, not every document, so it must not claim completeness.
# The closing line states what was found and that a missing document can be checked by its name, which then searches
# inside it.
_LIST_TEMPLATE = """
This question asks which documents, or how many, have something. You see only the passages the search found, not \
every document in the library: name the documents these passages show, each document once (several passages from \
one document are one document), never claim completeness ("all", "the only", "the library has N papers on …"), \
when you give a number, write the list first and the number after it ({found}), counting the documents you listed, \
and end with this short sentence: "{close}\"{language}"""


# more documents qualified than the scan may add (LIST_SCAN_MAX): the answer says it shows the best ones
_LIST_MORE_TEMPLATE = """ More documents matched than fit in one answer: instead of the count above, write "{best}" \
(with the kind asked for), and end with "{more}" instead."""


# the sentences in the question's language only (with both, answers switched language)
_LIST_PHRASES = {
    "en": dict(found='"I found N documents on this topic", with the kind of document asked for, e.g. "I found 9 '
                     'dissertations on this topic"; for one: "I found 1 document on this topic"',
               close="If one you expected is missing, ask about it by name and I'll check it.",
               best="These are the N best-matching documents",
               more="The library has more on this topic; a narrower question or a document's name finds them.",
               # with many German documents in the context, an English answer may come back in German (a German question
               # has DE_ANSWER_ADDENDUM)
               language=" The question is in English: write the answer in English, also when the documents are in "
                        "another language. Keep document titles and quotations as written."),
    "de": dict(found='"Ich habe N Dokumente zu diesem Thema gefunden", mit der gefragten Art von Dokument, z. B. "Ich '
                     'habe 9 Dissertationen zu diesem Thema gefunden"; für eines: "Ich habe 1 Dokument zu diesem Thema '
                     'gefunden"',
               close="Wenn ein von Ihnen erwartetes Dokument fehlt, fragen Sie bitte nach dem Namen, und ich prüfe es.",
               best="Das sind die N am besten passenden Dokumente",
               more="Die Bibliothek enthält mehr zu diesem Thema; eine engere Frage oder der Name eines Dokuments "
                    "findet sie.", language="")}


LIST_ADDENDUM = _LIST_TEMPLATE.format(**_LIST_PHRASES["en"])


LIST_MORE_ADDENDUM = _LIST_MORE_TEMPLATE.format(**_LIST_PHRASES["en"])


def nothing_relevant(best: list, query_intent: str, *, rerank_enabled: bool, low: float, dc: bool, paper,
                      named_docs) -> bool:
    """Even the best passage scores below `low` (the "nothing found" mark): a plain reply instead of the model, for
    questions over the library; never in Document Chat or for a named document; no scores, no judgement. Not for list
    questions, whose real scores sit near the mark (LIST_ADDENDUM keeps their wording honest)."""
    return (rerank_enabled and bool(best) and max(best) < low
            and query_intent in ("ranked_retrieval", "multi_hop")
            and not dc and not (paper or named_docs))


def list_scan_applies(query_intent: str, *, enabled: bool, dc: bool, paper, named_docs, image_intent: bool) -> bool:
    """A list or count question over the whole library (no document named, not Document Chat, not a figure request):
    its context gets the best passage of every matching document (list_extra)."""
    return enabled and query_intent == "set_query" and not dc and not paper and not named_docs and not image_intent


class ListExtra(list):
    """The passages a list scan adds; `capped`: more documents qualified than `cap` (the answer then says it shows the
    best-matching ones)."""

    def __init__(self, items=(), capped: bool = False):
        super().__init__(items)
        self.capped = capped


def list_extra(hits: list, heads: list, ratio: float, cap: int) -> ListExtra:
    """The passages of documents not yet among `hits` that score at least `ratio` × the best passage (relative to the
    best, which works across wordings), the documents best first, at most `cap` documents, each with every passage that
    passes (LIST_SCAN_PER_DOC per document); icons never; nothing when the reranker failed (no scores to judge by)."""
    if not heads or any(h.get("rerank_fallback") for h in heads):
        return ListExtra()
    top = max((h.get("rerank_score") or 0.0) for h in heads)
    have = {h["source"] for h in hits}
    docs: dict = {}                                  # source → its passing passages, best first (insertion order)
    for h in sorted(heads, key=lambda h: -(h.get("rerank_score") or 0.0)):
        if (h.get("rerank_score") or 0.0) < ratio * top:
            break
        if h["source"] in have or presentation.icon_image(h):
            continue
        docs.setdefault(h["source"], []).append(h)
    kept = list(docs.values())[:cap]
    return ListExtra([h for passages in kept for h in passages], capped=len(docs) > cap)


# the kind of document a list question asks for (router "doc_kind") → the library title's opening word(s)
_KIND_PREFIXES = {"dissertation": ("Dissertation",), "master's thesis": ("Master's thesis", "Masterarbeit"),
                  "bachelor's thesis": ("Bachelor's thesis", "Bachelorarbeit"), "talk": ("Talk", "Vortrag"),
                  "chapter": ("Chapter", "Kapitel")}


_KIND_PREFIXES["thesis"] = (_KIND_PREFIXES["dissertation"] + _KIND_PREFIXES["master's thesis"]
                            + _KIND_PREFIXES["bachelor's thesis"])


def scan_kind_sources(kind: str, collection: str, db_engine) -> list | None:
    """This library's documents of `kind` (_list_kind_sources), by their stored titles, each source as stored in NFC
    and NFD (both occur); None without a kind, without such titles, or on any failure (then no restriction)."""
    if not _KIND_PREFIXES.get(kind or ""):
        return None
    try:
        from memento.library import documents as ldb
        rows = ldb.list_documents(db_engine(), collection)
    except Exception:
        return None
    import unicodedata
    found = _list_kind_sources(kind, [(r["source"], r.get("title")) for r in rows])
    return sorted({f for s in found for f in (unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s))}) \
        if found else None


def list_folds(query_intent: str, *, whole_doc: bool, image_intent: bool) -> bool:
    """Does this answer's long list fold after LIST_FIRST entries? A list or count question over the library or the
    attached files, not a whole-document read and not a figure request (its figures go between the paragraphs)."""
    return query_intent == "set_query" and not whole_doc and not image_intent


def _list_kind_sources(kind: str, docs) -> list | None:
    """The documents of `kind` by their library title ("Dissertation: …", "Talk (2014): …"), for the list scan; None (no
    restriction) without a kind or when no title in this library names one."""
    prefixes = _KIND_PREFIXES.get(kind or "")
    if not prefixes:
        return None
    rx = re.compile(r"^(?:%s)(?:\s+[\d.]+)?(?=\s*[:(]|\s*$)" % "|".join(re.escape(p) for p in prefixes), re.I)
    out = sorted(src for src, title in docs if title and rx.match(title.strip()))
    return out or None


def list_addendum(query_intent: str, whole_doc: bool, dc: bool = False, capped: bool = False) -> str:
    """LIST_ADDENDUM for a list or count question answered from search passages, with LIST_MORE_ADDENDUM when the scan
    had more documents than it may add; never in Document Chat, which reads only the attached files."""
    if query_intent != "set_query" or whole_doc or dc:
        return ""
    words = _LIST_PHRASES["de" if request.LANG.get() == "de" else "en"]
    return _LIST_TEMPLATE.format(**words) + (_LIST_MORE_TEMPLATE.format(**words) if capped else "")


async def scan(turn) -> None:
    """A list or count question: every matching document of the asked-for kind and topic added to the passages."""
    try:
        import unicodedata
        topic = turn.list_topic or turn.query
        tvec = turn.qvec
        if topic != turn.query:
            from memento.common.model_servers import vl_embed
            tvec = await asyncio.to_thread(vl_embed, turn.valves.EMBEDDER_URL, turn.valves.EMBEDDER_MODEL,
                                           turn.dom.get("instruction") or request.FALLBACK_INSTRUCTION, topic, None, 30)
        kinds = await asyncio.to_thread(scan_kind_sources, turn.list_kind, turn.collection, turn.engine)
        scope_docs = list(turn.author_docs or [])          # a person's documents: those of the kind, if any
        if kinds and scope_docs:
            scope_docs = [s for s in scope_docs if s in set(kinds)] or scope_docs
        elif kinds:
            scope_docs = kinds
        heads = await asyncio.to_thread(list_heads, turn.valves, turn.client, tvec, topic,
                                        library_listing.author_scope(scope_docs, turn.in_progress) if scope_docs else
                                        library_state.exclude_keywords_section(library_state.with_exclusions(None, turn.in_progress)))
        extra = list_extra(turn.hits, heads, turn.valves.LIST_SCAN_RATIO, turn.valves.LIST_SCAN_MAX)
        turn.list_capped = extra.capped
        if extra:
            turn.hits = turn.hits + extra
        import logging
        logging.getLogger("memento.query").info(
            "list scan: %d documents added (%d passages; %d documents searched%s; topic %r; kind %r: %s)",
            len({h["source"] for h in extra}), len(extra), getattr(heads, "found", len(heads)),
            ", more matched" if turn.list_capped else "", topic[:100], turn.list_kind,
            f"{len({unicodedata.normalize('NFC', s) for s in kinds})} documents" if kinds else "no restriction")
    except Exception as e:
        import logging
        logging.getLogger("memento.errors").warning("E-list-scan: %r", e)


def list_heads(valves, client, qvec, topic: str, query_filter) -> list:
    """A list question's candidates: LIST_SCAN_PER_DOC passages of each of the LIST_SCAN_DOCS documents closest to
    the topic, no table-of-contents lines, reranked against the topic."""
    import logging
    from memento.retrieval import search
    t0 = time.monotonic()
    heads = search.list_heads(client, valves.COLLECTION, qvec, query_filter, valves.LIST_SCAN_DOCS,
                              valves.LIST_SCAN_PER_DOC)
    out = search.rerank(topic, list(heads), valves.RERANKER_URL, len(heads), 0.0, 0.0,
                        instruction=request.rerank_instruction(request.domain(valves.DOMAIN_NAME))) if heads else []
    logging.getLogger("memento.query").info("list: search %.2f s (%d documents%s, %d passages reranked)",
                                            time.monotonic() - t0, heads.found, ", capped" if heads.capped else "",
                                            len(heads))
    return search.RelatedHeads(out, capped=heads.capped, found=heads.found)
