"""The search for a question: which passage types a part-of-paper request asks for, the extra table and equation
search, the image filter, the merged results, and whether the reranked passages are balanced across documents.
"""
import asyncio
import re

from memento.query import library_state, presentation, request, texts
from memento.query.flows import figures, library_listing
from memento.query.scope import documents, resolve


TABLE_WORD_RE = re.compile(r"\b(?:tables?|tabellen?)\b", re.I)


def passage_types(element: str, figure_request: bool, query: str) -> list | None:
    """The passage types a request for a part of a paper searches: table → tables, equation → equations, a figure
    request → images and charts, also tables when it mentions a table; else None (no type filter)."""
    if element == "table":
        return ["table"]
    if element == "equation":
        return ["equation"]
    if figure_request:
        return ["image", "chart"] + (["table"] if TABLE_WORD_RE.search(query or "") else [])
    return None


def element_union(element: str) -> bool:
    """Table and equation requests add those passages to the normal search instead of replacing it (the text often holds
    the answer); figure requests keep their own search."""
    return element in ("table", "equation")


def merge_points(*lists) -> list:
    """Search results merged, each passage once, in order."""
    seen, out = set(), []
    for pts in lists:
        for p in pts or []:
            if p.id not in seen:
                seen.add(p.id)
                out.append(p)
    return out


def add_image_filter(base_filter, paper, types: list | None = None):
    """Constrain retrieval to image/chart chunks (or `types`), composed with any existing
    paper filter (Qdrant allows nested Filters in `must`)."""
    from qdrant_client.models import Condition, Filter, FieldCondition, MatchValue, MatchAny

    type_cond = FieldCondition(key="type", match=MatchAny(any=list(types or ["image", "chart"])))
    if base_filter is not None:
        must: list[Condition] = [base_filter, type_cond]
        return Filter(must=must)
    must: list[Condition] = [type_cond]
    if paper:
        must.insert(0, FieldCondition(key="source", match=MatchValue(value=paper)))
    return Filter(must=must)


def balance_applies(cap: int, *, paper, image_intent: bool, query_intent: str) -> bool:
    """The soft per-document cap (memento/retrieval/search.py: balance_papers) is for questions across the library; a named
    document, a figure request and "show the section" are about one document on purpose."""
    return cap > 0 and not paper and not image_intent and query_intent != "verbatim_section"


def filters(turn) -> tuple:
    """(the search filter, the filter of the tables / equations searched besides it or None, whether the search is
    limited to the asked-for kind of passage): the named documents, a person's documents, the asked-for kind of passage,
    documents being uploaded and keyword lines excluded; sets the figure request and the documents searched each on
    their own (file_order, even_files)."""
    turn.image_intent = (figures.figure_request(turn.query_intent, turn.element)
                         or (not turn.router_ok and figures.detect_image_intent(turn.query)))
    # a request for a part of a paper searches that kind of passage: tables, equations, figures
    element_types = passage_types(turn.element, turn.image_intent, turn.query)
    query_filter = None
    if turn.paper:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=turn.paper))])
    elif turn.named_docs:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        query_filter = Filter(must=[FieldCondition(key="source", match=MatchAny(any=turn.named_docs))])
    elif turn.author_docs:                      # a person's documents with a topic
        query_filter = library_listing.author_scope(turn.author_docs, set())
    union_filter = None                    # table / equation passages searched besides the normal search
    type_filtered = False                  # the main search is limited to the asked-for kind of passage
    if element_types:
        try:
            if element_union(turn.element):
                union_filter = library_state.exclude_keywords_section(library_state.with_exclusions(
                    add_image_filter(query_filter, turn.paper, element_types), turn.in_progress))
            else:                          # a figure request: images and charts only
                query_filter = add_image_filter(query_filter, turn.paper, element_types)
                type_filtered = True
        except Exception:
            union_filter = None
    query_filter = library_state.with_exclusions(query_filter, turn.in_progress)
    query_filter = library_state.exclude_keywords_section(query_filter)
    # A question about several documents is about each of them: Document Chat's files when none is named, or the two
    # or more documents a question names. Each is searched on its own and gets an equal share of the passages, and
    # the model is told to cover each; a summary reads each one whole (6b).
    if turn.doc_chat is not None and not turn.paper and not turn.named_docs:
        turn.file_order = [x for x in turn.doc_chat.documents if x in turn.sources]
    else:
        turn.file_order = list(turn.named_docs) if len(turn.named_docs) >= 2 else []
    turn.even_files = (len(turn.file_order) >= 2 and not turn.image_intent and turn.query_intent in ("ranked_retrieval", "multi_hop"))
    return query_filter, union_filter, type_filtered


async def run(turn):
    """The search: the filter (filters), the embedding, dense or hybrid search, graph candidates, rerank and the
    per-document share."""
    query_filter, union_filter, type_filtered = filters(turn)

    # ── 3. Embed query ────────────────────────────────────────────────────
    await turn.status("Searching your files..." if turn.doc_chat is not None else "Searching the library...")
    try:
        from memento.common.model_servers import vl_embed
        turn.qvec = await asyncio.to_thread(
            vl_embed, turn.valves.EMBEDDER_URL, turn.valves.EMBEDDER_MODEL,
            turn.dom.get("instruction") or request.FALLBACK_INSTRUCTION, turn.query, None, 30,
        )
    except Exception as e:
        yield texts.user_error("search", e)
        await turn.status("Done", done=True)
        turn.done = True
        return

    # ── 4. Qdrant dense search ────────────────────────────────────────────
    try:
        if turn.even_files:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            per_file = max(turn.valves.RERANK_TOP, turn.valves.RETRIEVE_K // len(turn.file_order))
            results = []
            for f in turn.file_order:
                one = library_state.exclude_keywords_section(Filter(must=[FieldCondition(key="source", match=MatchValue(value=f))]))
                results += (await asyncio.to_thread(
                    library_state.qdrant_query, turn.client, turn.collection, turn.qvec, turn.query, one, per_file,
                    turn.valves.HYBRID_ENABLED, turn.valves.HYBRID_PREFETCH_K,
                )).points
        else:
            results = (await asyncio.to_thread(
                library_state.qdrant_query, turn.client, turn.collection, turn.qvec, turn.query,
                query_filter, turn.valves.RETRIEVE_K,
                turn.valves.HYBRID_ENABLED, turn.valves.HYBRID_PREFETCH_K,
            )).points
        if union_filter is not None:       # the asked-for tables / equations among the candidates
            results = merge_points((await asyncio.to_thread(
                library_state.qdrant_query, turn.client, turn.collection, turn.qvec, turn.query, union_filter, turn.valves.RERANK_TOP,
                turn.valves.HYBRID_ENABLED, turn.valves.HYBRID_PREFETCH_K,
            )).points, results)
        if type_filtered and not results:
            # nothing of the asked-for kind: the same documents' other passages, said plainly; the requested
            # documents are kept, never another document's figures
            from memento.common.text_language import question_language
            scope_now = resolve.scope_sources(turn.paper, turn.named_docs, turn.author_docs)
            yield figures.no_kind_note("figure" if turn.image_intent else turn.element,
                                presentation.shown(scope_now[0]) if scope_now and len(scope_now) == 1 else
                                (("den angefragten Dokumenten" if question_language(turn.original_query) == "de"
                                  else "the documents asked about") if scope_now else None),
                                question_language(turn.original_query) == "de")
            results = (await asyncio.to_thread(
                library_state.qdrant_query, turn.client, turn.collection, turn.qvec, turn.query, resolve.scope_filter(scope_now, turn.in_progress),
                turn.valves.RETRIEVE_K,
                turn.valves.HYBRID_ENABLED, turn.valves.HYBRID_PREFETCH_K,
            )).points
        from memento.retrieval.search import point_to_hit
        turn.hits = [
            point_to_hit(r.payload or {}, r.score)
            for r in results
            if (r.payload or {}).get("text", "").strip()
        ]
        n_found = len(turn.hits)
        turn.hits = presentation.without_icons(turn.hits)          # badges and logos never take a place
        if len(turn.hits) < n_found:
            import logging
            logging.getLogger("memento.query").info("icons: %d image passages dropped", n_found - len(turn.hits))
    except Exception as e:
        yield texts.user_error("search", e)
        await turn.status("Done", done=True)
        turn.done = True
        return

    if not turn.hits:
        yield "No relevant chunks found in the knowledge base."
        await turn.status("Done", done=True)
        turn.done = True
        return

    # ── Graph-traversal candidates (Valves.GRAPH_RETRIEVAL) ──
    # Best effort: extends `hits` with chunks reached via the knowledge graph (memento/graph/query/graph_retrieval.py). Any
    # failure (Neo4j unreachable, graph not built) falls back to vector-only search.
    if turn.valves.GRAPH_RETRIEVAL and turn.doc_chat is None:
        try:
            from memento.graph.query.graph_retrieval import graph_candidate_ids

            existing_ids = {str(r.id) for r in results}
            dense_sources = list({
                (r.payload or {}).get("source", "")
                for r in results
                if (r.payload or {}).get("source")
            })
            exclude_ids = {int(qid) for qid in existing_ids if qid.lstrip("-").isdigit()}
            graph_ids = await asyncio.to_thread(
                graph_candidate_ids,
                turn.query,
                # dense rank order (REFERENCES rows are ranked by it)
                dense_chunk_ids=[str(r.id) for r in results],
                dense_sources=dense_sources,
                exclude_ids=exclude_ids,
                enabled=True,
                query_vector=turn.qvec,
                qdrant_client=turn.client,
            )
            if graph_ids:
                graph_points = await asyncio.to_thread(
                    turn.client.retrieve,
                    collection_name=turn.collection,
                    ids=graph_ids,
                    with_payload=True,
                )
                turn.hits.extend(
                    point_to_hit(p.payload or {}, 0.0)
                    for p in graph_points
                    if (p.payload or {}).get("text", "").strip()
                )
        except Exception:
            pass

    # ── 5. Rerank ──
    # The reranker returns exactly RERANK_TOP hits however relevant they are, so a small or lopsided collection
    # backfills with near-zero noise: RERANK_MIN_SCORE drops it (keeping the set if everything is below the floor).
    # IMAGE_RERANK_RATIO drops image/chart hits below a fraction of the top image/chart score, so an unrelated
    # figure does not fill a figure request. Both floors and the fallback to vector order on reranker failure live
    # in memento/retrieval/search.py:rerank, shared with the evaluation.
    from memento.retrieval.search import balance_papers as _balance_papers, rerank as _rerank
    # with the per-document cap, the reranker returns its whole order (it scores every candidate anyway) and the
    # RERANK_TOP passages are chosen from it
    share = documents.share_cap(turn.valves.PER_PAPER_CAP, turn.valves.RERANK_TOP, len(turn.named_docs))
    balance = balance_applies(share, paper=turn.paper, image_intent=turn.image_intent, query_intent=turn.query_intent)
    if turn.valves.RERANK_ENABLED:
        turn.hits = await asyncio.to_thread(
            _rerank, turn.query, turn.hits, turn.valves.RERANKER_URL,
            len(turn.hits) if (balance or turn.even_files) else turn.valves.RERANK_TOP,
            0.0 if turn.even_files else turn.valves.RERANK_MIN_SCORE,     # each file's best passages count
            turn.valves.IMAGE_RERANK_RATIO,
            instruction=request.rerank_instruction(turn.dom),
        )
        if turn.even_files:
            from memento.retrieval.search import share_evenly as _share_evenly
            turn.hits = _share_evenly(turn.hits, turn.valves.RERANK_TOP, turn.file_order)
        elif balance:
            import logging
            from collections import Counter
            plain = turn.hits[:turn.valves.RERANK_TOP]
            turn.hits = _balance_papers(turn.hits, turn.valves.RERANK_TOP, share, turn.valves.PER_PAPER_SWAP_RATIO)
            share = lambda hs: max(Counter(h["source"] for h in hs).values(), default=0)
            swapped = sum(1 for h in turn.hits if not any(h is p for p in plain))
            if swapped:
                logging.getLogger("memento.query").info(
                    "per-paper cap: %d passages swapped, papers %d -> %d, largest share %d -> %d", swapped,
                    len({h["source"] for h in plain}), len({h["source"] for h in turn.hits}), share(plain), share(turn.hits))
    else:
        turn.hits = turn.hits[:turn.valves.RERANK_TOP]
