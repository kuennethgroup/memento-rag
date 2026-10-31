"""Which documents a question covers: the picked, named or referenced documents and their search filter, a yes to
"search the whole library instead", where the Thorough check may search again, and documents still being uploaded.
"""
import asyncio
import re

from memento.query import library_state, presentation, texts
from memento.query.scope import conversation, documents


def search_whole_library(scope: str, intent: str) -> bool:
    """True when a document named in the question must not limit the search ("which other papers use X?", where X is
    also a document's short name): the router says the question is about the library; without a scope from the router, a
    list question is never about one document."""
    return scope == "library" or (scope == "none" and intent == "set_query")


def scope_sources(paper, named_docs, author_docs) -> list | None:
    """The documents a question is restricted to: the picked, named or referenced document, two or more named ones, or a
    person's documents; None for the whole library (one scope for every search)."""
    if paper:
        return [paper]
    if named_docs:
        return list(named_docs)
    if author_docs:
        return list(author_docs)
    return None


def scope_filter(scope, in_progress: set):
    """The search filter for a scope (scope_sources; a single source as a string too): its documents only, else the
    library; never documents mid-upload or "Keywords:" fragments."""
    if isinstance(scope, str):
        scope = [scope]
    base = None
    if scope:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
        match = MatchValue(value=scope[0]) if len(scope) == 1 else MatchAny(any=list(scope))
        base = Filter(must=[FieldCondition(key="source", match=match)])
    return library_state.exclude_keywords_section(library_state.with_exclusions(base, in_progress))


def check_search_filter(scope, in_progress: set):
    """Where the check's targeted re-searches may look: the question's own documents (a summary's or gap answer's
    document, the named documents, a person's documents), else the library."""
    return scope_filter(scope, in_progress)


# the replies "yes" answers, older wordings included, so a "yes" in an older chat still works
_NAMED_NOT_FOUND_PREFIXES = (texts.NAMED_NOT_FOUND_MSG[:40], texts.NAMED_NOT_FOUND_DE[:40],
                             "I could not find the document you named", "Ich konnte das genannte Dokument")


_SEARCH_ALL_YES_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|please(?:\s+do)?|go\s+ahead|do\s+it|search(?:\s+it)?(?:\s+the\s+(?:whole\s+)?library)?|"
    r"ja|jo|gerne|klar|bitte|mach(?:\s+das)?|such(?:e)?(?:\s+in\s+der\s+(?:ganzen\s+)?bibliothek)?)"
    r"(?:[\s,]+(?:please|bitte|thanks|thank\s+you|danke))?\s*[.!]*\s*$", re.I)


def search_all_request(question: str, history: list) -> str | None:
    """The previous question, when this message says yes to NAMED_NOT_FOUND_MSG: it is then asked again over the whole
    library. None for anything else."""
    if len(history or []) < 2 or not _SEARCH_ALL_YES_RE.match(question or ""):
        return None
    asked, reply = history[-2], history[-1]
    text = (reply.get("content") or "").lstrip()
    if asked.get("role") != "user" or reply.get("role") != "assistant" or not text.startswith(_NAMED_NOT_FOUND_PREFIXES):
        return None
    return (asked.get("content") or "").strip() or None


def processing_msg(names: list, *, all_busy: bool, de: bool) -> str:
    """The requested documents still being uploaded: all of them (the whole reply) or some (a note before the answer
    from the others)."""
    shown = " and ".join(f"**{n}**" for n in names) if not de else " und ".join(f"**{n}**" for n in names)
    one = len(names) == 1
    if de:
        verb = "wird" if one else "werden"
        return (f"{shown} {verb} noch verarbeitet, daher kann ich noch nicht daraus antworten. Bitte versuchen Sie es "
                f"später noch einmal." if all_busy else
                f"_{shown} {verb} noch verarbeitet und fließ{'t' if one else 'en'} nicht in diese Antwort ein._\n\n")
    verb = "is" if one else "are"
    return (f"{shown} {verb} still being processed, so I cannot answer from {'it' if one else 'them'} yet. Please try "
            f"again later." if all_busy else
            f"_{shown} {verb} still being processed and not included in this answer._\n\n")


async def detect(turn):
    """The documents the question is about: the library's (or the chat's) documents, those being uploaded, a picked,
    named or referenced document, two or more named ones, the asker's own uploads."""
    from qdrant_client import QdrantClient
    turn.client = turn.doc_chat.client() if turn.doc_chat is not None else QdrantClient(url=turn.valves.QDRANT_URL, timeout=30)
    turn.paper = None
    turn.named_docs = []          # two or more documents named in the question
    turn.sources = set()
    turn.in_progress = set()
    turn.picked, turn.docs = None, None
    turn.resolve_failed = False         # the lookup of a requested document failed: never the whole library instead
    try:
        def _scroll_sources():
            sources = set()
            offset = None
            while True:
                points, offset = turn.client.scroll(
                    collection_name=turn.collection,
                    with_payload=["source"],
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                for p in points:
                    src = (p.payload or {}).get("source")
                    if src:
                        sources.add(src)
                if offset is None:
                    break
            return sources

        # the library's documents from their table (memento/library/documents.py); an empty list is built once from the
        # collection, an unreadable table falls back to scrolling it (sync calls run off the event loop)
        if turn.doc_chat is not None:                 # the chat's ready files; no library list is read or written
            turn.docs = turn.doc_chat.documents
            turn.sources = set(turn.docs)
        else:
            turn.docs = await asyncio.to_thread(library_state.load_documents, turn.valves.COLLECTION, turn.dom, turn.engine)
            if turn.docs:
                turn.sources = set(turn.docs)
            else:
                turn.sources = await asyncio.to_thread(_scroll_sources)
                if turn.docs is not None and turn.sources:
                    await asyncio.to_thread(library_state.rebuild_documents, turn.valves.COLLECTION, turn.dom,
                                            _scroll_sources, turn.engine)
            # the picker's list, off the answer's path (re-exported only when it differs from the table)
            asyncio.get_running_loop().run_in_executor(None, documents.publish_library, turn.valves.COLLECTION,
                                                       set(turn.sources), turn.engine)

        # documents ingestion is rewriting are excluded from detection and retrieval (library_state.with_exclusions
        # below); library uploads only: Document Chat's files are ready
        turn.in_progress = set() if turn.doc_chat is not None else await asyncio.to_thread(library_state.in_progress_sources, turn.engine)
        if turn.in_progress:
            turn.sources = turn.sources - turn.in_progress
            n = len(turn.in_progress)
            noun, verb = ("document", "is") if n == 1 else ("documents", "are")
            warning = (
                f"⚠️ {n} {noun} {verb} still being processed "
                f"and {verb} temporarily excluded from results\n\n"
            )
            await turn.status(warning.strip())
            # also in the reply itself: status lines fold away once the answer is done, and this must stay visible
            yield warning

        # a requested document may still be uploading: names are resolved against those too, and such a document is
        # said to be processing instead of the question searching the rest of the library
        requestable = set(turn.sources) | set(turn.in_progress)
        # a document picked on the start screen: exact, from the picker's own list; the button decides the route
        turn.picked = await asyncio.to_thread(documents.picked_document, turn.original_query, turn.sources, turn.valves.COLLECTION)
        if turn.picked:
            turn.paper, turn.query_intent = turn.picked
        else:
            # "the papers I just uploaded": the asker's own latest upload
            mine = []
            if turn.doc_chat is None and documents.mentions_my_uploads(turn.original_query):
                try:
                    mine = await asyncio.to_thread(documents.latest_uploads, turn.valves.COLLECTION,
                                                   (turn.user or {}).get("id"), requestable, turn.engine)
                except Exception as e:
                    import logging
                    logging.getLogger("memento.errors").warning("E-my-uploads: %r", e)
                    turn.resolve_failed = True
            if turn.doc_chat is None and documents.mentions_my_uploads(turn.original_query):
                # the asker's own uploads only: a name in the question is looked up among them, never in the whole
                # library; none: the no-uploads reply below. Several named among them ("compare my uploads A and B")
                # before one best match
                many = await asyncio.to_thread(documents.named_documents, turn.query, list(mine), turn.dom) if len(mine) >= 2 else []
                within = None if len(many) >= 2 or len(mine) < 2 else \
                    await asyncio.to_thread(documents.resolve_reference, turn.query, list(mine), turn.dom)
                turn.paper = within or (mine[0] if len(mine) == 1 else None)
                turn.named_docs = many if len(many) >= 2 else ([] if within else (mine if len(mine) >= 2 else []))
            else:
                turn.paper = await asyncio.to_thread(documents.resolve_reference, turn.query, list(requestable), turn.dom)
                # two or more documents named ("compare A with B"): search those together, not just the best
                turn.named_docs = await asyncio.to_thread(documents.named_documents, turn.query, list(requestable), turn.dom)
                # "Which of them …?" after an answer about several documents: one search within those documents (as
                # the 1c rule below), not the names the router's rewrite happens to contain (conversation.them_scope)
                if turn.history and turn.doc_chat is None and conversation.PLURAL_BACKREF_RE.search(turn.original_query or ""):
                    them = conversation.them_scope(turn.original_query, await conversation.conversation_documents(turn.sources, turn.metadata))
                    if them:
                        import logging
                        logging.getLogger("memento.query").info("which of them: the previous answer's %d "
                                                                "documents", len(them))
                        turn.paper, turn.named_docs, turn.author_docs = None, [], them
            if mine:
                import logging
                logging.getLogger("memento.query").info("my uploads: %s", " | ".join(mine))
            if len(turn.named_docs) >= 2:
                turn.paper = None
                import logging
                logging.getLogger("memento.query").info("named documents: %s", " | ".join(turn.named_docs))
            else:
                turn.named_docs = []
            # a method that is also a document's name ("which other papers use X?"): the whole library
            if (turn.paper or turn.named_docs) and search_whole_library(turn.scope, turn.query_intent):
                import logging
                logging.getLogger("memento.query").info("named documents %s not used as a filter (scope=%s, %s)",
                                                        turn.paper or " | ".join(turn.named_docs), turn.scope, turn.query_intent)
                turn.paper = None
                if turn.scope == "library":
                    turn.named_docs = []
        # Document Chat with one ready file: "the attached file" and "this paper" name it, so a summary reads the
        # whole file; questions search only this file anyway
        if turn.doc_chat is not None and not turn.paper and not turn.named_docs and len(turn.sources) == 1:
            turn.paper = next(iter(turn.sources))
    except Exception as e:
        # detection is best effort: never block retrieval on it, unless the question asked for documents (below)
        import logging
        logging.getLogger("memento.errors").warning("E-resolve: %r", e)
        turn.paper, turn.named_docs = None, []
        turn.resolve_failed = True


async def decide(turn):
    """A question whose documents cannot be searched gets its reply here: the lookup failed, the asker has no uploads,
    the documents are still uploading, or "this paper" has nothing behind it."""
    if turn.search_all:                     # the reader chose the whole library
        turn.paper, turn.named_docs, turn.scope, turn.resolve_failed = None, [], "library", False

    # ── 1. A requested document that cannot be searched: its lookup failed, or it is still uploading ──
    if turn.doc_chat is None and turn.resolve_failed and (turn.scope in ("document", "documents")
                                          or documents.mentions_my_uploads(turn.original_query)):
        async for piece in presentation.type_out(texts.SCOPE_FAILED_MSG, *presentation.typing_speed(turn.valves, turn.metadata)):
            yield piece
        await turn.status("Done", done=True)
        turn.done = True
        return
    # ── 1. "my uploads" from someone with none in the library: said, nothing searched ──
    if (turn.doc_chat is None and not turn.search_all and not turn.picked and not turn.paper and not turn.named_docs
            and documents.mentions_my_uploads(turn.original_query) and (turn.user or {}).get("id")):
        from memento.common.text_language import question_language
        import logging
        logging.getLogger("memento.query").info("my uploads: none in the library")
        de = question_language(turn.original_query) == "de"
        async for piece in presentation.type_out(texts.NO_UPLOADS_DE if de else texts.NO_UPLOADS_MSG, *presentation.typing_speed(turn.valves, turn.metadata)):
            yield piece
        await turn.status("Done", done=True)
        turn.done = True
        return
    busy = [s for s in ([turn.paper] if turn.paper else list(turn.named_docs)) if s in turn.in_progress] if turn.doc_chat is None else []
    if busy:
        from memento.common.text_language import question_language
        ready = [s for s in ([turn.paper] if turn.paper else list(turn.named_docs)) if s not in turn.in_progress]
        de = question_language(turn.original_query) == "de"
        import logging
        logging.getLogger("memento.query").info("requested documents still processing: %s", " | ".join(busy))
        if not ready:
            async for piece in presentation.type_out(processing_msg([presentation.shown(s) for s in busy], all_busy=True, de=de),
                                      *presentation.typing_speed(turn.valves, turn.metadata)):
                yield piece
            await turn.status("Done", done=True)
            turn.done = True
            return
        yield processing_msg([presentation.shown(s) for s in busy], all_busy=False, de=de)
        turn.paper, turn.named_docs = (ready[0], []) if len(ready) == 1 else (None, ready)

    # ── 1. "this paper" with nothing behind it: ask which document, never the whole library ──
    if (turn.query_intent not in ("general", "chitchat", "corpus_listing", "rework")
            and conversation.unnamed_document(turn.original_query, dc=turn.doc_chat is not None, paper=turn.paper, named_docs=turn.named_docs,
                                  history=turn.history)):
        from memento.common.text_language import question_language
        import logging
        logging.getLogger("memento.query").info("unnamed document: asked which one")
        yield conversation.unnamed_document_msg(question_language(turn.original_query) == "de")
        await turn.status("Done", done=True)
        turn.done = True
        return

    # ── 1. "this paper" after an earlier turn: the document the previous answer cited, when its question agrees;
    # else ask, never the whole library ──
    if (turn.query_intent not in ("general", "chitchat", "corpus_listing", "rework") and turn.doc_chat is None and turn.history
            and not turn.paper and not turn.named_docs and conversation.THIS_DOC_RE.search(turn.original_query or "")):
        target = await conversation.conversation_target(turn.valves, turn.sources, turn.dom, turn.metadata)
        if target:
            import logging
            logging.getLogger("memento.query").info("this document: %s (from the previous answer)", target)
            turn.paper = target
        else:
            from memento.common.text_language import question_language
            yield conversation.unnamed_document_msg(question_language(turn.original_query) == "de")
            await turn.status("Done", done=True)
            turn.done = True
            return


async def named_not_found(turn):
    """A named document that matches nothing: the document the previous answer cited, or several, when the question
    refers back; else the reader is asked (a yes searches the whole library)."""
    import logging
    # only for a question that names no document and refers back ("its limitations"); a newly named one that
    # matches nothing is asked about, never swapped for the previous one
    refers_back = not documents.document_phrases(turn.original_query) and bool(conversation.BACKREF_RE.search(turn.original_query or ""))
    target = await conversation.conversation_target(turn.valves, turn.sources, turn.dom, turn.metadata) if turn.history and refers_back else None
    # "which of them" after an answer about several documents: within those documents
    several = (await conversation.conversation_documents(turn.sources, turn.metadata)
               if turn.history and refers_back and not target and turn.scope == "documents" else None)
    if target:
        logging.getLogger("memento.query").info("named document not found; the previous answer's: %s", target)
        turn.paper = target
    elif several:
        logging.getLogger("memento.query").info("named documents not found; the previous answer's: %s",
                                                several)
        turn.author_docs = several
    else:
        from memento.common.text_language import question_language
        logging.getLogger("memento.query").info("named document not found: asked (scope=%s)", turn.scope)
        de = question_language(turn.original_query) == "de"
        async for piece in presentation.type_out(texts.NAMED_NOT_FOUND_DE if de else texts.NAMED_NOT_FOUND_MSG, *presentation.typing_speed(turn.valves, turn.metadata)):
            yield piece
        await turn.status("Done", done=True)
        turn.done = True
        return
