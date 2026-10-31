"""The library listed: every document (a corpus listing), the documents by a person (from their stored front matter),
and the latest uploads, as the reply itself without the answer model.
"""
import asyncio
import re

from memento.query import library_state, presentation, request, texts
from memento.query.scope import documents


# ── Corpus-listing intent ──
# "List all the papers in the corpus" is answered from the `sources` set the pipe already has; per-chunk retrieval would
# surface only the closest matches. Conservative: a detected document name or an analysis word makes it a content
# question.
_CORPUS_LISTING_RE = re.compile(
    r"\b(list|show|display|give|name|what|which"
    r"|liste|zeig|zeige|nenne|welche|was)\w*\b.*"
    r"\b(paper|papers|document|documents|title|titles|source|sources"
    r"|corpus|database|collection|library"
    r"|papier|papiere|dokument|dokumente|titel|korpus|datenbank|sammlung|bibliothek)\b",
    re.IGNORECASE,
)


def _wants_corpus_listing(query: str, paper: str | None) -> bool:
    """Cheap pre-check: does the query ask "what's in the corpus" rather than
    a question about specific content? If so, the caller can answer directly
    from `sources` — no retrieval or LLM call needed."""
    if paper:
        return False
    nq = query.lower()
    if any(w in nq for w in request.ANALYSIS_WORDS):
        return False
    return bool(_CORPUS_LISTING_RE.search(nq))


# ── Documents by a person ──
# Answered from each document's stored front matter (memento/library/front_matter.py), complete and without the answer model,
# unless the question adds a topic.
_AUTHOR_GENERIC = frozenset("""are there any documents document papers paper publications publication works work articles
article from by of written authored coauthored co author authors list show me which what all the in library does do
did have has had write wrote give find is a an any many how number count can you tell about regarding on his her their
von vom gibt es dokumente arbeiten veröffentlichungen liste zeig zeige mir alle welche hat haben geschrieben in der
bibliothek die das und wie viele was kannst du über zu sind ich
please bitte could would kindly könntest würdest zeigen auflisten
supervise supervised supervision supervisor review reviewed reviewer examined theses thesis dissertations dissertation
master masters bachelor betreut betreute betreuten begutachtet begutachtete wurden wurde abschlussarbeiten
masterarbeiten bachelorarbeiten dissertationen
this these contain contains containing include includes including anything everything something published publish our
enthält enthalten beinhaltet veröffentlicht veröffentlichte etwas irgendetwas alles dies diese dieser diesem wir uns
unserer""".split())    # polite words, supervision and everyday wording ("Does this library contain …"): no topic


def author_scope(sources: list, in_progress: set):
    """The search filter for a person's documents (their documents only; never ones mid-upload or keyword lines)."""
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    return library_state.exclude_keywords_section(library_state.with_exclusions(
        Filter(must=[FieldCondition(key="source", match=MatchAny(any=list(sources)))]), in_progress))


def author_topic(question: str, author: str) -> bool:
    """Does the question ask about more than the person's documents ("Which of X's papers are about Y?")? Words
    left after the person's name, the generic ones and possessive endings."""
    from memento.library.front_matter import _fold
    name = set(re.findall(r"[a-z]+", _fold(author)))
    words = [w for w in re.findall(r"[a-z]+", _fold(question)) if w not in name and w not in ("s",)]
    generic = {_fold(w) for w in _AUTHOR_GENERIC}
    return any(len(w) >= 3 and w not in generic for w in words)


def author_listing(author: str, titles: list, links: dict, *, de: bool, missing: int, possible=(),
                    role_unclear=(), supervised=()) -> str:
    """The answer to "documents by <person>": how many documents name them as an author and their titles as links
    (sorted); "missing" documents have no stored front matter yet, which is said. Documents where only an initial or a
    short form of the first name matches (`possible`) or the name appears outside the author lines (`role_unclear`) are
    listed apart."""
    n = len(titles)
    if de:
        head = (f"**{n} Dokument{'e' if n != 1 else ''}** in der Bibliothek {'nennen' if n != 1 else 'nennt'} {author} "
                f"als Autor:\n\n" if n else f"Kein Dokument in der Bibliothek nennt **{author}** als Autor.")
        note = (f"\n\n_Für {missing} Dokument{'e' if missing != 1 else ''} ist die Autorenliste noch nicht verfügbar, "
                f"{'sie wurden' if missing != 1 else 'es wurde'} daher nicht geprüft._" if missing else "")
    else:
        head = (f"**{n} document{'s' if n != 1 else ''}** in the library name{'s' if n == 1 else ''} {author} as an "
                f"author:\n\n" if n else f"No document in the library names **{author}** as an author.")
        note = (f"\n\n_For {missing} document{'s' if missing != 1 else ''} the author list is not available yet, so "
                f"{'they were' if missing != 1 else 'it was'} not checked._" if missing else "")
    body = listing_markdown(sorted(titles, key=str.lower), links=links, de=de) if n else ""
    extra = ""
    # the documents the person supervised or reviewed, as read from the title page (doc_people)
    for group, label_en, label_de in ((supervised, f"**Supervised or reviewed by {author}**",
                                       f"**Betreut oder begutachtet von {author}**"),
                                      (possible, "**Possibly the same person** (only an initial or a short form of the "
                                                 "first name matches)",
                                       "**Möglicherweise dieselbe Person** (nur eine Initiale oder eine Kurzform des "
                                       "Vornamens passt)"),
                                      (role_unclear, "**Named at the start of the document, role unclear**",
                                       "**Am Anfang des Dokuments genannt, Rolle unklar**")):
        k = len(group or ())
        if k:
            what = (f"{k} Dokument{'e' if k != 1 else ''}" if de else f"{k} document{'s' if k != 1 else ''}")
            extra += (f"\n\n{label_de if de else label_en}, {what}:\n\n"
                      + listing_markdown(sorted(group, key=str.lower), links=links, de=de))
    return head + body + extra + note


def author_possible_note(author: str, k: int, *, de: bool) -> str:
    """Above an answer about a person's work: the documents where only an initial or a short form matches, not searched.
    Above an answer about a person's work: the documents where only an initial or a short form matches, not searched."""
    if not k:
        return ""
    if de:
        return (f"_{k} weitere{'s' if k == 1 else ''} Dokument{'e' if k != 1 else ''} "
                f"könnte{'n' if k != 1 else ''} von {author} sein (nur eine Initiale oder eine Kurzform des Vornamens passt) und "
                f"wurde{'n' if k != 1 else ''} nicht durchsucht._\n\n")
    return (f"_{k} more document{'s' if k != 1 else ''} may be by {author} (only an initial or a short form of the "
            f"first name matches) and "
            f"{'were' if k != 1 else 'was'} not searched._\n\n")


def listing_intro(n: int, dom: dict, de: bool = False) -> str:
    """The sentence above the library listing. "documents", not "papers": uploads are Word, PowerPoint, Excel and images
    too; the noun stays configurable per domain (document_noun). German for a German question."""
    if de:
        return f"Die Bibliothek enthält derzeit **{n}** {'Dokument' if n == 1 else 'Dokumente'}:\n\n"
    nouns = (dom or {}).get("document_noun", {})
    noun = nouns.get("singular", "document") if n == 1 else nouns.get("plural", "documents")
    return f"The library currently contains **{n}** {noun}:\n\n"


RECENT_N = 10


def recent_listing(rows: dict, titles: dict, links: dict, n: int = RECENT_N, de: bool = False) -> str:
    """The newest documents first, with the day each was added (rows: {source: {"added_at": datetime}})."""
    dated = sorted(((r.get("added_at"), s) for s, r in (rows or {}).items() if r.get("added_at")), reverse=True)[:n]
    lines = []
    for i, (at, s) in enumerate(dated, start=1):
        name = titles.get(s) or s
        shown = f"[{re.sub(r'([\[\]])', r'\\\1', name)}]({links[s]})" if links.get(s) else name
        lines.append(f"{i}. {shown} · {at.day} {at:%b %Y}")
    head = "Zuletzt hinzugefügte Dokumente:" if de else "The most recently added documents:"
    return f"{head}\n\n" + "\n".join(lines) + "\n"


def listing_markdown(names, links: dict | None = None, de: bool = False) -> str:
    """The numbered library list; `links` {title: url}: each title links to its document; `de`: German labels for the
    collapsed rest."""
    names = sorted(names, key=str.lower)
    n = len(names)
    links = links or {}

    def item(name: str) -> str:
        url = links.get(name)
        return f"[{re.sub(r'([\[\]])', r'\\\1', name)}]({url})" if url else name

    def items(a: int, b: int) -> str:
        return "\n".join(f"{i}. {item(names[i - 1])}" for i in range(a + 1, b + 1))

    text = items(0, min(presentation.LIST_FIRST, n)) + "\n\n"
    chain = ""
    for a in reversed(range(presentation.LIST_FIRST, n, presentation.LIST_STEP)):      # built from the last step inwards
        b = min(a + presentation.LIST_STEP, n)
        more = f"{b - a} weitere anzeigen ({a + 1}–{b} von {n})" if de else f"Show {b - a} more ({a + 1}–{b} of {n})"
        chain = f"<details>\n<summary>{more}</summary>\n\n{items(a, b)}\n\n{chain}</details>\n\n"
    text += chain
    if n - presentation.LIST_FIRST > presentation.LIST_STEP:
        every = f"Alle {n} anzeigen" if de else f"Show all {n}"
        text += (f'<details type="{presentation.LIST_ALL_TYPE}">\n<summary>{every}</summary>\n\n'
                 f'{items(presentation.LIST_FIRST, n)}\n\n</details>\n\n')
    return text


async def person_reply(turn):
    """Documents by a person, from the stored front matter: the listing, or with a topic the person's documents for the
    normal search."""
    import logging
    try:
        from memento.library import front_matter as doc_authors
        engine = turn.engine()
        await asyncio.to_thread(doc_authors.ensure_table, engine)
        rows = await asyncio.to_thread(doc_authors.load, engine, turn.valves.COLLECTION, with_hash=True)
        # an author list counts only for the version in the library: its fingerprint known and equal to the
        # library's; any other document is "not checked yet", like one without a list
        current, unverified = doc_authors.fresh(
            {s: r for s, r in rows.items() if s in turn.sources and s not in turn.in_progress},
            {s: ((turn.docs or {}).get(s) or {}).get("content_hash") for s in turn.sources if s not in turn.in_progress})
        # the people the model read from each title page first (memento/library/people.py; the front-matter rules
        # read a paper's author line, not a thesis' title page); the front-matter rules for the documents it has
        # not read yet
        from memento.library import people as doc_people
        await asyncio.to_thread(doc_people.ensure_table, engine)
        people = await asyncio.to_thread(
            doc_people.current, engine, turn.valves.COLLECTION,
            {s: ((turn.docs or {}).get(s) or {}).get("content_hash") for s in turn.sources if s not in turn.in_progress})
        read = doc_people.classify(people, turn.author)
        kinds = doc_authors.classify({s: f for s, f in current.items() if s not in people}, turn.author)
        theirs = sorted(set(read["author"]) | set(kinds["author"]))
        possible = sorted(set(read["possible_identity"]) | set(kinds["possible_identity"]))
        unclear, supervised = kinds["possible_role"], read["supervisor"]
        missing = len(unverified)
        logging.getLogger("memento.query").info(
            "author %r: %d documents, %d supervised, %d possibly, %d role unclear (%d without front matter; "
            "%d read by the model)", turn.author, len(theirs), len(supervised), len(possible), len(unclear), missing,
            len(people))
        if author_topic(turn.original_query, turn.author) and theirs:
            # the question as asked, searched (and list-scanned) within the person's documents only; documents
            # where only an initial matches are named, not searched
            from memento.common.text_language import question_language
            note_possible = author_possible_note(turn.author, len(possible),
                                                  de=question_language(turn.original_query) == "de")
            if note_possible:
                yield note_possible
            if len(theirs) >= 2:
                turn.author_docs = theirs
            else:
                turn.paper = theirs[0]
        else:
            from memento.common.text_language import question_language
            every = sorted(set(theirs) | set(possible) | set(unclear) | set(supervised))
            shown = await asyncio.to_thread(documents.display_titles, every, turn.dom) if every else {}
            links = {shown[s]: presentation.source_file_url(turn.valves.FILES_URL, s, 0) for s in every}
            async for piece in presentation.type_out(author_listing(turn.author, [shown[s] for s in theirs], links,
                                                      de=question_language(turn.original_query) == "de",
                                                      missing=missing, possible=[shown[s] for s in possible],
                                                      role_unclear=[shown[s] for s in unclear],
                                                      supervised=[shown[s] for s in supervised]),
                                      *presentation.typing_speed(turn.valves, turn.metadata)):
                yield piece
            await turn.status("Done", done=True)
            turn.done = True
            return
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-author: %r", e)
        # the question asked for a person's documents: never the whole library instead
        async for piece in presentation.type_out(texts.SCOPE_FAILED_MSG, *presentation.typing_speed(turn.valves, turn.metadata)):
            yield piece
        await turn.status("Done", done=True)
        turn.done = True
        return


async def corpus_reply(turn):
    """The library listed: every document, the latest uploads with dates, or Document Chat's files."""
    if turn.sources:
        try:
            shown = (dict(turn.doc_chat.names) if turn.doc_chat is not None
                     else await asyncio.to_thread(documents.display_titles, sorted(turn.sources), turn.dom))
        except Exception:
            shown = {s: s for s in turn.sources}
        from memento.common.text_language import question_language
        de = question_language(turn.original_query) == "de"
        if turn.doc_chat is None and documents.wants_recent(turn.original_query):     # newest first, with dates
            rows = await asyncio.to_thread(library_state.load_documents, turn.valves.COLLECTION, turn.dom, turn.engine)
            if rows:
                rows = {s: r for s, r in rows.items() if s in turn.sources}
                links = {s: presentation.source_file_url(turn.valves.FILES_URL, s, 0) for s in rows}
                async for piece in presentation.type_out(recent_listing(rows, shown, links, de=de),
                                          *presentation.typing_speed(turn.valves, turn.metadata)):
                    yield piece
                await turn.status("Done", done=True)
                turn.done = True
                return
        intro = "Files in this chat:\n\n" if turn.doc_chat is not None else listing_intro(len(turn.sources), turn.dom, de=de)
        # each library document links to its first page; not Document Chat's
        links = {} if turn.doc_chat is not None else {t: presentation.source_file_url(turn.valves.FILES_URL, s, 0)
                                           for s, t in shown.items()}
        # typed out like other finished answers; only the first titles are typed, the collapsed rest arrives
        # whole
        async for piece in presentation.type_out(intro + listing_markdown(sorted(shown.values(), key=str.lower), links=links,
                                                            de=de),
                                  *presentation.typing_speed(turn.valves, turn.metadata)):
            yield piece
    else:
        yield "The library is currently empty."
    await turn.status("Done", done=True)          # closes the "Looking for answer..." line
    turn.done = True
