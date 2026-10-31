"""Documents a question names: titles and other names, a named or picked document, an existence question, the asker's
latest uploads, the library list the start screen shows, and a document resolved by title (graph or names).
"""
import os
import re

from memento.query import library_state, presentation, request


def _detect_paper(query: str, sources: list) -> str | None:
    """Return the `source` whose normalized name appears in the query (longest match wins)."""
    nq = request.normalize(query)
    best = None
    for src in sources:
        nsrc = request.normalize(src)
        if nsrc and nsrc in nq and (best is None or len(nsrc) > len(request.normalize(best))):
            best = src
    return best


def _domain_graph(domain: dict | None) -> str | None:
    """The domain's own Neo4j URI if it has one and something answers on its port (0.3 s), else None. Callers must use
    this uri (config.NEO4J_URI follows the .env DOMAIN_NAME, not the Valve) and must not reach Neo4j otherwise: a
    stopped store makes the driver retry for about a minute."""
    try:
        import socket
        from urllib.parse import urlparse
        uri = (domain or {}).get("neo4j_uri")
        if not uri:
            return None
        u = urlparse(uri)
        with socket.create_connection((u.hostname or "localhost", u.port or 7687), timeout=0.3):
            return uri
    except Exception:
        return None



# words that join two named documents in a question: "compare A with B", "A vs. B", "how does A differ from B"
_NAME_JOIN = re.compile(r"\s*(?:,|;|\band\b|\bwith\b|\bvs\.?|\bversus\b|\bcompared (?:to|with)\b|\bagainst\b|"
                        r"\bdiffer(?:s)? from\b|\bund\b|\bmit\b|\bgegenüber\b|\bim vergleich zu\b|"
                        r"\bverglichen mit\b)\s*", re.I)


def named_documents(query: str, sources: list, domain: dict | None = None) -> list[str]:
    """Every document a question names, when it names two or more (a comparison); else [] or the one match. The question
    is split at joining words and each part resolved on its own, so a comparison keeps both documents (a title
    containing "and" may be split too: its parts still resolve to it, or to nothing)."""
    found: list[str] = []
    parts = [p for p in _NAME_JOIN.split(query or "") if p and len(p.split()) >= 2]
    if len(parts) < 2:
        return []
    for part in parts:
        src = _resolve_paper(part, sources, domain)
        if src and src not in found:
            found.append(src)
    return found


def share_cap(cap: int, top: int, named: int) -> int:
    """Passages per document: the soft cap, or with two or more named documents an equal share of the `top` slots."""
    return max(cap, -(-top // named)) if named >= 2 else cap


def _doc_aliases(sources: list, domain: dict | None) -> dict:
    """{source: [file name, parsed title, PDF title]} for the documents of this library."""
    import importlib
    try:
        in_dir, out_dir = importlib.import_module("memento.config.paths").corpus_dirs(domain or {})
    except Exception:
        in_dir = out_dir = None
    out = {}
    for src in sources:
        key = (str(out_dir), src)
        if key not in library_state.ALIASES:
            parsed_title, pdf_title = (None, None)
            try:
                from memento.library.documents import raw_titles
                parsed_title, pdf_title = raw_titles(src, in_dir, out_dir)
            except Exception:
                pass
            library_state.ALIASES[key] = [src] + [n for n in (parsed_title, pdf_title) if n]
            library_state.TITLES[key] = (parsed_title, pdf_title)
        out[src] = library_state.ALIASES[key] + [n for n in _listed_names(key) if n not in library_state.ALIASES[key]]
    return out


def _listed_names(key: tuple) -> list:
    """The library's own title of a document (`key` = (out_dir, source)), as stored and listed: people name documents as
    listed, and it separates documents that share a first-page title."""
    lib = library_state.LIB_TITLES.get(key)
    if not lib:
        return []
    from memento.library.documents import plain_title, same_name
    if same_name(lib, key[1]):
        return []
    return list(dict.fromkeys(n for n in (lib, plain_title(lib)) if n))


def _plausible_title(title: str | None, section_hints: dict | None) -> str | None:
    """A title candidate fit to SHOW as the document's title, else None (memento/library/documents.py)."""
    from memento.library.documents import plausible_title
    return plausible_title(title, section_hints)


def display_titles(sources: list, domain: dict | None) -> dict:
    """{source: text for the library listing}: the title (first-page title, else the PDF's own), with the file name
    in brackets when it differs — citations show file names, so the two stay connectable; else the file name."""
    import importlib
    _doc_aliases(sources, domain)
    from memento.library.documents import best_title, plain_title, same_name
    try:
        out_dir = str(importlib.import_module("memento.config.paths").corpus_dirs(domain or {})[1])
    except Exception:
        out_dir = "None"
    hints = (domain or {}).get("section_hints") or {}
    shown = {}
    for src in sources:
        # the library's own title first, as the picker shows it; else the first-page or PDF title
        lib = library_state.LIB_TITLES.get((out_dir, src))
        title = lib if lib and not same_name(lib, src) else best_title(*library_state.TITLES.get((out_dir, src), (None, None)), hints)
        title = plain_title(title) if title else None      # as the picker shows it
        shown[src] = src if not title else title if same_name(title, src) else f"{title} ({src})"
    return shown


# The start-screen picker's questions (loader.js): the document named exactly as the picker listed it (its title, or its
# file name when two documents share a title). Looked up in the picker's list, not by the title resolver, which abstains
# on shared or formula titles.
PICKED = re.compile(r'^\s*(Summarize the document|What research gaps does the document) ["“](.+)["”](?: name\?)?\s*$')


def picked_document(query: str, sources, collection: str | None = None) -> tuple[str, str] | None:
    """(source, intent) for a picker question, else None. The title is looked up in this library's own list
    (`collection`; None: the single-library list)."""
    m = PICKED.match(query or "")
    root = os.getenv("OUTPUT_ROOT")
    if not m:
        return None
    intent = "summary" if m.group(1).startswith("Summarize") else "gap_query"
    name, sources = m.group(2).strip(), set(sources)
    if name in sources:
        return name, intent
    try:
        from memento.library.documents import read_library
        docs = (read_library(root, collection) or {}).get("documents", []) if root else []
    except Exception:
        return None
    hits = {e.get("source") for e in docs
            if isinstance(e, dict) and (e.get("title") or "").replace('"', "'") == name and e.get("source") in sources}
    return (hits.pop(), intent) if len(hits) == 1 else None


# "Is document X in the library?": answered from the document list, not by the model, which only sees the top passages
# and hedges for a missing title. Strict on purpose: caught only with a document noun, quotes or the library named, and
# never with a topic or author after the noun (that is a topic question).
_DOC_NOUN = r"(?:paper|document|article|publication|study|file|pdf|report|thesis|dissertation|manuscript)"


_DOC_NOUN_DE = r"(?:paper|dokument|artikel|arbeit|datei|publikation|studie|veröffentlichung|aufsatz|dissertation|bericht)"


_LIB = r"(?:library|collection|database|corpus)"


_LIB_DE = r"(?:bibliothek|sammlung|datenbank)"


_IN_LIB = rf"(?:(?:already |currently )?(?:in|part of|included in|available in|stored in|contained in) (?:the|our|my|this|your) {_LIB})"


_IN_LIB_DE = rf"(?:(?:in|im) (?:der|unserer|dieser) {_LIB_DE}(?: (?:vorhanden|enthalten|verfügbar|drin))?)"


_EXISTENCE = [   # (pattern over the whole question, language, the library is named by the frame itself)
    (rf"(?:is|are) (?P<name>.+?) {_IN_LIB}", "en", True),
    (rf"do (?:we|you|i) (?:already )?(?:have|own|store|keep) (?P<name>.+?)(?P<lib> {_IN_LIB})?", "en", False),
    (rf"does (?P<name>.+?) exist(?P<lib> {_IN_LIB})?", "en", False),
    (rf"is there (?:a|an|the|any) {_DOC_NOUN} (?:called|titled|named|with the title) (?P<name>.+?)(?: {_IN_LIB})?",
     "en", True),
    (rf"(?:does|do) (?:the|our|my|this|your) {_LIB} (?:contain|have|include|hold) (?P<name>.+?)", "en", True),
    (rf"gibt es (?P<name>.+?)(?P<lib> {_IN_LIB_DE})?", "de", False),
    (rf"haben wir (?:schon |bereits )?(?P<name>.+?)(?P<lib> {_IN_LIB_DE})?", "de", False),
    (rf"(?:ist|sind) (?P<name>.+?) {_IN_LIB_DE}", "de", True),
    (rf"existiert (?P<name>.+?)(?P<lib> {_IN_LIB_DE})?", "de", False),
    (rf"enthält (?:die|unsere|diese) {_LIB_DE} (?P<name>.+?)", "de", True),
]


_EXISTENCE = [(re.compile(p + r"\s*[?.!]*", re.I), lang, lib, "called|titled" in p) for p, lang, lib in _EXISTENCE]


_QUOTES = "\"'„“”‚‘’«»‹›`"


_LEAD_ARTICLE = re.compile(r"^(?:the|a|an|that|das|der|den|die|ein|eine|einen)\s+", re.I)


_LEAD_NOUN = re.compile(rf"^(?:{_DOC_NOUN}|{_DOC_NOUN_DE})(?P<called>\s+(?:called|titled|named|with the title|namens|"
                        rf"mit dem titel|titel))?(?:\s+|$)", re.I)


_FILE_EXT = re.compile(r"\.(?:pdf|docx?|pptx?|xlsx?)$", re.I)


_TAIL_NOUN = re.compile(rf"(?:\s+|-)(?:{_DOC_NOUN}|{_DOC_NOUN_DE})$", re.I)


# "is field cycling discussed in the library" asks about content, not a document
_CONTENT_VERB = re.compile(r"\s(?:discussed|mentioned|covered|described|explained|addressed|studied|investigated|"
                           r"used|reported|treated|erwähnt|beschrieben|behandelt|erklärt|untersucht)$", re.I)


# not a name: a topic, author or clause after the noun, a pronoun, or a general word ("data on …", "papers on …")
_NOT_A_NAME = re.compile(
    r"^(?:about|on|regarding|concerning|covering|related to|by|from|that|which|where|discussing|describing|with|"
    r"for|über|zu|zum|zur|von|mit|bezüglich|betreffend|zum thema|it|this|that|these|those|they|them|one|es|das|dies|"
    r"diese|dieses|there|any|anything|something|some|papers|documents|articles|publications|studies|files|"
    r"information|info|data|research|literature|work|results|material|materials|arbeiten|dokumente|papiere|daten|"
    r"informationen|literatur|etwas|irgendwas|welche|was|it's(?=$))(?=\s|$)", re.I)


def existence_name(question: str) -> tuple[str, str] | None:
    """(the document name, "en"/"de") when the question asks whether one named document is in the library, else None."""
    q = " ".join(str(question or "").split())
    for rx, lang, lib, titled in _EXISTENCE:
        m = rx.fullmatch(q)
        if not m:
            continue
        name = _LEAD_ARTICLE.sub("", m.group("name").strip())
        named = _LEAD_NOUN.match(name)
        if named:
            name = name[named.end():]
        quoted = len(name) > 2 and name[0] in _QUOTES and name[-1] in _QUOTES
        tail = _TAIL_NOUN.search(name) if not quoted else None
        if tail:
            name = name[:tail.start()]
        name = name.strip().strip(_QUOTES).strip()
        has_lib = lib or bool(m.groupdict().get("lib"))
        # quoted, after "called"/"titled" or a file name: a name even when it starts like a topic or a pronoun
        as_name = quoted or titled or bool(named and named.group("called")) or bool(_FILE_EXT.search(name))
        if (not (named or tail or quoted or has_lib) or len(name) < 2 or len(name.split()) > 40
                or (_NOT_A_NAME.match(name) and not as_name) or (_CONTENT_VERB.search(name) and not as_name)):
            continue                      # not this frame; "is there a document titled X in …" fits a later one
        return name, lang
    return None


def lookup_document(name: str, sources: list, domain: dict | None) -> str | None:
    """The library document `name` names (title, file name or short name), by the resolver the questions use; else a
    cut-off title: four or more words that start exactly one document's name (too few distinctive words for the
    resolver's fuzzy match)."""
    src = _resolve_paper(name, sources, domain) or _resolve_paper(f"the {name} paper", sources, domain)
    if src or len(name.split()) < 4:
        return src
    try:
        from memento.query.scope.paper_resolve import normalize
        start = normalize(name)
        owners = {s for s, names in _doc_aliases(list(sources), domain).items()
                  if start and any(f"{normalize(n)} ".startswith(f"{start} ") for n in names if n)}
        return owners.pop() if len(owners) == 1 else None
    except Exception:
        return None


def existence_reply(name: str, lang: str, *, title: str | None = None, related=(), n_docs: int = 0) -> str:
    """The fixed reply: Yes + the document's title; No + related documents; No + how to add it."""
    name, de = presentation.plain(name), lang == "de"
    if title:
        return (f"**Ja**, *{presentation.plain(title)}* ist in der Bibliothek. Stellen Sie eine Frage dazu, oder nutzen Sie "
                f"**Summarize a document** auf der Startseite für einen Überblick." if de else
                f"**Yes**, *{presentation.plain(title)}* is in the library. Ask a question about it, or use **Summarize a "
                f"document** on the start screen for an overview.")
    if related:
        items = "\n".join(f"- {presentation.plain(t)}" for t in related)
        return (f"**Nein**, kein Dokument mit genau dem Titel *{name}* ist in der Bibliothek. Verwandte Dokumente:\n\n"
                f"{items}\n\nIst es ein anderes Dokument, kann es mit **Library Upload** hinzugefügt werden." if de else
                f"**No** document titled exactly *{name}* is in the library. Related documents:\n\n{items}\n\n"
                f"If it is a different document, you can add it with **Library Upload**.")
    noun = ("Dokument" if n_docs == 1 else "Dokumente") if de else ("document" if n_docs == 1 else "documents")
    return (f"**Nein**, kein Dokument mit genau dem Titel *{name}* ist in der Bibliothek ({n_docs} {noun}). Es kann mit "
            f"**Library Upload** hinzugefügt werden." if de else
            f"**No** document titled exactly *{name}* is in the library ({n_docs} {noun}). You can add it with "
            f"**Library Upload**.")


# The start-screen picker's list: OUTPUT_ROOT/_runtime/library.json, exported from the document table by its write
# path (memento/library/documents.py); publish_library re-exports it when it differs from the table.
def publish_library(collection: str, sources: set, db_engine) -> None:
    """Re-export the picker's list from the document table when the file is missing or lists another set (a reset,
    a crash between the table write and the export, another writer). The write path exports on every change."""
    import logging
    root = os.getenv("OUTPUT_ROOT")
    if not root:
        return
    try:
        from memento.library import documents as ldb, documents as ll
        on_file = ll.read_library(root, collection)          # this library's own list
        if on_file is not None and {e.get("source") for e in on_file["documents"]} == set(sources):
            return
        ldb.export_library(db_engine(), collection, root)
        logging.getLogger("memento.query").info("library list: exported for %s", collection)
    except Exception as e:
        logging.getLogger("memento.errors").warning("E-library-list: %r", e)


# the words a question uses for the one document it is about ("the X paper", "die X-Arbeit")
_DOC_PHRASE_RE = re.compile(r"\bthe\s+((?:[\w‐–:-]+\s+){0,3}?[\w‐–:-]+)\s+"
                            r"(paper|study|thesis|article|review|book|document|dissertation)\b", re.I)


_DOC_PHRASE_DE_RE = re.compile(r"\b(?:die|das|der|den|dem)\s+[\w‐–]+[-\s](?:Arbeit|Paper|Studie|Dissertation|"
                               r"Artikel|Buch)\b")


# words that name no document by themselves ("the first paper", "the same study", "the original X paper")
_GENERIC_DOC_WORDS = {"first", "second", "third", "last", "same", "previous", "next", "following", "above", "original",
                      "other", "this", "that", "whole", "entire", "new", "latest", "recent", "cited", "attached",
                      "uploaded", "mentioned", "referenced", "said", "aforementioned", "one"}


def document_phrases(query: str) -> list:
    """The phrases naming a document in a question ("the X paper"), in order; generic words dropped,
    phrases of generic words only ("the first paper") left out."""
    out = []
    for m in _DOC_PHRASE_RE.finditer(query or ""):
        words = [w for w in m.group(1).split() if w.lower() not in _GENERIC_DOC_WORDS]
        if words:
            out.append(f"the {' '.join(words)} {m.group(2)}")
    return out + [m.group(0) for m in _DOC_PHRASE_DE_RE.finditer(query or "")]


# references to the asker's own recent upload ("the papers I just uploaded", "my uploads")
_MY_UPLOADS_RE = re.compile(
    r"\b(?:just|recently|newly)\s+(?:uploaded|added)\b|\bmy\s+(?:own\s+)?(?:uploads?|uploaded)\b|"
    # "my latest uploads", "my most recent uploaded files"
    r"\bmy\s+(?:latest|last|newest|recent|most\s+recent)\s+(?:uploads?|uploaded)\b|"
    r"\b(?:papers?|documents?|files?)\s+i\s+(?:just\s+|have\s+|'ve\s+)?(?:uploaded|added)\b|"
    r"\b(?:soeben|gerade|eben|neu|kürzlich)\s+(?:hoch|up)ge?(?:laden|loade)\w*|\bupgeloadet\w*|\bhochgeladene\w*|"
    r"\bmeine\w*\s+(?:uploads?|hochgeladen\w*)|"
    # "the latest uploaded file": one file, not the list of newest documents
    r"\b(?:latest|last|newest|most\s+recent(?:ly)?)\s+(?:uploaded|added)\s+(?:file|paper|document|pdf|thesis)\b|"
    r"\b(?:file|paper|document|pdf)\s+i\s+(?:last|just)\s+(?:uploaded|added)\b", re.I)


UPLOAD_BATCH_MINUTES = 30      # one upload: the documents a user added within this long before their newest


UPLOAD_BATCH_MAX = 10


def mentions_my_uploads(query: str) -> bool:
    return bool(_MY_UPLOADS_RE.search(query or ""))


def _upload_batch(rows, user_id: str | None) -> list:
    """The user's latest upload: (source, uploaded_by, added_at) rows of the library list → the sources this user
    added within UPLOAD_BATCH_MINUTES before their newest, sorted, at most UPLOAD_BATCH_MAX."""
    if not user_id:
        return []
    mine = [(src, at) for src, who, at in rows if who == user_id and at is not None]
    if not mine:
        return []
    newest = max(at for _, at in mine)
    from datetime import timedelta
    keep = sorted(src for src, at in mine if at >= newest - timedelta(minutes=UPLOAD_BATCH_MINUTES))
    return keep[:UPLOAD_BATCH_MAX]


def latest_uploads(collection: str, user_id: str | None, sources: set, db_engine) -> list:
    """The user's latest upload batch among the searchable documents (memento_documents.uploaded_by)."""
    if not user_id:
        return []
    import sqlalchemy as sa
    with db_engine().connect() as c:
        rows = c.execute(sa.text("select source, uploaded_by, added_at from memento_documents "
                                 "where collection = :c and uploaded_by = :u"), {"c": collection, "u": user_id}).fetchall()
    return [s for s in _upload_batch([tuple(r) for r in rows], user_id) if s in sources]


def resolve_reference(query: str, sources: list, domain: dict | None = None) -> str | None:
    """The document a question is about: a full name typed word for word first, then the one its own phrase names ("the X
    paper"), then the whole question. When the question describes a document that matches none, no single
    document is chosen (a method name in the question could otherwise resolve to a document's short name)."""
    exact = _exact_title(query, sources, domain)
    if exact:
        return exact
    phrases = document_phrases(query)
    for ph in phrases:
        src = _resolve_paper(ph, sources, domain)
        if src:
            return src
    return None if phrases else _resolve_paper(query, sources, domain)


def _exact_title(query: str, sources: list, domain: dict | None = None) -> str | None:
    """The document whose full name the question contains word for word (PaperResolver's title_exact), else None."""
    try:
        from memento.query.scope.paper_resolve import PaperResolver
        names = _doc_aliases(list(sources), domain)
        hit = PaperResolver((src, alias) for src, aliases in names.items() for alias in aliases).resolve(query)
        return hit.source if hit and hit.method == "title_exact" else None
    except Exception:
        return None


def _resolve_paper(query: str, sources: list, domain: dict | None = None) -> str | None:
    """The `source` of a document the question names, by title as well as by id (with a graph: its Paper titles; without
    one: _doc_aliases, then a guarded short name). Abstains when ambiguous; falls back to id matching when the title
    index is unreachable."""
    # Titles live in the domain's own graph (neo4j_uri). No graph for this domain, or the graph down: file-name/id
    # matching at once.
    uri = _domain_graph(domain)
    if not uri:
        # no graph: file name, first-page title and PDF title, then a guarded short name
        try:
            from memento.query.scope.paper_resolve import PaperResolver, resolve_short_name
            names = _doc_aliases(list(sources), domain)
            hit = PaperResolver((src, alias) for src, aliases in names.items() for alias in aliases).resolve(query)
            if hit:
                return hit.source
            short = resolve_short_name(query, names)
            if short:
                return short
        except Exception:
            pass
        return _detect_paper(query, sources)
    try:
        from memento.graph.store import Neo4jStore
        from memento.query.scope.paper_resolve import PaperResolver
        keep = set(sources)
        with Neo4jStore(uri=uri) as store:
            rows = [r.data() for r in store._run(
                "MATCH (p:Paper) RETURN p.source AS src, p.title AS title").records]
        hit = PaperResolver((r["src"], r["title"]) for r in rows
                            if r["src"] in keep).resolve(query)
        if hit:
            return hit.source
    except Exception:
        pass
    return _detect_paper(query, sources)


def entity_scoped_listing(query: str, paper: str | None, domain: dict | None = None,
                           enabled: bool = False) -> str | None:
    """Answer "list all <entity> papers" from the graph's MENTIONS index (the corpus listing below lists every document
    and would drop the constraint). None when it is not a set query or no entity resolves, so the caller falls through.
    A mention threshold drops documents that mention the entity only in passing.

    Graph feature: only with the GRAPH_RETRIEVAL Valve on and the domain's own, reachable graph."""
    if paper or not enabled:
        return None
    try:
        from memento.graph.query.query_intent import parse_intent
        intent = parse_intent(query)
        if not intent.is_set_query:
            return None
        uri = _domain_graph(domain)
        if not uri:
            return None
        from memento.graph.store import Neo4jStore
        from memento.graph.query.graph_aggregate import format_answer, set_query
        with Neo4jStore(uri=uri) as store:
            result = set_query(query, intent=intent, store=store, min_hits=3)
        if result.get("abstained") or not result.get("n_papers"):
            return None
        return format_answer(result, show=25)
    except Exception:
        return None


# "uploaded most recently", "the latest documents", "zuletzt hochgeladen": newest first, not alphabetical
_RECENT_RE = re.compile(r"\b(?:recent(?:ly)?|latest|newest|last\s+(?:uploaded|added)|zuletzt|neueste\w*|"
                        r"kürzlich)\b", re.I)


def wants_recent(query: str) -> bool:
    return bool(_RECENT_RE.search(query or ""))
