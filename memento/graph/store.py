#!/usr/bin/env python3
"""GraphStore interface and its Neo4j implementation: memento/graph/build/build_graph.py writes through it,
memento/graph/query/graph_retrieval.py reads through it, and only GraphStore subclasses talk to Neo4j (a fake replaces it in
tests).

Nodes:
- Paper {id, title, authors, year, doi, source, …}: id = document stem = the Qdrant `source`; title/authors/year/doi
from memento/ingestion/extract_metadata.py.
- Section {id, name, level, paper_id, …}: id = f"{paper_id}::{section_path}".
- Chunk {id, type, page, page_end, text, source, img_path, …}: id = the deterministic content hash used as the Qdrant
point id, so graph nodes match Qdrant points; `type` mirrors the payload (text/image/table/equation).
- Entity {id, name, type, aliases, …}: id = the registry canonical_id after alias resolution, so one material or process
across documents and languages is one node.
- Reference {id, marker, raw, authors, title, year, doi, paper_id}: one per bibliography entry, id =
f"{paper_id}::ref::{marker}"; authors/title only from an optional LLM pass.
- Author {id, name}: id = graph.normalize(name), shared across documents.

Edges: (Paper)-[:HAS_SECTION]->(Section); (Section)-[:PARENT_SECTION]->(Section); (Chunk)-[:IN_SECTION]->(Section);
(Chunk)-[:MENTIONS]->(Entity); (Chunk)-[:REFERENCES]->(Chunk) (a text chunk to the figure/table/equation it cites, see
build_graph.py); (Entity)-[:RELATION {rel_type, paper_id, chunk_id, …}]->(Entity) (every extracted relation shares one
relationship type with its name in `rel_type`, since Cypher cannot parameterize type names);
(Paper)-[:AUTHORED_BY]->(Author); (Paper)-[:HAS_REFERENCE]->(Reference); (Chunk)-[:CITES]->(Reference) (an in-text
"[n]"); (Reference)-[:REFERS_TO {score}]->(Paper) and (Paper)-[:CITES_PAPER {via_ref}]->(Paper) (a reference matched by
title to another document of the corpus).

Cross-document resolution (build_graph.py --resolve) is a separate, re-runnable corpus pass: REFERS_TO / CITES_PAPER are
derived again from Reference.raw and Paper.title each time, and clear_paper's DETACH DELETE also removes such edges
pointing at a rebuilt document, so re-run --resolve after rebuilding any document.

Built for corpora of 1000+ documents: uniqueness constraints and indexes (ensure_schema()) and batched UNWIND writes
(BATCH_SIZE)."""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from memento.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

log = logging.getLogger(__name__)

# ── Schema constants ───────────────────────────────────────────────────────
NODE_LABELS = {"Paper", "Section", "Chunk", "Entity", "Reference", "Author"}

EDGE_TYPES = {
    "HAS_SECTION",
    "PARENT_SECTION",
    "IN_SECTION",
    "MENTIONS",
    "REFERENCES",
    "RELATION",
    "AUTHORED_BY",
    "HAS_REFERENCE",
    "CITES",
    "REFERS_TO",
    "CITES_PAPER",
    "CO_OCCURS",     # Entity ↔ Entity co-occurrence in a chunk, `weight` = chunks shared: the relation-free arm
                     # (LinearRAG's design), built by memento/graph/build/build_graph.py
    "SIMILAR",       # Entity ↔ Entity embedding similarity (synonymy), written by memento/graph/build/embed_entities.py
                     # --synonymy, traversed by PPR (an alias bridge for entities the registry does not know)
    "SAME_WORK",     # Paper ↔ Paper: one work (duplicate copy / chapter / defense / title duplicate), `kind`
    "SAME_AUTHOR",   # Paper ↔ Paper: same doctoral author (person folder), `student`
}

# Rows per UNWIND transaction. Keeps writes bounded for the 1000+ doc
# large corpus without per-row round trips.
BATCH_SIZE = 1000


def _chunked(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ── Data types ────────────────────────────────────────────────────────────
@dataclass
class GraphNode:
    id: str       # unique within `label` — see schema docstring above
    label: str    # one of NODE_LABELS
    props: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    start_id: str
    start_label: str
    end_id: str
    end_label: str
    type: str     # one of EDGE_TYPES
    props: dict = field(default_factory=dict)


# ── Interface ─────────────────────────────────────────────────────────────
class GraphStore(ABC):
    """Backend-agnostic interface for the Memento knowledge graph."""

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create constraints/indexes. Idempotent — safe to call every run."""

    @abstractmethod
    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> int:
        """MERGE nodes by (label, id), merging in `props`. Returns count."""

    @abstractmethod
    def upsert_edges(self, edges: Iterable[GraphEdge]) -> int:
        """MERGE edges by (start, type, end), merging in `props`. Returns the
        number of edges actually written; an edge whose endpoint node is
        missing is dropped and counted in `self.last_dropped`."""

    @abstractmethod
    def clear_paper(self, paper_id: str) -> None:
        """Delete a paper's Section/Chunk subgraph (for re-ingestion).

        Entity nodes are shared across papers and are not deleted outright:
        RELATION edges provenanced to this paper (`props.paper_id ==
        paper_id`) are removed, and any Entity left with no remaining edges
        afterwards is removed too.
        """

    @abstractmethod
    def match_entities(self, names: list[str]) -> list[dict]:
        """Resolve free-text names to Entity nodes via exact/alias match
        (case-insensitive). Returns node property dicts (including `id`)."""

    @abstractmethod
    def subgraph_chunks(self, seed_entity_ids: list[str], max_hops: int = 2,
                         limit: int = 50, max_df: Optional[int] = None) -> list[dict]:
        """Traverse outward from seed Entity ids along RELATION edges (up to
        `max_hops`), then collect Chunk nodes that MENTION any visited
        entity. Returns dicts with `chunk_id`, `source`, `score`, `via`
        (up to 5 contributing entity ids), ordered by score desc, where
        score = sum over the distinct reached entities the chunk mentions of
        ln(N_chunks / df(entity)). Entities with df > `max_df` are skipped
        (hub cap; None = no cap) — used by graph_retrieval.py as extra
        dense-search candidates."""

    @abstractmethod
    def get_neighbors(self, node_ids: list[str], rel_types: Optional[list[str]] = None,
                       max_hops: int = 1, limit: int = 200) -> list[dict]:
        """Generic neighborhood lookup (inspection/debugging and Phase 2
        cross-modal traversal). Returns dicts with `node` (property map plus
        `_label`) and `hops`."""

    @abstractmethod
    def paper_exists(self, paper_id: str) -> bool:
        """Cheap indexed lookup — True iff a Paper node with this id already
        has a graph built. Used by memento/graph/build/build_graph.py's --all loop to
        skip already-built papers on incremental (re-)runs over a growing
        corpus, without the cost of `all_papers()`'s full scan."""

    @abstractmethod
    def all_papers(self) -> list[dict]:
        """All Paper nodes, as dicts with `id`, `title` (may be None),
        `year` (may be None). Used by Phase 3 cross-paper resolution to
        build the fuzzy title-match candidate set."""

    @abstractmethod
    def all_references(self) -> list[dict]:
        """All Reference nodes, as dicts with `id`, `paper_id` (the citing
        paper, via HAS_REFERENCE), `raw`, `year` (may be None). Used by
        Phase 3 cross-paper resolution."""

    @abstractmethod
    def expand_via_references(self, chunk_ids: list[str], limit: int = 50) -> list[dict]:
        """Phase 4: for each given Chunk id, traverse REFERENCES edges in
        either direction and return connected Chunk dicts with
        `chunk_id`, `source`, `via_chunk`.

        This surfaces the Phase 1 exhaustive figure/table/equation↔text
        links: a retrieved text chunk pulls in the figure it discusses, and
        a retrieved figure chunk pulls in all the text paragraphs that
        reference it. Only chunks not already in `chunk_ids` are returned,
        each once, ordered by the position in `chunk_ids` (dense rank) of the
        best-ranked chunk linking to it; `limit` applies after ordering."""

    @abstractmethod
    def cross_paper_chunks(self, paper_ids: list[str], entity_ids: list[str],
                            limit: int = 50) -> list[dict]:
        """Hop 1 step via CITES_PAPER from the given papers (in
        either direction), then collect Chunk nodes in those neighbor papers
        that MENTION any of the given entity ids. Returns dicts with
        `chunk_id`, `source`, `via_paper`.

        Bridges the cross-paper citation graph into retrieval: a query that
        reaches paper Y also surfaces relevant chunks from the papers Y cites
        (or that cite Y), provided they share entity vocabulary with the query. Rows also carry
        `score` (same IDF scoring as subgraph_chunks) and are ordered by it.
        Returns `[]` if either argument is empty."""

    # ── PPR / dense-seeding support (optional — concrete raising defaults so
    #    existing GraphStore fakes/backends without PPR keep working; the
    #    callers in memento/graph/query/graph_retrieval.py are best-effort and treat
    #    NotImplementedError like any other failure) ────────────────────────
    def entity_edges(self, types: str = "RELATION|SIMILAR",
                     min_weight: int = 0) -> list[tuple[str, str]]:
        """Entity–Entity edge pairs, each pair once. The PPR traversal's graph.

        `types` selects the axis-C arm (LLM relations vs CO_OCCURS);
        `min_weight` thresholds CO_OCCURS by chunks-shared."""
        raise NotImplementedError

    def chunk_mentions_for(self, entity_ids: list[str]) -> list[dict]:
        """(chunk_id, source, entity_id) rows for every Chunk that MENTIONS
        any of the given entities. Used to score chunks by PPR entity mass."""
        raise NotImplementedError

    def all_entities(self) -> list[dict]:
        """All Entity nodes as dicts with `id`, `name`, `type`, `aliases`.
        Used by memento/graph/build/embed_entities.py to build the entity-embedding
        collection for dense seeding."""
        raise NotImplementedError

    def chunk_entities(self, chunk_ids: list[str], max_df: Optional[int] = None) -> dict[str, float]:
        """Entities MENTIONed by the given chunks, weighted ln(N_chunks / df);
        entities with df > `max_df` are skipped. Seeds the multi-hop
        first-hop stream. Optional — the base class raises."""
        raise NotImplementedError

    def citing_neighbors(self, paper_ids: list[str]) -> list[str]:
        """Paper ids linked to any of `paper_ids` by CITES_PAPER (either
        direction), excluding the inputs. The structural (entity-free)
        cross-paper stream ranks these papers' chunks by query similarity in
        memento/graph/query/graph_retrieval.py. Optional — the base class raises."""
        raise NotImplementedError

    def set_meta(self, key: str, value: str) -> None:
        """Store a graph-level fact, e.g. `build_id` (changed by every write
        step; memento/graph/ppr_cache.py keys its matrix cache on it). Optional."""
        raise NotImplementedError

    def get_meta(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def same_work_neighbors(self, paper_ids: list[str]) -> list[str]:
        """Papers linked to `paper_ids` by SAME_WORK (G1 structural stream). Optional."""
        raise NotImplementedError

    def relations_among(self, entity_ids: list[str], limit: int = 10) -> list[dict]:
        raise NotImplementedError

    def entities_by_ids(self, ids: list[str]) -> dict[str, str]:
        raise NotImplementedError

    def existing_entity_ids(self, ids: list[str]) -> set[str]:
        raise NotImplementedError

    def papers_for_entities(self, eids: list[str], need: int = 1) -> list[dict]:
        raise NotImplementedError

    def mark_changed(self) -> None:
        """Give the graph a fresh `build_id` after a write, so caches keyed on
        it (memento/graph/ppr_cache.py) never serve a stale matrix. Optional."""
        raise NotImplementedError

    identity = ""   # which database this store points at; scopes on-disk caches

    def all_authors(self) -> list[dict]:
        """All Author nodes as dicts with `id`, `name`. Used by
        memento/graph/query/predicates.py's author resolver. Optional — the base class
        raises; a backend without Author nodes just can't offer that
        predicate (memento/graph/query/predicates.py's menu build degrades to
        skipping it, same as any other best-effort schema read)."""
        raise NotImplementedError

    def relationship_types(self) -> list[str]:
        """Relationship types live in the graph right now. Optional — see
        all_authors() above; a backend that can't introspect its own schema
        just yields an empty structural-predicate menu, which degrades to
        the pre-existing mentions/temporal-only behaviour."""
        raise NotImplementedError

    def relationship_type_stats(self, rel_type: str) -> Optional[tuple[str, str, int]]:
        """(start_label, end_label, count) for one relationship type, or
        None. Optional, see relationship_types() above."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── Neo4j implementation ─────────────────────────────────────────────────
class Neo4jStore(GraphStore):
    """Neo4j-backed GraphStore.

    `driver` may be injected (e.g. a fake in unit tests) to avoid requiring
    a live Neo4j instance; if omitted, a real `neo4j.GraphDatabase` driver is
    created from `uri`/`user`/`password` (defaults from memento.config.settings /
    config/.env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).
    """

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER,
                 password: str = NEO4J_PASSWORD, database: str = "neo4j",
                 driver=None):
        if driver is not None:
            self._driver = driver
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self.identity = f"{uri}/{database}"
        self.last_dropped = 0   # edges dropped by the last upsert_edges call

    def close(self) -> None:
        self._driver.close()

    def _run(self, query: str, **params):
        return self._driver.execute_query(query, parameters_=params, database_=self._database)

    # ── schema ────────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        for stmt in (
            "CREATE CONSTRAINT memento_paper_id  IF NOT EXISTS FOR (n:Paper)   REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT memento_section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT memento_chunk_id   IF NOT EXISTS FOR (n:Chunk)   REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT memento_entity_id  IF NOT EXISTS FOR (n:Entity)  REQUIRE n.id IS UNIQUE",
            "CREATE INDEX memento_entity_name     IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX memento_entity_type     IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX memento_chunk_source    IF NOT EXISTS FOR (n:Chunk)  ON (n.source)",
            "CREATE INDEX memento_relation_type   IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.rel_type)",
            "CREATE CONSTRAINT memento_reference_id IF NOT EXISTS FOR (n:Reference) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT memento_author_id    IF NOT EXISTS FOR (n:Author)    REQUIRE n.id IS UNIQUE",
            "CREATE INDEX memento_author_name       IF NOT EXISTS FOR (n:Author)    ON (n.name)",
        ):
            self._run(stmt)

    # ── writes ────────────────────────────────────────────────────────────
    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> int:
        nodes = list(nodes)
        by_label: dict[str, list[dict]] = {}
        for n in nodes:
            if n.label not in NODE_LABELS:
                raise ValueError(f"Unknown node label: {n.label!r}")
            by_label.setdefault(n.label, []).append({"id": n.id, "props": n.props})

        total = 0
        for label, rows in by_label.items():
            query = (
                f"UNWIND $rows AS row "
                f"MERGE (n:{label} {{id: row.id}}) "
                f"SET n += row.props"
            )
            for batch in _chunked(rows, BATCH_SIZE):
                self._run(query, rows=batch)
                total += len(batch)
        return total

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> int:
        edges = list(edges)
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for e in edges:
            if e.type not in EDGE_TYPES:
                raise ValueError(f"Unknown edge type: {e.type!r}")
            if e.start_label not in NODE_LABELS or e.end_label not in NODE_LABELS:
                raise ValueError(
                    f"Unknown node label in edge: {e.start_label!r}/{e.end_label!r}")
            groups.setdefault((e.type, e.start_label, e.end_label), []).append(
                {"start_id": e.start_id, "end_id": e.end_id, "props": e.props})

        # MATCH drops a row silently when an endpoint node is missing, so count
        # what was actually written instead of what was requested.
        written = 0
        for (etype, start_label, end_label), rows in groups.items():
            # RELATION is keyed per statement (rel_type = free-text description, chunk_id): merging per entity pair
            # would overwrite earlier statements text and provenance. PPR is unaffected: entity_edges() returns DISTINCT
            # pairs.
            merge = ("MERGE (a)-[r:RELATION {rel_type: row.props.rel_type, chunk_id: row.props.chunk_id}]->(b) "
                     if etype == "RELATION" else f"MERGE (a)-[r:{etype}]->(b) ")
            query = (
                f"UNWIND $rows AS row "
                f"MATCH (a:{start_label} {{id: row.start_id}}) "
                f"MATCH (b:{end_label} {{id: row.end_id}}) "
                + merge +
                f"SET r += row.props "
                f"RETURN count(r) AS n"
            )
            for batch in _chunked(rows, BATCH_SIZE):
                written += self._run(query, rows=batch).records[0]["n"]
        self.last_dropped = len(edges) - written
        if written:
            self.mark_changed()
        if self.last_dropped:
            log.warning("upsert_edges: %d of %d edges dropped (missing endpoint node)",
                        self.last_dropped, len(edges))
        return written

    # ── deletes ───────────────────────────────────────────────────────────
    def clear_paper(self, paper_id: str, orphan_cleanup: bool = True) -> None:
        # Detach-delete this paper's own Section/Chunk/Reference subgraph.
        # Reference ids are paper-scoped (f"{paper_id}::ref::{marker}", not
        # shared like Entity/Author), so they're safe to delete outright —
        # any (Reference)-[:REFERS_TO]->(Paper) edge from Phase 3 goes with it.
        self._run(
            "MATCH (p:Paper {id: $paper_id}) "
            "OPTIONAL MATCH (p)-[:HAS_SECTION]->(s:Section) "
            "OPTIONAL MATCH (c:Chunk)-[:IN_SECTION]->(s) "
            "OPTIONAL MATCH (p)-[:HAS_REFERENCE]->(ref:Reference) "
            "DETACH DELETE p, s, c, ref",
            paper_id=paper_id,
        )
        # Remove entity-relation edges provenanced to this paper...
        self._run(
            "MATCH ()-[r:RELATION {paper_id: $paper_id}]->() DELETE r",
            paper_id=paper_id,
        )
        # ...and any Entity/Author left with no remaining edges (orphaned by
        # the paper's removal) — keeps the graph from accumulating dead nodes.
        # Parallel build workers must NOT sweep orphans per paper: another
        # worker's freshly upserted Entity nodes have no edges yet and would be
        # deleted, silently dropping that worker's MENTIONS edges. build_graph
        # passes orphan_cleanup=False and calls delete_orphans() once at the end.
        self.mark_changed()
        if orphan_cleanup:
            self.delete_orphans()

    def delete_orphans(self) -> None:
        """Delete Entity/Author nodes left without any relationship. A
        concurrent writer attaching an edge makes Neo4j refuse the delete
        (ConstraintError); retrying re-evaluates "no relationships"."""
        from neo4j.exceptions import ConstraintError, TransientError
        for label in ("Entity", "Author"):
            for attempt in range(5):
                try:
                    self._run(f"MATCH (e:{label}) WHERE NOT (e)--() DELETE e")
                    break
                except (ConstraintError, TransientError):
                    if attempt == 4:
                        raise
        self.mark_changed()

    # ── reads ─────────────────────────────────────────────────────────────
    def match_entities(self, names: list[str]) -> list[dict]:
        if not names:
            return []
        result = self._run(
            "UNWIND $names AS name "
            "MATCH (e:Entity) "
            "WHERE toLower(e.name) = toLower(name) "
            "   OR toLower(name) IN [a IN coalesce(e.aliases, []) | toLower(a)] "
            "RETURN DISTINCT e{.*} AS entity",
            names=names,
        )
        return [record["entity"] for record in result.records]

    def _n_chunks(self) -> int:
        return self._run("MATCH (c:Chunk) RETURN count(c) AS n").records[0]["n"]

    def subgraph_chunks(self, seed_entity_ids: list[str], max_hops: int = 2,
                         limit: int = 50, max_df: Optional[int] = None) -> list[dict]:
        if not seed_entity_ids:
            return []
        max_hops = max(0, min(int(max_hops), 5))  # bound the traversal depth
        query = (
            "UNWIND $seed_ids AS sid MATCH (e:Entity {id: sid}) "
            f"OPTIONAL MATCH (e)-[:RELATION*0..{max_hops}]-(r:Entity) "
            "WITH collect(DISTINCT e) + collect(DISTINCT r) AS ents "
            "UNWIND ents AS ent WITH DISTINCT ent WHERE ent IS NOT NULL "
            "WITH ent, COUNT { (ent)<-[:MENTIONS]-(:Chunk) } AS df "
            "WHERE df > 0 AND ($max_df IS NULL OR df <= $max_df) "
            "MATCH (c:Chunk)-[:MENTIONS]->(ent) "
            # IDF-weighted: a chunk mentioning a rare reached entity outranks
            # one mentioning only a hub (e.g. "temperature")
            "WITH c, sum(log(toFloat($n) / df)) AS score, collect(ent.id)[..5] AS via "
            "RETURN c.id AS chunk_id, c.source AS source, score, via "
            "ORDER BY score DESC, chunk_id LIMIT $limit"
        )
        result = self._run(query, seed_ids=list(seed_entity_ids), max_df=max_df,
                           n=max(self._n_chunks(), 1), limit=limit)
        return [dict(r) for r in result.records]

    def get_neighbors(self, node_ids: list[str], rel_types: Optional[list[str]] = None,
                       max_hops: int = 1, limit: int = 200) -> list[dict]:
        if not node_ids:
            return []
        max_hops = max(1, min(int(max_hops), 5))  # bound the traversal depth

        rel_filter = ""
        params: dict = {"node_ids": node_ids, "limit": limit}
        if rel_types:
            rel_filter = (
                "WHERE all(rel IN r WHERE type(rel) IN $rel_types "
                "OR rel.rel_type IN $rel_types) "
            )
            params["rel_types"] = rel_types

        query = (
            "UNWIND $node_ids AS nid "
            "MATCH (n {id: nid}) "
            f"MATCH p = (n)-[r*1..{max_hops}]-(m) "
            f"{rel_filter}"
            "RETURN DISTINCT m{.*, _label: labels(m)[0]} AS node, length(p) AS hops "
            "ORDER BY hops ASC "
            "LIMIT $limit"
        )
        result = self._run(query, **params)
        return [{"node": r["node"], "hops": r["hops"]} for r in result.records]

    def paper_exists(self, paper_id: str) -> bool:
        result = self._run(
            "MATCH (p:Paper {id: $paper_id}) RETURN true AS exists LIMIT 1",
            paper_id=paper_id,
        )
        return bool(result.records)

    # ── Phase 3: cross-paper resolution reads ───────────────────────────────
    def all_papers(self) -> list[dict]:
        result = self._run("MATCH (p:Paper) RETURN p.id AS id, p.title AS title, p.year AS year")
        return [dict(r) for r in result.records]

    def all_references(self) -> list[dict]:
        result = self._run(
            "MATCH (p:Paper)-[:HAS_REFERENCE]->(ref:Reference) "
            "RETURN ref.id AS id, p.id AS paper_id, ref.raw AS raw, ref.year AS year"
        )
        return [dict(r) for r in result.records]

    # ── Phase 4: multi-hop retrieval expansion ───────────────────────────────
    def expand_via_references(self, chunk_ids: list[str], limit: int = 50) -> list[dict]:
        if not chunk_ids:
            return []
        # `chunk_ids` arrive in dense rank order: each linked chunk takes the
        # best rank of the chunks that reference it, and the limit is applied
        # after ordering (not to an arbitrary DISTINCT slice).
        result = self._run(
            "UNWIND range(0, size($chunk_ids) - 1) AS i "
            "MATCH (a:Chunk {id: $chunk_ids[i]})-[:REFERENCES]-(b:Chunk) "
            "WHERE NOT b.id IN $chunk_ids "
            "WITH b, min(i) AS rank "
            "RETURN b.id AS chunk_id, b.source AS source, $chunk_ids[rank] AS via_chunk "
            "ORDER BY rank, chunk_id LIMIT $limit",
            chunk_ids=chunk_ids,
            limit=limit,
        )
        return [dict(r) for r in result.records]

    # ── PPR / dense-seeding support ──────────────────────────────────────────
    def entity_edges(self, types: str = "RELATION|SIMILAR",
                     min_weight: int = 0) -> list[tuple[str, str]]:
        """Entity-Entity pairs for PPR, once per undirected pair.

        `types` selects the axis-C arm: "RELATION|SIMILAR" is the LLM-extracted
        relation graph, "CO_OCCURS" the relation-free co-occurrence graph
        (LinearRAG's design). They are kept separable because measured relation
        precision is ~0.4 while co-occurrence has no extraction error by
        construction -- and PPR reads only adjacency, never `rel_type`, so the
        comparison isolates whether the labels buy anything.

        `min_weight` applies to CO_OCCURS, whose `weight` is the number of
        chunks a pair shares; ~90% of pairs co-occur exactly once.
        """
        # elementId(a) < elementId(b) keeps each undirected pair to one row
        # (the undirected -[..]- match would otherwise return both directions).
        where = "elementId(a) < elementId(b)"
        if min_weight:
            where += " AND coalesce(r.weight, 1) >= $min_weight"
        result = self._run(
            f"MATCH (a:Entity)-[r:{types}]-(b:Entity) "
            f"WHERE {where} "
            "RETURN DISTINCT a.id AS a, b.id AS b",
            min_weight=min_weight,
        )
        return [(r["a"], r["b"]) for r in result.records]

    def chunk_mentions_for(self, entity_ids: list[str]) -> list[dict]:
        if not entity_ids:
            return []
        result = self._run(
            "UNWIND $ids AS eid "
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: eid}) "
            "RETURN c.id AS chunk_id, c.source AS source, eid AS entity_id",
            ids=entity_ids,
        )
        return [dict(r) for r in result.records]

    def all_entities(self) -> list[dict]:
        result = self._run(
            "MATCH (e:Entity) "
            "RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases"
        )
        return [dict(r) for r in result.records]

    def all_authors(self) -> list[dict]:
        """All Author nodes, `id`/`name`. Used by memento/graph/query/predicates.py's
        author-name resolver (memento/graph/query/predicates.py's structural
        predicate registry)."""
        result = self._run("MATCH (a:Author) RETURN a.id AS id, a.name AS name")
        return [dict(r) for r in result.records]

    # ── Schema introspection (memento/graph/query/predicates.py) ──────────────────
    # Live-schema reads, not a hand-written edge-type list: predicates.py
    # derives its query-time menu from what the graph actually contains, so a
    # new edge type added to EDGE_TYPES/build_graph.py becomes queryable
    # without an edit here or there. neo4j:5-community ships no APOC plugin
    # (docker-compose.yml has no NEO4J_PLUGINS), so this uses only the
    # built-in `db.*` procedures, not `apoc.meta.schema()`.
    def relationship_types(self) -> list[str]:
        """Relationship types actually present in the graph right now."""
        result = self._run("CALL db.relationshipTypes() YIELD relationshipType "
                           "RETURN relationshipType")
        return [r["relationshipType"] for r in result.records]

    def relationship_type_stats(self, rel_type: str) -> Optional[tuple[str, str, int]]:
        """(start_label, end_label, count) for one relationship type, or None
        if it has no live instances. One full-scan count per call -- callers
        should only call this for types not already excluded by name (see
        predicates._OWNED_BY_MENTIONS_PATH), so this never runs against
        MENTIONS/RELATION's hundred-thousand-edge scale."""
        result = self._run(
            f"MATCH (a)-[r:`{rel_type}`]->(b) "
            "RETURN labels(a)[0] AS s, labels(b)[0] AS e, count(*) AS n"
        )
        if not result.records:
            return None
        row = result.records[0]
        return row["s"], row["e"], row["n"]

    def cross_paper_chunks(self, paper_ids: list[str], entity_ids: list[str],
                            limit: int = 50) -> list[dict]:
        if not paper_ids or not entity_ids:
            return []
        result = self._run(
            "UNWIND $paper_ids AS pid "
            "MATCH (:Paper {id: pid})-[:CITES_PAPER]-(nb:Paper) "
            "WHERE NOT nb.id IN $paper_ids WITH DISTINCT nb "
            "MATCH (c:Chunk {source: nb.id})-[:MENTIONS]->(e:Entity) WHERE e.id IN $entity_ids "
            "WITH c, nb, e, COUNT { (e)<-[:MENTIONS]-(:Chunk) } AS df "
            "WITH c, nb, sum(log(toFloat($n) / df)) AS score "
            "RETURN c.id AS chunk_id, c.source AS source, nb.id AS via_paper, score "
            "ORDER BY score DESC, chunk_id LIMIT $limit",
            paper_ids=paper_ids, entity_ids=entity_ids,
            n=max(self._n_chunks(), 1), limit=limit,
        )
        return [dict(r) for r in result.records]

    def chunk_entities(self, chunk_ids: list[str], max_df: Optional[int] = None) -> dict[str, float]:
        if not chunk_ids:
            return {}
        res = self._run(
            "UNWIND $ids AS cid MATCH (:Chunk {id: cid})-[:MENTIONS]->(e:Entity) "
            "WITH DISTINCT e, COUNT { (e)<-[:MENTIONS]-(:Chunk) } AS df "
            "WHERE df > 0 AND ($max_df IS NULL OR df <= $max_df) "
            "RETURN e.id AS id, log(toFloat($n) / df) AS w",
            ids=list(chunk_ids), max_df=max_df, n=max(self._n_chunks(), 1))
        return {r["id"]: float(r["w"]) for r in res.records}

    def set_meta(self, key: str, value: str) -> None:
        self._run("MERGE (m:Meta {id: $k}) SET m.value = $v", k=key, v=value)

    def mark_changed(self) -> None:
        self.set_meta("build_id", f"write:{time.time_ns()}:{uuid.uuid4().hex[:8]}")

    def get_meta(self, key: str) -> Optional[str]:
        rec = self._run("MATCH (m:Meta {id: $k}) RETURN m.value AS v", k=key).records
        return rec[0]["v"] if rec else None

    def same_work_neighbors(self, paper_ids: list[str]) -> list[str]:
        if not paper_ids:
            return []
        res = self._run("UNWIND $pids AS pid MATCH (:Paper {id: pid})-[:SAME_WORK]-(nb:Paper) "
                        "WHERE NOT nb.id IN $pids RETURN DISTINCT nb.id AS id ORDER BY id", pids=list(paper_ids))
        return [r["id"] for r in res.records]

    def relations_among(self, entity_ids: list[str], limit: int = 10) -> list[dict]:
        if not entity_ids:
            return []
        res = self._run("MATCH (a:Entity)-[r:RELATION]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids "
                        "RETURN a.name AS src, b.name AS tgt, r.rel_type AS description, r.paper_id AS paper_id "
                        "ORDER BY src, tgt, description LIMIT $limit", ids=list(entity_ids), limit=limit)
        return [dict(r) for r in res.records]

    def entities_by_ids(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        res = self._run("UNWIND $ids AS i MATCH (e:Entity {id: i}) RETURN e.id AS id, e.name AS name", ids=list(ids))
        return {r["id"]: r["name"] for r in res.records}

    def existing_entity_ids(self, ids: list[str]) -> set[str]:
        return set(self.entities_by_ids(ids))

    def papers_for_entities(self, eids: list[str], need: int = 1) -> list[dict]:
        """Documents mentioning at least `need` of the seed entities; `hits` = distinct
        chunks mentioning any seed (the corpus-level lookup of G2/G3)."""
        if not eids:
            return []
        res = self._run("UNWIND $eids AS eid MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: eid}) "
                        "WITH c.source AS src, collect(DISTINCT eid) AS seen, count(DISTINCT c) AS hits "
                        "WHERE size(seen) >= $need RETURN src, hits ORDER BY hits DESC, src",
                        eids=list(eids), need=need)
        return [dict(r) for r in res.records]

    def citing_neighbors(self, paper_ids: list[str]) -> list[str]:
        if not paper_ids:
            return []
        result = self._run(
            "UNWIND $pids AS pid MATCH (:Paper {id: pid})-[:CITES_PAPER]-(nb:Paper) "
            "WHERE NOT nb.id IN $pids RETURN DISTINCT nb.id AS id",
            pids=paper_ids,
        )
        return [r["id"] for r in result.records]


def get_store() -> Neo4jStore:
    """Convenience factory for the current domain's own Neo4j instance.

    Refuses domains without a `neo4j_uri` in their YAML: they would fall back
    to the .env default (bolt://localhost:7687), which holds the old
    229-paper graph that must stay untouched, and graphs of different corpora
    would mix. Tests and one-off tools pass an explicit uri to Neo4jStore."""
    from memento.config.settings import DOMAIN
    if not DOMAIN.get("neo4j_uri"):
        raise RuntimeError(
            f"domain {DOMAIN.get('name')!r} has no neo4j_uri in config/domains/*.yaml; "
            "add one (its own Neo4j instance) before building or querying a graph")
    return Neo4jStore()
