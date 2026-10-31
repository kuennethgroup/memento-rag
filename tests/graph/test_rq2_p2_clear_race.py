# tests/graph/test_rq2_p2_clear_race.py — clear_paper must survive a concurrent writer
from neo4j.exceptions import ConstraintError

from memento.graph.store import Neo4jStore


class FlakyDriver:
    """execute_query succeeds, except the first orphan-entity delete raises the
    ConstraintError Neo4j gives when another transaction attached a relationship."""
    def __init__(self):
        self.queries, self.failed = [], False

    def execute_query(self, query, parameters_=None, database_=None):
        self.queries.append(query)
        if "WHERE NOT (e)--() DELETE e" in query and not self.failed:
            self.failed = True
            raise ConstraintError("Cannot delete node<794>, because it still has relationships.")
        return type("R", (), {"records": []})()

    def close(self):
        pass


def test_clear_paper_retries_orphan_cleanup_after_race():
    d = FlakyDriver()
    Neo4jStore(driver=d).clear_paper("P")
    assert sum("MATCH (e:Entity) WHERE NOT (e)--() DELETE e" in q for q in d.queries) == 2   # failed once, retried
    assert any("MATCH (e:Author) WHERE NOT (e)--() DELETE e" in q for q in d.queries)


class RecordingDriver:
    def __init__(self):
        self.queries = []

    def execute_query(self, query, parameters_=None, database_=None):
        self.queries.append(query)
        return type("R", (), {"records": []})()

    def close(self):
        pass


def test_clear_paper_can_skip_orphan_cleanup_for_parallel_builds():
    d = RecordingDriver()
    Neo4jStore(driver=d).clear_paper("P", orphan_cleanup=False)
    assert not any("WHERE NOT (e)--() DELETE e" in q for q in d.queries)
    assert any("DETACH DELETE p, s, c, ref" in q for q in d.queries)


def test_delete_orphans_sweeps_entities_and_authors():
    d = RecordingDriver()
    Neo4jStore(driver=d).delete_orphans()
    assert any("MATCH (e:Entity) WHERE NOT (e)--() DELETE e" in q for q in d.queries)
    assert any("MATCH (e:Author) WHERE NOT (e)--() DELETE e" in q for q in d.queries)
