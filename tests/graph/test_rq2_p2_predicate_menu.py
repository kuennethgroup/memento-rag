# Graph-internal work links stay out of the predicate menu.
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.query.predicates import build_menu  # noqa: E402


class _Store:
    def relationship_types(self):
        return ["SAME_WORK", "SAME_AUTHOR", "CITES_PAPER"]

    def relationship_type_stats(self, rel):
        return ("Paper", "Paper", 50)

    def _run(self, *a, **k):
        raise RuntimeError("no examples in tests")


def test_same_work_and_same_author_are_not_query_predicates():
    names = {p.edge for p in build_menu(_Store()).predicates}
    assert "SAME_WORK" not in names and "SAME_AUTHOR" not in names
