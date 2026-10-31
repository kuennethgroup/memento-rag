# tests/graph/test_rq2_p2_citations.py
import random
import sys

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.citations import resolve_citations, resolve_citations_all_pairs  # noqa: E402

WORDS = ("alumina zirconia sintering flash spark plasma grain growth toughness hardness boride carbide "
         "nitride thermal conductivity oxidation coating microstructure densification kinetics").split()


def _corpus(seed=0, n_papers=60, n_refs=400):
    rng = random.Random(seed)
    papers = [{"id": f"p{i}", "title": " ".join(rng.sample(WORDS, rng.randint(3, 8))),
               "year": rng.randint(2000, 2024)} for i in range(n_papers)]
    papers.append({"id": "short", "title": "Introduction", "year": 2010})
    refs = []
    for j in range(n_refs):
        citing = rng.choice(papers)["id"]
        if rng.random() < 0.4:
            cited = rng.choice(papers)
            raw = f"A. Author, {cited['title']}, J. Eur. Ceram. Soc. 12 ({cited['year']}) 1-10."
            year = cited["year"]
        else:
            raw = "B. Other, " + " ".join(rng.sample(WORDS, 6)) + ", Acta Mater. (2001)."
            year = 2001
        refs.append({"id": f"r{j}", "paper_id": citing, "raw": raw, "year": year})
    return papers, refs


def test_indexed_resolver_equals_all_pairs():
    for seed in range(3):
        papers, refs = _corpus(seed)
        fast = resolve_citations(papers, refs)
        slow = resolve_citations_all_pairs(papers, refs)
        assert sorted((m["ref_id"], m["cited"]) for m in fast["matches"]) == \
               sorted((m["ref_id"], m["cited"]) for m in slow["matches"])
        assert fast["n_nondistinctive"] == slow["n_nondistinctive"]
        assert all(m["cited"] != "short" for m in fast["matches"])
        assert all(m["cited"] != m["citing"] for m in fast["matches"])
