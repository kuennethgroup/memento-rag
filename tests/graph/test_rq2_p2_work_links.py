# tests/graph/test_rq2_p2_work_links.py
import sys

sys.argv = [sys.argv[0], "--domain", "polymer-engineering"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.build_graph import build_paper  # noqa: E402
from memento.graph.build.work_links import title_duplicate_edges, work_link_edges  # noqa: E402

REG = {
    "Diss_A": {"type": "dissertation", "student": "A, Anna"},
    "Diss_A_druck": {"type": "dissertation", "student": "A, Anna", "duplicate_of": "Diss_A"},
    "Vortrag_A": {"type": "defense", "student": "A, Anna", "parent_diss": "Diss_A"},
    "03_Kap_A": {"type": "chapter", "student": "A, Anna", "parent": "Diss_A"},
    "Paper_A": {"type": "document", "student": "A, Anna"},
    "Diss_B": {"type": "dissertation", "student": "B, Bert"},
}


def _pairs(edges, t):
    return {(e.start_id, e.end_id, e.props.get("kind")) for e in edges if e.type == t}


def test_work_link_edges_from_registry():
    e = work_link_edges(REG, set(REG))
    assert _pairs(e, "SAME_WORK") == {("Diss_A", "Diss_A_druck", "duplicate"),
                                      ("Diss_A", "Vortrag_A", "defense"),
                                      ("03_Kap_A", "Diss_A", "chapter")}
    same_author = {tuple(sorted((x.start_id, x.end_id))) for x in e if x.type == "SAME_AUTHOR"}
    assert ("Diss_A", "Paper_A") in same_author and ("Diss_A", "Vortrag_A") not in same_author
    assert all("Diss_B" not in p for p in same_author)


def test_title_duplicates_need_long_near_identical_titles():
    papers = [{"id": "p1", "title": "Flash sintering of 3YSZ ceramics under AC fields", "year": 2019},
              {"id": "p2", "title": "Flash sintering of 3YSZ ceramics under AC fields.", "year": 2020},
              {"id": "p3", "title": "Introduction", "year": 2019},
              {"id": "p4", "title": "Introduction", "year": 2019}]
    assert {(x.start_id, x.end_id) for x in title_duplicate_edges(papers)} == {("p1", "p2")}


class FakeStore:
    def __init__(self):
        self.nodes, self.edges = [], []

    def clear_paper(self, pid):
        pass

    def upsert_nodes(self, nodes):
        self.nodes += list(nodes)
        return len(self.nodes)

    def upsert_edges(self, edges):
        self.edges += list(edges)
        return len(self.edges)


ITEMS = [{"type": "text", "text": "Alumina and zirconia composites were sintered by SPS.", "page_idx": 0}]


def test_every_parsed_document_gets_entities(tmp_path):
    # the graph covers exactly the files that were parsed and embedded; skipping belongs to the parsing phase, not to
    # graph building (no registry-based skipping)
    for info in ({"type": "dissertation", "duplicate_of": "Diss_A"}, {"type": "image_artifact"},
                 {"type": "chapter", "parent": "Diss_A", "copy_of": "Diss_A"}, {"type": "dissertation"}, None):
        s = FakeStore()
        summary = build_paper("X", store=s, items=ITEMS, paper_dir=tmp_path, use_llm=False, doc_info=info)
        assert summary["n_chunks"] == 1 and [e for e in s.edges if e.type == "MENTIONS"], info


def test_chapter_without_text_overlap_gets_no_same_work_link():
    reg = {"Diss_J": {"type": "dissertation", "student": "J, Jo"},
           "04_Schaum": {"type": "chapter", "student": "J, Jo", "parent": "Diss_J", "overlap_with_parent": 0.0},
           "03_Draft": {"type": "chapter", "student": "J, Jo", "parent": "Diss_J", "overlap_with_parent": 0.4}}
    e = work_link_edges(reg, set(reg))
    assert _pairs(e, "SAME_WORK") == {("03_Draft", "Diss_J", "chapter")}
    assert ("04_Schaum", "Diss_J") in {(x.start_id, x.end_id) for x in e if x.type == "SAME_AUTHOR"}


def test_title_that_is_a_subset_of_a_longer_title_is_not_a_duplicate():
    # token-set ratio scores subset titles 1.0
    papers = [{"id": "p1", "title": "Flash sintering of 3YSZ ceramics under AC fields", "year": 2019},
              {"id": "p2", "title": "Flash sintering of 3YSZ ceramics under AC fields: grain growth, "
                                    "defect chemistry and electrode reactions", "year": 2019}]
    assert title_duplicate_edges(papers) == []
