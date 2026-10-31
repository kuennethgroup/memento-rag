# tests/graph/test_rq2_p2_documents.py
from memento.graph.documents import extracts_entities, is_person_folder, work_id


def test_person_folder():
    assert is_person_folder("Garcia, Pablo") and is_person_folder("Fendler,  Michael")
    assert is_person_folder("Schulz Michael")          # archive folder without a comma
    assert not is_person_folder("1_Promotionsordnung") and not is_person_folder(None)


def test_extracts_entities_policy():
    assert extracts_entities(None)                                    # unknown -> research
    assert extracts_entities({"type": "dissertation"})
    assert not extracts_entities({"type": "image_artifact"})
    assert not extracts_entities({"type": "admin"})
    assert not extracts_entities({"type": "dissertation", "duplicate_of": "Diss_X"})


def test_work_id_prefers_duplicate_then_parent():
    assert work_id("A", {"duplicate_of": "D", "parent": "P"}) == "D"
    assert work_id("A", {"parent": "P"}) == "P"
    assert work_id("A", {"parent_diss": "Q"}) == "Q"
    assert work_id("A", None) == "A"


def test_copy_of_parent_creates_no_entities():
    assert not extracts_entities({"type": "chapter", "parent": "Diss_X", "copy_of": "Diss_X"})
    assert extracts_entities({"type": "chapter", "parent": "Diss_X"})          # rewritten chapter stays research


def test_containment_share():
    from memento.graph.documents import containment
    a = ["one two three four five six seven eight nine ten"]
    assert containment(a, ["zero one two three four five six seven eight nine ten eleven"]) == 1.0
    assert containment(a, ["completely different words in this other document here now"]) == 0.0
    assert containment([], ["x"]) == 0.0
