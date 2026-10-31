# tests/evaluation/test_rq2_p3_hardness.py
from evaluation.RQ2.questions.hardness import hardness


def test_hardness_label():
    b = ("P3", 4)
    assert hardness(b, [("P1", 0), ("P3", 4)], []) == "easy"
    assert hardness(b, [], [("P3", 4)]) == "easy"
    assert hardness(b, [("P3", 5)], [("P2", 4)]) == "hard"
