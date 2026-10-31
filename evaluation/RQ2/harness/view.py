"""What an arm may see of a question: id, text, language — never the metadata."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionView:
    qid: str
    question: str
    lang: str


def view_of(q) -> QuestionView:
    return QuestionView(q.qid, q.question, q.lang or "")
