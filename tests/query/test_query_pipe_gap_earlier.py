# "Possible extensions" build on the study's own points, not on the gaps in earlier work it addresses.
import asyncio
from memento.query import generation

import openwebui.functions.query_pipe as qp
from memento.query.flows import gap_answer


def test_no_extensions_from_gaps_in_earlier_work_alone(monkeypatch):
    pipe = qp.Pipe()

    async def no_model(*a, **kw):
        raise AssertionError("the model must not be asked")
    monkeypatch.setattr(generation, "open_answer", lambda valves, *a, **k: no_model(*a, **k))
    authors = "### Gaps in earlier work it addresses\n1. Inverse design of multi-phase ceramics is unexplored [1].\n"
    out = asyncio.run(gap_answer.gap_suggestions(pipe.valves, source="Rayan_Thesis", docs={}, variant=None, authors=authors, chunks=[],
                                            question="suggest future work", think=False, budget=None, db_engine=qp._db_engine))
    assert out == ""
