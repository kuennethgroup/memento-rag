"""RQ1 GT v4 queries as the RQ2 control set, unchanged."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.RQ2.questions.common import Question


def from_rq1_v4(path, domain, universe):
    d = json.loads(Path(path).read_text())
    out = []
    for q in d["queries"]:
        out.append(Question(
            qid=f"{domain}:control:{q['query_id']}", domain=domain, qtype="control", question=q["query"],
            reference_answer="", lang="", universe=universe,
            gold_pages=[{"source": r["paper_id"], "page": r["page"], "role": "pooled", "fact": None}
                        for r in q.get("relevant_chunks", [])],
            meta={k: q.get(k) for k in ("query_id", "target_type", "category", "target_paper")}))
    return out
