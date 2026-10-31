#!/usr/bin/env python3
"""Plan (and optionally apply) alias merges for one corpus graph.

  python memento/graph/build/merge_entities.py --domain polymer-engineering --threshold 0.90 --out data/rq2/merge_polymer_0.90.json
  ... --apply    # rewire the graph with the planned mapping
Needs Neo4j (domain instance) and the VL embedder (port 8021)."""
import argparse
import json
from pathlib import Path

from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
from memento.graph.build.entity_merge import apply_merges, plan_merges  # noqa: E402
from memento.graph.store import get_store  # noqa: E402
from memento.graph.build.embed_entities import embed_texts, entity_text  # noqa: E402

_ENTS = ("MATCH (e:Entity) OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e) "
         "RETURN e.id AS id, e.name AS name, e.type AS type, coalesce(e.canonical, false) AS canonical, "
         "coalesce(e.aliases, []) AS aliases, count(c) AS n_mentions ORDER BY id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    with get_store() as store:
        ents = [dict(r) for r in store._run(_ENTS).records]
        vecs = embed_texts([entity_text(e) for e in ents])
        plan = plan_merges(ents, vecs, a.threshold)
        names = {e["id"]: e["name"] for e in ents}
        plan["pairs"] = [p | {"name_a": names[p["a"]], "name_b": names[p["b"]]} for p in plan["pairs"]]
        plan |= {"domain": a.domain, "threshold": a.threshold, "n_entities": len(ents),
                 "n_members_merged": len(plan["mapping"])}
        if a.apply:
            plan["applied"] = apply_merges(store, plan["mapping"])   # bumps build_id itself
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(json.dumps({k: plan[k] for k in ("n_entities", "n_members_merged")} | plan.get("applied", {})))


if __name__ == "__main__":
    main()
