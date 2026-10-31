"""Multi-hop bridge questions: graph-blind anchors, MuSiQue-style composition,
bridge-leak check, necessity filter."""
from __future__ import annotations

import random
from collections import defaultdict

from evaluation.RQ2.questions.common import lang_of, language_instruction, llm_json
from evaluation.RQ2.questions.terms import candidate_terms, hub_cap, term_key
from memento.graph.documents import work_id


def anchor_pairs(chunks, df_full, registry, *, n_research_docs, seed, max_pairs, exclude_pairs=frozenset()):
    rng = random.Random(seed)
    cap = hub_cap(n_research_docs)
    by_term: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        for t in candidate_terms(c["text"]):
            if 2 <= len(df_full.get(t, ())) <= cap:
                by_term[t].append(c)
    out, used_docs = [], set()
    terms = sorted(by_term)
    rng.shuffle(terms)
    for t in terms:
        cs = sorted(by_term[t], key=lambda c: (-len(c["text"]), c["point_id"]))
        best = None
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                if a["source"] == b["source"]:
                    continue
                if work_id(a["source"], registry.get(a["source"])) == work_id(b["source"], registry.get(b["source"])):
                    continue
                key = tuple(sorted((a["point_id"], b["point_id"])))
                docs = tuple(sorted((a["source"], b["source"])))
                if key in exclude_pairs or docs in used_docs:
                    continue
                best = (a, b, key, docs)
                break
            if best:
                break
        if best:
            a, b, key, docs = best
            if rng.random() < 0.5:
                a, b = b, a
            used_docs.add(docs)
            out.append({"bridge": t, "a": a, "b": b, "pair_key": key})
            if len(out) >= max_pairs:
                break
    return out


COMPOSE_SYSTEM = (
    "You write two-hop research questions from two passages of different documents. Passage A and passage B both "
    "mention the BRIDGE term. Write ONE question that (1) describes the bridge only through a fact stated in "
    "passage A, never by its name, abbreviation, formula or translation, and (2) asks for a fact that only passage B "
    "states about the bridge. A reader must need A to identify the bridge and B to answer. Ask about a scientific "
    "fact (a result, value, property, material, method or condition), never about equipment suppliers, funding, "
    "authors or affiliations. The question must stand alone: never mention the passages, 'passage A' or 'passage "
    "B'. Use the language the request names. Output ONLY JSON: {\"question\": str, "
    "\"answer\": str (short, from B), \"fact_a\": str (the A fact used to identify the bridge), \"fact_b\": str "
    "(the B fact that answers), \"bridge_forms\": [every surface form of the bridge you know: abbreviation, full "
    "name, German/English forms]}")


def compose(client, model, pair):
    user = (f"BRIDGE: {pair['bridge']}\n\nPassage A ({pair['a']['source']}, page {pair['a']['page'] + 1}):\n"
            f"{pair['a']['text'][:2500]}\n\nPassage B ({pair['b']['source']}, page {pair['b']['page'] + 1}):\n"
            f"{pair['b']['text'][:2500]}")
    lang = lang_of([pair["a"]["text"], pair["b"]["text"]])
    user += "\n\n" + language_instruction(lang)
    r = llm_json(client, model, COMPOSE_SYSTEM, user, temperature=0.3)
    need = ("question", "answer", "fact_a", "fact_b")
    if not isinstance(r, dict) or not all(isinstance(r.get(k), str) and r[k].strip() for k in need):
        return None
    forms = r.get("bridge_forms") if isinstance(r.get("bridge_forms"), list) else []
    return {k: r[k].strip() for k in need} | {"bridge_forms": [f for f in forms if isinstance(f, str)], "lang": lang}


def bridge_leaks(question, bridge, forms):
    q = f" {term_key(question)} "
    return any(k and f" {k}" in q for k in {term_key(bridge), *(term_key(f) for f in forms)})


ANSWER_SYSTEM = ("Answer the question using ONLY the passages given. If they do not contain enough information, "
                 "answer exactly: NOT ANSWERABLE. Be brief.")
JUDGE_SYSTEM = ("You grade an answer against a reference answer for a research question. Output ONLY JSON "
                "{\"verdict\": \"correct\"|\"incorrect\"}: correct if the answer states the reference's key fact "
                "(wording may differ, extra detail is fine); incorrect if it is missing, wrong, or NOT ANSWERABLE.")


def answer_from(client, model, question, passages):
    user = "\n\n".join(f"Passage {i + 1}:\n{p[:2500]}" for i, p in enumerate(passages)) + f"\n\nQuestion: {question}"
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=400,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return (r.choices[0].message.content or "").strip()


def judge_answer(client, model, question, reference, candidate):
    r = llm_json(client, model, JUDGE_SYSTEM,
                 f"Question: {question}\nReference answer: {reference}\nAnswer to grade: {candidate}")
    v = r.get("verdict") if isinstance(r, dict) else None
    return v if v in ("correct", "incorrect") else None


def necessity(client, model, q, a_text, b_text):
    answers = {"ab": answer_from(client, model, q["question"], [a_text, b_text]),
               "a_only": answer_from(client, model, q["question"], [a_text]),
               "b_only": answer_from(client, model, q["question"], [b_text])}
    labels = {k: judge_answer(client, model, q["question"], q["answer"], v) for k, v in answers.items()}
    passes = (None not in labels.values() and labels["ab"] == "correct"
              and labels["a_only"] != "correct" and labels["b_only"] != "correct")
    return labels | {"passes": passes, "answers": answers}
