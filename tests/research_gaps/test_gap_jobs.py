# Extracting, checking and merging open research questions (gap ledger step 1).
import json
import sys
import pytest
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import jobs as gj  # noqa: E402


def _p(i, text, page=0, tag="limitation", signal="sig"):
    return {"point_id": f"pt{i}", "page": page, "text": text, "gap_type": tag, "gap_signal": signal}


PASSAGES = [_p(1, "The effect of La doping above 10 % on wake-up has not been examined.", 3),
            _p(2, "In this work we only used 10 nm films.", 4, "untested_condition"),
            _p(3, "The films were annealed at 600 °C.", 5)]


def test_the_extraction_prompt_numbers_the_passages_with_page_and_tag():
    msgs = gj.extract_messages("Doc title", PASSAGES)
    assert msgs[0]["role"] == "system" and "scope" in msgs[0]["content"] and "field" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert user.startswith("Document: Doc title")
    assert "[1] page 4 · tag: limitation · sig\nThe effect of La doping" in user
    assert "[2] page 5 · tag: untested_condition" in user


def test_extraction_output_is_validated():
    raw = json.dumps([
        {"i": 1, "type": "field", "question": "How does La doping above 10 % affect wake-up in HfO2?",
         "claim": "The effect of La doping above 10 % on wake-up has not been examined.", "kind": "limitation"},
        {"i": 2, "type": "scope", "claim": "Only 10 nm films were used.", "reason": "the study's own choice"},
        {"i": 3, "type": "none"},
        {"i": 9, "type": "field", "question": "Q?", "claim": "C."},                 # no such passage
        {"i": 1, "type": "field", "question": "", "claim": "C."},                   # empty question
        {"i": 1, "type": "field", "question": "Q2?", "claim": "C2.", "kind": "weird"},  # kind → the passage's tag
    ])
    items = gj.parse_extract("```json\n" + raw + "\n```", PASSAGES)
    assert [(x["i"], x["type"]) for x in items] == [(0, "field"), (1, "scope"), (2, "none"), (0, "field")]
    assert items[0]["kind"] == "limitation" and items[3]["kind"] == "limitation"
    assert items[1]["reason"] == "the study's own choice"


def test_unreadable_extraction_output_raises():
    import pytest
    with pytest.raises(ValueError):
        gj.parse_extract("I think the gaps are …", PASSAGES)


def test_extraction_asks_again_for_passages_the_model_skipped():
    calls = []

    def llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps([{"i": 1, "type": "none"}])                       # skipped 2 and 3
        return json.dumps([{"i": 1, "type": "scope", "claim": "c", "reason": "r"}, {"i": 2, "type": "none"}])
    items = gj.extract_document(llm, "Doc", PASSAGES, batch=10)
    assert len(calls) == 2 and "[1] page 5" in calls[1][1]["content"]           # only the missing two, renumbered
    assert sorted((x["i"], x["type"]) for x in items) == [(0, "none"), (1, "scope"), (2, "none")]


def test_long_documents_are_read_in_batches():
    many = [_p(i, f"text {i}") for i in range(1, 8)]
    seen = []

    def llm(messages):
        import re
        n = len(re.findall(r"^\[\d+\] page ", messages[1]["content"], re.M))
        seen.append(n)
        return json.dumps([{"i": k, "type": "none"} for k in range(1, n + 1)])
    items = gj.extract_document(llm, "Doc", many, batch=3)
    assert seen == [3, 3, 1] and sorted(x["i"] for x in items) == list(range(7))


class FakeVerifier:
    def __init__(self, labels):
        self.labels, self.pairs = labels, None

    def verify(self, pairs):
        from types import SimpleNamespace
        self.pairs = pairs
        return {cid: SimpleNamespace(label=self.labels[k], rationale=f"why {k}") for k, (cid, _, _) in enumerate(pairs)}


def test_only_questions_the_passage_supports_are_kept():
    items = [{"i": 0, "type": "field", "question": "Q1?", "claim": "C1.", "kind": "limitation"},
             {"i": 0, "type": "field", "question": "Q2?", "claim": "C2.", "kind": "limitation"},
             {"i": 2, "type": "field", "question": "Q3?", "claim": "C3.", "kind": "uncertainty"},
             {"i": 1, "type": "scope", "claim": "c", "reason": "r"}]
    v = FakeVerifier(["supported", "unsupported", "unjudged"])
    kept, dropped = gj.ground(v, items, PASSAGES)
    assert [x["question"] for x in kept] == ["Q1?"]
    assert [(x["question"], x["reason"]) for x in dropped] == [("Q2?", "unsupported: why 1"), ("Q3?", "unjudged: why 2")]
    assert [p[1] for p in v.pairs] == ["C1.", "C2.", "C3."] and v.pairs[0][2] == PASSAGES[0]["text"]  # claim vs passage


def test_merge_adds_a_mention_to_the_same_question_or_creates_a_gap():
    store = {"gaps": {}, "mentions": [], "vectors": {}}

    def add_gap(statement, kind):
        gid = len(store["gaps"]) + 1
        store["gaps"][gid] = statement
        return gid

    def search(vec, k):
        return [(gid, store["gaps"][gid], 0.9) for gid in store["vectors"]][:k]
    judged = []

    def relate(question, near):
        judged.append((question, near))
        return (near[0][0] if "again" in question else None), []
    deps = dict(embed=lambda t: [1.0], search=search, relate=relate, add_gap=add_gap,
                add_mention=lambda gid, item: store["mentions"].append((gid, item["question"])),
                upsert=lambda gid, vec, statement: store["vectors"].__setitem__(gid, vec))
    item = {"question": "How does X affect Y?", "kind": "limitation"}
    assert gj.merge(item, **deps) == (1, True)                                  # nothing near: new gap, no judge call
    assert judged == []
    assert gj.merge({"question": "How does X affect Y, again?", "kind": "limitation"}, **deps) == (1, False)
    assert gj.merge({"question": "Something else?", "kind": "future_work"}, **deps) == (2, True)
    assert store["mentions"] == [(1, "How does X affect Y?"), (1, "How does X affect Y, again?"), (2, "Something else?")]


def test_merge_ignores_far_neighbours():
    deps = dict(embed=lambda t: [1.0], search=lambda v, k: [(5, "far away?", 0.3)],
                relate=lambda q, near: (_ for _ in ()).throw(AssertionError("judge must not be asked")),
                add_gap=lambda s, k: 6, add_mention=lambda gid, item: None, upsert=lambda gid, v, s: None)
    assert gj.merge({"question": "Q?", "kind": "limitation"}, floor=0.6, **deps) == (6, True)


def test_a_limitation_this_document_itself_solves_is_not_an_open_question():
    # a problem the document itself solves ("earlier models cannot do X; we will show how") is no open question: the
    # library check excludes the stating document, so it would stay "open"
    msgs = gj.extract_messages("Doc", PASSAGES)
    assert '"solved"' in msgs[0]["content"] and "we show" in msgs[0]["content"].lower()
    raw = json.dumps([{"i": 1, "type": "solved", "claim": "Earlier models ignore X.", "reason": "the document shows how"},
                      {"i": 2, "type": "scope", "claim": "Only 10 nm films.", "reason": "own choice"},
                      {"i": 3, "type": "field", "question": "Q?", "claim": "C.", "kind": "uncertainty"}])
    items = gj.parse_extract(raw, PASSAGES)
    assert [x["type"] for x in items] == ["solved", "scope", "field"]
    left = gj.left_out(items)
    assert left == [(0, "Earlier models ignore X.", "solved in this document: the document shows how"),
                    (1, "Only 10 nm films.", "study scope only: own choice")]


def test_the_extraction_prompt_asks_for_natural_questions_about_the_science():
    p = gj.extract_messages("Doc", PASSAGES)[0]["content"]
    assert "about the science" in p and "not about a study" in p


def test_the_same_question_rule_is_strict():
    # a wrong merge inflates "stated by N documents", so merging is conservative
    p = gj.relation_messages("Q?", [(1, "E?")])[0]["content"]
    assert "necessarily" in p and "when in doubt" in p.lower() and "input" in p
    assert "names" in p.lower() and "images" in p.lower()                          # the counter-example


def test_relation_judge_output_same_and_related():
    # strict "same" (counts as one gap) plus "related" (a theme; never claimed identical)
    assert gj.parse_relation('{"same": 2, "related": [1, 3]}', [11, 12, 13]) == (12, [11, 13])
    assert gj.parse_relation('{"same": 0, "related": [2]}', [11, 12]) == (None, [12])
    assert gj.parse_relation('{"same": 0, "related": [9, "x", 1, 1]}', [11, 12]) == (None, [11])   # invalid ignored
    assert gj.parse_relation("garbage", [11]) == (None, [])
    p = gj.relation_messages("Q?", [(1, "E?")])[0]["content"]
    assert '"related"' in p and "necessarily" in p


def test_merge_records_related_gaps():
    links = []
    deps = dict(embed=lambda t: [1.0], search=lambda v, k: [(5, "close?", 0.9), (6, "also close?", 0.8)],
                relate=lambda q, near: (None, [6]), add_gap=lambda s, k: 7, add_mention=lambda gid, item: None,
                upsert=lambda gid, v, s: None, add_relation=lambda a, b: links.append((a, b)))
    assert gj.merge({"question": "Q?", "kind": "limitation"}, **deps) == (7, True)
    assert links == [(7, 6)]
    deps["relate"] = lambda q, near: (5, [6])                                      # same as 5, related to 6
    links.clear()
    assert gj.merge({"question": "Q?", "kind": "limitation"}, **deps) == (5, False)
    assert links == [(5, 6)]


def test_extraction_rules_from_the_independent_audit():
    # "…beyond the scope of this study" followed by what others could do is future work for the field, not scope; vague
    # questions are left out
    p = gj.extract_messages("Doc", PASSAGES)[0]["content"]
    assert "beyond the scope of this study" in p and "future_work" in p
    assert "specifically" in p and "comprehensive" in p


def test_related_needs_the_same_system_and_the_same_property():
    # "related" is a shared theme, not a loose association
    p = gj.relation_messages("Q?", [(1, "E?")])[0]["content"]
    assert "AND" in p and "same property or phenomenon" in p


def test_one_question_asks_one_thing_and_capabilities_are_not_gaps():
    # one question per item (two-part questions make merges fuzzy); expectations and conditions are no open questions
    p = gj.extract_messages("Doc", PASSAGES)[0]["content"]
    assert "one thing" in p and "two items" in p
    assert "capability" in p and "untested or unresolved" in p


# ── step 2: the library check ──
def _h(src, score, pid, text="t", page=0):
    return {"source": src, "rerank_score": score, "point_id": pid, "text": text, "page": page}


def test_check_candidates_leave_out_the_stating_documents_and_merge_both_searches():
    hits = [_h("A", 0.9, "1"), _h("B", 0.8, "2"), _h("C", 0.5, "3")]
    heads = [_h("B", 0.8, "2"), _h("D", 0.7, "4"), _h("A", 0.6, "5"), _h("E", 0.1, "6")]
    got = gj.check_candidates(hits, heads, stating={"A"}, cap=3)
    assert [h["point_id"] for h in got] == ["2", "4", "3"]                          # best first, no A, no duplicates


def test_check_output_is_validated_and_missing_passages_are_unknown():
    # a passage the model did not judge is "unknown", never "does not concern" (that would make a question "open" /
    # "nothing found" when the check had simply not finished)
    raw = json.dumps([{"i": 1, "relation": "addresses", "reason": "reports it"},
                      {"i": 2, "relation": "weird"}, {"i": 4, "relation": "partly"}])
    labels = gj.parse_check("```json\n" + raw + "\n```", 3)
    assert [lb["relation"] for lb in labels] == ["addresses", "unknown", "unknown"]
    assert not gj.complete(labels) and gj.complete(labels[:1])
    assert labels[0]["reason"] == "reports it"
    import pytest
    with pytest.raises(ValueError):                  # unreadable: no status at all (never a false "open")
        gj.parse_check("garbage", 2)


def test_library_status_follows_fixed_rules():
    st = lambda *rels: gj.library_status([{"relation": r} for r in rels])
    assert st("not_concern", "not_concern") == "open"
    assert st("partly", "not_concern") == "partly"
    assert st("addresses", "partly") == "addressed"
    assert st("addresses", "contradicts") == "contested"
    assert st() == "open"


def test_open_questions_keep_their_closest_near_misses():
    cands = [_h("B", 0.8, "2"), _h("D", 0.7, "4"), _h("C", 0.5, "3"), _h("F", 0.4, "7")]
    labels = [{"relation": "not_concern", "reason": "other topic"}] * 4
    near = gj.near_misses(cands, labels, n=3)
    assert [h["point_id"] for h, _ in near] == ["2", "4", "3"]
    labels2 = [{"relation": "partly", "reason": "r"}] + labels[1:]
    assert [h["point_id"] for h, _ in gj.near_misses(cands, labels2, n=2)] == ["4", "3"]   # only not_concern ones


def test_the_check_prompt_numbers_passages_and_states_the_question_and_claim():
    msgs = gj.check_messages("How does X affect Y?", "X's effect on Y is unknown.", [_h("B", 0.8, "2", "Some text.", 4)])
    assert "addresses" in msgs[0]["content"] and "contradicts" in msgs[0]["content"]
    assert "Question: How does X affect Y?" in msgs[1]["content"] and "[1] B, page 5\nSome text." in msgs[1]["content"]


def test_partly_requires_part_of_an_answer():
    # passages that only mention the topic or say the question is hard are not "partly", which would wash out "open"
    p = gj.check_messages("Q?", "C.", [_h("B", 0.8, "2")])[0]["content"]
    assert "part of the answer" in p and "restates the problem" in p


# ── step 3: the literature check ──

def test_the_search_phrase_is_short_plain_and_one_line():
    msgs = gj.query_messages("How does La doping affect the ferroelectric phase fraction of HfO2 thin films?")
    assert msgs[0]["role"] == "system" and "HfO2" in msgs[1]["content"]
    assert gj.parse_query('"La doping HfO2 thin film ferroelectric phase fraction"\n') == \
        "La doping HfO2 thin film ferroelectric phase fraction"
    assert gj.parse_query("Query: hafnia wake-up AND oxygen vacancies") == "hafnia wake-up oxygen vacancies"
    assert len(gj.parse_query(" ".join(f"w{i}" for i in range(30))).split()) == 12
    with pytest.raises(ValueError):
        gj.parse_query("  \n ")


def test_abstracts_are_judged_with_the_same_relations():
    from memento.research_gaps.literature import Work
    works = [Work("Wake-up in HfO2", 2021, "10.1/x", "Abstract " * 400, "https://doi.org/10.1/x", "openalex", "W1",
                  {"openalex"}),
             Work("No abstract", None, "", "", "", "semantic_scholar", "p2", {"semantic_scholar"})]
    msgs = gj.web_check_messages("Q?", "claim", works)
    text = msgs[1]["content"]
    assert "[1] Wake-up in HfO2 (2021)" in text and "[2] No abstract (year unknown)" in text
    assert "(no abstract: judge from the title only" in text and len(text) < 4000
    assert "abstract" in msgs[0]["content"].lower() and "own knowledge" in msgs[0]["content"]


@pytest.mark.parametrize("rels, status", [(["not_concern"], "nothing_found"), (["partly", "not_concern"], "partly"),
                                         (["addresses", "partly"], "possibly_addressed"),
                                         (["contradicts", "addresses"], "contested"), ([], "no_results")])
def test_web_status(rels, status):
    assert gj.web_status([{"relation": r} for r in rels]) == status


def test_a_shorter_phrase_when_few_papers_were_found():
    assert gj.shorter_query("3D capacitors perovskite memories 100nm groundrules integration challenges") == \
        "3D capacitors perovskite memories 100nm"
    assert gj.shorter_query("hafnia wake-up") is None                    # already short


def test_no_papers_read_is_no_result_not_nothing_found():
    assert gj.web_status([]) == "no_results"
    assert gj.web_status([{"relation": "not_concern"}]) == "nothing_found"


def test_the_closest_papers_read_are_kept_when_nothing_was_found():
    works = ["w1", "w2", "w3", "w4", "w5"]                      # reranker order, best first
    labels = [{"relation": r, "reason": f"r{k}"} for k, r in enumerate(
        ["not_concern", "partly", "not_concern", "not_concern", "not_concern"])]
    assert [w for w, _lb in gj.web_near_misses(works, labels)] == ["w1", "w3", "w4"]


# ── step 5: a new document checked against the existing questions ──

@pytest.mark.parametrize("old, new, status", [
    ([], ["not_concern"], "open"),
    ([], ["partly"], "partly"),
    (["partly"], ["addresses"], "addressed"),
    (["addresses"], ["not_concern"], "addressed"),
    (["partly"], ["contradicts"], "contested"),
])
def test_a_new_documents_labels_combine_with_the_stored_evidence(old, new, status):
    assert gj.combined_status(old, [{"relation": r} for r in new]) == status


def test_only_passages_above_the_threshold_reach_the_model():
    hits = [{"rerank_score": 0.2}, {"rerank_score": 0.04}, {"rerank_score": 0.039}]
    assert gj.worth_checking(hits) == hits[:2]
    assert gj.worth_checking([{"rerank_score": 0.01}]) == []



# ── an incomplete judgment is asked again for the missing passages only ──
def _llm(replies):
    calls = []

    def llm(msgs):
        calls.append(msgs)
        return replies[len(calls) - 1]
    return llm, calls


def test_only_the_unjudged_passages_are_asked_again():
    cands = ["p1", "p2", "p3"]
    llm, calls = _llm([json.dumps([{"i": 1, "relation": "addresses", "reason": "a"}]),
                       json.dumps([{"i": 1, "relation": "not_concern", "reason": "b"},
                                   {"i": 2, "relation": "partly", "reason": "c"}])])
    labels = gj.judge_labels(llm, lambda sub: [{"role": "user", "content": "|".join(sub)}], cands)
    assert [lb["relation"] for lb in labels] == ["addresses", "not_concern", "partly"] and gj.complete(labels)
    assert calls[1][0]["content"] == "p2|p3"                    # the second call has only the two unjudged passages


def test_still_missing_after_one_more_call_stays_unknown():
    llm, calls = _llm([json.dumps([{"i": 2, "relation": "partly"}]), json.dumps([])])
    labels = gj.judge_labels(llm, lambda sub: [{"role": "user", "content": "|".join(sub)}], ["p1", "p2"])
    assert [lb["relation"] for lb in labels] == ["unknown", "partly"] and len(calls) == 2


def test_a_complete_first_answer_needs_one_call():
    llm, calls = _llm([json.dumps([{"i": 1, "relation": "not_concern"}])])
    assert gj.complete(gj.judge_labels(llm, lambda sub: [{"role": "user", "content": "x"}], ["p1"])) and len(calls) == 1


def test_unreadable_output_is_retried_once_as_before():
    llm, calls = _llm(["garbage", json.dumps([{"i": 1, "relation": "addresses"}])])
    labels = gj.judge_labels(llm, lambda sub: [{"role": "user", "content": "x"}], ["p1"])
    assert labels[0]["relation"] == "addresses" and calls[1][-1]["content"] == "Output ONLY the JSON list."
