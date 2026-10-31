# The people a document names (authors; supervisors and reviewers), read once by the library's answer model from its
# title page. The front-matter rules (memento/library/front_matter.py) read a paper's author line; a thesis title page
# ("genehmigte Dissertation von …", "Betreut von: …", a colloquium line with date and name) does not have one. A name
# the model gives is kept only when the text has it.
import json
import re
import types

import pytest
from sqlalchemy import create_engine

from memento.library import people as dp
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.query import cache, library_state, workers
from memento.query.flows import gap_answer, library_listing

VOGEL = ("# Co-Injektion im Spritzgießen: Sandwichbauteile mit geschäumtem Kern und weichelastischer Haut\n\nDer Fakultät "
          "für Ingenieurwissenschaften der Universität Bayreuth zur Erlangung der Würde einer Doktor-Ingenieurin "
          "(Dr.-Ing.) genehmigte Dissertation von Dipl.-Ing. Katharina Vogel (geb. Lindner) aus Musterstadt\n\n"
          "Erstgutachter: Prof. Dr.-Ing. Holger Rückert\nZweitgutachter: Prof. Dr.-Ing. Volker Sandhäger")
FISCHER = ("# Masterarbeit\n\nEinfluss der Kettenverlängerung von recyceltem Polyethylenterephthalat\n\nVorgelegt von: "
           "Lena Schubert\nBetreut von: Prof. Dr.-Ing. Holger Rückert\nChristian Köhler, M.Sc.")


def _reply(obj):
    return json.dumps(obj)


# ── what the model's reply may add ──

def test_names_come_without_titles_and_each_person_once():
    authors, supervisors = dp.parse_people(_reply({
        "authors": ["Dipl.-Ing. Katharina Vogel", "Katharina Vogel"],
        "supervisors": ["Prof. Dr.-Ing. Holger Rückert", "Volker Sandhäger", "Katharina Vogel"]}), VOGEL)
    assert authors == ["Katharina Vogel"]
    assert supervisors == ["Holger Rückert", "Volker Sandhäger"]           # an author is never also a supervisor


def test_a_name_the_text_does_not_have_is_dropped():
    authors, supervisors = dp.parse_people(_reply({"authors": ["Katharina Vogel", "Max Mustermann"],
                                                   "supervisors": ["Hans Meier"]}), VOGEL)
    assert authors == ["Katharina Vogel"] and supervisors == []


def test_the_library_title_counts_as_text():
    # a chapter file has no title page; its library title names the author
    authors, _ = dp.parse_people(_reply({"authors": ["Lindemann"], "supervisors": []}), "**1. Einleitung** Unter dem …",
                                 title="Chapter 1 (Lindemann): Einleitung")
    assert authors == ["Lindemann"]


@pytest.mark.parametrize("reply", ['```json\n{"authors": ["Lena Schubert"], "supervisors": []}\n```',
                                   '<think>the page says</think>{"authors": ["Lena Schubert"], "supervisors": []}'])
def test_fences_and_thinking_around_the_json_are_read(reply):
    assert dp.parse_people(reply, FISCHER) == (["Lena Schubert"], [])


@pytest.mark.parametrize("reply", ["", "no json here", '{"authors": "Lena Schubert"}', "[1, 2]"])
def test_an_unreadable_reply_is_none(reply):
    assert dp.parse_people(reply, FISCHER) is None


# ── stored per document version, like the front matter ──

def _library(engine, rows):
    from memento.library import documents as ldb
    ldb.ensure_tables(engine)
    for src, h in rows:
        ldb.add_document(engine, collection="c", entry={"source": src, "title": src}, content_hash=h, uploaded_by="u",
                         output_root=None)


def test_people_are_stored_only_for_the_version_in_the_library():
    e = create_engine("sqlite://")
    _library(e, [("Diss", "h1")])
    dp.ensure_table(e)
    assert dp.put(e, collection="c", source="Diss", content_hash="h1", authors=["Katharina Vogel"],
                  supervisors=["Holger Rückert"], model="m") is True
    assert dp.put(e, collection="c", source="Diss", content_hash="old", authors=["X"], supervisors=[], model="m") is False
    assert dp.put(e, collection="c", source="Gone", content_hash="h1", authors=["X"], supervisors=[], model="m") is False
    assert dp.current(e, "c", {"Diss": "h1"}) == {"Diss": (["Katharina Vogel"], ["Holger Rückert"])}
    assert dp.current(e, "c", {"Diss": "h2"}) == {}                          # a newer version: not checked yet


def test_an_unreadable_reply_is_kept_as_such_and_not_asked_again():
    e = create_engine("sqlite://")
    _library(e, [("Diss", "h1")])
    dp.ensure_table(e)
    assert dp.put(e, collection="c", source="Diss", content_hash="h1", authors=None, supervisors=None, model="m")
    assert dp.current(e, "c", {"Diss": "h1"}) == {}                          # the front-matter rules decide for it
    assert dp.todo(e, "c", {"Diss": "h1"}) == []                             # and it is not asked again


# ── who is named how ──

PEOPLE = {"Diss": (["Katharina Vogel"], ["Holger Rückert", "Volker Sandhäger"]),
          "MA": (["Lena Schubert"], ["Holger Rückert", "Christian Köhler"]),
          "Talk": (["Holger Rückert"], []),
          "Chap": (["Lindemann"], []),
          "Paper": (["C. Brückner", "A. Kessler"], [])}


def test_authors_supervisors_and_initials_are_told_apart():
    k = dp.classify(PEOPLE, "Holger Rückert")
    assert (k["author"], k["supervisor"], k["possible_identity"]) == (["Talk"], ["Diss", "MA"], [])
    assert dp.classify(PEOPLE, "Holger Rueckert")["author"] == ["Talk"]          # spellings fold alike
    assert dp.classify(PEOPLE, "Christopher Brückner")["possible_identity"] == ["Paper"]   # only an initial matches
    assert dp.classify(PEOPLE, "Lindemann")["author"] == ["Chap"]                      # a surname asked: it decides
    assert dp.classify(PEOPLE, "Nils Lindemann")["possible_identity"] == ["Chap"]       # no first name on record
    assert dp.classify(PEOPLE, "Katharina Vogel") == {"author": ["Diss"], "supervisor": [], "possible_identity": []}


# ── read by the model, a few documents at a time ──

class FakeModel:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=r))])


def test_the_fill_reads_documents_without_people_a_few_at_a_time():
    e = create_engine("sqlite://")
    _library(e, [("Diss", "h1"), ("MA", "h2"), ("Done", "h3")])
    dp.ensure_table(e)
    dp.put(e, collection="c", source="Done", content_hash="h3", authors=["A B"], supervisors=[], model="m")
    fronts = {"Diss": VOGEL, "MA": FISCHER, "Done": "x"}
    model = FakeModel([_reply({"authors": ["Katharina Vogel"], "supervisors": ["Holger Rückert"]})])
    stored, failed = dp.fill(e, collection="c", library={"Diss": "h1", "MA": "h2", "Done": "h3"}, fronts=fronts,
                             titles={"Diss": "Dissertation: Co-Injektion", "MA": "Master's thesis: Einfluss"},
                             client=model, model="flash", limit=1)
    assert (stored, failed) == (1, 0) and len(model.calls) == 1
    call = model.calls[0]
    assert call["model"] == "flash" and call["temperature"] == 0
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "Dissertation: Co-Injektion" in call["messages"][-1]["content"] and "Katharina Vogel" in \
        call["messages"][-1]["content"]
    assert dp.todo(e, "c", {"Diss": "h1", "MA": "h2", "Done": "h3"}) == ["MA"]


def test_a_failed_call_stores_nothing_and_is_tried_again_later():
    e = create_engine("sqlite://")
    _library(e, [("MA", "h2")])
    dp.ensure_table(e)
    stored, failed = dp.fill(e, collection="c", library={"MA": "h2"}, fronts={"MA": FISCHER}, titles={},
                             client=FakeModel([RuntimeError("down")]), model="m", limit=5)
    assert (stored, failed) == (0, 1) and dp.todo(e, "c", {"MA": "h2"}) == ["MA"]


def test_a_document_without_front_matter_is_not_asked():
    e = create_engine("sqlite://")
    _library(e, [("NoText", "h")])
    dp.ensure_table(e)
    model = FakeModel([])
    assert dp.fill(e, collection="c", library={"NoText": "h"}, fronts={}, titles={}, client=model, model="m",
                   limit=5) == (0, 0)
    assert model.calls == []


# ── cases from real libraries ──

def test_a_trailing_initial_is_not_taken_for_the_surname():
    # a trailing initial is not taken for the surname
    front = "# Development of Novel Halogen-free Thermosetting Systems\n\nRaul M. Garcia G."
    assert dp.parse_people(_reply({"authors": ["Pablo M. Garcia G."], "supervisors": []}), front) == (["Pablo M. Garcia"], [])


def test_the_file_name_counts_as_text_and_reaches_the_model():
    # a talk whose first slide has only its title; the file name has the speaker
    front = "# Entwicklung neuer zellulärer Polymerelektrete durch gezielte Morphologiesteuerung"
    assert dp.parse_people(_reply({"authors": ["Lindemann"], "supervisors": []}), front, title="Talk: Entwicklung …",
                           source="Vortrag_Lindemann_100607") == (["Lindemann"], [])
    user = dp.messages(front, "Talk: Entwicklung …", "Vortrag_Lindemann_100607")[-1]["content"]
    assert "Vortrag_Lindemann_100607" in user and "file name" in user


def test_a_name_written_surname_first_still_matches():
    people = {"Diss": (["Böhmer Christian"], ["Rückert Holger"])}
    assert dp.classify(people, "Christian Böhmer")["author"] == ["Diss"]
    assert dp.classify(people, "Holger Rückert")["supervisor"] == ["Diss"]


# ── in the answer and in the worker ──

def _qp():
    from pathlib import Path
    root = PROJECT_ROOT
    mod = load_function(root / "openwebui/functions/query_pipe.py", "memento_query")
    return mod


def test_the_listing_names_the_documents_a_person_supervised_apart():
    qp = _qp()
    links = {"Diss": "/files/view/Diss/0", "Talk": "/files/view/Talk/0", "MA": "/files/view/MA/0"}
    en = library_listing.author_listing("Holger Rückert", ["Talk"], links, de=False, missing=0, supervised=["MA", "Diss"])
    assert en.startswith("**1 document** in the library names Holger Rückert as an author:")
    assert "**Supervised or reviewed by Holger Rückert**, 2 documents:" in en
    assert en.index("[Talk]") < en.index("Supervised") < en.index("[Diss]") < en.index("[MA]")
    de = library_listing.author_listing("Holger Rückert", [], links, de=True, missing=0, supervised=["MA"])
    assert de.startswith("Kein Dokument in der Bibliothek nennt **Holger Rückert** als Autor.")
    assert "**Betreut oder begutachtet von Holger Rückert**, 1 Dokument:" in de
    assert "—" not in en + de


def test_the_person_step_takes_the_read_people_first_and_the_front_matter_rules_for_the_rest():
    import inspect
    src = inspect.getsource(library_listing.person_reply)
    assert "doc_people.current, engine" in src and "doc_people.classify(" in src and "supervised=" in src
    # the front-matter rules only for documents the model has not read
    assert "doc_authors.classify({s: f for s, f in current.items() if s not in people}, turn.author)" in src


def test_the_worker_reads_people_while_the_model_is_idle():
    import inspect
    qp = _qp()
    cycle = inspect.getsource(workers.warmup_cycle)
    assert cycle.index("fill_people(valves, engine, st") < cycle.index("warmup_step(valves, engine")
    fill = inspect.getsource(workers.fill_people)
    assert "warmup_idle(" in fill and "doc_people.fill(" in fill and "limit=PEOPLE_PER_ROUND" in fill


def test_the_worker_fill_waits_when_busy_and_rests_when_done(monkeypatch):
    qp = _qp()
    e = create_engine("sqlite://")
    _library(e, [("MA", "h2")])
    from memento.library import front_matter as da
    da.ensure_table(e)
    da.put(e, collection="c", source="MA", content_hash="h2", front=FISCHER)
    pipe = types.SimpleNamespace(valves=types.SimpleNamespace(COLLECTION="c", METRICS_URL=None, METRICS_API_KEY_ENV=None,
                                                              WARMUP_MAX_RUNNING=2, LLM_BASE_URL="http://x/v1",
                                                              LLM_MODEL="flash", LLM_API_KEY_ENV=""))
    model = FakeModel([_reply({"authors": ["Lena Schubert"], "supervisors": ["Holger Rückert"]})])
    monkeypatch.setattr(library_state, "people_client", lambda v: model)
    monkeypatch.setattr(workers, "vllm_metrics", lambda *a, **k: None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(workers, "warmup_idle", lambda *a, **k: False)              # live questions: wait
    st = {}
    workers.fill_people(pipe.valves, e, st, function_id="memento_query", db_engine=lambda: e)
    assert model.calls == []
    monkeypatch.setattr(workers, "warmup_idle", lambda *a, **k: True)
    workers.fill_people(pipe.valves, e, st, function_id="memento_query", db_engine=lambda: e)
    assert len(model.calls) == 1
    assert dp.current(e, "c", {"MA": "h2"}) == {"MA": (["Lena Schubert"], ["Holger Rückert"])}
    workers.fill_people(pipe.valves, e, st, function_id="memento_query", db_engine=lambda: e)   # nothing left: rests
    assert st["people_next"] > 0 and len(model.calls) == 1


# ── "Which theses did X supervise?": the router takes the supervisor as the person asked for ──

def test_the_router_takes_a_supervisor_as_the_person_asked_for():
    from memento.query.router import ROUTER_SYSTEM_PROMPT as P
    assert "supervised or reviewed" in P
    assert '"Which theses did Holger Rückert supervise?" → "Holger Rückert"' in P
    assert '"Welche Arbeiten hat Christian Böhmer betreut?" → "Christian Böhmer"' in P


@pytest.mark.parametrize("q,person", [("Which theses did Holger Rückert supervise?", "Holger Rückert"),
                                      ("Welche Arbeiten hat Christian Böhmer betreut?", "Christian Böhmer"),
                                      ("Which documents did Volker Sandhäger review?", "Volker Sandhäger"),
                                      ("Welche Masterarbeiten wurden von Holger Rückert begutachtet?",
                                       "Holger Rückert")])
def test_a_supervision_question_is_a_listing_not_a_topic(q, person):
    assert library_listing.author_topic(q, person) is False


def test_a_topic_beside_supervision_is_still_a_topic():
    assert library_listing.author_topic("Which theses did Holger Rückert supervise on PET foams?", "Holger Rückert")


# ── research gaps: a talk's own aims are not gaps ──
# Measured with a blind judge: the rule helps talks but costs theses, so only a talk (its library title starts with
# "Talk") is asked with the rule; every other document keeps the prompt and its stored answers keep their keys.

def test_a_talk_is_asked_with_the_aims_rule_and_every_other_document_without():
    from memento.research_gaps import passages as gr
    rule = "own aims, objectives or research questions are not gaps"
    assert rule not in gr.GAP_SYSTEM_PROMPT
    assert rule in gr.GAP_SYSTEM_PROMPT_TALK and "also not as future work" in gr.GAP_SYSTEM_PROMPT_TALK
    assert gr.GAP_SYSTEM_PROMPT_TALK.replace(gr.TALK_AIMS_RULE, "") == gr.GAP_SYSTEM_PROMPT    # nothing else differs
    assert gr.gap_system_prompt("Talk (2015): Struktur und Eigenschaften") == gr.GAP_SYSTEM_PROMPT_TALK
    assert gr.gap_system_prompt("Talk: Micro- and nanostructured polymer blends") == gr.GAP_SYSTEM_PROMPT_TALK
    for title in ("Dissertation: Talking polymers", "Master's thesis: …", "", None):
        assert gr.gap_system_prompt(title) == gr.GAP_SYSTEM_PROMPT


def test_only_a_talks_stored_gap_answer_gets_a_new_key():
    from types import SimpleNamespace
    from memento.research_gaps import passages as gr
    qp = _qp()
    v = SimpleNamespace(GAP_BUDGET_TOKENS=20000, TEMPERATURE=0.15, MAX_TOKENS=16384)
    before = cache.gap_prompt_version(2000, v)
    assert cache.gap_prompt_version(2000, v, prompt=gr.gap_system_prompt("Dissertation: X")) == before
    assert cache.gap_prompt_version(2000, v, prompt=gr.gap_system_prompt("Talk: X")) != before


def test_the_gap_answer_its_key_and_its_check_use_the_documents_prompt():
    import inspect
    src = inspect.getsource(gap_answer.reply)
    assert "gap_prompt = gap_system_prompt(gap_title)" in src
    assert re.search(r'cache\.prompt_version\("gaps", "capped", turn\.dom, turn\.valves, think=turn\.gap_base_think,\s+title=gap_title\)', src)
    assert '[{"role": "system", "content": gap_prompt}]' in src and "check_prompt=gap_prompt" in src
    assert "GAP_SYSTEM_PROMPT}" not in src and "check_prompt=GAP_SYSTEM_PROMPT" not in src


def test_a_supervisor_on_record_with_an_initial_is_possibly_the_same_person():
    # talks name "V Sandhäger" in the footer; a full-name question must not silently leave them out
    people = {"Talk": (["Elif Yilmaz"], ["V Sandhäger"]), "Diss": (["Tarik Begic"], ["Volker Sandhäger"])}
    k = dp.classify(people, "Volker Sandhäger")
    assert (k["supervisor"], k["possible_identity"]) == (["Diss"], ["Talk"])


def test_two_workers_never_read_the_same_documents_at_once(monkeypatch):
    # two workers of a library must not read the same documents at once (each model call twice)
    import threading
    import time as _time
    qp = _qp()
    e = create_engine("sqlite://", connect_args={"check_same_thread": False},
                      poolclass=__import__("sqlalchemy.pool", fromlist=["StaticPool"]).StaticPool)
    _library(e, [("MA", "h2")])
    from memento.library import front_matter as da
    da.ensure_table(e)
    da.put(e, collection="c", source="MA", content_hash="h2", front=FISCHER)
    dp.ensure_table(e)

    class Slow(FakeModel):
        def _create(self, **kw):
            _time.sleep(0.3)
            return super()._create(**kw)
    model = Slow([_reply({"authors": ["Lena Schubert"], "supervisors": []})] * 2)
    monkeypatch.setattr(library_state, "people_client", lambda v: model)
    monkeypatch.setattr(workers, "vllm_metrics", lambda *a, **k: None)
    monkeypatch.setattr(library_state, "in_progress_sources", lambda *a: set())
    monkeypatch.setattr(workers, "warmup_idle", lambda *a, **k: True)
    pipe = types.SimpleNamespace(valves=types.SimpleNamespace(COLLECTION="c", METRICS_URL=None, METRICS_API_KEY_ENV=None,
                                                              WARMUP_MAX_RUNNING=2, LLM_BASE_URL="http://x/v1",
                                                              LLM_MODEL="flash", LLM_API_KEY_ENV=""))
    threads = [threading.Thread(target=workers.fill_people, args=(pipe.valves, e, {}),
                                kwargs=dict(function_id="memento_query", db_engine=lambda: e)) for _ in range(2)]
    for w in threads:
        w.start()
    for w in threads:
        w.join()
    assert len(model.calls) == 1
