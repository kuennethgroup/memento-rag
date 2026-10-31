# Stored answers for summaries and research gaps.
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.library import answers as ac  # noqa: E402

FIELDS = dict(collection="col", source="doc", content_hash="h1", action="summary", language="en",
              model="local:qwen3.8-27b", mode="whole", prompt_version="p1")


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    ac.ensure_table(e)
    return e


def test_key_fields_all_matter():
    base = ac.cache_key(**FIELDS)
    for field in FIELDS:
        assert ac.cache_key(**{**FIELDS, field: FIELDS[field] + "x"}) != base, field


def test_put_get_roundtrip(engine):
    key = ac.cache_key(**FIELDS)
    cites = [{"type": "citation", "data": {"document": ["passage"], "metadata": [{"name": "[1] doc (p. 1)"}]}}]
    assert ac.get(engine, key) is None
    ac.put(engine, key, fields=FIELDS, answer="The summary. [1]", citations=cites, generated_by="Qwen 3.8 27B (local)")
    row = ac.get(engine, key)
    assert row["answer"] == "The summary. [1]" and row["citations"] == cites
    assert row["generated_by"] == "Qwen 3.8 27B (local)" and row["hits"] == 1
    assert ac.get(engine, key)["hits"] == 2


def test_put_replaces(engine):
    key = ac.cache_key(**FIELDS)
    ac.put(engine, key, fields=FIELDS, answer="first", citations=[], generated_by="m")
    ac.put(engine, key, fields=FIELDS, answer="second", citations=[], generated_by="m")
    with engine.connect() as c:
        assert c.execute(ac.ANSWERS.select()).fetchall().__len__() == 1
    assert ac.get(engine, key)["answer"] == "second"


def test_delete_for_source(engine):
    for src, action in (("doc", "summary"), ("doc", "gaps"), ("other", "summary")):
        f = {**FIELDS, "source": src, "action": action}
        ac.put(engine, ac.cache_key(**f), fields=f, answer="a", citations=[], generated_by="m")
    assert ac.delete_for_source(engine, "col", "doc") == 2
    with engine.connect() as c:
        assert [r.source for r in c.execute(ac.ANSWERS.select())] == ["other"]


def test_prompt_version_changes_with_parts():
    assert ac.prompt_version("a", 1) != ac.prompt_version("a", 2) and len(ac.prompt_version("a")) == 16


def test_put_if_absent_keeps_existing(engine):
    key = ac.cache_key(**FIELDS)
    ac.put(engine, key, fields=FIELDS, answer="live answer", citations=[], generated_by="m")
    ac.put(engine, key, fields=FIELDS, answer="warm-up answer", citations=[], generated_by="m", replace=False)
    assert ac.get(engine, key)["answer"] == "live answer"
    other = ac.cache_key(**{**FIELDS, "source": "doc2"})
    ac.put(engine, other, fields={**FIELDS, "source": "doc2"}, answer="warm-up", citations=[], generated_by="m",
           replace=False)
    assert ac.get(engine, other)["answer"] == "warm-up"


def test_a_stored_answer_says_whether_it_was_checked(engine):
    # "Thorough check" replays only a checked answer; the warm-up writes checked ones
    key = ac.cache_key(**FIELDS)
    ac.put(engine, key, fields=FIELDS, answer="a", citations=[], generated_by="m")
    assert ac.get(engine, key)["checked"] is False
    ac.put(engine, key, fields=FIELDS, answer="b", citations=[], generated_by="m", checked=True)
    assert ac.get(engine, key)["checked"] is True


def test_a_checked_warmup_answer_replaces_only_an_unchecked_one(engine):
    key = ac.cache_key(**FIELDS)
    ac.put(engine, key, fields=FIELDS, answer="live, unchecked", citations=[], generated_by="m")
    ac.put(engine, key, fields=FIELDS, answer="warm-up, unchecked", citations=[], generated_by="m", replace=False)
    assert ac.get(engine, key)["answer"] == "live, unchecked"                    # as before
    ac.put(engine, key, fields=FIELDS, answer="warm-up, checked", citations=[], generated_by="m", replace=False,
           checked=True)
    assert ac.get(engine, key)["answer"] == "warm-up, checked"                   # the upgrade
    ac.put(engine, key, fields=FIELDS, answer="another checked", citations=[], generated_by="m", replace=False,
           checked=True)
    assert ac.get(engine, key)["answer"] == "warm-up, checked"                   # a checked one stays


def test_an_existing_table_gets_the_checked_column():
    from sqlalchemy import text
    e = create_engine("sqlite://")
    with e.begin() as c:                                                           # the table in its first form
        c.execute(text("CREATE TABLE memento_answer_cache (key VARCHAR(64) PRIMARY KEY, collection VARCHAR(200) NOT NULL, "
                       "source VARCHAR(500) NOT NULL, content_hash VARCHAR(128) NOT NULL, action VARCHAR(20) NOT NULL, "
                       "language VARCHAR(10) NOT NULL, model VARCHAR(200) NOT NULL, mode VARCHAR(20) NOT NULL, "
                       "prompt_version VARCHAR(32) NOT NULL, generated_by VARCHAR(200) NOT NULL, answer TEXT NOT NULL, "
                       "citations TEXT NOT NULL, status VARCHAR(10) NOT NULL, created_at TIMESTAMP, hits INTEGER NOT NULL, "
                       "last_hit_at TIMESTAMP)"))
        key = ac.cache_key(**FIELDS)
        c.execute(text("INSERT INTO memento_answer_cache VALUES (:k,'col','doc','h1','summary','en','local:qwen3.8-27b',"
                       "'whole','p1','m','old answer','[]','ready',NULL,0,NULL)"), {"k": key})
    ac.ensure_table(e)
    ac.ensure_table(e)                                                             # twice: nothing breaks
    row = ac.get(e, key)
    assert row["answer"] == "old answer" and row["checked"] is False
