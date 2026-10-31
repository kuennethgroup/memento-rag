# Gap ledger step 3: the literature search. Only a short search phrase leaves the server.
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from memento.config.paths import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from memento.research_gaps import literature as ls  # noqa: E402

OA_WORK = {"id": "https://openalex.org/W1", "title": "Wake-up in HfO2", "doi": "https://doi.org/10.1002/PSSR.1",
           "publication_year": 2021, "abstract_inverted_index": {"Wake-up": [0], "is": [1], "explained": [2]}}
S2_PAPER = {"paperId": "abc", "title": "Wake-up in HfO2.", "year": 2021, "abstract": "A longer abstract here.",
            "externalIds": {"DOI": "10.1002/pssr.1"}, "url": "https://www.semanticscholar.org/paper/abc"}


def test_openalex_and_semantic_scholar_records_are_read():
    a = ls.from_openalex(OA_WORK)
    assert (a.title, a.year, a.doi, a.abstract, a.source) == ("Wake-up in HfO2", 2021, "10.1002/pssr.1",
                                                              "Wake-up is explained", "openalex")
    assert a.url == "https://doi.org/10.1002/pssr.1"
    b = ls.from_s2(S2_PAPER)
    assert (b.doi, b.source, b.url) == ("10.1002/pssr.1", "semantic_scholar", "https://doi.org/10.1002/pssr.1")
    c = ls.from_s2({"paperId": "x", "title": "No DOI", "url": "https://www.semanticscholar.org/paper/x"})
    assert c.doi == "" and c.url.endswith("/x") and c.abstract == ""


def test_the_same_paper_from_both_services_is_kept_once_with_the_longer_abstract():
    merged = ls.merge([ls.from_openalex(OA_WORK), ls.from_s2(S2_PAPER),
                       ls.from_s2({**S2_PAPER, "externalIds": {}, "title": "Wake-up in HfO2"}),      # by title
                       ls.from_openalex({**OA_WORK, "id": "W2", "doi": None, "title": "Another paper"})])
    assert [w.title for w in merged] == ["Wake-up in HfO2", "Another paper"]
    assert merged[0].abstract == "A longer abstract here." and merged[0].sources == {"openalex", "semantic_scholar"}


def test_the_librarys_own_documents_are_left_out():
    works = ls.merge([ls.from_openalex(OA_WORK), ls.from_openalex({**OA_WORK, "id": "W3", "doi": None,
                                                                  "title": "Polymer informatics: current status"})])
    kept = ls.drop_library(works, ["POLYMER INFORMATICS - Current Status and Critical Next Steps"[:36],
                                   "Something else"])
    assert [w.title for w in kept] == ["Wake-up in HfO2"]
    assert ls.drop_library(works, []) == works


def test_openalex_budget_from_its_headers():
    h = {"x-ratelimit-remaining-usd": "0.097", "x-ratelimit-cost-usd": "0.001"}
    assert ls.openalex_searches_left(h) == 97
    assert ls.openalex_searches_left({"x-ratelimit-remaining-usd": "0.0005", "x-ratelimit-cost-usd": "0.001"}) == 0
    assert ls.openalex_searches_left({}) is None                              # unknown: try


class FakeHTTP:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status, self.headers, self.calls = payload, status, headers or {}, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return SimpleNamespace(status_code=self.status, headers=self.headers, json=lambda: self.payload,
                               raise_for_status=lambda: None if self.status < 400 else (_ for _ in ()).throw(
                                   RuntimeError(str(self.status))))


@pytest.fixture
def engine():
    e = create_engine("sqlite://")
    ls.ensure_tables(e)
    return e


def test_searches_send_only_the_phrase_and_are_cached(engine):
    oa = FakeHTTP({"results": [OA_WORK]}, headers={"x-ratelimit-remaining-usd": "0.05",
                                                   "x-ratelimit-cost-usd": "0.001"})
    works, left = ls.search_openalex("hafnia wake-up", engine=engine, http=oa, n=25)
    assert [w.title for w in works] == ["Wake-up in HfO2"] and left == 50
    (url, params, _h), = oa.calls
    assert params["search"] == "hafnia wake-up" and params["per_page"] == 25
    works2, left2 = ls.search_openalex("hafnia wake-up", engine=engine, http=oa, n=25)    # from the cache
    assert len(oa.calls) == 1 and [w.title for w in works2] == ["Wake-up in HfO2"] and left2 is None
    s2 = FakeHTTP({"data": [S2_PAPER]})
    w3 = ls.search_s2("hafnia wake-up", engine=engine, http=s2, api_key="k", n=25)
    assert w3[0].source == "semantic_scholar" and s2.calls[0][2] == {"x-api-key": "k"}
    assert s2.calls[0][1]["query"] == "hafnia wake-up"


def test_an_old_cache_entry_is_searched_again(engine):
    oa = FakeHTTP({"results": [OA_WORK]})
    ls.search_openalex("q", engine=engine, http=oa)
    ls.age_cache(engine, timedelta(days=31))
    ls.search_openalex("q", engine=engine, http=oa)
    assert len(oa.calls) == 2


def test_a_failed_search_is_not_cached(engine):
    bad = FakeHTTP({}, status=429)
    with pytest.raises(ls.SearchFailed):
        ls.search_s2("q", engine=engine, http=bad, api_key="k", sleep=lambda s: None)
    good = FakeHTTP({"data": [S2_PAPER]})
    assert ls.search_s2("q", engine=engine, http=good, api_key="k")


class Flaky(FakeHTTP):
    """429 for the first `fails` calls, then the payload."""

    def __init__(self, payload, fails):
        super().__init__(payload)
        self.fails = fails

    def get(self, url, params=None, headers=None, timeout=None):
        self.status = 429 if self.fails > 0 else 200
        self.fails -= 1
        return super().get(url, params, headers, timeout)


def test_a_rate_limited_search_is_retried_with_growing_pauses(engine):
    waits = []
    s2 = Flaky({"data": [S2_PAPER]}, fails=2)
    works = ls.search_s2("q", engine=engine, http=s2, api_key="k", sleep=waits.append)
    assert len(works) == 1 and len(s2.calls) == 3 and waits == [2, 4]
    with pytest.raises(ls.SearchFailed):
        ls.search_s2("q2", engine=engine, http=Flaky({"data": []}, fails=9), api_key="k", sleep=waits.append)


def test_abstracts_by_work_id_from_the_cache(engine):
    ls.search_openalex("q", engine=engine, http=FakeHTTP({"results": [OA_WORK]}))
    ls.search_s2("q", engine=engine, http=FakeHTTP({"data": [S2_PAPER]}), api_key="k")
    assert ls.cached_abstracts(engine) == {"https://openalex.org/W1": "Wake-up is explained",
                                           "abc": "A longer abstract here."}


def test_search_syntax_characters_are_removed_for_openalex(engine):
    # "mu*" (the Coulomb pseudopotential) is no wildcard: the search would be refused (HTTP 400)
    oa = FakeHTTP({"results": [OA_WORK]})
    ls.search_openalex('electron-phonon parameter mu* "Tc" (ambient): pressure?', engine=engine, http=oa)
    assert oa.calls[0][1]["search"] == "electron-phonon parameter mu Tc ambient pressure"
    assert ls.openalex_phrase("a | b ~2 !x") == "a b 2 x"


def test_a_failed_search_never_shows_the_key(engine):
    # no secret in plain text anywhere; an HTTP error's message carries the request URL
    import httpx

    class Refusing:
        def get(self, url, params=None, headers=None, timeout=None):
            req = httpx.Request("GET", url, params=params)
            raise httpx.HTTPStatusError("400 Bad Request for url " + str(req.url), request=req,
                                        response=httpx.Response(400, request=req))
    with pytest.raises(ls.SearchFailed) as e:
        ls.search_openalex("q", engine=engine, http=Refusing(), api_key="SECRET123")
    assert "SECRET123" not in str(e.value) and "400" in str(e.value)
    with pytest.raises(ls.SearchFailed) as e:
        ls.search_s2("q", engine=engine, http=Refusing(), api_key="SECRET456", sleep=lambda s: None)
    assert "SECRET456" not in str(e.value)
