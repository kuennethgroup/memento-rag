# tests/graph/test_rq2_fact_sheets.py — per-document fact sheets (RQ2 fact-sheets arm)
import sys
import uuid

sys.argv = [sys.argv[0], "--domain", "ceramics"]
from memento.config.cli_domain import apply_domain_arg  # noqa: E402
apply_domain_arg()
import pytest  # noqa: E402

from memento.graph.build import fact_sheets as fs  # noqa: E402


def test_split_parts_keeps_order_and_limit():
    parts = fs.split_parts(["a" * 30, "b" * 30, "c" * 30, "d" * 100], part_chars=70)
    assert parts[0] == "a" * 30 + "\n\n" + "b" * 30
    assert parts[1] == "c" * 30
    assert parts[2:] == ["d" * 70, "d" * 30]
    assert all(len(p) <= 70 for p in parts)


def test_parse_part_handles_fences_thinking_and_garbage():
    assert fs.parse_part('<think>x</think>```json\n{"materials": ["alumina"]}\n```') == {"materials": ["alumina"]}
    assert fs.parse_part('Here: {"year": 2020} done') == {"year": 2020}
    assert fs.parse_part("no json") is None and fs.parse_part("[1, 2]") is None


def test_merge_parts_unions_case_insensitively_and_votes():
    sheet = fs.merge_parts([
        {"topic": "Sintering of alumina", "doc_type": "dissertation", "year": 2019,
         "materials": ["Alumina", "zirconia"], "properties": [{"name": "hardness", "value": "18", "unit": "GPa"}]},
        {"topic": "Results", "doc_type": "dissertation", "year": "2019", "materials": ["alumina", "YSZ", ""],
         "conditions": ["1450 °C"], "properties": [{"name": "Hardness", "value": "18", "unit": "GPa"}, "junk"]},
        {"doc_type": "not a type", "year": "about 2020", "processes": "not a list"},
    ])
    assert sheet["materials"] == ["Alumina", "zirconia", "YSZ"]
    assert sheet["topic"] == "Sintering of alumina" and sheet["part_topics"] == ["Sintering of alumina", "Results"]
    assert sheet["doc_type"] == "dissertation" and sheet["year"] == 2019
    assert sheet["processes"] == [] and sheet["conditions"] == ["1450 °C"]
    assert len(sheet["properties"]) == 1


class _LLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0
        self.chat = self; self.completions = self

    def create(self, **kw):
        self.calls += 1
        out = self.outputs.pop(0)
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": out})})]})


def test_extract_sheet_retries_once_then_counts_failed_part(tmp_path):
    from memento.graph.build.extraction_cache import RawResponseCache
    llm = _LLM(['{"materials": ["alumina"]}', "garbage", "still garbage"])
    cache = RawResponseCache(tmp_path / "c.json")
    sheet = fs.extract_sheet(llm, "m", "D", ["x" * 60, "y" * 60], cache=cache, part_chars=70)
    assert sheet["n_parts"] == 2 and sheet["n_failed_parts"] == 1 and llm.calls == 3
    assert sheet["usage"]["calls"] == 3 and sheet["usage"]["cache_hits"] == 0
    assert sheet["materials"] == ["alumina"] and sheet["source"] == "D"
    again = fs.extract_sheet(_LLM(["garbage", "garbage"]), "m", "D", ["x" * 60, "y" * 60], cache=cache, part_chars=70)
    assert again["materials"] == ["alumina"]          # first part comes from the cache


def test_payload_canonicalizes_with_the_registry():
    from memento.graph import canonicalize
    p = fs.payload({"source": "D", "materials": ["3YSZ", "Unobtainium X"], "properties": [
        {"name": "fracture toughness", "value": "5", "unit": "MPa m^0.5", "material": "3YSZ"}]}, canonicalize)
    assert "unobtainium x" in p["materials_canon"]
    assert p["materials_raw"] == "3YSZ | Unobtainium X"
    assert p["properties_raw"] == "fracture toughness" and p["year"] is None


def test_sheets_collection_name():
    assert fs.sheets_collection("memento-ceramics-rag") == "memento-ceramics-sheets"


def test_qdrant_index_and_match(request, throwaway_qdrant):
    from memento.graph import canonicalize
    client = throwaway_qdrant
    name = f"test-fact-sheets-{uuid.uuid4().hex[:8]}"
    request.addfinalizer(lambda: client.delete_collection(name))
    sheets = [{"source": "A", "materials": ["alumina"], "characterization": ["X-ray diffraction"], "year": 2020},
              {"source": "B", "materials": ["polypropylene"], "characterization": ["EXAFS spectroscopy"]},
              {"source": "C", "materials": ["zirconia"], "processes": ["spark plasma sintering"]}]
    assert fs.index_sheets(client, name, [fs.payload(s, canonicalize) for s in sheets]) == 3
    assert fs.matching_sources(client, name, ["EXAFS"], canonicalize) == {"B"}
    assert fs.matching_sources(client, name, ["Alumina", "Al2O3"], canonicalize) == {"A"}
    assert fs.matching_sources(client, name, ["spark plasma sintering"], canonicalize) == {"C"}
    assert fs.matching_sources(client, name, ["graphene"], canonicalize) == set()


def test_part_over_the_context_limit_is_split_and_retried(tmp_path):
    from memento.graph.build.extraction_cache import RawResponseCache

    class TooLong(Exception):
        pass

    class LLM:
        def __init__(self):
            self.chat = self; self.completions = self; self.sizes = []

        def create(self, **kw):
            n = len(kw["messages"][1]["content"])
            self.sizes.append(n)
            if n > 8000:
                raise TooLong("Error code: 400 - This model's maximum context length is 32768 tokens")
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {
                "content": '{"materials": ["m%d"]}' % len(self.sizes)})})], "usage": None})

    text = ("alpha " * 300 + "\n\n") * 6          # one part of ~10.8k chars
    llm, cache = LLM(), RawResponseCache(tmp_path / "c.json")
    sheet = fs.extract_sheet(llm, "m", "D", [text], cache=cache, part_chars=20_000)
    assert sheet["n_parts"] == 1 and sheet["n_split_parts"] == 1 and sheet["n_failed_parts"] == 0
    assert len(sheet["materials"]) == 2 and sheet["usage"]["calls"] == 2
    assert (tmp_path / "c.json").exists()             # saved after each part, not only at the end


def test_other_errors_still_raise(tmp_path):
    class LLM:
        chat = completions = None

        def __init__(self):
            self.chat = self; self.completions = self

        def create(self, **kw):
            raise RuntimeError("connection refused")
    with pytest.raises(RuntimeError):
        fs.extract_sheet(LLM(), "m", "D", ["x" * 100])


def test_match_counts_drops_documents_whose_names_only_match_across_fields(request, throwaway_qdrant):
    import uuid
    from memento.graph import canonicalize
    client = throwaway_qdrant
    name = f"test-fact-sheets-{uuid.uuid4().hex[:8]}"
    request.addfinalizer(lambda: client.delete_collection(name))
    sheets = [{"source": "A", "processes": ["spark plasma sintering"]},
              {"source": "B", "processes": ["plasma treatment", "spark erosion", "sintering"]}]
    fs.index_sheets(client, name, [fs.payload(s, canonicalize) for s in sheets])
    assert fs.match_counts(client, name, ["spark plasma sintering"], canonicalize) == {"A": 1}
