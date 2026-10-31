# Fakes at the service boundaries for the contract tests: the model API (openai), Qdrant, the embedder and the
# reranker (memento.common.model_servers), and a recorder of what the pipe sent to each. Nothing inside the pipes is patched,
# so the same tests hold when the pipes' code moves between modules.
import hashlib
import json
import os
import re
import types
from pathlib import Path

CONTRACTS = Path(__file__).parent / "fixtures" / "contracts"
UPDATE = os.getenv("MEMENTO_UPDATE_CONTRACTS") == "1"     # rewrite the golden files from the current code


def _cond(payload, c):
    v, m = payload.get(c.key), c.match
    if hasattr(m, "value"):
        return v == m.value
    if hasattr(m, "any"):
        return v in m.any
    if hasattr(m, "except_"):
        return v not in m.except_
    raise AssertionError(f"unexpected match {m!r}")


def allows(flt, payload) -> bool:
    """Does a Qdrant filter let this payload through? (must / must_not / should, nested filters)"""
    from qdrant_client.models import Filter
    if flt is None:
        return True
    test = lambda c: allows(c, payload) if isinstance(c, Filter) else _cond(payload, c)
    if not all(test(c) for c in (flt.must or [])):
        return False
    if any(test(c) for c in (flt.must_not or [])):
        return False
    return not flt.should or any(test(c) for c in flt.should)


def filter_repr(flt):
    """A Qdrant filter as plain data, for the transcript."""
    if flt is None:
        return None
    if hasattr(flt, "model_dump"):
        return {k: v for k, v in flt.model_dump(exclude_none=True).items() if v}
    return repr(flt)


class FakeQdrant:
    """QdrantClient over in-memory collections {name: [payload, …]}; every query is recorded."""
    collections: dict = {}
    log: list = []

    def __init__(self, *a, **k):
        pass

    def _points(self, name, flt):
        return [types.SimpleNamespace(id=i, payload=p, score=round(0.9 - i / 100, 4))
                for i, p in enumerate(FakeQdrant.collections.get(name, [])) if allows(flt, p)]

    def collection_exists(self, name):
        return name in FakeQdrant.collections

    def count(self, name, exact=True, count_filter=None, **k):
        return types.SimpleNamespace(count=len(self._points(name, count_filter)))

    def query_points(self, collection_name, query=None, using=None, prefetch=None, query_filter=None, limit=10, **k):
        flt = query_filter if query_filter is not None else (prefetch[0].filter if prefetch else None)
        FakeQdrant.log.append({"op": "query", "collection": collection_name, "filter": filter_repr(flt),
                               "limit": limit, "hybrid": bool(prefetch)})
        return types.SimpleNamespace(points=self._points(collection_name, flt)[:limit])

    def query_points_groups(self, collection_name, query=None, using=None, query_filter=None, group_by="source",
                            group_size=1, limit=10, **k):
        FakeQdrant.log.append({"op": "groups", "collection": collection_name, "filter": filter_repr(query_filter),
                               "group_size": group_size, "limit": limit})
        groups = {}
        for p in self._points(collection_name, query_filter):
            groups.setdefault(p.payload.get(group_by), []).append(p)
        return types.SimpleNamespace(groups=[types.SimpleNamespace(id=g, hits=hits[:group_size])
                                             for g, hits in list(groups.items())[:limit]])

    def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, **k):
        FakeQdrant.log.append({"op": "scroll", "collection": collection_name, "filter": filter_repr(scroll_filter)})
        return self._points(collection_name, scroll_filter), None

    def delete(self, collection_name, points_selector=None, **k):
        FakeQdrant.log.append({"op": "delete", "collection": collection_name})


# the date and time some prompts carry ("Friday, 23 October 2026, 03:12 CEST")
_NOW = re.compile(r"[A-Z][a-z]+day, \d{1,2} [A-Z][a-z]+ \d{4}, \d{2}:\d{2}(?: \S+)?")


class FakeLLM:
    """openai.AsyncOpenAI / openai.OpenAI: the router (a non-streamed call) gets the scenario's route as JSON, a
    streamed call the scenario's answer text. Every call is recorded."""
    calls: list = []
    route: dict = {}
    answer: str = "Answer [1]."

    def __init__(self, *a, **k):
        self.base_url = k.get("base_url")
        self.chat = self
        self.completions = self

    def _record(self, kw):
        msgs = kw.get("messages") or []
        system = next((m["content"] for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str)), "")
        last = msgs[-1].get("content") if msgs else ""
        if isinstance(last, list):
            last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
        FakeLLM.calls.append({"stream": bool(kw.get("stream")), "model": kw.get("model"), "messages": len(msgs),
                              "system_sha": hashlib.sha256(_NOW.sub("<now>", system).encode()).hexdigest()[:12],
                              "last_message": (last or "")[-300:], "max_tokens": kw.get("max_tokens"),
                              "extra_body": sorted((kw.get("extra_body") or {}).keys())})

    def _reply(self, kw):
        self._record(kw)
        if kw.get("stream"):
            text = FakeLLM.answer

            async def gen():
                delta = types.SimpleNamespace(content=text, reasoning_content=None, reasoning=None)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=None)])
            return gen()
        msg = types.SimpleNamespace(content=json.dumps(FakeLLM.route), reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=None)

    async def create(self, **kw):
        return self._reply(kw)


class SyncFakeLLM(FakeLLM):
    def create(self, **kw):
        out = self._reply(kw)
        if kw.get("stream"):
            raise AssertionError("a synchronous streamed call")
        return out


def fake_rerank(base_url, model, query, docs, instruction, top_n=None, multimodal=True, timeout=300):
    ranked = [(i, round(0.9 - i / 100, 4)) for i in range(len(docs))]
    return ranked[:top_n] if top_n else ranked


_VOLATILE = [(re.compile(r"\b\d+(?:\.\d+)? ?(?:s|ms|min)\b"), "<t>"),
             (re.compile(r"[0-9a-f]{24,}"), "<id>"),
             (re.compile(r"\b20\d\d-\d\d-\d\d[T ][0-9:.]+\S*"), "<time>")]


def normalize(obj):
    """Durations, random ids and timestamps replaced, so a transcript compares across runs."""
    if isinstance(obj, str):
        for rx, rep in _VOLATILE:
            obj = rx.sub(rep, obj)
        return obj
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    return obj


def check_golden(kind: str, name: str, transcript: dict):
    """Compare a transcript with tests/fixtures/contracts/<kind>/<name>.json (MEMENTO_UPDATE_CONTRACTS=1 writes it)."""
    path = CONTRACTS / kind / f"{name}.json"
    got = json.loads(json.dumps(normalize(transcript), ensure_ascii=False, default=str))
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
        if not UPDATE:
            raise AssertionError(f"{path} written; inspect it and run again")
        return
    want = json.loads(path.read_text())
    if got != want:
        diffs = "\n".join(_diff(want, got)[:12])
        raise AssertionError(f"{kind}/{name}: the pipe's behaviour differs from {path}\n{diffs}\n"
                             "(MEMENTO_UPDATE_CONTRACTS=1 rewrites it after an intended change)")


def _diff(want, got, at="") -> list:
    """The paths where two transcripts differ, with both values (shortened)."""
    if isinstance(want, dict) and isinstance(got, dict):
        return [d for k in sorted(set(want) | set(got)) for d in _diff(want.get(k), got.get(k), f"{at}.{k}")]
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        return [d for i, (a, b) in enumerate(zip(want, got)) for d in _diff(a, b, f"{at}[{i}]")]
    return [] if want == got else [f"  {at or '.'}: expected {str(want)[:160]!r}, got {str(got)[:160]!r}"]
