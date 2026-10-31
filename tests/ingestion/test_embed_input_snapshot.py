# What each element is embedded with, under which point id and with which payload, for both builders: the command
# line's (embed.enumerate_embeddable, called by run_pipeline) and Library Upload's (document.Ingester.embed_one). Recorded
# on the code before the reorganization, so moving the code cannot change it; memento/ingestion/README.md lists where the
# two builders differ ("Library Upload and the command line").
# Both builders read the same items: the content list the upload wrote, after its enrichment.
# Run through scripts/test_offline.sh.
import base64
import itertools
import json
import shutil

import pytest

import test_ingestion_adapter_contract as upload_contract
from _snapshots import check_snapshot_entry
from test_ingestion_adapter_contract import world  # noqa: F401  (the upload harness's fixture)

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _items(form: str) -> list[dict]:
    """A content list with every element kind: a heading, a paragraph citing a table, a paragraph across pages, a
    list, a table, an equation, a figure with a caption, a figure without any text (alone on its page), a chart, a header and a low-info
    fragment. `pdf`: MinerU's PDF parse (boxes); `office`: an Office parse (no boxes, an anchor on text)."""
    box = (lambda *b: {"bbox": list(b)}) if form == "pdf" else (lambda *b: {})
    anchor = {} if form == "pdf" else {"anchor": "slide-1"}
    return [
        {"type": "text", "text": "2. Results", "text_level": 1, "page_idx": 0, **box(10, 10, 300, 30)},
        {"type": "text", "text": "The samples reach 12 GPa (Table 1) after annealing at 900 °C.", "page_idx": 0,
         **box(10, 40, 500, 80), **anchor},
        {"type": "text", "text": "Hardness rises with the treatment temperature and", "page_idx": 0,
         **box(10, 700, 500, 760)},
        {"type": "text", "text": "falls again above 1100 °C.", "page_idx": 1, **box(10, 40, 500, 60)},
        {"type": "list", "list_items": ["first condition", "second condition"], "page_idx": 1, **box(10, 70, 500, 90)},
        {"type": "table", "table_caption": ["Table 1. Hardness after annealing"],
         "table_body": "<table><tr><td>900 °C</td><td>12 GPa</td></tr></table>", "page_idx": 1,
         **box(10, 100, 500, 200)},
        {"type": "equation", "text": "$$H = F / A$$", "text_format": "latex", "page_idx": 1, **box(10, 210, 500, 230)},
        {"type": "text", "text": "Equation 1 gives the hardness as force over area.", "page_idx": 1,
         **box(10, 235, 500, 260)},
        {"type": "image", "img_path": "images/fig1.png", "image_caption": ["Figure 1. Grain structure"],
         "image_footnote": [], "page_idx": 2, **box(10, 10, 300, 300)},
        {"type": "text", "text": "Figure 1 shows fine grains after annealing.", "page_idx": 2, **box(10, 310, 500, 340)},
        {"type": "image", "img_path": "images/fig2.png", "image_caption": [], "page_idx": 4, **box(310, 10, 600, 300)},
        {"type": "chart", "img_path": "images/chart1.png", "chart_caption": ["Figure 2. Hardness over temperature"],
         "content": "A rising line reaching 13 GPa.", "page_idx": 3, **box(10, 10, 300, 300)},
        {"type": "header", "text": "Journal of Samples", "page_idx": 3, **box(10, 0, 200, 8)},
        {"type": "text", "text": "Keywords", "page_idx": 3, **box(10, 900, 100, 910)},
    ]


class _ContextClient:
    """The contextualizer's model: one situating sentence per element id it may ask about."""
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kw):
        import types
        text = json.dumps({str(i): f"Situating sentence {i}." for i in range(64)})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


def _norm(value, tmp):
    return json.loads(json.dumps(value, default=str).replace(str(tmp), "<TMP>"))


def _run_upload(w, monkeypatch, form: str, domain: str, no_enrich: bool) -> dict:
    import memento.common.model_servers as model_servers
    import memento.ingestion.embed as sparse
    items = _items(form)
    monkeypatch.setattr(upload_contract, "CONTENT", items)
    real_get = upload_contract.FakeMinerU.get.__func__

    def get(cls, url, *a, **k):
        r = real_get(cls, url, *a, **k)
        if url.endswith("/result"):
            data = r.json()
            data["results"]["doc"]["images"] = {n: "data:image/png;base64," + base64.b64encode(PNG).decode()
                                                for n in ("fig1.png", "fig2.png", "chart1.png")}
            return upload_contract.FakeMinerU._resp(data)
        return r
    monkeypatch.setattr(upload_contract.FakeMinerU, "get", classmethod(get))
    import requests
    monkeypatch.setattr(requests, "get", upload_contract.FakeMinerU.get)   # the harness bound the old one
    embedded, sparse_texts, recorded = [], [], []

    def vl_embed_many(base_url, model, inputs, workers=16):
        embedded.extend(dict(i) for i in inputs)
        return [[0.0] * 8 for _ in inputs]
    monkeypatch.setattr(model_servers, "vl_embed_many", vl_embed_many)
    real_sparse = sparse.sparse_encode

    def sparse_encode(texts):
        sparse_texts.extend(texts)
        return real_sparse(texts)
    monkeypatch.setattr(sparse, "sparse_encode", sparse_encode)

    def upsert(self, collection_name, points, **k):     # noqa: F811 (the keyword the upload passes)
        for p in points:
            recorded.append({"id": p.id, "payload": p.payload, "vectors": sorted(p.vector)})
    monkeypatch.setattr(upload_contract.FakeQdrantStore, "upsert", upsert)
    w.pipe.valves.DOMAIN_NAME = domain
    w.pipe.valves.NO_ENRICH = no_enrich
    f = upload_contract._upload(w, "alice", "f1", "doc.pdf", f"embed snapshot {form}")
    reply = upload_contract._send(w, "alice", [f])["reply"]
    assert recorded, reply
    return {"embed_inputs": embedded, "sparse_texts": sparse_texts, "points": recorded}


def _run_pipeline_rows(content_list, domain: str, ctx: bool, setting: str, monkeypatch) -> list:
    from memento.config.domain_loader import load_domain
    from memento.ingestion import embed
    from memento.ingestion.contextualize import contextualize_items
    monkeypatch.setattr(embed, "EMBED_RETRIEVE_RERANK_INSTRUCTION", load_domain(domain)["instruction"])
    paper_dir = content_list.parent
    (paper_dir / "vlm_captions.json").unlink(missing_ok=True)       # each combination from the same state
    items = json.loads(content_list.read_text())
    if ctx:
        contextualize_items(items, enabled=True, client=_ContextClient(), model="fake", workers=1)
    kw = {"default": {}, "text_only": {"text_only": True}, "no_enrichment": {"enrichment_enabled": False},
          "no_captions": {"generate_captions": False}}[setting]
    rows = embed.enumerate_embeddable(items, paper_dir, "doc", **kw)
    return [{"chunk_id": r["chunk_id"], "item_idx": r["item_idx"], "embed_input": r["embed_input"],
             "payload": r["payload"]} for r in rows]


CASES = list(itertools.product(("pdf", "office"), ("generic", "polymer-engineering"), (False, True)))


@pytest.mark.parametrize("form,domain,no_enrich", CASES,
                         ids=[f"{f}-{d}-{'raw' if n else 'enriched'}" for f, d, n in CASES])
def test_embed_inputs_unchanged(world, monkeypatch, form, domain, no_enrich):  # noqa: F811
    upload = _run_upload(world, monkeypatch, form, domain, no_enrich)
    lists = sorted((world.tmp / "out").rglob("*_content_list_enriched.json" if not no_enrich else "*_content_list.json"))
    assert len(lists) == 1, lists
    shutil.copy(lists[0], world.tmp / "content_list_used.json")
    command_line = {}
    for ctx in (False, True):
        for setting in ("default", "text_only", "no_enrichment", "no_captions"):
            if no_enrich and setting != "no_enrichment":
                continue                      # the raw list is what run_pipeline embeds only with --no-enrich
            command_line[f"ctx={ctx}|{setting}"] = _run_pipeline_rows(lists[0], domain, ctx, setting, monkeypatch)
    key = f"{form}|{domain}|{'raw' if no_enrich else 'enriched'}"
    check_snapshot_entry("embed_inputs_snapshot", key,
                         _norm({"upload": upload, "command_line": command_line}, world.tmp))


def test_the_two_builders_differ_only_where_the_readme_says():
    # the recorded outputs, compared as each path stores them (run_pipeline adds source and img_path at upsert and stores
    # no caption): equal dense inputs, point ids and payloads, but an element without a box (Office) is stored with
    # bbox [] by the command line and null by Library Upload
    from _snapshots import FIXTURES
    recorded = json.loads((FIXTURES / "embed_inputs_snapshot.json").read_text())
    assert len(recorded) == len(CASES)
    for key, entry in recorded.items():
        rows = entry["command_line"]["ctx=False|" + ("no_enrichment" if key.endswith("raw") else "default")]
        upload = entry["upload"]
        assert [r["embed_input"] for r in rows] == upload["embed_inputs"], key
        assert [int(r["chunk_id"]) for r in rows] == [p["id"] for p in upload["points"]], key
        for row, point in zip(rows, upload["points"]):
            m = row["payload"]
            stored = {"source": "doc", **{k: m.get(k) for k in ("type", "text", "page", "img_path", "section",
                                                               "section_raw", "bbox")}}
            stored.update({k: m[k] for k in ("page_end",) if m.get(k) is not None})
            stored.update({k: m[k] for k in ("equation_context", "table_context", "mineru_text", "caption_source")
                           if m.get(k)})
            if key.startswith("office"):
                assert stored.pop("bbox") == [] and point["payload"].get("bbox") is None, key
                assert stored == {k: v for k, v in point["payload"].items() if k != "bbox"}, key
            else:
                assert stored == point["payload"], key
