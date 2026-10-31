# The Open WebUI pipes and the command-line backend stay in sync: embedding instruction from the ingestion pipe's own
# DOMAIN_NAME Valve, page ranges from both ingestion paths, the command line's document-list update, and verification
# re-searches scoped like the main search.
import json
import sys
import types
from pathlib import Path

from qdrant_client.models import SparseVector
from memento.config.paths import PROJECT_ROOT
from _owui_loader import load_function
from memento.ingestion.upload import document

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))

from memento.config.domain_loader import load_domain  # noqa: E402

ip = load_function(ROOT / "openwebui/functions/ingestion_pipe.py", "memento_ingestion")

ITEMS = [
    {"type": "text", "text": "Band gaps were computed with HSE06.", "page_idx": 2, "page_end": 3, "bbox": [1, 2, 3, 4]},
    {"type": "text", "text": "A second paragraph.", "page_idx": 3, "bbox": [1, 2, 3, 4]},
]


class _FakeQdrant:
    def __init__(self, *a, **kw):
        self.points = []

    def count(self, *a, **kw):
        raise RuntimeError("no collection yet")

    def upsert(self, collection_name, points):
        self.points.extend(points)


def _sparse(texts):
    return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


def test_ingestion_pipe_embeds_with_its_valve_domain_and_keeps_page_ranges(tmp_path, monkeypatch):
    out = tmp_path / "doc"
    out.mkdir()
    (out / "doc_content_list.json").write_text(json.dumps(ITEMS))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    seen, client = [], _FakeQdrant()
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: client))
    import memento.common.model_servers as vc
    import memento.ingestion.embed as sp
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: seen.extend(inputs) or [[0.1]] * len(inputs))
    monkeypatch.setattr(sp, "sparse_encode", _sparse)
    monkeypatch.setattr(document, "set_paper_in_progress", lambda *a: None)
    monkeypatch.setattr(document.Ingester, "register_document", lambda *a, **k: None)
    p = document.Ingester(ip.Pipe().valves)
    p.valves.DOMAIN_NAME = "computational-materials"          # the OpenWebUI process's .env may say another domain
    p.valves.NO_ENRICH, p.valves.RESEARCH_GAP_TAG, p.valves.GRAPH_BUILD = True, False, False
    monkeypatch.setattr(p, "find_paper_dir", lambda stem: out)
    res = [r for r in p.embed_one(pdf) if isinstance(r, dict)][-1]
    assert res["ok"] and res["vectors"] == 2
    want = load_domain("computational-materials")["instruction"]
    assert want != load_domain("polymer-engineering")["instruction"]
    assert {e["instruction"] for e in seen} == {want}
    pages = [(pt.payload["page"], pt.payload.get("page_end")) for pt in client.points]
    assert pages == [(2, 3), (3, None)]


def test_command_line_stores_page_ranges_like_the_pipe(tmp_path, monkeypatch):
    import memento.ingestion.run_pipeline as rp
    (tmp_path / "doc_content_list.json").write_text(json.dumps(ITEMS))
    monkeypatch.setattr(rp, "call_embedder", lambda inputs, url, text_only=False: ([[0.1]] * len(inputs), 0.0))
    monkeypatch.setattr(rp, "sparse_encode", _sparse)
    monkeypatch.setattr(rp, "_record_in_manifest", lambda *a: None)
    added = []
    monkeypatch.setattr(rp, "_add_to_document_list", lambda paper_id, **kw: added.append(paper_id))
    for flag in ("GAP_TAG_ENABLED", "GRAPH_BUILD_ENABLED", "CONTEXTUALIZE_ENABLED"):
        monkeypatch.setattr(rp, flag, False)

    class Client(_FakeQdrant):
        def count(self, *a, **kw):
            return types.SimpleNamespace(count=len(self.points))
    client = Client()
    m = rp.ingest_paper(tmp_path, "doc", "http://embedder", client, enrichment_enabled=False)
    assert m.n_vectors == 2 and added == ["doc"]
    assert [(pt.payload["page"], pt.payload.get("page_end")) for pt in client.points] == [(2, 3), (3, None)]


COL = "memento-computational-materials-rag"
from sqlalchemy import create_engine as _REAL_CREATE_ENGINE  # noqa: E402
from memento.ingestion.upload import workflow


class _SourcesClient:
    """A Qdrant stand-in whose collection holds passages of `sources` (scroll only)."""
    def __init__(self, sources):
        self.sources = list(sources)

    def scroll(self, collection_name=None, with_payload=None, with_vectors=None, limit=None, offset=None, **kw):
        return [types.SimpleNamespace(payload={"source": s}) for s in self.sources], None


def _run_doc_list(monkeypatch, served, built=True, url="postgresql://x", in_collection=("paper_a", "paper_b")):
    """The command line's side of the document list, on a real (SQLite) table: which rows it adds."""
    import memento.ingestion.run_pipeline as rp
    import sqlalchemy
    from memento.library import documents as db
    e = _REAL_CREATE_ENGINE("sqlite://")                  # not the patched one from an earlier call in this test
    with e.begin() as c:
        c.execute(sqlalchemy.text("CREATE TABLE function (id TEXT, valves TEXT)"))
        c.execute(sqlalchemy.text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": served})
    db.ensure_tables(e)
    if built:
        db.add_document(e, collection=COL, entry={"source": "old", "title": "old"}, content_hash="h", uploaded_by="u",
                        output_root=None)
    monkeypatch.setattr(rp, "_doc_list", {})
    monkeypatch.setattr(rp, "COLLECTION", COL)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda u: e)
    monkeypatch.setenv("DATABASE_URL", url) if url else monkeypatch.delenv("DATABASE_URL", raising=False)
    client = _SourcesClient(in_collection)
    rp._add_to_document_list("paper_a", client=client)
    rp._add_to_document_list("paper_b", client=client)
    from memento.library import warmup_queue as wq
    with e.connect() as c:
        jobs = sorted((r.source, r.action) for r in c.execute(wq.JOBS.select()))
    added = sorted(r["source"] for r in db.list_documents(e, COL) if r["source"] != "old")
    assert jobs == sorted((s, a) for s in added for a in ("summary", "gaps"))     # every added document is warmed
    return added


def test_command_line_adds_to_the_list_of_the_served_collection(monkeypatch):
    served = json.dumps({"COLLECTION": COL})
    assert _run_doc_list(monkeypatch, served) == ["paper_a", "paper_b"]
    assert _run_doc_list(monkeypatch, json.dumps({"COLLECTION": "memento-ceramics-rag-ocr"})) == []   # eval collection
    assert _run_doc_list(monkeypatch, "gAAAAAB-encrypted") == []                                        # unknown
    # an empty list is built whole from the collection, as Memento Ingestion does
    assert _run_doc_list(monkeypatch, served, built=False) == ["paper_a", "paper_b"]
    assert _run_doc_list(monkeypatch, served, url="") == []


def test_ingestion_pipe_adds_only_to_a_built_list():
    src = (ROOT / "memento/ingestion/upload/document.py").read_text()
    assert "ldb.has_documents(engine, self.valves.COLLECTION)" in src and "ldb.rebuild_documents(" in src


def test_verification_re_searches_use_the_main_search_scope():
    src = (ROOT / "openwebui/functions/query_pipe.py").read_text()
    check = (ROOT / "memento/query/checking.py").read_text()               # Thorough check's own re-search
    body = check[check.index("def _retrieve(q: str)"):check.index("return found", check.index("def _retrieve(q: str)"))]
    assert "query_filter=retrieve_filter" in body
    # one rule for every answer type (test_pipe_logic: summaries and gaps stay in their document)
    assert "retrieve_filter=resolve.check_search_filter(" in (ROOT / "memento/query/flows/answer.py").read_text()
    scope = (ROOT / "memento/query/scope/resolve.py").read_text()
    assert "return library_state.exclude_keywords_section(library_state.with_exclusions(base, in_progress))" in scope


def test_ingestion_pipe_reuses_the_saved_figure_caption(tmp_path, monkeypatch):
    # a figure without author text: the saved caption (vlm_captions.json, shared with run_pipeline.py) is reused
    out = tmp_path / "doc"
    (out / "images").mkdir(parents=True)
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
    pix.save(str(out / "images" / "fig.png"))
    (out / "doc_content_list.json").write_text(json.dumps([{"type": "image", "img_path": "images/fig.png",
                                                            "page_idx": 0, "bbox": [0, 0, 10, 10]}]))
    (out / "vlm_captions.json").write_text(json.dumps({"images/fig.png": "SEM image of a porous foam."}))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = _FakeQdrant()
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: client))
    import memento.common.model_servers as vc
    import memento.ingestion.embed as sp
    import memento.ingestion.caption_image as ci
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: [[0.1]] * len(inputs))
    monkeypatch.setattr(sp, "sparse_encode", _sparse)
    monkeypatch.setattr(ci, "caption_image", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("regenerated")))
    monkeypatch.setattr(document, "set_paper_in_progress", lambda *a: None)
    monkeypatch.setattr(document.Ingester, "register_document", lambda *a, **k: None)
    p = document.Ingester(ip.Pipe().valves)
    p.valves.NO_ENRICH, p.valves.RESEARCH_GAP_TAG, p.valves.GRAPH_BUILD = False, False, False
    monkeypatch.setattr(p, "find_paper_dir", lambda stem: out)
    import memento.ingestion.enrich_parsed_content_list as en
    monkeypatch.setattr(en, "enrich_content_list", lambda path, **kw: path)  # keep the item as written
    res = [r for r in p.embed_one(pdf) if isinstance(r, dict)][-1]
    assert res["ok"]
    assert client.points[0].payload["text"] == "SEM image of a porous foam."
    assert client.points[0].payload["caption_source"] == "vlm"


def test_content_lists_get_text_and_noise_lists_are_flagged():
    # lists are embedded (they carry a `text` field)
    from memento.ingestion.enrich_parsed_content_list import prepare_lists
    L = lambda items, **kw: {"type": "list", "list_items": items, "section": "Conclusion", "section_raw": "5 Conclusions", **kw}
    concl = L(["(1) The MPB region shows coexisting rhombohedral and tetragonal phases.",
               "(2) The composition reaches a Curie temperature of 550 °C after sintering at 1100 °C for 2 h"])
    results = L(["Sample A was sintered at 1450 °C for 3", "Sample B was sintered at 1500 °C for 4"])   # ends in numbers
    refs = L(["[1] A. Author, J. Mater. Sci. 12 (2001) 3."], sub_type="ref_text")
    refs_sec = L(["Smith, J.: Polymer foams. Springer 2010, pp. 1-20"], section="References")
    toc = L(["2.1 Einteilung polymerbasierter Schäume......5", "2.2 Grundlagen zum Schäumen....7"], section="Background")
    toc_head = L(["Figure 1 Schematic of the extruder", "Figure 2 Cell morphology"], section_raw="List of Figures")
    tiny = L(["a)", "b)"])
    items = [concl, results, refs, refs_sec, toc, toc_head, tiny, {"type": "text", "text": "x"}]
    stats = prepare_lists(items)
    assert stats == {"content": 2, "references": 2, "toc": 2, "empty": 1}
    assert concl["text"].startswith("(1) The MPB") and "\n(2) " in concl["text"] and not concl.get("low_info")
    assert not results.get("low_info")
    assert all(x.get("low_info") for x in (refs, refs_sec, toc, toc_head, tiny))


def test_a_content_list_is_embedded_and_gap_tagged_by_both_paths():
    import memento.ingestion.run_pipeline as rp
    from memento.research_gaps.tagging import GAP_TAG_TYPES
    item = {"type": "list", "text": "Future work:\n- test at higher temperatures", "page_idx": 4, "bbox": [0, 0, 1, 1]}
    from memento.ingestion import embed
    embed_input, payload = embed.build_embed_input(item, Path("/nonexistent"))
    assert embed_input["text"].startswith("Future work") and payload["type"] == "list"
    assert GAP_TAG_TYPES == {"text", "list"}
    src = (ROOT / "memento/ingestion/upload/document.py").read_text()
    assert 'item.get("type") in GAP_TAG_TYPES' in src and '"image_analysis":      "true"' in src


def test_paragraph_ending_with_colon_gets_the_following_equation_or_list():
    # "…is given by:" and its equation are separate chunks: the equation is attached to the paragraph
    from memento.ingestion.enrich_parsed_content_list import attach_continuations
    items = [
        {"type": "text", "text": "The PDF of a log-normal distribution is given by:", "page_idx": 1},
        {"type": "page_number", "text": "4"},                                            # page furniture between
        {"type": "equation", "text": "$$f(M)=\\frac{1}{M\\sigma\\sqrt{2\\pi}}$$", "page_idx": 2},
        {"type": "text", "text": "The model performs the following steps:"},
        {"type": "list", "text": "1. generate\n2. fit"},
        {"type": "text", "text": "Results are shown below:"},
        {"type": "image", "img_path": "images/a.jpg"},                                   # not an equation/list
        {"type": "text", "text": "Contents:"},
        {"type": "list", "text": "1 Intro ....3", "low_info": True},                      # a table of contents
    ]
    assert attach_continuations(items) == 2
    assert items[0]["text"].endswith("\\sqrt{2\\pi}}$$") and items[0]["continued_by"] == "equation"
    assert items[3]["text"] == "The model performs the following steps:\n1. generate\n2. fit"
    assert items[5]["text"] == "Results are shown below:" and items[7]["text"] == "Contents:"
    assert items[2]["text"].startswith("$$")                                              # equation keeps its chunk


def test_ingestion_registration_enqueues_the_warm_up(monkeypatch, tmp_path):
    from memento.library import documents as db, warmup_queue as wq
    import open_webui.internal.db as owui                      # the per-test in-memory database (conftest)
    e = owui.engine
    db.ensure_tables(e)
    db.add_document(e, collection="col", entry={"source": "old", "title": "old"}, content_hash="h", uploaded_by="u",
                    output_root=None)                         # a built list: the upload is added, not rebuilt
    p = document.Ingester(ip.Pipe().valves)
    p.valves.COLLECTION, p.valves.DOMAIN_NAME = "col", "computational-materials"
    p.valves.INPUT_ROOT, p.valves.OUTPUT_ROOT = str(tmp_path / "in"), str(tmp_path / "out")
    monkeypatch.setattr(p, "load_manifest", lambda: {"documents": {"new": {"content_hash": "hn"}}})
    p.register_document("new", "hn")
    assert db.get_document(e, "col", "new")["content_hash"] == "hn"
    assert [(j["source"], j["action"]) for j in (wq.claim_next(e), wq.claim_next(e))] == [("new", "summary"), ("new", "gaps")]


# ── Multi-panel figures: both paths hand the parsed PDF to the enrichment, which crops merged figures ──
def test_ingest_paper_passes_source_pdf(tmp_path, monkeypatch):
    import memento.ingestion.run_pipeline as rp
    (tmp_path / "doc_content_list.json").write_text(json.dumps(ITEMS))
    seen = []

    def fake_enrich(path, source_pdf=None):
        seen.append(source_pdf)
        return path
    monkeypatch.setattr(rp, "enrich_content_list", fake_enrich)
    monkeypatch.setattr(rp, "call_embedder", lambda inputs, url, text_only=False: ([[0.1]] * len(inputs), 0.0))
    monkeypatch.setattr(rp, "sparse_encode", _sparse)
    monkeypatch.setattr(rp, "_record_in_manifest", lambda *a: None)
    monkeypatch.setattr(rp, "_add_to_document_list", lambda *a, **kw: None)
    for flag in ("GAP_TAG_ENABLED", "GRAPH_BUILD_ENABLED", "CONTEXTUALIZE_ENABLED"):
        monkeypatch.setattr(rp, flag, False)

    class Client(_FakeQdrant):
        def count(self, *a, **kw):
            return types.SimpleNamespace(count=len(self.points))
    pdf = tmp_path / "doc.pdf"
    rp.ingest_paper(tmp_path, "doc", "http://embedder", Client(), source_pdf=pdf)
    assert seen == [pdf]


def test_upload_pipe_passes_source_pdf(tmp_path, monkeypatch):
    out = tmp_path / "doc"
    out.mkdir()
    (out / "doc_content_list.json").write_text(json.dumps(ITEMS))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = _FakeQdrant()
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: client))
    import memento.common.model_servers as vc
    import memento.ingestion.embed as sp
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: [[0.1]] * len(inputs))
    monkeypatch.setattr(sp, "sparse_encode", _sparse)
    monkeypatch.setattr(document, "set_paper_in_progress", lambda *a: None)
    monkeypatch.setattr(document.Ingester, "register_document", lambda *a, **k: None)
    seen = []
    import memento.ingestion.enrich_parsed_content_list as en

    def fake_enrich(path, source_pdf=None):
        seen.append(source_pdf)
        return path
    monkeypatch.setattr(en, "enrich_content_list", fake_enrich)
    p = document.Ingester(ip.Pipe().valves)
    p.valves.NO_ENRICH, p.valves.RESEARCH_GAP_TAG, p.valves.GRAPH_BUILD = False, False, False
    monkeypatch.setattr(p, "find_paper_dir", lambda stem: out)
    [r for r in p.embed_one(pdf) if isinstance(r, dict)]
    assert seen == [pdf]


# ── An empty library: the command line registers a document exactly like Memento Ingestion ──
def test_empty_library_registration_is_the_same_in_both_paths(monkeypatch, tmp_path):
    """After a library reset the list is empty. Memento Ingestion builds it from the collection and queues the
    document's warm-up; the command line did nothing (no list row, no stored summary or gaps). Both must leave the same
    rows and the same warm-up jobs: the collection already holds `earlier` (stored, not yet registered) and `new`."""
    import sqlalchemy
    from memento.library import documents as db, warmup_queue as wq
    import memento.ingestion.run_pipeline as rp
    import open_webui.internal.db as owui
    in_dir, out_dir = tmp_path / "in" / "ComputationalMaterials", tmp_path / "out" / "ComputationalMaterials"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    manifest = {"documents": {"new": {"content_hash": "hn", "uploaded_by": "u1"},
                              "earlier": {"content_hash": "he", "uploaded_by": "u1"}}}
    client = _SourcesClient(["earlier", "new"])

    def state(e):
        with e.connect() as c:
            jobs = sorted((r.source, r.action) for r in c.execute(wq.JOBS.select().where(wq.JOBS.c.collection == COL)))
        rows = sorted((r["source"], r["content_hash"]) for r in db.list_documents(e, COL))
        return rows, jobs

    # Memento Ingestion
    pe = owui.engine
    db.ensure_tables(pe)
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: client))
    p = document.Ingester(ip.Pipe().valves)
    p.valves.COLLECTION, p.valves.DOMAIN_NAME = COL, "computational-materials"
    p.valves.INPUT_ROOT, p.valves.OUTPUT_ROOT = str(tmp_path / "in"), str(tmp_path / "out")
    monkeypatch.setattr(p, "load_manifest", lambda: manifest)
    p.register_document("new", "hn")
    pipe_state = state(pe)

    # the command line, on its own empty database
    ce = _REAL_CREATE_ENGINE("sqlite://")
    with ce.begin() as c:
        c.execute(sqlalchemy.text("CREATE TABLE function (id TEXT, valves TEXT)"))
        c.execute(sqlalchemy.text("INSERT INTO function VALUES ('memento_query', :v)"), {"v": json.dumps({"COLLECTION": COL})})
    db.ensure_tables(ce)
    monkeypatch.setattr(rp, "_doc_list", {})
    monkeypatch.setattr(rp, "COLLECTION", COL)
    monkeypatch.setattr(rp, "INPUT_DIR", in_dir)
    monkeypatch.setattr(rp, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda u: ce)
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(manifest))
    monkeypatch.setattr(rp, "_MANIFEST_FILE", mf)
    rp._add_to_document_list("new", client=client)
    cli_state = state(ce)

    assert pipe_state == cli_state == ([("earlier", "he"), ("new", "hn")], [("new", "gaps"), ("new", "summary")])


def test_a_document_is_shown_and_registered_only_after_its_gap_tags(tmp_path, monkeypatch):
    # the document is unhidden and registered only after its gap tagging: the gap updater (every 60 s, waits only while
    # a document is hidden), the Limitations warm-up and a "Draft a proposal" click must not read it before its tags
    # exist
    out = tmp_path / "doc"
    out.mkdir()
    (out / "doc_content_list.json").write_text(json.dumps(ITEMS))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    events = []

    class Q(_FakeQdrant):
        def set_payload(self, collection, payload, ids):
            events.append("tag written")
    client = Q()
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: client))
    import memento.common.model_servers as vc
    import memento.ingestion.embed as sp
    import memento.research_gaps.tagging as rgt
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: [[0.1]] * len(inputs))
    monkeypatch.setattr(sp, "sparse_encode", _sparse)
    monkeypatch.setattr(rgt, "tag_paper_chunks",
                        lambda chunks, **kw: events.append("tagged") or {0: {"gap_signal": "s", "gap_type": "t"}})
    monkeypatch.setattr(document, "set_paper_in_progress", lambda stem, on: events.append("hidden" if on else "shown"))
    monkeypatch.setattr(document.Ingester, "register_document", lambda *a, **k: events.append("registered"))
    p = document.Ingester(ip.Pipe().valves)
    p.valves.DOMAIN_NAME = "computational-materials"
    p.valves.NO_ENRICH, p.valves.RESEARCH_GAP_TAG, p.valves.GRAPH_BUILD = True, True, False
    monkeypatch.setattr(p, "find_paper_dir", lambda stem: out)
    res = [r for r in p.embed_one(pdf) if isinstance(r, dict)][-1]
    assert res["ok"]
    assert events == ["hidden", "tagged", "tag written", "tag written", "shown", "registered"]


def test_a_failed_gap_tagging_still_shows_and_registers_the_document(tmp_path, monkeypatch):
    out = tmp_path / "doc"
    out.mkdir()
    (out / "doc_content_list.json").write_text(json.dumps(ITEMS))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    events = []
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kw: _FakeQdrant()))
    import memento.common.model_servers as vc
    import memento.ingestion.embed as sp
    import memento.research_gaps.tagging as rgt
    monkeypatch.setattr(vc, "vl_embed_many", lambda url, model, inputs: [[0.1]] * len(inputs))
    monkeypatch.setattr(sp, "sparse_encode", _sparse)
    monkeypatch.setattr(rgt, "tag_paper_chunks", lambda chunks, **kw: (_ for _ in ()).throw(RuntimeError("model down")))
    monkeypatch.setattr(document, "set_paper_in_progress", lambda stem, on: events.append("hidden" if on else "shown"))
    monkeypatch.setattr(document.Ingester, "register_document", lambda *a, **k: events.append("registered"))
    p = document.Ingester(ip.Pipe().valves)
    p.valves.DOMAIN_NAME = "computational-materials"
    p.valves.NO_ENRICH, p.valves.RESEARCH_GAP_TAG, p.valves.GRAPH_BUILD = True, True, False
    monkeypatch.setattr(p, "find_paper_dir", lambda stem: out)
    res = [r for r in p.embed_one(pdf) if isinstance(r, dict)][-1]
    assert res["ok"] and events == ["hidden", "shown", "registered"]
