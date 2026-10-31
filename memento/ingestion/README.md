# Ingestion

How documents become searchable passages, how to add them, and how to keep a library clean. The chat (Library Upload,
`upload/`, started by `openwebui/functions/ingestion_pipe.py`) and the command line (`run_pipeline.py`) store a
document the same way (see [Library Upload and the command line](#library-upload-and-the-command-line)).

## Pipeline

```
file → 1 convert → 2 parse (MinerU) → 3 enrich → 4 embed and store (Qdrant) → 5 research-gap tags
```

1. **Convert** (`convert.py`): PDF, DOCX, PPTX, XLSX and images (PNG, JPG, TIF, BMP, WEBP) are read as they are;
   `.doc`, `.ppt`, `.xls`, `.odt`, `.odp`, `.ods` and `.rtf` are converted with LibreOffice first.
2. **Parse** with MinerU 3.4.0 (`vlm-engine`): text, titles, tables, figures and equations with page and position.
   Output: `data/output/<corpus_dir>/<document>/vlm/` (Office files: `office/`).
3. **Enrich** (`enrich_parsed_content_list.py`, no model): corrects look-alike characters against a born-digital PDF's text layer (`text_layer.py`), joins paragraphs split across pages, merges the panels of a
   figure into one figure, links figures, tables and equations to the sentences that discuss them, assigns each
   passage its section, and marks noise (page furniture, tables of contents, reference lists, QR codes) so it is not
   embedded. A figure without any text gets a short description from the answer model (`caption_image.py`). Output:
   `<document>_content_list_enriched.json` next to MinerU's file.
4. **Embed and store** (`run_pipeline.py`, `embed.py`): one point per passage, figure, table or equation, with a dense
   vector (Qwen3-VL-Embedding-8B; figures as images) and a BM25 vector. Point ids are fixed per document and
   position, so storing a document again replaces its points. A table's dense vector also holds the passages citing it
   by label, an equation's the sentences around it; both are stored (`table_context`, `equation_context`) and shown to
   the answer model under the table or formula.
5. **Research-gap tags** (`memento/research_gaps/tagging.py`, when `RESEARCH_GAP_TAG` is on): the answer model marks passages that
   state limitations, future work, untested conditions or uncertainties. The gap ledger is built from them.

Each document also gets a row in the library's document list (`memento_documents`: title, fingerprint, uploader) and
its front matter for author questions (`memento_front_matter`).

## Adding documents

**From the chat:** Library Upload ([`openwebui/README.md`](../../openwebui/README.md#library-upload)).

**From the command line:** put the files in `data/input/<corpus_dir>/` (any folder depth; `corpus_dir` is set in
`config/domains/<domain>.yaml`), then:

```bash
bash run_ingestion.sh --domain <domain>                                   # all new documents
bash run_ingestion.sh --domain <domain> --pdf <file>                      # one document
python -m memento.ingestion.run_pipeline --domain <domain> --all --skip-parse --force   # embed again from the parsed output
```

Other options: `--staggered` (parse everything, then embed; for a parser and embedder on one GPU), `--parse-only`,
`--reset` (empty the collection first), `--backend`, `--collection`, `--no-enrich`, `--no-images`, `--text-only`,
`--input-dir`, `--output-dir`, `--parse-workers`. The command line does not hide a document while it is being stored,
so run it when nobody uses the library, or use Library Upload.

**What happens to a file** (`library_index.py`, the same rule for both ways, kept in `data/manifest.json`):

| The file | Result |
|---|---|
| has the same content as a document already in the library, under another name | not added (duplicate) |
| has the same name and content as a library document | skipped (unchanged) |
| has the same name as one of your own documents, with new content | replaces it (new version) |
| has a name already used by someone else's document or another library | stored under a new name |
| anything else | added |

## Library Upload and the command line

Library Upload (`upload/document.py`, `Ingester.embed_one`) and the command line (`run_pipeline.py` with
`embed.enumerate_embeddable`) build a document's embed inputs each in their own code. From the same content list they
store the same dense inputs, point ids and payloads, for PDF and Office parses, enriched or raw
(`tests/ingestion/test_embed_input_snapshot.py`), with one difference:

| | Command line | Library Upload |
|---|---|---|
| `bbox` of an element without a box (Office parses) | `[]` | `null` |

Contextual retrieval (`CONTEXTUALIZE=true`) and `--text-only` exist only on the command line.

## Research-gap ledger

The answers on open research questions read a ledger built from the research-gap tags (step 5): every open question of
the library, checked against the other documents and the literature. For a library that already holds documents, build
it once, in three resumable steps:

```bash
set -a; . config/.env; set +a
.venv/bin/python scripts/build_gap_ledger.py         # the questions of each document: extract, judge, merge
.venv/bin/python scripts/check_gap_ledger.py         # each question against the other documents
.venv/bin/python scripts/check_gap_literature.py     # each question against the literature
```

The gap updater (`scripts/update_gap_ledger.py`, started with Open WebUI) then keeps it current as documents are added
and removed.

## Display titles

The document list's `title` is what users see in the picker, in lists and in the model's source labels. A prefix
names the kind of document ("Dissertation: …", "Master's thesis: …", "Talk: …"); list questions use it to find
documents of the kind asked for. Titles must be unique within a library. To change titles, update the rows of
`memento_documents` and re-export the picker list with `memento.library.documents.export_library(engine, collection, OUTPUT_ROOT)`;
stored answers are kept.

## Collections

`<prefix>` comes from the domain profile.

| Collection | Content |
|---|---|
| `<prefix>-rag` | the library |
| `<prefix>-rag-gaps` | the gap ledger's questions |
| `<prefix>-temp` | Document Chat files |
| `<prefix>-rag-ocr`, `-rag-text`, `-rag-noenrich`, `-rag-noenrich-text`, `-rag-context` | evaluation variants (OCR parser, text-only embedder, no enrichment, contextual retrieval) |
| `<prefix>-sheets` | per-document fact sheets (evaluation) |

## Optional and evaluation modules

Off by default, used when switched on or by the evaluations:

- the knowledge graph (`memento/graph/build/`: `build_graph.py`, `extract_entities.py`, `merge_entities.py`,
  `embed_entities.py`, `work_links.py`, `llm_enrich.py`): `GRAPH_BUILD=true`, or
  `python -m memento.graph.build.build_graph --domain <name> --all` in bulk;
- contextual retrieval (`contextualize.py`): `CONTEXTUALIZE=true`, the contextual-retrieval collection;
- fact sheets (`memento/graph/build/fact_sheets.py`) and the evaluation collections (`evaluation/RQ1/collections/`:
  `build_text_collection.py`, `build_noenrich_text_collection.py`, `build_colqwen_collection.py`).

The OCR baseline uses the same pipeline with the OCR parser: `run_ingestion.sh --ocr`.

## Known limits

- Standalone micrographs and photos without text are found only by their visible text, such as scale bars.
- Tables continued across pages are stored as separate tables; the continuation loses its column headers.
- Corrupt files fail to parse and are reported as skipped.
- MinerU 4.0 changed its interface and output; Memento runs on 3.4.0.
