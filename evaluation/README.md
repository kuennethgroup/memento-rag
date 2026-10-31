# Evaluation

The scripts that build the ground truth and run the four evaluations of the thesis, for your own documents. No data
or results are included: each script reads a corpus that Memento has parsed and indexed (`config/domains/<name>.yaml`),
the model endpoints in `config/.env`, and writes its output under `data/` or a path you give. RQ1 runs on any corpus
as written; RQ2 and RQ3 keep the thesis' corpus names, models and endpoints in their scripts, listed in their sections,
and need adapting first. Run them from the
repository root with the project's `.venv`, after installing their extra packages once:
`.venv/bin/pip install -r requirements/evaluation.txt`. The scripts with options print them with `--help`; the RQ3
steps' scripts take the corpus or run name as written below.

The thesis' results come from an earlier revision of this code. Later fixes changed parts of the search, the routing,
the answer prompt and the Thorough check, so a new run can give other numbers than the thesis.

| Folder | Question | Needs |
|---|---|---|
| `RQ1/` | Do vision-language parsing and multimodal retrieval find the right page? | the parsed corpus in six collections, embedders, reranker, a judge model |
| `RQ2/` | Does a knowledge graph improve retrieval and answers? | the library collection, a graph (Neo4j), the answer and judge models |
| `RQ3/` | Does checking each claim against its sources improve the answers? | RQ1's ground truth, the answer, verifier and judge models |
| `RQ4/` | How does the deployed system work for its users? | a running Memento, the study questionnaire |

## RQ1: parsing and retrieval

1. **Collections:** the same corpus in six variants (names from the domain's prefix, `memento/config/settings.py`), all with
   `python -m memento.ingestion.run_pipeline --domain <d> --all --reset --collection <name>`:

   | Collection | Extra options |
   |---|---|
   | `<prefix>-rag` | `--backend vlm-engine` |
   | `<prefix>-rag-ocr` | `--backend pipeline` |
   | `<prefix>-rag-context` | `--backend vlm-engine`, with `CONTEXTUALIZE=true` |
   | `<prefix>-rag-noenrich` | `--backend vlm-engine --no-enrich` |
   | `<prefix>-rag-text` | `--backend vlm-engine --text-only --embedder-url <text embedder>` |
   | `<prefix>-rag-noenrich-text` | `evaluation/RQ1/collections/build_noenrich_text_collection.py --domain <d> --all --reset` instead |

   After the first, `--skip-parse` reuses the parsed output.
2. **Ground truth:** `evaluation/RQ1/auto_annotate.py --domain <d> --out evaluation/RQ1/ground_truth_auto.json`
   samples passages per content type (text, table, figure, equation), writes a question for each with the answer
   model and has an independent judge label the relevant pages. The judge is `JUDGE_MODEL` (default `glm-5.3-flash`)
   at `OPENAI_BASE_URL` with `CLOUD_API_KEY`; set the three in `config/.env`.
3. **Pooling:** `evaluation/RQ1/pool_relevance_all.py --domain <d> --sources vlm ocr ctx noenrich text8b noenrichtxt
   --in evaluation/RQ1/ground_truth_auto.json --out evaluation/RQ1/ground_truth_auto_pooled_all.json` retrieves with
   every collection and judges the pages not labelled yet (TREC-style), so no condition is scored against pages only
   another one found. `--judge-base-url` and `--judge-model` use another judge.
4. **Retrieval runs:** `evaluation/RQ1/rq1_eval.py --domain <d> --no-colqwen --exclude textrerank
   --in evaluation/RQ1/ground_truth_auto_pooled_all.json --out evaluation/RQ1/rq1_results.json` runs every condition
   (parser, dense / hybrid / reranked, modality ablations) on the pooled ground truth: Hit@k, MRR, recall and NDCG per
   question. The four text-reranker conditions run with the text reranker served, into the same results:
   `--only <conditions> --merge-into evaluation/RQ1/rq1_results.json`.
5. **Analysis:** `evaluation/RQ1/analyze_rq1_results.py --in evaluation/RQ1/rq1_results.json --out-dir
   evaluation/RQ1/analysis` builds the per-condition tables and paired tests (Benjamini-Hochberg corrected);
   `evaluation/RQ1/plot_rq1_results.py --in evaluation/RQ1/rq1_results.json --out-dir evaluation/RQ1/analysis/figures`
   draws them.
6. **Parsing quality:** `parse_quality.py` compares the parsing backends' outputs; `parse_judge.py` lets a
   vision-language judge decide their disagreements.

The scripts' defaults follow the same chain, so the `--in` and `--out` options can be left out.

## RQ2: graph-assisted retrieval

1. **Inputs** (`build/`): `rq2_pilot_slice.py` (a link-dense pilot of about 50 documents), `build_document_registry.py`
   (documents, works and their people), `rq2_dump_chunk_texts.py` and `rq2_gliner_extract.py` (entities with GLiNER,
   in its own environment), `rq2_fact_sheets_build.py` (per-document records for list questions).
2. **Graph:** `python -m memento.graph.build.build_graph --domain <d> --all` (the pilot: `--papers-file data/rq2/pilot_<d>.txt`);
   `graph_audit.py` samples entities, relations and merges and has a judge check them.
3. **Questions:** `questions/build.py --domain <d> --universe <pilot|full>` with `multihop` or `corpus-level` writes
   questions from the graph-blind text, pools their gold evidence and labels their hardness; RQ1's questions are the
   control set.
4. **Harness:** `harness/run.py --domain <d> --universe <pilot|full> --run <name> <stage>`, the stages in order:
   `arms` (every retrieval arm), `pool`, `answers`, `judge`, `score`, `stats`, `report`.

**Fixed in the scripts.** `questions/build.py` takes RQ1's questions as the control set from `GT_V4`, which names
the thesis' ground-truth files for `ceramics` and `polymer-engineering`; add your corpus there. The harness answers
and judges with `qwen3.8-27b` at `RQ2_LLM_URLS` (default `http://localhost:8091/v1`); the no-LLM graph arms read one
Neo4j per corpus from `LITE_PORTS`.

## RQ3: answer verification

RQ3 is a research harness for the thesis' two corpora: it runs on another corpus only after adapting the settings
below, which sit in the scripts, not in `config/.env`.

| Setting | Where | Thesis value |
|---|---|---|
| Corpora and their collections | `COLL` in `eval/run_eval.py`, `eval/freeze.py` and `eval/build_unanswerable.py` | `ceramics`, `polymer-engineering` |
| RQ1's ground truth | `GT` in `eval/select_questions.py` and `eval/build_nuggets.py` | `evaluation/RQ1/ground_truth_auto_v4_qwenjudge.json` and its polymer twin |
| RQ2's questions | `eval/select_questions.py` | `data/rq2/runs/final/<corpus>/gold.jsonl` (RQ2's `final` run) |
| Answer and verifier model | `QWEN_URL` in `eval/run_eval.py` (generation also `--qwen-url`), `eval/freeze.py`, `eval/build_*.py` | `qwen3.8-27b` at `http://localhost:8090/v1`, key `REASONING_API_KEY` |
| Judge | `eval/run_eval.py`, `eval/judge.py`, `eval/build_*.py` | `gemma-4-12b` at `http://localhost:8095/v1` |
| Search services | `eval/run_eval.py`, `eval/freeze.py`, `eval/build_unanswerable.py` | embedder 8021, reranker 8022, Qdrant 6333 on localhost |

The steps, once adapted:

1. **Questions:** `eval/select_questions.py` (the set of both corpora, no arguments), `eval/build_nuggets.py <corpus>`
   (reference facts), `eval/build_unanswerable.py <corpus>` (verified unanswerable questions).
2. **Freeze:** `eval/freeze.py <corpus>` runs the Library Chat pipe up to the answer call and stores routing and
   sources, so every strategy answers from the same retrieval.
3. **Runs:** `eval/run_eval.py --corpus <name> --run <name> --stage gen`, then `--stage judge` (the judge in
   `eval/judge.py`, blind to the strategy); `eval/stats.py <run>` and `eval/report.py <run>` for the paired statistics
   and the report.
4. **Checker benchmark:** `benchmark/build_claims.py` (a labelled claim set, written crosswise by two model families),
   `benchmark/review_figures.py list|record <corpus> …` and `benchmark/extract_figure_text.py` (figure claims, no
   arguments), `benchmark/run_benchmark.py` (a verifier or judge measured on it).

## RQ4: the deployed system

`feedback_form/create_study_form.gs`: a Google Apps Script that creates the study's questionnaire.
