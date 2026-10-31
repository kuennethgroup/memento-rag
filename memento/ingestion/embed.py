"""What each element of a parsed document is embedded with, and under which id: its embed input (text and, for a
figure, the image, with the domain's instruction), its deterministic Qdrant point id, its BM25 sparse vector, and the
authors' text a table is embedded and answered with.

Point ids: graph `Chunk.id` and the Qdrant point id of the same chunk must be the same value, or graph retrieval returns
ids that resolve to nothing; every backend path derives them here. Library Upload (upload/document.py) computes the
same formula inline (tests assert they agree).

BM25: FastEmbed's `Qdrant/bm25` model (CPU-only, no model server), shared by ingestion and the one-time backfill; BM25
won every comparison against BM42 in the ablation on the real corpus.

Tables: enrichment links each table to up to three passages citing its label ("Table 4", "Tabelle 4"); only
label-cited text is used. It goes first in the dense vector, before caption and body, and to the answer model under the
table (payload `table_context`); keyword search and the displayed, cited text stay caption and body.
"""

from __future__ import annotations

import hashlib

import logging
from pathlib import Path

from qdrant_client.models import SparseVector

from memento.config.settings import DOMAIN


# ── Point ids ─────────────────────────────────────────────────────────────────
def chunk_point_id(paper_id: str, page_idx: int, item_type: str, done: int) -> int:
    """`done` is the chunk's 0-based index in document order."""
    digest = hashlib.sha256(
        f"{paper_id}_{page_idx}_{item_type}_{done}".encode()).hexdigest()
    return int(digest, 16) % (2 ** 63)


# ── BM25 sparse vectors ───────────────────────────────────────────────────────
_MODEL_NAME = "Qdrant/bm25"
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding
        _model = SparseTextEmbedding(model_name=_MODEL_NAME)
    return _model


def sparse_encode(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of texts into Qdrant SparseVector objects (BM25)."""
    model = _get_model()
    embeddings = list(model.embed(texts))
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def sparse_encode_query(text: str) -> SparseVector:
    """Encode a single query string — same model, query-side call."""
    model = _get_model()
    embeddings = list(model.query_embed([text]))
    e = embeddings[0]
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


# ── The authors' text under a table ───────────────────────────────────────────
def table_context(item: dict) -> str:
    """The passages citing this table by its label, or "" (no label match)."""
    if item.get("type") != "table" or item.get("enrichment_method") != "label_match":
        return ""
    return (item.get("enriched_context") or "").strip()


def table_embed_text(context: str, combined: str) -> str:
    """The dense-vector text of a table: the label-cited passages, then caption and body."""
    return "\n".join(filter(None, [context, combined]))


# ── Embed inputs of a parsed document ─────────────────────────────────────────
log = logging.getLogger("pipeline")    # the command line's log (run_pipeline configures it)

# aside_text, page_footnote and footer are watermark and licence boilerplate (arXiv stamps, DOI links, correspondence
# lines): skipped
SKIP_TYPES  = {'page_number', 'header', 'footer', 'aside_text', 'page_footnote', 'ref_text'}
TEXT_TYPES  = {'text', 'list', 'equation'}
IMAGE_TYPES = {'image', 'chart'}
TABLE_TYPES = {'table'}

EMBED_RETRIEVE_RERANK_INSTRUCTION = DOMAIN["instruction"]


# ── Build embed inputs ────────────────────────────────────────────────────────
def _fold_context(item: dict, embed_text: str) -> str:
    """Prepend the Contextual-Retrieval situating sentence (when contextualize_items wrote one) to the embedded text;
    the displayed `text` is never touched."""
    ctx = (item.get('chunk_context') or '').strip()
    return f"{ctx}\n{embed_text}" if ctx else embed_text


def build_embed_input(item: dict, images_dir: Path, text_only: bool = False,
                      enrichment_enabled: bool = True, generate_captions: bool = True):
    item_type = item.get('type')
    page      = item.get('page_idx', -1)
    bbox      = item.get('bbox', [])
    # canonical section bucket + nearest raw heading, annotated by
    # enrich_content_list() (assign_sections) before embedding
    section     = item.get('section')
    section_raw = item.get('section_raw')

    if item_type in SKIP_TYPES:
        return None

    if item.get('low_info'):
        return None

    if item_type in TEXT_TYPES:
        text = item.get('text', '').strip()
        if not text:
            return None
        embed_text = text
        equation_context = None
        if item_type == 'equation':
            # Bare LaTeX has no words a query would use: the author's surrounding sentences (enriched_context) are
            # folded in, so an equation is found by what it represents. The displayed text stays the equation.
            ref_text = item.get('enriched_context', '').strip()
            if ref_text:
                embed_text = '\n'.join(filter(None, [ref_text, text]))
                # BM25 uses the displayed `text`, which is real prose for every other type; for equations it is bare
                # LaTeX, so ref_text is carried separately for the sparse encoding, without changing what is displayed.
                equation_context = ref_text
        payload = {'type': item_type, 'page': page, 'bbox': bbox, 'text': text,
                   'section': section, 'section_raw': section_raw}
        if equation_context:
            payload['equation_context'] = equation_context
        if item.get('page_end') is not None:
            payload['page_end'] = item['page_end']
        return (
            {'text': _fold_context(item, embed_text), 'instruction': EMBED_RETRIEVE_RERANK_INSTRUCTION},
            payload,
        )

    if item_type in TABLE_TYPES:
        caption  = ' '.join(item.get('table_caption', []))
        body     = item.get('table_body', '')
        combined = f"{caption}\n{body}".strip()
        if not combined:
            return None
        # the passages citing the table by its label go first in the dense vector and to the answer model
        # (table_context above); BM25 and the displayed/cited `text` stay caption + body
        context = table_context(item)
        payload = {'type': 'table', 'page': page, 'bbox': bbox,
                   'caption': caption, 'text': combined,
                   'section': section, 'section_raw': section_raw}
        if context:
            payload['table_context'] = context
        return (
            {'text': _fold_context(item, table_embed_text(context, combined)),
             'instruction': EMBED_RETRIEVE_RERANK_INSTRUCTION},
            payload,
        )

    if item_type in IMAGE_TYPES:
        caption = ' '.join(
            item.get('image_caption', []) if item_type == 'image'
            else item.get('chart_caption', [])
        )
        vlm_content = item.get('content', '').strip()
        ref_text    = item.get('enriched_context', '').strip()
        # display_text is stored as the chunk's `text`: shown in citations and given to the LLM as context. MinerU's own
        # figure reconstruction (`vlm_content`) is kept out of it and out of embed_text (stored apart as `mineru_text`,
        # display only): on complex charts it can invent values that would look like real data once embedded. Only the
        # author's own text (enriched_context, caption) is searchable.
        display_text = '\n'.join(filter(None, [ref_text, caption]))
        embed_text    = display_text

        payload_extra = {'mineru_text': vlm_content} if vlm_content else {}
        payload_extra['section'] = section
        payload_extra['section_raw'] = section_raw

        img_path = images_dir / item.get('img_path', '')

        # VLM captioning fallback: image exists but has no author text
        caption_source = None
        if not display_text and img_path.is_file() and not text_only and enrichment_enabled:
            from memento.ingestion.caption_image import cached_caption
            generated = cached_caption(images_dir, item.get('img_path', ''), generate=generate_captions)
            if generated:
                display_text = generated
                embed_text = generated
                caption_source = "vlm"

        if img_path.is_file() and not text_only:
            embed_input = {'image': str(img_path), 'instruction': EMBED_RETRIEVE_RERANK_INSTRUCTION}
            if embed_text:
                embed_input['text'] = _fold_context(item, embed_text)
            payload = {'type': item_type, 'page': page, 'bbox': bbox,
                 'caption': caption, 'text': display_text, 'img_path': str(img_path),
                 **payload_extra}
            if caption_source:
                payload['caption_source'] = caption_source
            return (embed_input, payload)
        # image missing (or text_only ablation mode) — fall back to text-only
        # so the caption/context stays searchable instead of losing the chunk
        if embed_text:
            if not text_only:
                log.warning(f"image not found, embedding text-only: {img_path}")
            payload = {'type': item_type, 'page': page, 'bbox': bbox,
                 'caption': caption, 'text': display_text, 'img_path': str(img_path),
                 **payload_extra}
            if caption_source:
                payload['caption_source'] = caption_source
            return (
                {'text': _fold_context(item, embed_text), 'instruction': EMBED_RETRIEVE_RERANK_INSTRUCTION},
                payload,
            )
        if not text_only:
            log.warning(f"image not found and no caption/context: {img_path}")
        return None

    return None


def enumerate_embeddable(items: list[dict], paper_dir: Path, paper_id: str, *,
                         text_only: bool = False, enrichment_enabled: bool = True,
                         generate_captions: bool = True) -> list[dict]:
    """Single source of truth for what ingestion embeds and under which point id.
    build_graph.py uses this too, so graph Chunk ids cannot drift from Qdrant.

    Every item that ingestion embeds, in document order. Each row:
    {"chunk_id": str, "done": int, "item_idx": int, "item": dict,
     "embed_input": dict, "payload": dict}. chunk_id is the Qdrant point id
    (decimal string) = chunk_point_id(paper_id, payload.get('page', 0), payload['type'], done)."""
    rows: list[dict] = []
    for item_idx, item in enumerate(items):
        if item.get('text_level'):
            # headings are not embedded as near-empty chunks: enrich_content_list() folded them into each item's
            # section/section_raw
            continue
        result = build_embed_input(item, paper_dir, text_only=text_only,
                                   enrichment_enabled=enrichment_enabled,
                                   generate_captions=generate_captions)
        if result is None:
            continue
        embed_input, payload = result
        done = len(rows)
        rows.append({
            "chunk_id": str(chunk_point_id(paper_id, payload.get('page', 0), payload.get('type', ''), done)),
            "done": done, "item_idx": item_idx, "item": item,
            "embed_input": embed_input, "payload": payload,
        })
    return rows
