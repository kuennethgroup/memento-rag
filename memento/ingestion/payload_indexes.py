# memento/ingestion/payload_indexes.py: Qdrant indexes on the passage fields Memento filters by
"""Filters on `source`, `section`, `type`, `gap_relevant` and `page` would scan every passage without an index, growing
with the library. Created when a collection is made and added to an existing one at its next upload; creating one is
online (no re-upload, search keeps working)."""
from __future__ import annotations

from qdrant_client.models import PayloadSchemaType

PAYLOAD_INDEXES = {
    "source": PayloadSchemaType.KEYWORD,
    "section": PayloadSchemaType.KEYWORD,
    "type": PayloadSchemaType.KEYWORD,
    "gap_relevant": PayloadSchemaType.BOOL,
    "page": PayloadSchemaType.INTEGER,
}


def ensure_payload_indexes(client, collection: str) -> list[str]:
    """Create the indexes this collection lacks; returns the fields indexed now (empty when all were there)."""
    have = set((client.get_collection(collection).payload_schema or {}).keys())
    made = []
    for field, schema in PAYLOAD_INDEXES.items():
        if field not in have:
            client.create_payload_index(collection_name=collection, field_name=field, field_schema=schema, wait=True)
            made.append(field)
    return made
