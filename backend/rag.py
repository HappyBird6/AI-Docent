"""Retrieve and merge artwork metadata and scoped external-document vectors."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .corpus import Corpus, CorpusUnavailableError, get_corpus, paragraphs
from .document_index import document_collection, document_records
from .metadata_index import IndexUnavailableError, SearchSettings, embed_query, metadata_collection, search_settings


@dataclass
class RetrievalContext:
    corpus: Corpus
    settings: SearchSettings
    metadata: Any
    documents: Any


def prepare_retrieval(corpus: Corpus | None = None) -> RetrievalContext:
    corpus, settings = corpus or get_corpus(), search_settings()
    return RetrievalContext(corpus, settings, metadata_collection(settings, list(corpus.artworks.values())),
                            document_collection(settings, corpus))


def metadata_source(artwork: dict, collection) -> dict:
    row = collection.get(ids=[artwork["id"]], include=["documents", "metadatas"])
    if row["ids"] != [artwork["id"]] or not row["documents"][0]:
        raise IndexUnavailableError("Current artwork metadata is missing")
    return {"number": 1, "document_id": artwork["id"], "chunk_ids": [artwork["id"]],
            "title": "작품 메타데이터 · " + artwork["title"], "url": artwork["source_url"],
            "kind": "metadata", "source_type": "artwork_metadata", "contextual": False,
            "license": artwork.get("metadata_license", "CC0-1.0"), "license_url": artwork["license_url"],
            "attribution": "Art Institute of Chicago · 작품 메타데이터",
            "transformations": "Museum metadata fields and image alternative text combined; no generated prose or image analysis.",
            "excerpt": row["documents"][0], "retrieved_at": artwork.get("retrieved_at", "")}


def external_source(document: dict, records: list[dict], number: int) -> dict:
    # Union overlapping spans from the same original paragraph; preserve omissions.
    spans = []
    for record in sorted(records, key=lambda row: (row["metadata"]["paragraph_index"], row["metadata"]["start_char"])):
        meta = record["metadata"]
        part, start, end = meta["paragraph_index"], meta["start_char"], meta["end_char"]
        if spans and spans[-1][0] == part and start <= spans[-1][2]:
            spans[-1] = (part, spans[-1][1], max(spans[-1][2], end))
        else:
            spans.append((part, start, end))
    original = paragraphs(document)
    excerpt = "\n\n[…]\n\n".join(original[part]["text"][start:end] for part, start, end in spans)
    return {"number": number, "document_id": document["id"], "chunk_ids": [record["id"] for record in records],
            "title": document["title"], "url": document["source_url"], "kind": document["kind"],
            "source_type": document["source_type"], "license": document["license"], "license_url": document["license_url"],
            "attribution": document["attribution"], "transformations": document.get("transformations", "") +
            " Retrieved spans selected; overlapping spans merged; gaps marked […]; original language retained.",
            "excerpt": excerpt, "retrieved_at": document.get("retrieved_at", ""),
            "contextual": document["kind"] == "historical_context"}


def retrieve_sources(artwork_id: str, keywords: str, focus: str = "artwork", context: RetrievalContext | None = None) -> list[dict]:
    context = context or prepare_retrieval()
    corpus, settings = context.corpus, context.settings
    artwork = corpus.artworks[artwork_id]
    sources = [metadata_source(artwork, context.metadata)]
    allowed = set(corpus.eligible_document_ids(artwork_id, focus))
    all_records = document_records(corpus, settings.document_chunk_size, settings.document_chunk_overlap)
    records = {record["id"]: record for record in all_records if record["metadata"]["document_id"] in allowed}
    vector = list(embed_query(keywords, settings.embedding_model))

    def query(ids: list[str], count: int) -> list[tuple[float, dict]]:
        count = min(count, sum(record["metadata"]["document_id"] in ids for record in records.values()))
        if not ids or not count:
            return []
        result = context.documents.query(query_embeddings=[vector], n_results=count,
                                         where={"document_id": {"$in": ids}}, include=["distances"])
        ranked = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            if chunk_id not in records or records[chunk_id]["metadata"]["document_id"] not in ids:
                raise IndexUnavailableError("Document outside artwork scope")
            if not math.isfinite(distance):
                raise IndexUnavailableError("Invalid vector distance")
            ranked.append((distance, records[chunk_id]))
        return sorted(ranked, key=lambda row: (row[0], row[1]["id"]))

    own = query([artwork["artwork_document_id"]], 1)
    if not own:
        raise IndexUnavailableError("Missing artwork explanation")
    ranked = query(sorted(allowed), settings.retrieval_top_k * 4)
    preferred_kind = "artist" if focus == "artist" else "style" if focus == "period" else None
    preferred = query([doc_id for doc_id in allowed if corpus.documents[doc_id]["kind"] == preferred_kind], 1) if preferred_kind else []
    if focus == "guide":
        preferred = [row for kind in ("artist", "style")
                     for row in query([doc_id for doc_id in allowed if corpus.documents[doc_id]["kind"] == kind], 1)]
    historical = query([doc_id for doc_id in allowed if corpus.documents[doc_id]["kind"] == "historical_context"], 1) if focus in ("period", "guide") else []
    selected, seen_ids, seen_text, per_document = [], set(), set(), Counter()
    for distance, record in own + preferred + historical + ranked:
        doc_id = record["metadata"]["document_id"]
        content_hash = sha256(re.sub(r"\s+", " ", record["document"]).strip().encode()).hexdigest()
        if record["id"] in seen_ids or content_hash in seen_text:
            continue
        if selected and distance > settings.retrieval_max_distance:
            continue
        if per_document[doc_id] >= settings.retrieval_chunks_per_source:
            continue
        if doc_id not in per_document and len(per_document) >= settings.retrieval_max_sources - 1:
            continue
        selected.append(record)
        seen_ids.add(record["id"])
        seen_text.add(content_hash)
        per_document[doc_id] += 1
        if len(selected) == settings.retrieval_top_k:
            break
    for doc_id in per_document:
        rows = [record for record in selected if record["metadata"]["document_id"] == doc_id]
        sources.append(external_source(corpus.documents[doc_id], rows, len(sources) + 1))
    return sources
