"""Separate, persistent document vectors with configurable size and overlap."""

import json
from functools import lru_cache
from hashlib import sha256

from .corpus import Corpus, get_corpus, paragraphs
from .metadata_index import SearchSettings, ready_collection, search_settings, sync_records
from .collection_store import collection_guard


def split_document(document: dict, size: int, overlap: int) -> list[dict]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk size or overlap")
    result = []
    for paragraph_index, paragraph in enumerate(paragraphs(document)):
        text, start = paragraph["text"], 0
        while start < len(text):
            end = min(start + size, len(text))
            result.append({"id": f"{document['id']}:p{paragraph_index}:c{start}",
                           "text": text[start:end], "paragraph_index": paragraph_index,
                           "start_char": start, "end_char": end, "source_line": paragraph["source_line"]})
            if end == len(text):
                break
            start = end - overlap
    return result


@lru_cache(maxsize=4)
def document_records(corpus: Corpus, size: int, overlap: int) -> list[dict]:
    records = []
    for document in corpus.documents.values():
        for chunk in split_document(document, size, overlap):
            metadata = {key: document.get(key, "") for key in ("title", "kind", "source_type", "source_url", "license", "license_url", "attribution", "transformations", "retrieved_at")}
            metadata.update({"document_id": document["id"], "related_artwork_ids_json": json.dumps(document.get("related_artwork_ids", [])),
                             "paragraph_index": chunk["paragraph_index"], "start_char": chunk["start_char"], "end_char": chunk["end_char"],
                             "source_line": chunk["source_line"], "chunk_size": size, "chunk_overlap": overlap})
            record = {"id": chunk["id"], "document": chunk["text"], "metadata": metadata,
                      "embedding_text": document["title"] + "\n" + chunk["text"]}
            metadata["content_hash"] = sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            records.append(record)
    return records


def sync_document_index(settings: SearchSettings | None = None, corpus: Corpus | None = None, embedder=None) -> dict:
    settings = settings or search_settings()
    with collection_guard(settings):
        corpus = corpus or get_corpus(settings)
        records = document_records(corpus, settings.document_chunk_size, settings.document_chunk_overlap)
        result = sync_records(settings, settings.document_collection, records, embedder,
                              {"chunk_size": settings.document_chunk_size, "chunk_overlap": settings.document_chunk_overlap,
                               "source_documents": len(corpus.documents)})
        return {**result, "source_documents": len(corpus.documents), "chunk_size": settings.document_chunk_size,
                "chunk_overlap": settings.document_chunk_overlap}


def document_collection(settings: SearchSettings | None = None, corpus: Corpus | None = None):
    settings = settings or search_settings()
    corpus = corpus or get_corpus(settings)
    return ready_collection(settings, settings.document_collection,
                            document_records(corpus, settings.document_chunk_size, settings.document_chunk_overlap))


def document_index_status() -> dict:
    from .metadata_index import IndexUnavailableError

    settings, corpus = search_settings(), get_corpus()
    expected = len(document_records(corpus, settings.document_chunk_size, settings.document_chunk_overlap))
    try:
        collection = document_collection(settings, corpus)
        count, ready = collection.count(), True
    except IndexUnavailableError:
        count, ready = 0, False
    return {"ready": ready, "collection": settings.document_collection, "count": count,
            "expected_count": expected, "source_documents": len(corpus.documents),
            "chunk_size": settings.document_chunk_size, "chunk_overlap": settings.document_chunk_overlap,
            "embedding_model": settings.embedding_model}
