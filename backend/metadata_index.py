"""One stable, embedded record per artwork in a persistent Chroma collection."""

import json
import math
import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provider import GMS_BASE_URL, MissingAPIKeyError, configured_key

ROOT = Path(__file__).resolve().parents[1]
ARTWORKS_PATH = ROOT / "data" / "art" / "artworks.json"
CONFIG_PATH = ROOT / "config" / "search.json"
INDEX_VERSION = 1


class IndexUnavailableError(RuntimeError):
    def __init__(self, message="Index unavailable", *, collection=None):
        super().__init__(message)
        self.collection = collection


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata_collection: str = Field(min_length=3, max_length=100)
    persist_directory: str
    embedding_model: str
    embedding_batch_size: int = Field(ge=1, le=128)
    candidate_count: int = Field(ge=1, le=100)
    max_results: int = Field(ge=1, le=20)
    document_collection: str = Field(default="artwork_documents", min_length=3, max_length=100)
    document_chunk_size: int = Field(default=1000, ge=200, le=1400)
    document_chunk_overlap: int = Field(default=200, ge=0)
    retrieval_top_k: int = Field(default=6, ge=1, le=8)
    retrieval_max_sources: int = Field(default=5, ge=2, le=5)
    retrieval_chunks_per_source: int = Field(default=2, ge=1, le=2)
    retrieval_max_distance: float = Field(default=0.9, ge=0, le=2)
    collection_cache_directory: str = ".cache/collection"
    external_queries: int = Field(default=2, ge=1, le=3)
    external_candidates_per_query: int = Field(default=4, ge=1, le=8)
    external_max_selected: int = Field(default=3, ge=1, le=6)
    external_min_chars: int = Field(default=300, ge=0, le=5000)

    @model_validator(mode="after")
    def compatible_settings(self):
        if self.document_chunk_overlap >= self.document_chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        if self.metadata_collection == self.document_collection:
            raise ValueError("Metadata and documents require separate collections")
        return self


def search_settings() -> SearchSettings:
    return SearchSettings.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))


def load_artworks() -> list[dict]:
    artworks = json.loads(ARTWORKS_PATH.read_text(encoding="utf-8"))
    if not isinstance(artworks, list) or not artworks or len({art["id"] for art in artworks}) != len(artworks):
        raise ValueError("Empty or duplicate artwork metadata")
    if not all(isinstance(art.get("search_text"), str) and art["search_text"].strip() for art in artworks):
        raise ValueError("Missing artwork index text")
    return artworks


def index_record(artwork: dict) -> dict:
    text = artwork["search_text"]
    if artwork.get("image_alt"):
        text += "\nImage description: " + artwork["image_alt"]
    metadata = {
        "artwork_id": artwork["id"], "title": artwork["title"], "artist": artwork["artist"],
        "date_display": artwork.get("date_display", ""),
        "source_url": artwork["source_url"], "source_type": "artwork_metadata",
        "license": artwork.get("metadata_license", "CC0-1.0"),
    }
    for key in ("date_start", "date_end"):
        if isinstance(artwork.get(key), int):
            metadata[key] = artwork[key]
    payload = {"id": artwork["id"], "document": text, "metadata": metadata}
    metadata["content_hash"] = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def signature(records: list[dict]) -> str:
    value = [(record["id"], record["metadata"]["content_hash"]) for record in sorted(records, key=lambda item: item["id"])]
    return sha256(json.dumps(value).encode()).hexdigest()


@lru_cache(maxsize=4)
def chroma_client(directory: str):
    return chromadb.PersistentClient(path=directory, settings=Settings(anonymized_telemetry=False))


def client_for(settings: SearchSettings):
    return chroma_client(str((ROOT / settings.persist_directory).resolve()))


def embedding_client() -> OpenAI:
    return _embedding_client(configured_key())


@lru_cache(maxsize=1)
def _embedding_client(key: str) -> OpenAI:
    return OpenAI(api_key=key, base_url=GMS_BASE_URL, timeout=30, max_retries=1)


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    response = embedding_client().embeddings.create(model=model, input=texts)
    rows = sorted(response.data, key=lambda item: item.index)
    vectors = [row.embedding for row in rows]
    if [row.index for row in rows] != list(range(len(texts))) or not vectors:
        raise RuntimeError("Invalid embedding response")
    dimension = len(vectors[0])
    if not dimension or any(len(vector) != dimension or not all(math.isfinite(value) for value in vector) for vector in vectors):
        raise RuntimeError("Invalid embedding vectors")
    return vectors


@lru_cache(maxsize=128)
def embed_query(query: str, model: str) -> tuple[float, ...]:
    return tuple(embed_texts([query], model)[0])


def sync_metadata_index(settings: SearchSettings | None = None, artworks: list[dict] | None = None, embedder=None) -> dict:
    settings = settings or search_settings()
    records = [index_record(artwork) for artwork in (load_artworks() if artworks is None else artworks)]
    return sync_records(settings, settings.metadata_collection, records, embedder)


def sync_records(settings: SearchSettings, collection_name: str, records: list[dict], embedder=None, extra_metadata: dict | None = None) -> dict:
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("Empty or duplicate artwork records")
    client = client_for(settings)
    metadata = {"embedding_model": settings.embedding_model, "index_version": INDEX_VERSION,
                "provider": "ssafy-gms", "hnsw:space": "cosine", **(extra_metadata or {})}
    collection = client.get_or_create_collection(collection_name, embedding_function=None, metadata=metadata)
    if any(collection.metadata.get(key) != metadata[key] for key in ("embedding_model", "index_version", "provider")):
        raise IndexUnavailableError("Embedding settings changed; use a new metadata_collection name")
    old = collection.get(include=["metadatas"])
    existing = dict(zip(old["ids"], old["metadatas"]))
    changed = [record for record in records if existing.get(record["id"], {}).get("content_hash") != record["metadata"]["content_hash"]]
    stale = sorted(set(existing) - {record["id"] for record in records})
    # Until all batches finish, API requests must treat this collection as unavailable.
    mutable_meta = {key: value for key, value in collection.metadata.items() if key != "hnsw:space"}
    collection.modify(metadata={**mutable_meta, "state": "building"})
    embedder = embedder or embed_texts
    for start in range(0, len(changed), settings.embedding_batch_size):
        batch = changed[start:start + settings.embedding_batch_size]
        vectors = embedder([record.get("embedding_text", record["document"]) for record in batch], settings.embedding_model)
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding count mismatch")
        collection.upsert(ids=[record["id"] for record in batch],
                          documents=[record["document"] for record in batch],
                          metadatas=[record["metadata"] for record in batch], embeddings=vectors)
    if stale:
        collection.delete(ids=stale)
    count = collection.count()
    if count != len(records):
        raise RuntimeError("Metadata index count mismatch")
    collection.modify(metadata={**collection.metadata, **(extra_metadata or {}), "state": "ready", "corpus_signature": signature(records), "expected_count": len(records)})
    return {"collection": collection_name, "count": count, "embedded": len(changed),
            "unchanged": len(records) - len(changed), "removed": len(stale), "ready": True}


def metadata_collection(settings: SearchSettings | None = None, artworks: list[dict] | None = None):
    settings = settings or search_settings()
    records = [index_record(art) for art in (load_artworks() if artworks is None else artworks)]
    return ready_collection(settings, settings.metadata_collection, records)


def ready_collection(settings: SearchSettings, collection_name: str, records: list[dict]):
    try:
        collection = client_for(settings).get_collection(collection_name, embedding_function=None)
        meta = collection.metadata or {}
        if (meta.get("state") != "ready" or meta.get("embedding_model") != settings.embedding_model or
                meta.get("index_version") != INDEX_VERSION or meta.get("corpus_signature") != signature(records) or
                collection.count() != len(records)):
            raise IndexUnavailableError("Missing, stale, or incomplete metadata index", collection=collection_name)
        return collection
    except IndexUnavailableError:
        raise
    except Exception as error:
        raise IndexUnavailableError("Cannot open metadata index", collection=collection_name) from error


def metadata_index_status() -> dict:
    settings = search_settings()
    expected = len(load_artworks())
    try:
        collection = metadata_collection(settings)
        return {"ready": True, "collection": collection.name, "count": collection.count(),
                "expected_count": expected, "embedding_model": settings.embedding_model}
    except IndexUnavailableError:
        return {"ready": False, "collection": settings.metadata_collection, "count": 0,
                "expected_count": expected, "embedding_model": settings.embedding_model}
