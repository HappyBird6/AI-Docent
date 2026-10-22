"""Validated source documents and artwork links; never joins omitted passages."""

import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "art"


class CorpusUnavailableError(RuntimeError):
    pass


def paragraphs(document: dict) -> list[dict]:
    if document.get("passages"):
        return [{"text": part["text"].strip(), "source_line": part.get("source_line", -1)} for part in document["passages"] if part["text"].strip()]
    return [{"text": text.strip(), "source_line": -1} for text in re.split(r"\n\s*\n", document["text"]) if text.strip()]


class Corpus:
    def __init__(self, artworks: list[dict], documents: list[dict], extra_links: list[tuple] | None = None):
        self.artworks = {art["id"]: art for art in artworks}
        self.documents = {doc["id"]: doc for doc in documents}
        self.extra_links = extra_links or []
        if not self.artworks or not self.documents or len(self.documents) != len(documents):
            raise ValueError("Empty or duplicated corpus")
        for doc in documents:
            if not all(doc.get(key) for key in ("text", "title", "kind", "source_type", "source_url", "license", "license_url", "attribution")) or not paragraphs(doc):
                raise ValueError("Incomplete document or source metadata")
        for art in artworks:
            linked = [art["artwork_document_id"], *art.get("artist_document_ids", []), *art.get("style_document_ids", [])]
            if any(doc_id not in self.documents for doc_id in linked):
                raise ValueError("Missing linked artwork document")

    @staticmethod
    def historical_match(artwork: dict, document: dict) -> bool:
        start, end = artwork.get("date_start"), artwork.get("date_end")
        if not isinstance(start, int) or not isinstance(end, int):
            return False
        course = document.get("course_slug")
        if course == "early-modern-europe":
            return start <= 1780 and end >= 1500
        if course == "enlightenment":
            return start <= 1830 and end >= 1700
        if course == "dutch-painting":
            geography = " ".join(str(artwork.get(key, "")) for key in ("place_of_origin", "artist_display")).lower()
            return start <= 1700 and end >= 1600 and any(term in geography for term in ("netherland", "dutch", "holland"))
        return False

    def eligible_document_ids(self, artwork_id: str, focus: str) -> list[str]:
        art = self.artworks[artwork_id]
        linked = [art["artwork_document_id"], *art.get("artist_document_ids", []), *art.get("style_document_ids", [])]
        if focus in ("period", "guide"):
            linked.extend(doc["id"] for doc in self.documents.values() if doc["kind"] == "historical_context" and self.historical_match(art, doc))
        linked.extend(doc_id for art_id, question_focus, doc_id in self.extra_links
                      if art_id == artwork_id and (focus == "guide" or question_focus == focus) and doc_id in self.documents)
        return list(dict.fromkeys(linked))


@lru_cache(maxsize=1)
def _load_corpus(directory: str, file_signature: tuple) -> Corpus:
    path = Path(directory)
    return Corpus(json.loads((path / "artworks.json").read_text(encoding="utf-8")),
                  json.loads((path / "documents.json").read_text(encoding="utf-8")))


def get_corpus(settings=None) -> Corpus:
    try:
        stamp = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (DATA_DIR / "artworks.json", DATA_DIR / "documents.json"))
        seed = _load_corpus(str(DATA_DIR), stamp)
        from .collection_store import runtime_snapshot
        documents, links = runtime_snapshot(settings)
        if not links:
            return seed
        merged = dict(seed.documents)
        for doc in documents:
            # Curated seeds remain authoritative when a shared Wikipedia page
            # is encountered again. Only its artwork links are extended.
            merged.setdefault(doc["id"], doc)
        for artwork_id, _, doc_id in links:
            if artwork_id in seed.artworks and doc_id in merged:
                doc = dict(merged[doc_id])
                doc["related_artwork_ids"] = sorted(set(doc.get("related_artwork_ids", []) + [artwork_id]))
                merged[doc_id] = doc
        return Corpus(list(seed.artworks.values()), list(merged.values()), links)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CorpusUnavailableError("Cannot load collected corpus") from error
