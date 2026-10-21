"""Artwork-scoped BM25 retrieval over the licensed local corpus; no embeddings."""

import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "art"
STOP_WORDS = set("a an and are as at be by for from has in is it its of on or that the their this to was were which with about explain painting artwork artist period".split())


class CorpusUnavailableError(RuntimeError):
    pass


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [word for word in re.findall(r"[a-z0-9]+", normalized) if len(word) > 1 and word not in STOP_WORDS]


def chunks(document: dict) -> list[str]:
    paragraphs = [p["text"] for p in document.get("passages", [])] or re.split(r"\n\s*\n", document["text"])
    result = []
    for paragraph in paragraphs:
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
            for offset in range(0, len(sentence), 1400):
                part = sentence[offset:offset + 1400]
                if len(current) + len(part) + 1 > 1400 and current:
                    result.append(current)
                    current = ""
                current = f"{current} {part}".strip()
        if current:
            result.append(current)
    return result


class Corpus:
    def __init__(self, artworks: list[dict], documents: list[dict]):
        self.artworks = {item["id"]: item for item in artworks}
        self.documents = {item["id"]: item for item in documents}
        if not self.artworks or not self.documents or len(self.documents) != len(documents):
            raise ValueError("Empty or duplicated corpus")
        self.chunks = {}
        frequencies = Counter()
        lengths = []
        for document in documents:
            parts = chunks(document)
            if not parts or not all(document.get(k) for k in ("source_url", "license", "license_url", "attribution")):
                raise ValueError("Incomplete source metadata")
            rows = []
            for index, text in enumerate(parts):
                terms = Counter(tokens(document["title"] + " " + text))
                length = sum(terms.values())
                rows.append((index, text, terms, length))
                frequencies.update(terms.keys())
                lengths.append(length)
            self.chunks[document["id"]] = rows
        count = len(lengths)
        self.average_length = sum(lengths) / max(count, 1)
        self.idf = {word: math.log(1 + (count - frequency + .5) / (frequency + .5)) for word, frequency in frequencies.items()}

    def score(self, terms: list[str], row: tuple) -> float:
        _, _, frequencies, length = row
        return sum(self.idf.get(term, 0) * (frequencies[term] * 2.2) /
                   (frequencies[term] + 1.2 * (.25 + .75 * length / self.average_length))
                   for term in set(terms) if frequencies[term])

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

    def retrieve(self, artwork_id: str, keywords: str, focus: str = "artwork") -> list[dict]:
        artwork = self.artworks[artwork_id]
        linked = [artwork["artwork_document_id"], *artwork.get("artist_document_ids", []), *artwork.get("style_document_ids", [])]
        if any(document_id not in self.documents for document_id in linked):
            raise CorpusUnavailableError("Missing linked document")
        candidates = [self.documents[document_id] for document_id in dict.fromkeys(linked)]
        if focus == "period":
            candidates += [doc for doc in self.documents.values() if doc["kind"] == "historical_context" and self.historical_match(artwork, doc)]
        query = tokens(keywords)
        ranked = []
        for document in candidates:
            rows = sorted(self.chunks[document["id"]], key=lambda row: (-self.score(query, row), row[0]))
            best = self.score(query, rows[0])
            kind = document["kind"]
            if kind == "historical_context" and best <= 0:
                continue
            priority = 8 if kind == {"artwork": "artwork", "artist": "artist", "period": "style"}.get(focus) else 0
            if kind == "historical_context":
                priority = 4
            ranked.append((priority + best, document, rows))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        anchor = next(item for item in ranked if item[1]["id"] == artwork["artwork_document_id"])
        selected = [anchor]
        historical_count = 0
        for item in ranked:
            if item is anchor:
                continue
            if item[1]["kind"] == "historical_context":
                if historical_count >= 2:
                    continue
                historical_count += 1
            selected.append(item)
            if len(selected) == 5:
                break
        results = []
        for number, (_, document, rows) in enumerate(selected, 1):
            picked = sorted(rows[:2], key=lambda row: row[0])
            excerpt = "\n\n[…]\n\n".join(row[1] for row in picked)
            results.append({
                "number": number, "document_id": document["id"],
                "chunk_ids": [f"{document['id']}:{row[0]}" for row in picked],
                "title": document["title"], "url": document["source_url"],
                "kind": document["kind"], "source_type": document["source_type"],
                "license": document["license"], "license_url": document["license_url"],
                "attribution": document["attribution"], "transformations": document.get("transformations", ""),
                "excerpt": excerpt, "retrieved_at": document.get("retrieved_at", ""),
                "contextual": document["kind"] == "historical_context",
            })
        return results


@lru_cache(maxsize=1)
def _load_corpus(directory: str, signature: tuple) -> Corpus:
    path = Path(directory)
    return Corpus(json.loads((path / "artworks.json").read_text(encoding="utf-8")),
                  json.loads((path / "documents.json").read_text(encoding="utf-8")))


def get_corpus() -> Corpus:
    try:
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (DATA_DIR / "artworks.json", DATA_DIR / "documents.json"))
        return _load_corpus(str(DATA_DIR), signature)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CorpusUnavailableError("Cannot load collected corpus") from error
