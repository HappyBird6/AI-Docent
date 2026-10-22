"""Persistent query/selection journal; source data is separate from shipped seeds."""

import json
import sqlite3
import unicodedata
from contextlib import contextmanager
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from filelock import FileLock

from .metadata_index import ROOT, search_settings


def normalized_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cache_directory(settings=None) -> Path:
    return (ROOT / (settings or search_settings()).collection_cache_directory).resolve()


@lru_cache(maxsize=8)
def _lock(path: str):
    return FileLock(path, timeout=180)


@contextmanager
def collection_guard(settings=None):
    directory = cache_directory(settings)
    directory.mkdir(parents=True, exist_ok=True)
    # Thread-local reentrancy plus an OS file lock serialize retrieval/index writes
    # in this process and in other workers using the same cache directory.
    with _lock(str(directory / "collection.lock")):
        yield


class CollectionStore:
    def __init__(self, settings=None):
        directory = cache_directory(settings)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "journal.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS searches (
                    key TEXT PRIMARY KEY, query TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS selections (
                    key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS plans (
                    key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS links (
                    artwork_id TEXT NOT NULL, focus TEXT NOT NULL, document_id TEXT NOT NULL,
                    PRIMARY KEY(artwork_id, focus, document_id));
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def read(self, table: str, key: str):
        if table not in ("searches", "selections", "plans"):
            raise ValueError("Invalid journal table")
        with self.connect() as db:
            row = db.execute(f"SELECT payload FROM {table} WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_search(self, key: str, query: str, payload: dict):
        with self.connect() as db:
            db.execute("INSERT INTO searches VALUES (?, ?, ?)", (key, query, json.dumps(payload, ensure_ascii=False)))

    def save_plan(self, key: str, plan: dict):
        with self.connect() as db:
            db.execute("INSERT INTO plans VALUES (?, ?)", (key, json.dumps(plan, ensure_ascii=False)))

    def reports(self, artwork_id: str | None = None, limit: int = 10) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM selections ORDER BY rowid DESC")
            result = []
            for row in rows:
                report = json.loads(row[0])
                if artwork_id is None or report["artwork_id"] == artwork_id:
                    result.append(report)
                    if len(result) >= limit:
                        break
        return result

    def save_selection(self, key: str, report: dict, documents: list[dict], artwork_id: str, focus: str):
        with self.connect() as db:
            for doc in documents:
                db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?)", (doc["id"], json.dumps(doc, ensure_ascii=False)))
                db.execute("INSERT OR IGNORE INTO links VALUES (?, ?, ?)", (artwork_id, focus, doc["id"]))
            db.execute("INSERT INTO selections VALUES (?, ?)", (key, json.dumps(report, ensure_ascii=False)))
            # Set before indexing. Failed embeddings can resume from this journal
            # without searching the network or asking the selector again.
            db.execute("INSERT OR REPLACE INTO state VALUES ('index_dirty', '1')")

    def dirty(self) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE key='index_dirty'").fetchone()
        return bool(row and row[0] == "1")

    def mark_indexed(self):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO state VALUES ('index_dirty', '0')")

    def snapshot(self) -> tuple[list[dict], list[tuple]]:
        with self.connect() as db:
            docs = [json.loads(row[0]) for row in db.execute("SELECT payload FROM documents ORDER BY id")]
            links = db.execute("SELECT artwork_id, focus, document_id FROM links ORDER BY artwork_id, focus, document_id").fetchall()
        return docs, links


def runtime_snapshot(settings=None) -> tuple[list[dict], list[tuple]]:
    if not (cache_directory(settings) / "journal.sqlite3").is_file():
        return [], []
    return CollectionStore(settings).snapshot()
