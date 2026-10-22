"""Real Chroma document indexing with local test vectors and no provider calls."""

import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch

from backend.corpus import Corpus, get_corpus
from backend.document_index import document_collection, split_document, sync_document_index
from backend.metadata_index import IndexUnavailableError, client_for, search_settings


def local_vectors(texts, model):
    return [[float(len(text)), float(text.count("a") + 1), 1.0] for text in texts]


class DocumentIndexTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.settings = search_settings().model_copy(update={"persist_directory": directory.name,
                                                              "collection_cache_directory": str(Path(directory.name) / "collection")})
        self.client = client_for(self.settings)
        self.addCleanup(self.client.close)
        source = get_corpus(self.settings)
        arts = list(source.artworks.values())[:2]
        linked = {doc_id for art in arts for doc_id in source.eligible_document_ids(art["id"], "artwork")}
        self.corpus = Corpus(arts, [source.documents[doc_id] for doc_id in linked])

    def test_idempotent_separate_collection_has_source_rights_and_fixed_chunk_ids(self):
        first = sync_document_index(self.settings, self.corpus, local_vectors)
        with patch("backend.metadata_index.embed_texts", side_effect=AssertionError("No repeated embedding")):
            second = sync_document_index(self.settings, self.corpus)
        self.assertEqual(first["count"], second["count"])
        self.assertEqual(second["embedded"], 0)
        self.assertNotEqual(first["collection"], self.settings.metadata_collection)
        rows = document_collection(self.settings, self.corpus).get(include=["metadatas"])
        self.assertEqual(len(rows["ids"]), len(set(rows["ids"])))
        for meta in rows["metadatas"]:
            for field in ("document_id", "source_url", "license", "license_url", "attribution", "related_artwork_ids_json"):
                self.assertTrue(meta[field])

    def test_chunk_configuration_changes_count_and_requires_reindex(self):
        first = sync_document_index(self.settings, self.corpus, local_vectors)
        changed = self.settings.model_copy(update={"document_chunk_size": 400, "document_chunk_overlap": 100})
        with self.assertRaises(IndexUnavailableError):
            document_collection(changed, self.corpus)
        second = sync_document_index(changed, self.corpus, local_vectors)
        self.assertGreater(second["count"], first["count"])
        self.assertEqual(document_collection(changed, self.corpus).count(), second["count"])

    def test_configured_overlap_is_exact_and_never_crosses_omitted_passages(self):
        text = "0123456789" * 50
        document = {"id": "openlearn:test", "text": "unused", "passages": [{"text": text, "source_line": 10}, {"text": "SECOND PASSAGE", "source_line": 20}]}
        chunks = split_document(document, 200, 40)
        first = [chunk for chunk in chunks if chunk["paragraph_index"] == 0]
        self.assertEqual(first[0]["text"][-40:], first[1]["text"][:40])
        self.assertEqual(first[1]["start_char"], 160)
        self.assertTrue(all(len(chunk["text"]) <= 200 for chunk in chunks))
        self.assertEqual(chunks[-1]["text"], "SECOND PASSAGE")
        self.assertEqual(chunks[-1]["source_line"], 20)

    def test_failed_reindex_blocks_readers_and_can_resume(self):
        sync_document_index(self.settings, self.corpus, local_vectors)
        docs = deepcopy(list(self.corpus.documents.values()))
        docs[0]["text"] += "\n\nAdditional source paragraph."
        changed = Corpus(list(self.corpus.artworks.values()), docs)
        with self.assertRaises(RuntimeError):
            sync_document_index(self.settings, changed, lambda *_: (_ for _ in ()).throw(RuntimeError("provider failed")))
        with self.assertRaises(IndexUnavailableError):
            document_collection(self.settings, self.corpus)
        recovered = sync_document_index(self.settings, changed, local_vectors)
        self.assertTrue(recovered["ready"])
        self.assertEqual(document_collection(self.settings, changed).count(), recovered["count"])

    def test_invalid_overlap_or_identical_collection_names_are_rejected(self):
        values = search_settings().model_dump()
        for changes in ({"document_chunk_overlap": values["document_chunk_size"]}, {"document_collection": values["metadata_collection"]}):
            with self.assertRaises(ValueError):
                type(self.settings).model_validate({**values, **changes})


if __name__ == "__main__":
    unittest.main()
