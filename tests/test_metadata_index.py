"""Real persistent Chroma tests with deterministic local vectors; no paid API."""

import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from backend import metadata_index as index


def local_vectors(texts, model):
    return [[float(len(text)), float(text.count("a") + 1), 1.0] for text in texts]


class MetadataIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = index.search_settings().model_copy(update={"persist_directory": self.directory.name})
        self.artworks = deepcopy(index.load_artworks()[:3])
        self.client = index.client_for(self.settings)
        # Release the Windows SQLite/HNSW handles before temporary-directory cleanup.
        self.addCleanup(self.client.close)

    def test_idempotent_indexing_keeps_fixed_ids_and_skips_paid_embeddings(self):
        first = index.sync_metadata_index(self.settings, self.artworks, local_vectors)
        with patch.object(index, "embed_texts", side_effect=AssertionError("Must reuse unchanged embeddings")):
            second = index.sync_metadata_index(self.settings, self.artworks)
        self.assertEqual(first["count"], 3)
        self.assertEqual(first["embedded"], 3)
        self.assertEqual(second["count"], 3)
        self.assertEqual(second["embedded"], 0)
        collection = index.metadata_collection(self.settings, self.artworks)
        rows = collection.get(include=["documents", "metadatas", "embeddings"])
        self.assertEqual(set(rows["ids"]), {art["id"] for art in self.artworks})
        self.assertEqual(len(rows["embeddings"]), 3)
        self.assertEqual(rows["metadatas"][0]["source_type"], "artwork_metadata")

    def test_changed_metadata_updates_one_record_and_removes_obsolete_ids(self):
        index.sync_metadata_index(self.settings, self.artworks, local_vectors)
        self.artworks = self.artworks[:2]
        self.artworks[0]["search_text"] += "\nAdditional verified metadata"
        result = index.sync_metadata_index(self.settings, self.artworks, local_vectors)
        self.assertEqual((result["count"], result["embedded"], result["removed"]), (2, 1, 1))
        self.assertEqual(index.metadata_collection(self.settings, self.artworks).count(), 2)

    def test_missing_stale_or_partial_index_does_not_silently_search(self):
        with self.assertRaises(index.IndexUnavailableError):
            index.metadata_collection(self.settings, self.artworks)
        index.sync_metadata_index(self.settings, self.artworks, local_vectors)
        modified = deepcopy(self.artworks)
        modified[0]["date_end"] += 1
        with self.assertRaises(index.IndexUnavailableError):
            index.metadata_collection(self.settings, modified)
        with patch.object(index, "embed_texts", side_effect=RuntimeError("Simulated provider failure")):
            with self.assertRaises(RuntimeError):
                index.sync_metadata_index(self.settings, modified)
        with self.assertRaises(index.IndexUnavailableError):
            index.metadata_collection(self.settings, self.artworks)

    def test_changed_embedding_model_requires_a_different_collection(self):
        index.sync_metadata_index(self.settings, self.artworks, local_vectors)
        changed = self.settings.model_copy(update={"embedding_model": "different-model"})
        with self.assertRaises(index.IndexUnavailableError):
            index.sync_metadata_index(changed, self.artworks, local_vectors)


if __name__ == "__main__":
    unittest.main()
