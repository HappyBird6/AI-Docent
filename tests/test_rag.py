"""Real two-collection vector retrieval using deterministic local fixture vectors."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from backend import corpus as source_data, rag
from backend.corpus import Corpus
from backend.document_index import document_collection, split_document, sync_document_index
from backend.metadata_index import client_for, metadata_collection, search_settings, sync_metadata_index


def local_vectors(texts, model):
    return [[float(len(text)), float(text.count("a") + 1), 1.0] for text in texts]


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.settings = search_settings().model_copy(update={"persist_directory": cls.directory.name,
                                                           "collection_cache_directory": str(Path(cls.directory.name) / "collection")})
        cls.corpus = rag.get_corpus(cls.settings)
        cls.client = client_for(cls.settings)
        cls.addClassCleanup(cls.directory.cleanup)
        cls.addClassCleanup(cls.client.close)
        sync_metadata_index(cls.settings, list(cls.corpus.artworks.values()), local_vectors)
        sync_document_index(cls.settings, cls.corpus, local_vectors)
        cls.context = rag.RetrievalContext(cls.corpus, cls.settings,
                                          metadata_collection(cls.settings), document_collection(cls.settings, cls.corpus))

    def retrieve(self, artwork_id, keywords, focus="artwork", context=None):
        with patch.object(rag, "embed_query", return_value=tuple(local_vectors([keywords], "test")[0])):
            return rag.retrieve_sources(artwork_id, keywords, focus, context or self.context)

    def test_all_100_artworks_include_their_metadata_and_only_linked_documents(self):
        self.assertEqual(len(self.corpus.documents), 221)
        for art in self.corpus.artworks.values():
            with self.subTest(artwork=art["id"]):
                sources = self.retrieve(art["id"], "composition color biography")
                self.assertEqual(sources[0]["document_id"], art["id"])
                self.assertEqual(sources[0]["source_type"], "artwork_metadata")
                self.assertEqual(sources[1]["document_id"], art["artwork_document_id"])
                allowed = self.corpus.eligible_document_ids(art["id"], "artwork")
                self.assertTrue(all(source["document_id"] in allowed for source in sources[1:]))
                for source in sources:
                    self.assertTrue(all(source[key] for key in ("excerpt", "license", "license_url", "attribution", "url")))
                    self.assertLessEqual(len(source["excerpt"]), 3000)

    def test_modern_dutch_artist_cannot_pull_golden_age_or_enlightenment_context(self):
        sources = self.retrieve("aic:28560", "Dutch Golden Age Calvinism trade Enlightenment", "period")
        self.assertFalse(any(source["contextual"] for source in sources))
        self.assertTrue(any(source["kind"] == "style" for source in sources))

    def test_period_context_preserves_source_license_and_is_labeled_as_general(self):
        for course in ("dutch-painting", "early-modern-europe", "enlightenment"):
            art = next(art for art in self.corpus.artworks.values() if Corpus.historical_match(art, {"course_slug": course}))
            # Pin the candidate course in this test; production policy can admit multiple compatible courses.
            with patch.object(Corpus, "historical_match", side_effect=lambda art, doc: doc.get("course_slug") == course):
                sources = self.retrieve(art["id"], "history society trade religion reason", "period")
            contexts = [source for source in sources if source["contextual"]]
            self.assertTrue(contexts)
            self.assertTrue(contexts[0]["document_id"].startswith("openlearn:" + course + ":"))
            self.assertEqual(contexts[0]["license"], "CC-BY-NC-SA-4.0")

    def test_guide_retrieves_artwork_artist_and_style_within_current_work_scope(self):
        sources = self.retrieve("aic:28560", "bedroom composition color artist style", "guide")
        self.assertEqual(sources[0]["kind"], "metadata")
        self.assertTrue({"artwork", "artist", "style"}.issubset({source["kind"] for source in sources}))
        self.assertFalse(any(source["contextual"] for source in sources))

    def test_top_k_changes_evidence_without_reindexing_and_duplicate_ids_are_removed(self):
        small = replace(self.context, settings=self.settings.model_copy(update={"retrieval_top_k": 1}))
        first = self.retrieve("aic:28560", "bedroom color composition", context=small)
        second = self.retrieve("aic:28560", "bedroom color composition")
        self.assertEqual(len(first), 2)  # One metadata record plus one external chunk.
        self.assertGreater(sum(len(source["chunk_ids"]) for source in second), 2)
        ids = [chunk_id for source in second for chunk_id in source["chunk_ids"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_overlapping_spans_are_merged_in_the_displayed_excerpt(self):
        text = "abcdefghij" * 250
        document = {**next(iter(self.corpus.documents.values())), "text": text}
        chunks = split_document(document, 1000, 200)[:2]
        rows = [{"id": chunk["id"], "metadata": chunk} for chunk in chunks]
        source = rag.external_source(document, rows, 2)
        self.assertEqual(source["excerpt"], text[:1800])
        self.assertEqual(len(source["chunk_ids"]), 2)

    def test_real_vector_query_selects_a_different_passage_for_a_different_query(self):
        art = {**self.corpus.artworks["aic:28560"], "artwork_document_id": "test:colors", "artist_document_ids": [], "style_document_ids": []}
        doc = {**self.corpus.documents["aic:artwork:28560"], "id": "test:colors", "title": "Colors", "text": "Red pigment study.\n\nBlue pigment study."}
        corpus = Corpus([art], [doc])
        settings = self.settings.model_copy(update={"metadata_collection": "test_colors_meta", "document_collection": "test_colors_docs", "retrieval_top_k": 1})

        def colors(texts, model):
            return [[1., 0., 0.] if "Red" in text else [0., 1., 0.] if "Blue" in text else [0., 0., 1.] for text in texts]

        sync_metadata_index(settings, [art], colors)
        sync_document_index(settings, corpus, colors)
        context = rag.RetrievalContext(corpus, settings, metadata_collection(settings, [art]), document_collection(settings, corpus))
        for query in ("Red", "Blue"):
            with patch.object(rag, "embed_query", return_value=tuple(colors([query], "test")[0])):
                sources = rag.retrieve_sources(art["id"], query, context=context)
            self.assertIn(query, sources[1]["excerpt"])

    def test_missing_corpus_raises_without_general_knowledge_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(source_data, "DATA_DIR", Path(directory)):
            with self.assertRaises(source_data.CorpusUnavailableError):
                source_data.get_corpus()


if __name__ == "__main__":
    unittest.main()
