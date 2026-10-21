"""Corpus retrieval tests; no provider requests or network access."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import rag


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = rag.get_corpus()

    def test_all_100_artworks_are_scoped_to_their_own_documents(self):
        self.assertEqual(len(self.corpus.documents), 221)
        for artwork in self.corpus.artworks.values():
            with self.subTest(artwork=artwork["id"]):
                results = self.corpus.retrieve(artwork["id"], "composition color biography", "artwork")
                linked = {artwork["artwork_document_id"], *artwork["artist_document_ids"], *artwork["style_document_ids"]}
                self.assertEqual(results[0]["document_id"], artwork["artwork_document_id"])
                self.assertTrue(all(result["document_id"] in linked for result in results))
                for source in results:
                    self.assertTrue(source["excerpt"] and source["license"] and source["license_url"] and source["attribution"])
                    doc = self.corpus.documents[source["document_id"]]
                    for chunk_id in source["chunk_ids"]:
                        index = int(chunk_id.rsplit(":", 1)[1])
                        self.assertIn(self.corpus.chunks[doc["id"]][index][1], source["excerpt"])

    def test_19th_century_dutch_artist_does_not_retrieve_17th_century_course(self):
        sources = self.corpus.retrieve("aic:28560", "Dutch Golden Age Calvinism trade Enlightenment", "period")
        self.assertFalse(any(source["contextual"] for source in sources))
        self.assertTrue(any(source["kind"] == "style" for source in sources))

    def test_dutch_17th_century_context_is_eligible_and_has_original_license(self):
        artwork = next(art for art in self.corpus.artworks.values() if rag.Corpus.historical_match(art, {"course_slug": "dutch-painting"}))
        sources = self.corpus.retrieve(artwork["id"], "Dutch Republic Calvinism religion market trade", "period")
        dutch = [source for source in sources if source["document_id"].startswith("openlearn:dutch-painting:")]
        self.assertTrue(dutch)
        self.assertEqual(dutch[0]["license"], "CC-BY-NC-SA-4.0")
        self.assertTrue(dutch[0]["contextual"])
        self.assertIn("Jason Gaiger", dutch[0]["attribution"])

    def test_early_modern_and_enlightenment_can_be_retrieved_when_period_matches(self):
        for course, query in (("early-modern-europe", "Reformation social order religious Europe"), ("enlightenment", "Enlightenment reason science empiricism revolution")):
            artwork = next(art for art in self.corpus.artworks.values() if rag.Corpus.historical_match(art, {"course_slug": course}))
            results = self.corpus.retrieve(artwork["id"], query, "period")
            self.assertTrue(any(source["document_id"].startswith("openlearn:" + course + ":") for source in results))

    def test_question_selects_different_biography_passages(self):
        artwork = next(art for art in self.corpus.artworks.values() if any(len(self.corpus.chunks[doc]) > 4 for doc in art["artist_document_ids"]))
        doc_id = artwork["artist_document_ids"][0]
        rows = self.corpus.chunks[doc_id]
        # A term unique to a later chunk must rank that chunk above an unrelated opening.
        term = next((term for row in rows[2:] for term in row[2] if term not in rows[0][2] and term not in rows[1][2]), None)
        self.assertIsNotNone(term)
        result = next(source for source in self.corpus.retrieve(artwork["id"], term, "artist") if source["document_id"] == doc_id)
        self.assertIn(term, rag.tokens(result["excerpt"]))

    def test_corrupt_or_missing_corpus_raises_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag, "DATA_DIR", Path(directory)):
            with self.assertRaises(rag.CorpusUnavailableError):
                rag.get_corpus()


if __name__ == "__main__":
    unittest.main()
