"""F2004 journal, real local Chroma and mocked external/model boundaries."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.collection_store import CollectionStore, collection_guard, digest, normalized_query
from backend.corpus import get_corpus
from backend.document_index import document_collection, sync_document_index
from backend.external_documents import (CollectionUnavailableError, WikipediaSearch, collect_for_question,
                                        queries_for, recover_index, select_documents)
from backend.metadata_index import client_for, search_settings
from backend.docent import planned_search
from tests.test_document_index import local_vectors


class ExternalDocumentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.settings = search_settings().model_copy(update={"persist_directory": str(Path(directory.name) / "vectors"),
                                                              "collection_cache_directory": str(Path(directory.name) / "collection")})
        self.client = client_for(self.settings)
        self.addCleanup(self.client.close)
        self.corpus = get_corpus(self.settings)
        self.art = self.corpus.artworks["aic:28560"]
        self.plan = {"keywords": "Van Gogh complementary colours", "focus": "artwork",
                     "external_queries": ["Van Gogh complementary colours"]}
        self.doc = {"id": "wikipedia:en:999999991", "title": "Complementary colors", "text": "Historical colour theory and painters. " * 30,
                    "source_url": "https://en.wikipedia.org/wiki/Complementary_colors", "source_type": "wikipedia",
                    "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "attribution": "Wikipedia contributors, Complementary colors", "retrieved_at": "2026-10-22T00:00:00Z",
                    "transformations": "Plain-text lead extraction", "related_artwork_ids": []}
        self.other = {**self.doc, "id": "wikipedia:en:999999992", "title": "Unrelated document"}
        self.searcher = Mock()
        self.searcher.search.return_value = [self.doc, self.other]
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(content=json.dumps({"selected": [{"id": self.doc["id"], "kind": "style", "reason": "색채 질문에 필요한 이론"}]}))
        embed = patch("backend.metadata_index.embed_texts", side_effect=local_vectors)
        embed.start()
        self.addCleanup(embed.stop)

    def collect(self, plan=None, question="보색은 어떻게 사용했어?", art=None):
        with collection_guard(self.settings):
            return collect_for_question(art or self.art, question, plan or self.plan, self.model,
                                        self.settings, get_corpus(self.settings), self.searcher)

    def test_selection_indexes_only_chosen_and_always_keeps_own_document(self):
        report = self.collect()
        corpus = get_corpus(self.settings)
        self.assertIn(self.doc["id"], corpus.documents)
        self.assertNotIn(self.other["id"], corpus.documents)
        self.assertIn(self.doc["id"], corpus.eligible_document_ids(self.art["id"], "artwork"))
        another = next(key for key in corpus.artworks if key != self.art["id"])
        self.assertNotIn(self.doc["id"], corpus.eligible_document_ids(another, "artwork"))
        self.assertNotIn(self.doc["id"], corpus.eligible_document_ids(self.art["id"], "period"))
        collection = document_collection(self.settings, corpus)
        rows = collection.get(where={"document_id": self.doc["id"]}, include=["metadatas"])
        self.assertTrue(rows["ids"])
        self.assertTrue(all(self.art["id"] in m["related_artwork_ids_json"] and m["license"] == "CC-BY-SA-4.0" for m in rows["metadatas"]))
        self.assertTrue(collection.get(where={"document_id": self.art["artwork_document_id"]})["ids"])
        self.assertTrue(report["selected"][0]["forced"])
        self.assertEqual(report["rejected"][0]["id"], self.other["id"])
        self.assertIn(self.art["title"], self.searcher.search.call_args_list[0].args[0])

    def test_normalized_queries_are_persistent_and_skip_search_selection_and_embedding(self):
        first = self.collect()
        count = document_collection(self.settings, get_corpus(self.settings)).count()
        self.searcher.search.side_effect = AssertionError("Already collected query")
        self.model.invoke.side_effect = AssertionError("Already selected plan")
        plan = deepcopy(self.plan)
        plan["external_queries"] = ["  VAN GOGH   complementary COLOURS  "]
        with patch("backend.external_documents.sync_document_index", side_effect=AssertionError("Already indexed")):
            second = self.collect(plan)
        self.assertTrue(second["selection_cache_hit"])
        self.assertEqual(count, document_collection(self.settings, get_corpus(self.settings)).count())
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(len(CollectionStore(self.settings).snapshot()[0]), 2)  # mandatory + chosen

    def test_same_term_is_not_recollected_for_another_work_but_selection_is_scoped(self):
        self.collect()
        self.searcher.search.reset_mock()
        other_art = self.corpus.artworks["aic:27992"]
        second = self.collect(art=other_art)
        self.assertEqual(second["search_cache_hits"], 1)
        self.assertEqual(self.searcher.search.call_count, 1)  # only new mandatory work query
        self.assertEqual(self.model.invoke.call_count, 2)

    def test_empty_results_are_cached_and_own_document_bypasses_length_and_selector(self):
        self.searcher.search.return_value = []
        self.model.invoke.return_value = SimpleNamespace(content='{"selected":[]}')
        self.settings = self.settings.model_copy(update={"external_min_chars": 5000})
        first = self.collect()
        self.assertEqual([row["id"] for row in first["selected"]], [self.art["artwork_document_id"]])
        self.assertTrue(self.collect()["selection_cache_hit"])
        self.assertEqual(self.searcher.search.call_count, 2)

    def test_invalid_selection_retains_search_cache_without_indexing_and_can_retry(self):
        for content in ('{"selected":[{"id":"not-a-candidate","kind":"style","reason":"fake"}]}', 'not json'):
            self.model.invoke.return_value = SimpleNamespace(content=content)
            with self.assertRaises(CollectionUnavailableError):
                self.collect()
        self.assertEqual(self.searcher.search.call_count, 2)
        self.assertEqual(CollectionStore(self.settings).snapshot(), ([], []))
        self.model.invoke.return_value = SimpleNamespace(content='{"selected":[]}')
        self.collect()
        self.assertEqual(self.searcher.search.call_count, 2)

    def test_failed_embedding_resumes_from_selection_without_network_or_model(self):
        with patch("backend.external_documents.sync_document_index", side_effect=RuntimeError("embedding failed")):
            with self.assertRaises(RuntimeError):
                self.collect()
        self.assertTrue(CollectionStore(self.settings).dirty())
        self.searcher.search.side_effect = AssertionError("No recollection after embedding failure")
        self.model.invoke.side_effect = AssertionError("No reselection after embedding failure")
        report = self.collect()
        self.assertTrue(report["selection_cache_hit"])
        self.assertFalse(CollectionStore(self.settings).dirty())
        self.assertTrue(document_collection(self.settings, get_corpus(self.settings)).count())

    def test_failed_search_is_not_cached_and_only_failed_term_is_retried(self):
        self.searcher.search.side_effect = [[self.doc], CollectionUnavailableError("offline")]
        with self.assertRaises(CollectionUnavailableError):
            self.collect()
        self.searcher.search.side_effect = None
        self.searcher.search.return_value = [self.doc]
        self.collect()
        self.assertEqual(self.searcher.search.call_count, 3)

    def test_concurrent_identical_requests_collect_once_and_leave_a_ready_index(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(lambda _: self.collect(), range(2)))
        self.assertEqual(self.searcher.search.call_count, 2)
        self.assertEqual(self.model.invoke.call_count, 1)
        self.assertEqual(sum(row["selection_cache_hit"] for row in reports), 1)
        self.assertTrue(document_collection(self.settings, get_corpus(self.settings)).count())

    def test_provider_filters_disambiguation_and_preserves_source_revision_and_rights(self):
        pages = [{"pageid": 1, "title": "Color", "fullurl": "https://en.wikipedia.org/wiki/Color", "extract": "A colour.", "revisions": [{"revid": 42}]},
                 {"pageid": 2, "title": "Ambiguous", "pageprops": {"disambiguation": ""}},
                 {"pageid": 3, "title": "Injected URL", "fullurl": "http://127.0.0.1/private", "extract": "No"}]
        docs = WikipediaSearch.documents(pages, "https://en.wikipedia.org/w/api.php?action=query")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["revision_id"], 42)
        self.assertEqual(docs[0]["license"], "CC-BY-SA-4.0")
        self.assertIn("oldid=42", docs[0]["revision_url"])

    def test_provider_requires_identified_user_agent_and_never_caches_errors(self):
        with patch.dict("os.environ", {"WIKI_CONTACT": ""}), self.assertRaises(CollectionUnavailableError):
            WikipediaSearch()
        with patch.dict("os.environ", {"WIKI_CONTACT": "test@example.org"}), patch("backend.external_documents.time.sleep"), patch("backend.external_documents.requests.get") as get:
            response = Mock(status_code=429, headers={"Retry-After": "120"})
            get.return_value = response
            with self.assertRaises(CollectionUnavailableError):
                WikipediaSearch().search("color", 4)
            self.assertEqual(get.call_count, 1)
            self.assertIn("test@example.org", get.call_args.kwargs["headers"]["User-Agent"])

    def test_search_plan_persists_for_same_question_but_separates_followup_context(self):
        store = CollectionStore(self.settings)
        metadata = {key: self.art[key] for key in ("id", "title", "artist", "date_display")}
        planner = Mock(return_value=self.plan)
        with patch("backend.docent.search_plan", planner):
            first = planned_search(store, self.model, metadata, "색채?", [])
            second = planned_search(CollectionStore(self.settings), self.model, metadata, "  색채?  ", [])
            self.assertEqual(first, second)
            self.assertEqual(planner.call_count, 1)
            planned_search(store, self.model, metadata, "색채?", [{"role": "user", "content": "작가의 생애"}])
            self.assertEqual(planner.call_count, 2)

    def test_collection_failure_has_safe_api_message_and_no_provider_trace(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        with patch("backend.main.generate_answer", side_effect=CollectionUnavailableError("private internal message")):
            response = TestClient(app).post("/api/chat", json={"artwork_id": self.art["id"], "question": "작품 설명", "history": []})
        self.assertEqual(response.status_code, 503)
        self.assertIn("외부 문서", response.json()["detail"])
        self.assertNotIn("private", response.text)

    def test_fresh_and_cached_collection_emit_truthful_stages_and_prepared_titles(self):
        from threading import Event
        from backend.progress import progress_scope
        events = []
        with progress_scope(events.append, Event()):
            self.collect()
        stages = [event["stage"] for event in events]
        self.assertIn("collecting", stages)
        self.assertIn("selecting", stages)
        self.assertIn("indexing", stages)
        self.assertEqual(stages[-1], "collected")
        self.assertIn(self.doc["title"], events[-1]["titles"])
        events.clear()
        with progress_scope(events.append, Event()):
            self.collect()
        self.assertEqual([event["stage"] for event in events], ["selection_cached", "collected"])


if __name__ == "__main__":
    unittest.main()
