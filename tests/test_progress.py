"""Public streaming events, isolated requests and actionable offline setup errors."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import provider
from backend.main import app
from backend.metadata_index import IndexUnavailableError, client_for, search_settings
from backend.progress import MESSAGES, ProgressCancelled, progress_scope, report_progress
from backend.readiness import DOCUMENTS_MISSING, KEY_MISSING, METADATA_MISSING


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.payload = {"artwork_id": "aic:28560", "question": "이 작품에 대해 알려줘.", "history": []}

    def stream(self, payload=None):
        response = self.client.post("/api/chat/stream", json=payload or self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        return [json.loads(line) for line in response.text.splitlines()]

    def test_actual_stage_events_precede_unchanged_final_chat_result(self):
        def generate(art, question, history):
            for stage in ("checking", "collecting", "selecting", "indexing"):
                report_progress(stage)
            report_progress("collected", titles=["Museum explanation", "Artist biography"])
            report_progress("retrieving")
            report_progress("generating")
            report_progress("validating")
            return {"answer": "문서에 근거한 답변입니다.", "sources": []}
        with patch("backend.main.generate_answer", side_effect=generate):
            events = self.stream()
            ordinary = self.client.post("/api/chat", json=self.payload).json()
        self.assertEqual([event["stage"] for event in events[:-1]],
                         ["checking", "collecting", "selecting", "indexing", "collected", "retrieving", "generating", "validating"])
        self.assertEqual(events[-1], {"artwork_id": self.payload["artwork_id"], "type": "result", "data": ordinary})
        self.assertEqual(events[4]["titles"], ["Museum explanation", "Artist biography"])

    def test_missing_key_and_each_missing_index_have_specific_safe_error_frames(self):
        settings = search_settings()
        for error, expected in ((provider.MissingAPIKeyError(), KEY_MISSING),
                                (IndexUnavailableError("private provider trace", collection=settings.metadata_collection), METADATA_MISSING),
                                (IndexUnavailableError("private provider trace", collection=settings.document_collection), DOCUMENTS_MISSING)):
            with self.subTest(message=expected), patch("backend.main.generate_answer", side_effect=error):
                normal = self.client.post("/api/chat", json=self.payload)
                streamed = self.stream()
            self.assertEqual(normal.status_code, 503)
            self.assertEqual(normal.json()["detail"], expected)
            self.assertEqual(streamed[-1]["type"], "error")
            self.assertEqual(streamed[-1]["status"], 503)
            self.assertEqual(streamed[-1]["detail"], expected)
            self.assertNotIn("private provider trace", json.dumps(streamed))

    def test_unknown_provider_error_is_sanitized_and_does_not_emit_success(self):
        with patch("backend.main.generate_answer", side_effect=RuntimeError("private model prompt and key")):
            events = self.stream()
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("private model", json.dumps(events))
        self.assertNotIn("result", [event["type"] for event in events])

    def test_validation_errors_and_unknown_work_do_not_start_generation(self):
        with patch("backend.main.generate_answer") as generate:
            response = self.client.post("/api/chat/stream", json={**self.payload, "question": ""})
            self.assertEqual(response.status_code, 422)
            events = self.stream({**self.payload, "artwork_id": "missing"})
        self.assertEqual(events[-1]["status"], 404)
        generate.assert_not_called()

    def test_parallel_requests_have_separate_progress_and_terminal_artwork_ids(self):
        def generate(art, question, history):
            report_progress("collected", titles=[art["id"]])
            report_progress("generating")
            return {"answer": art["id"], "sources": []}
        def run(art_id):
            with TestClient(app) as client:
                response = client.post("/api/chat/stream", json={**self.payload, "artwork_id": art_id})
                return [json.loads(line) for line in response.text.splitlines()]
        with patch("backend.main.generate_answer", side_effect=generate), ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(run, ["aic:28560", "aic:27992"]))
        for art_id, events in zip(["aic:28560", "aic:27992"], responses):
            self.assertTrue(all(event["artwork_id"] == art_id for event in events))
            self.assertEqual(events[0]["titles"], [art_id])
            self.assertEqual(events[-1]["data"]["answer"], art_id)

    def test_progress_is_bounded_and_cancelled_work_stops_at_next_stage(self):
        events, cancelled = [], Event()
        with progress_scope(events.append, cancelled):
            report_progress("collected", titles=["X" * 1000] * 2 + ["a", "b", "c", "d"])
            cancelled.set()
            with self.assertRaises(ProgressCancelled):
                report_progress("generating")
        report_progress("generating")  # Scope has been reset; no event leaks.
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0]["titles"]), 4)
        self.assertEqual(len(events[0]["titles"][0]), 160)

    def test_actual_empty_indexes_and_missing_env_key_are_reported_without_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = search_settings().model_copy(update={"persist_directory": str(Path(directory) / "vectors"),
                                                              "collection_cache_directory": str(Path(directory) / "collection")})
            config = Path(directory) / "search.json"
            config.write_text(settings.model_dump_json(), encoding="utf-8")
            with patch("backend.metadata_index.CONFIG_PATH", config), patch.dict("os.environ", {"GMS_KEY": ""}), \
                 patch("backend.provider.ChatOpenAI", side_effect=AssertionError("No readiness model call")), \
                 patch("backend.metadata_index.OpenAI", side_effect=AssertionError("No readiness embedding call")):
                data = self.client.get("/api/docent/status").json()
            self.assertFalse(data["ready"])
            self.assertEqual({issue["code"] for issue in data["issues"]},
                             {"GMS_KEY_MISSING", "METADATA_INDEX_UNAVAILABLE", "DOCUMENT_INDEX_UNAVAILABLE"})
            self.assertIn("index_artworks", str(data))
            self.assertIn("index_documents", str(data))
            client_for(settings).close()

    def test_cached_clients_do_not_hide_a_removed_key(self):
        from backend import metadata_index
        provider._model.cache_clear()
        metadata_index._embedding_client.cache_clear()
        self.addCleanup(provider._model.cache_clear)
        self.addCleanup(metadata_index._embedding_client.cache_clear)
        with patch.dict("os.environ", {"GMS_KEY": "test-only-key"}), patch("backend.provider.ChatOpenAI", return_value=Mock()), \
             patch("backend.metadata_index.OpenAI", return_value=Mock()):
            provider.get_model()
            metadata_index.embedding_client()
            with patch.dict("os.environ", {"GMS_KEY": ""}):
                with self.assertRaises(provider.MissingAPIKeyError):
                    provider.get_model()
                with self.assertRaises(provider.MissingAPIKeyError):
                    metadata_index.embedding_client()
