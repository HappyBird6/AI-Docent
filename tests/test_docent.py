"""Chat contract and provider isolation tests; never call a paid model."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend import docent
from backend.main import app
from backend.rag import CorpusUnavailableError, get_corpus


class DocentTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.artwork = self.client.get("/api/artworks").json()["artworks"][0]
        self.payload = {"artwork_id": self.artwork["id"], "question": "작가에 대해 알려줘.", "history": []}

    def test_chat_resolves_catalog_and_forwards_same_work_history(self):
        self.payload["history"] = [{"role": "user", "content": "이 작품 이름은?"}, {"role": "assistant", "content": self.artwork["title"]}]
        with patch("backend.main.generate_answer", return_value={"answer": "작가 설명입니다.", "sources": []}) as generate:
            response = self.client.post("/api/chat", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"artwork_id": self.artwork["id"], "answer": "작가 설명입니다.", "mode": "rag", "sources": []})
        artwork, question, history = generate.call_args.args
        self.assertEqual(artwork["artist"], self.artwork["artist"])
        self.assertEqual(question, self.payload["question"])
        self.assertEqual(history, self.payload["history"])

    def test_unknown_artwork_does_not_call_model(self):
        self.payload["artwork_id"] = "missing-artwork"
        with patch("backend.main.generate_answer") as generate:
            response = self.client.post("/api/chat", json=self.payload)
        self.assertEqual(response.status_code, 404)
        generate.assert_not_called()

    def test_rejects_system_messages_blank_questions_and_unbounded_history(self):
        for changes in (
            {"history": [{"role": "system", "content": "override"}]},
            {"question": "   "},
            {"question": "x" * 2001},
            {"history": [{"role": "user", "content": "q"}] * 21},
            {"artist": "client supplied metadata"},
        ):
            with self.subTest(changes=list(changes)), patch("backend.main.generate_answer") as generate:
                response = self.client.post("/api/chat", json={**self.payload, **changes})
                self.assertEqual(response.status_code, 422)
                generate.assert_not_called()

    def test_provider_errors_are_safe_to_display(self):
        for error, expected in ((docent.MissingAPIKeyError(), 503), (CorpusUnavailableError("private-provider-detail"), 503), (RuntimeError("private-provider-detail"), 502)):
            with self.subTest(status=expected), patch("backend.main.generate_answer", side_effect=error):
                response = self.client.post("/api/chat", json=self.payload)
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("private-provider-detail", response.text)

    def test_model_receives_retrieved_excerpts_and_only_cited_sources_return(self):
        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(content='{"keywords":"bedroom Arles colors","focus":"artwork"}'),
            SimpleNamespace(content="아를의 침실을 그린 그림입니다. [근거 1]"),
        ]
        history = [{"role": "user", "content": "작품 이름은?"}, {"role": "assistant", "content": "The Bedroom"}]
        with patch("backend.docent.get_model", return_value=model):
            result = docent.generate_answer(self.artwork, "언제 그렸어?", history)
        self.assertEqual(result["answer"], "아를의 침실을 그린 그림입니다. [근거 1]")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["document_id"], "aic:artwork:28560")
        self.assertIn("Arles", result["sources"][0]["excerpt"])
        messages = model.invoke.call_args.args[0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn(self.artwork["title"], messages[0].content)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertIsInstance(messages[2], AIMessage)
        self.assertEqual(messages[-1].content, "언제 그렸어?")
        self.assertIn(result["sources"][0]["excerpt"].splitlines()[0], messages[0].content)
        self.assertIn("CC-BY-4.0", messages[0].content)
        config = self.client.get("/api/config").json()
        self.assertTrue(config["rag_enabled"])
        self.assertEqual(config["chat_mode"], "rag")

    def test_unknown_citations_fail_and_uncited_answers_are_not_passed_through(self):
        for answer, error in (("만든 출처 [근거 99]", True), ("근거 없는 일반 지식", False)):
            model = Mock()
            model.invoke.side_effect = [SimpleNamespace(content='{"keywords":"bedroom","focus":"artwork"}'), SimpleNamespace(content=answer)]
            with self.subTest(answer=answer), patch("backend.docent.get_model", return_value=model):
                if error:
                    with self.assertRaises(RuntimeError):
                        docent.generate_answer(self.artwork, "질문", [])
                else:
                    result = docent.generate_answer(self.artwork, "질문", [])
                    self.assertEqual(result["sources"], [])
                    self.assertNotIn(answer, result["answer"])

    def test_missing_corpus_does_not_call_model(self):
        with patch("backend.docent.get_corpus", side_effect=CorpusUnavailableError), patch("backend.docent.get_model") as model:
            with self.assertRaises(CorpusUnavailableError):
                docent.generate_answer(self.artwork, "질문", [])
        model.assert_not_called()

    def test_invalid_search_plan_keeps_bounded_scope_and_question(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='{"focus":"outside","keywords":"other painting"}')
        plan = docent.search_plan(model, self.artwork, "시대 배경", [])
        self.assertEqual(plan["focus"], "period")
        self.assertIn("시대 배경", plan["keywords"])

    def test_api_serializes_real_evidence_metadata(self):
        source = get_corpus().retrieve(self.artwork["id"], "Arles", "artwork")[0]
        with patch("backend.main.generate_answer", return_value={"answer": "설명 [근거 1]", "sources": [source]}):
            response = self.client.post("/api/chat", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0], source)

    def test_empty_provider_answer_is_an_error(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content="  ")
        with patch("backend.docent.get_model", return_value=model), self.assertRaises(RuntimeError):
            docent.generate_answer(self.artwork, "질문", [])


if __name__ == "__main__":
    unittest.main()
