"""Grounded viewing-guide structure, citations, API routing and missing evidence."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.guide import NO_GUIDE, NO_SECTION, generate_guide
from backend.main import app


class GuideTests(unittest.TestCase):
    def setUp(self):
        self.art = {"id": "aic:28560", "title": "The Bedroom", "artist": "Vincent van Gogh", "date_display": "1889"}
        self.sources = [{"number": 1, "document_id": "aic:28560", "kind": "metadata", "excerpt": "The Bedroom. Vincent van Gogh. 1889."},
                        {"number": 2, "document_id": "museum:test", "kind": "artwork", "excerpt": "The painter understood the bedroom as a restful image."},
                        {"number": 3, "document_id": "unused:test", "kind": "style", "excerpt": "Unused evidence."}]
        self.guide = {"look": {"text": "문서에서 소개하는 침실 공간을 먼저 바라보세요.", "citations": [2]},
                      "context": {"text": "작가는 이 장면을 편안한 이미지로 이해했습니다.", "citations": [2]},
                      "reflect": {"text": "편안함이라는 설명을 떠올리면 이 공간은 어떻게 느껴지나요?", "citations": [2]}}
        self.model = Mock()

    def generate(self):
        self.model.invoke.return_value = SimpleNamespace(content=json.dumps(self.guide, ensure_ascii=False))
        return generate_guide(self.model, self.art, self.sources)

    def test_three_viewing_steps_keep_only_cited_retrieved_sources(self):
        answer = self.generate()
        for heading in ("1. 먼저 바라보기", "2. 알고 보면 좋은 이야기", "3. 다시 바라보기"):
            self.assertIn(heading, answer["answer"])
        self.assertEqual(answer["answer"].count("[근거 2]"), 3)
        self.assertEqual(answer["sources"], [self.sources[1]])
        prompt = self.model.invoke.call_args.args[0]
        evidence = json.loads(prompt[1].content)
        self.assertEqual(evidence["artwork"]["id"], self.art["id"])
        self.assertEqual(evidence["sources"], self.sources)
        self.assertIn("이미지를 직접 보거나 분석하지 않았다", prompt[0].content)

    def test_uncited_section_is_replaced_without_invented_facts_or_fake_sources(self):
        self.guide["context"] = {"text": "근거 없이 만든 제작 동기", "citations": []}
        answer = self.generate()
        self.assertIn(NO_SECTION, answer["answer"])
        self.assertNotIn("근거 없이 만든 제작 동기", answer["answer"])
        self.assertEqual(answer["sources"], [self.sources[1]])

    def test_no_supported_step_returns_explicit_unavailable_notice(self):
        self.guide = {key: {"text": "", "citations": []} for key in ("look", "context", "reflect")}
        self.assertEqual(self.generate(), {"answer": NO_GUIDE, "sources": []})

    def test_no_retrieved_sources_never_generates_from_general_knowledge(self):
        result = generate_guide(self.model, self.art, [])
        self.assertEqual(result, {"answer": NO_GUIDE, "sources": []})
        self.model.invoke.assert_not_called()

    def test_unknown_duplicate_and_inline_citations_are_rejected(self):
        for step in ({"text": "허구 근거", "citations": [99]}, {"text": "중복", "citations": [2, 2]},
                     {"text": "임의 인용 [근거 1]", "citations": [2]}, {"text": "https://invented.example/", "citations": [2]}):
            self.guide["look"] = step
            with self.subTest(step=step), self.assertRaises(RuntimeError):
                self.generate()

    def test_missing_or_extra_steps_and_malformed_json_fail_closed(self):
        for content in ('{}', 'not JSON', json.dumps({**self.guide, "extra": {"text": "extra", "citations": []}})):
            self.model.invoke.return_value = SimpleNamespace(content=content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                generate_guide(self.model, self.art, self.sources)

    def test_guide_flag_is_optional_and_uses_existing_chat_response_contract(self):
        client = TestClient(app)
        payload = {"artwork_id": self.art["id"], "question": "작품 감상 가이드", "feature": "guide", "history": []}
        with patch("backend.main.generate_answer", return_value={"answer": "감상 가이드", "sources": []}) as generate:
            response = client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"artwork_id": self.art["id"], "answer": "감상 가이드", "mode": "rag", "sources": []})
        self.assertEqual(generate.call_args.kwargs, {"feature": "guide"})
        for invalid in ("unknown", {}, None):
            payload["feature"] = invalid
            with self.subTest(feature=invalid), patch("backend.main.generate_answer") as generate:
                self.assertEqual(client.post("/api/chat", json=payload).status_code, 422)
                generate.assert_not_called()
