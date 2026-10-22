"""Natural-language search contracts and constraint enforcement, without LLM calls."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import discovery
from backend.main import app
from backend.metadata_index import IndexUnavailableError, load_artworks


def response(value):
    return SimpleNamespace(content=json.dumps(value, ensure_ascii=False))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.artworks = load_artworks()
        self.bedroom = next(art for art in self.artworks if art["id"] == "aic:28560")

    def test_hard_filters_apply_to_date_ranges_artist_origin_and_exact_style(self):
        plan = discovery.SearchPlan(query_en="Van Gogh", year_start=1801, year_end=1900, artist_terms=["Van Gogh"], style_terms=["Post-Impressionism"])
        self.assertTrue(discovery.matches_constraints(self.bedroom, plan))
        self.assertFalse(discovery.matches_constraints(self.bedroom, plan.model_copy(update={"style_terms": ["Impressionism"]})))
        self.assertFalse(discovery.matches_constraints(self.bedroom, plan.model_copy(update={"year_end": 1800})))
        self.assertFalse(discovery.matches_constraints(self.bedroom, plan.model_copy(update={"artist_terms": ["Monet"]})))
        self.assertFalse(discovery.matches_constraints(self.bedroom, plan.model_copy(update={"artist_terms": ["고흐"]})))
        self.assertFalse(discovery.matches_constraints(self.bedroom, plan.model_copy(update={"origin_terms": ["France"]})))

    def search(self, plan, matches, candidate_ids=None):
        model = Mock()
        model.invoke.side_effect = [response(plan), response({"matches": matches})]
        collection = Mock()
        collection.query.return_value = {"ids": [candidate_ids or [self.bedroom["id"]]]}
        with patch.object(discovery, "metadata_collection", return_value=collection), \
             patch.object(discovery, "get_model", return_value=model), \
             patch.object(discovery, "embed_query", return_value=(1.0, 0.0)):
            result = discovery.search_artworks("고흐 작품 찾아줘", 8)
        return result, model, collection

    def test_vector_candidates_are_llm_filtered_and_return_real_gallery_metadata(self):
        result, model, collection = self.search({"query_en": "Vincent van Gogh", "artist_terms": ["van Gogh"]}, [{"id": self.bedroom["id"], "reason": "작가가 Vincent van Gogh로 표기되어 있습니다."}])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["results"][0]["title"], "The Bedroom")
        self.assertEqual(result["results"][0]["source_url"], self.bedroom["source_url"])
        allowed = collection.query.call_args.kwargs["where"]["artwork_id"]["$in"]
        self.assertTrue(all("gogh" in art["artist"].lower() for art in self.artworks if art["id"] in allowed))
        self.assertEqual(model.invoke.call_count, 2)

    def test_hallucinated_ids_are_rejected_and_no_unfiltered_fallback_is_used(self):
        with self.assertRaises(discovery.DiscoveryError):
            self.search({"query_en": "Van Gogh"}, [{"id": "made-up", "reason": "invented"}])
        result, _, _ = self.search({"query_en": "Van Gogh"}, [])
        self.assertEqual(result["results"], [])

    def test_unknown_artist_yields_empty_without_embedding_or_selection_call(self):
        model = Mock()
        model.invoke.return_value = response({"query_en": "Unknown Artist", "artist_terms": ["DoesNotExist"]})
        with patch.object(discovery, "metadata_collection"), patch.object(discovery, "get_model", return_value=model), patch.object(discovery, "embed_query") as embed:
            result = discovery.search_artworks("없는 작가 작품 찾아줘")
        self.assertEqual(result["total"], 0)
        self.assertEqual(model.invoke.call_count, 1)
        embed.assert_not_called()

    def test_invalid_plans_cannot_remove_constraints_by_silent_fallback(self):
        for value in ({"query_en": "x", "year_start": 2000, "year_end": 1400}, {"query_en": "x", "invented_field": True}):
            with self.subTest(value=value), self.assertRaises(discovery.DiscoveryError):
                discovery.parse_response(response(value), discovery.SearchPlan)

    def test_api_validates_input_and_sanitizes_errors(self):
        for body in ({"query": " "}, {"query": "x" * 1001}, {"query": "x", "limit": 0}, {"query": "x", "filter": "override"}):
            with patch("backend.main.search_artworks") as search:
                self.assertEqual(self.client.post("/api/artworks/search", json=body).status_code, 422)
                search.assert_not_called()
        for error, code in ((IndexUnavailableError("private detail"), 503), (RuntimeError("private detail"), 502)):
            with patch("backend.main.search_artworks", side_effect=error):
                result = self.client.post("/api/artworks/search", json={"query": "고흐 작품"})
            self.assertEqual(result.status_code, code)
            self.assertNotIn("private detail", result.text)

    def test_api_search_response_and_index_status(self):
        output = {"query": "고흐 작품", "results": [], "total": 0}
        with patch("backend.main.search_artworks", return_value=output):
            self.assertEqual(self.client.post("/api/artworks/search", json={"query": "고흐 작품"}).json(), output)
        with patch("backend.main.metadata_index_status", return_value={"ready": False, "count": 0}):
            self.assertEqual(self.client.get("/api/index/status").json(), {"ready": False, "count": 0})


if __name__ == "__main__":
    unittest.main()
