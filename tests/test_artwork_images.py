"""Image identity, integrity and serving checks; no external requests."""

import json
import unittest
from hashlib import sha256
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.artworks import DATA_DIR, gallery_catalog
from backend.main import app
from scripts.prepare_local_images import SOURCES, verified_image_url, webp_size


class ArtworkImageTests(unittest.TestCase):
    def test_three_verified_copies_are_first_candidates_and_serve_as_images(self):
        manifest = json.loads((DATA_DIR / "local_images.json").read_text(encoding="utf-8"))
        artworks = {art["id"]: art for art in gallery_catalog()}
        raw = {art["id"]: art for art in json.loads((DATA_DIR / "artworks.json").read_text(encoding="utf-8"))}
        client = TestClient(app)
        self.assertEqual(set(manifest), set(SOURCES))
        for artwork_id, info in manifest.items():
            with self.subTest(artwork=artwork_id):
                artwork = artworks[artwork_id]
                candidate = artwork["image_candidates"][0]
                self.assertEqual(candidate, "/static/artworks/" + info["file"])
                response = client.get(candidate)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.headers["content-type"].startswith("image/"))
                self.assertEqual(sha256(response.content).hexdigest(), info["sha256"])
                self.assertEqual(webp_size(response.content), (info["width"], info["height"]))
                self.assertGreaterEqual(min(info["width"], info["height"]), 1000)
                self.assertEqual(info["verified_museum_url"], raw[artwork_id]["source_url"])
                self.assertEqual(info["license"], "CC0-1.0")
                self.assertIn(raw[artwork_id]["image_url"], artwork["image_candidates"])

    def test_unverified_local_filename_is_not_accepted(self):
        original = json.loads

        def changed(text):
            data = original(text)
            if isinstance(data, dict) and "aic:16564" in data and data["aic:16564"].get("file"):
                data["aic:16564"]["file"] = "../../.env"
            return data

        with patch("backend.artworks.json.loads", side_effect=changed):
            artwork = next(art for art in gallery_catalog() if art["id"] == "aic:16564")
        self.assertTrue(all(not url.startswith("/static/") for url in artwork["image_candidates"]))

    def test_source_validation_rejects_similar_titles_other_museums_and_wrong_licenses(self):
        artwork = {"source_id": 16564, "title": "Branch of the Seine near Giverny (Mist)", "artist": "Claude Monet",
                   "source_url": "https://www.artic.edu/artworks/16564", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/"}
        image_url = "https://images.artheonmuseum.org/aic/full/branch-of-the-seine-near-giverny-mist-aic-16564.webp"
        schema = {"@type": "VisualArtwork", "name": artwork["title"], "creator": {"name": artwork["artist"]},
                  "image": {"contentUrl": image_url, "license": artwork["license_url"]}}

        def html(record, museum=artwork["source_url"]):
            return f'<a href="{museum}">Museum</a><p>Public Domain</p><script type="application/ld+json">{json.dumps(record)}</script>'

        self.assertEqual(verified_image_url(html(schema), artwork), image_url)
        for changed in (
            html(schema, "https://www.artic.edu/artworks/11111"),
            html({**schema, "name": "A similar painting"}),
            html({**schema, "image": {"contentUrl": image_url, "license": "All rights reserved"}}),
        ):
            with self.assertRaises(ValueError):
                verified_image_url(changed, artwork)


if __name__ == "__main__":
    unittest.main()
