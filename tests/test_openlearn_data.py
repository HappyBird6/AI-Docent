"""Licensed text selection, source integrity and repeatable data merging."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import prepare_openlearn_data as ol
from scripts.prepare_art_data import preserve_openlearn

AUTHOR_TEXT = "Trade and manufacturing changed the economic structure of European towns, while the development of commercial markets influenced the production and ownership of paintings."
OTHER_TEXT = "Religious institutions and political authorities competed for influence in early modern society, and this competition affected the kinds of cultural work commissioned by wealthy patrons."


def capture(text, slug="early-modern-europe"):
    lines = text.splitlines()
    return {"course_slug": slug, "page_url": "https://www.open.edu/openlearn/example", "format": "printable", "total_lines": len(lines),
            "lines": [{"number": i, "text": line} for i, line in enumerate(lines)]}


class OpenLearnTests(unittest.TestCase):
    def test_quotations_continuations_media_and_poetry_are_not_selected(self):
        sample = capture("\n".join([
            "# Course", "## 1 Trade", AUTHOR_TEXT,
            "> This quotation belongs to an external author and must not enter the selected corpus.",
            "A long continuation of an external quotation without a repeated quote marker must also be excluded rather than treated as the course author's own narrative.",
            "(External Author, 1965, p. 10)",
            "A mixed paragraph containing ‘quoted wording’ alongside course commentary must be dropped in its entirety even when the commentary is original.",
            "This paragraph ends with an unmatched closing quotation’ (Author, 1965, p. 5).",
            "Figure 1 Third-party image and caption. " + AUTHOR_TEXT,
            "Nature, and Nature's Laws lay hid in Night,", OTHER_TEXT, "## References", AUTHOR_TEXT,
        ]))
        sections, exclusions = ol.select_passages(sample)
        retained = "\n".join(p["text"] for section in sections for p in section["passages"])
        self.assertEqual(retained, AUTHOR_TEXT + "\n" + OTHER_TEXT)
        self.assertIn("blockquotation_continuation_or_attribution", {x["reason"] for x in exclusions})

    def test_activity_four_is_fully_excluded_but_author_discussion_can_be_kept(self):
        sample = capture("\n".join([
            "# Course", "## 6 Context", AUTHOR_TEXT,
            "### Activity 4 Copyrighted extract", OTHER_TEXT, "#### Discussion", OTHER_TEXT,
            "### Activity 5 Social order", "A supplied source extract must remain excluded even if the renderer does not mark it as a block quotation. " + OTHER_TEXT,
            "#### Discussion", OTHER_TEXT,
        ]))
        sections, _ = ol.select_passages(sample)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["passages"][0]["text"], AUTHOR_TEXT)
        self.assertEqual(sections[1]["passages"][0]["text"], OTHER_TEXT)
        self.assertNotIn("Activity 4", sections[1]["section_title"])

    def test_missing_source_lines_fail_closed(self):
        sample = capture("# Course\n## Section\n" + AUTHOR_TEXT)
        sample["lines"].pop(1)
        with self.assertRaises(ValueError):
            ol.select_passages(sample)

    def test_merge_preserves_existing_documents_and_is_idempotent(self):
        original = [{"id": "aic:one", "text": "original"}, {"id": "openlearn:other-course:one", "text": "unmanaged source"}]
        old = {"id": "openlearn:dutch-painting:old", "text": "obsolete supplement"}
        new = [{"id": "openlearn:dutch-painting:new", "text": "new supplement"}]
        before = copy.deepcopy(original)
        merged = ol.merge_documents(original + [old], new)
        self.assertEqual(merged, before + new)
        self.assertEqual(ol.merge_documents(merged, new), merged)
        self.assertEqual(original, before)

    def test_saved_corpus_has_complete_rights_and_integrity_metadata(self):
        docs = ol.read_json(ol.DATA / "openlearn_documents.json")
        ol.validate_documents(docs)
        self.assertEqual({d["course_slug"] for d in docs}, ol.SLUGS)
        merged = {d["id"]: d for d in ol.read_json(ol.DATA / "documents.json")}
        for doc in docs:
            self.assertEqual(merged[doc["id"]], doc)
            self.assertTrue(doc["noncommercial_only"])
            self.assertEqual(doc["related_artwork_ids"], [])
        text = "\n".join(d["text"] for d in docs)
        for excluded in ("We saw the ceremony of the Doges", "About 9 I saw the Doge", "The aim of an Encyclopédia is", "God said, Let Newton be", "There is no flesh in man's", "Since the people is a body with several heads"):
            self.assertNotIn(excluded, text)

    def test_base_collector_keeps_the_vetted_supplement(self):
        supplement = ol.read_json(ol.DATA / "openlearn_documents.json")
        base = [{"id": "aic:one", "text": "original"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openlearn_documents.json"
            path.write_text(json.dumps(supplement), encoding="utf-8")
            self.assertEqual(preserve_openlearn(base, path), base + supplement)

    def test_retained_passages_match_private_snapshot_when_available(self):
        path = ol.ROOT / ".cache" / "openlearn" / "snapshots.json"
        if not path.exists():
            self.skipTest("Private raw source snapshots are deliberately not distributed")
        snapshots = {c["page_url"]: c for c in ol.read_json(path)["captures"]}
        for doc in ol.read_json(ol.DATA / "openlearn_documents.json"):
            page = snapshots[doc["source_url"]]
            self.assertEqual(doc["source_snapshot_sha256"], ol.digest(json.dumps(page["lines"], ensure_ascii=False, sort_keys=True)))
            lines = {line["number"]: line["text"].strip() for line in page["lines"]}
            for passage in doc["passages"]:
                self.assertEqual(passage["text"], lines[passage["source_line"]])


if __name__ == "__main__":
    unittest.main()
