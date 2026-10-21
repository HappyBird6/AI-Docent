"""Select licensed author narrative from reviewed OpenLearn page-text snapshots.

No LLM, embedding, retrieval or chat code is imported. Live downloads currently
return a bot challenge, so collection uses public page text from a web reader.
Full snapshots stay in ignored .cache/openlearn; only selected text is published.
Run with --snapshots PATH to rebuild; --merge-only reapplies the checked supplement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "art"
LICENSE = "CC-BY-NC-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SLUGS = {"early-modern-europe", "dutch-painting", "enlightenment"}


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def section_key(title: str) -> str:
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def body_lines(capture: dict) -> list[dict]:
    lines = sorted(capture["lines"], key=lambda line: line["number"])
    if [line["number"] for line in lines] != list(range(capture["total_lines"])):
        raise ValueError(f"Incomplete source snapshot: {capture['page_url']}")
    starts = [i for i, line in enumerate(lines) if line["text"].startswith("# ")]
    if not starts:
        raise ValueError(f"Missing course body: {capture['page_url']}")
    lines = lines[starts[0]:]
    if capture["format"] == "section":
        for i, line in enumerate(lines):
            if re.match(r"^(?:Previous |Next |Print$|Take the next step)", line["text"]):
                return lines[:i]
        raise ValueError("Section footer boundary not found")
    return lines


def rejection(text: str) -> str | None:
    if re.match(r"^(?:Image\b|Figure\s+\d|Maximise\b|Show description|Hide description|Video\b|Audio\b|Description\b)", text, re.I):
        return "image_caption_or_media"
    if re.match(r"^This is (?:an? )?(?:painting|picture|illustration|map|engraving|photograph|sculpture|portrait|drawing|landscape)\b", text, re.I):
        return "image_description"
    if re.search(r"[‘“”\"]|’(?=[\s,.;:)\]]|$)|(?<!\w)'|'(?=\W|$)", text):
        # Exclude entire mixed paragraphs, including short quoted terminology.
        return "inline_quotation_mixed_paragraph"
    if re.match(r"^\([^)]*(?:\d{4}|p\.|pp\.|trans\.)[^)]*\)$", text):
        return "quotation_attribution"
    if re.search(r"\b(?:this (?:free |OpenLearn )?course|you (?:will|shall|should|can|are|have)|learning (?:objectives|outcomes)|exercise|activity|video|transcript)\b", text, re.I):
        return "learning_instruction_or_media_reference"
    if re.search(r"highlighted in bold|through the course|covered in the course|your (?:knowledge|response|notes|study)", text, re.I):
        return "learning_instruction_or_media_reference"
    if re.search(r"\b(?:you|your|we|our)\b|\bI(?: |\u2019)|this passage|in the extract|in his poem|in the period covered by the course", text, re.I):
        return "instruction_or_excluded_material_dependency"
    if re.match(r"^(?:Read |Look at |Watch |Study |Try |Make notes|What was your|Please |Click |In the (?:above|following) (?:passage|extract)|This (?:quotation|extract|passage)|The (?:above|following) (?:quotation|extract|passage))", text, re.I):
        return "instruction_or_excluded_material_dependency"
    if "©" in text or "Photo:" in text or "Photo ©" in text:
        return "copyright_credit_or_caption"
    if re.match(r"^\d+\.\s+(?:to|of|and)\b", text, re.I):
        return "dependent_list_fragment"
    if len(text.strip(" *-0123456789.\t")) < 100:
        return "short_fragment"
    return None


def select_passages(capture: dict) -> tuple[list[dict], list[dict]]:
    sections, excluded = [], []
    current = None
    skipped_level = None
    activity = None
    quote_open = False
    body = body_lines(capture)

    def omit(number, text, reason):
        excluded.append({"source_line": number, "reason": reason, "sha256": digest(text), "characters": len(text)})

    for i, line in enumerate(body):
        text, number = line["text"].strip(), line["number"]
        if not text:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", text)
        if heading:
            level, title = len(heading[1]), heading[2].strip()
            quote_open = False
            if re.match(r"^(?:References|Acknowledgements|Further reading|Glossary)\b", title, re.I):
                break
            if re.match(r"^(?:Activity|Exercise)\b", title, re.I):
                parent = activity["parent"] if activity else current
                activity = {"title": title, "level": level, "parent": parent,
                            "blocked": capture["course_slug"] == "early-modern-europe" and bool(re.match(r"^Activity 4\b", title))}
                skipped_level = level
                omit(number, text, "excluded_activity_or_exercise")
                continue
            if re.match(r"^Discussion\b", title, re.I):
                if activity and activity["parent"] and not activity["blocked"]:
                    current = {"section_title": f"{activity['parent']['section_title']} — {activity['title']} — Discussion",
                               "source_heading_line": number, "passages": []}
                    sections.append(current)
                    skipped_level = None
                else:
                    skipped_level = activity["level"] if activity else level
                    omit(number, text, "excluded_activity_discussion")
                continue
            if re.match(r"^(?:Transcript|Learning outcomes|Introduction)\b", title, re.I):
                skipped_level = min(level, skipped_level) if skipped_level is not None else level
                omit(number, text, "excluded_learning_or_media_section")
                continue
            if skipped_level is not None:
                if level > skipped_level:
                    omit(number, text, "inside_excluded_section")
                    continue
                skipped_level = None
            if activity and level <= activity["level"]:
                activity = None
            if capture["format"] == "printable" and level == 1:
                current = None  # Course title, not a content section.
                continue
            current = {"section_title": title, "source_heading_line": number, "passages": []}
            sections.append(current)
            continue
        if skipped_level is not None:
            omit(number, text, "inside_excluded_section")
            continue
        if text.startswith(">"):
            quote_open = True
            omit(number, text, "third_party_blockquotation")
            if re.search(r"\([^)]*\d{4}[^)]*\)\s*$", text):
                quote_open = False
            continue
        if quote_open:
            omit(number, text, "blockquotation_continuation_or_attribution")
            if re.match(r"^\([^)]*(?:\d{4}|p\.|pp\.|trans\.)[^)]*\)$", text):
                quote_open = False
            continue
        if current is None:
            continue
        # Drop a quote's introducing paragraph as well as the quotation itself.
        following = next((row["text"].strip() for row in body[i + 1:] if row["text"].strip()), "")
        introduces_quote = text.endswith(":") and (following.startswith(">") or len(following) < 100)
        reason = "quotation_lead_in" if introduces_quote else rejection(text)
        if re.match(r"^\([^)]*(?:\d{4}|p\.|pp\.|trans\.)[^)]*\)$", following):
            reason = "unmarked_quotation_before_attribution"
        if reason:
            omit(number, text, reason)
        else:
            current["passages"].append({"source_line": number, "text": text, "sha256": digest(text)})
    return sections, excluded


def build_documents(snapshot: dict, sources: dict) -> tuple[list[dict], dict]:
    documents, pages, course_counts = [], [], Counter()
    occurrences = Counter()
    acquired_at = snapshot["captured_at_utc"].replace(" UTC", "+00:00").replace(" ", "T", 1)
    for capture in snapshot["captures"]:
        course = sources[capture["course_slug"]]
        source_digest = digest(json.dumps(capture["lines"], ensure_ascii=False, sort_keys=True))
        if course["license"] != LICENSE or course["license_url"] != LICENSE_URL:
            raise ValueError("Unexpected license; re-review the source")
        sections, exclusions = select_passages(capture)
        for section in sections:
            passages = section["passages"]
            text = "\n\n".join(p["text"] for p in passages)
            if len(text) < 300:
                exclusions.extend({"source_line": p["source_line"], "reason": "section_too_short", "sha256": p["sha256"], "characters": len(p["text"])} for p in passages)
                continue
            slug = course["slug"]
            doc_id = f"openlearn:{slug}:{section_key(section['section_title'])}"
            occurrences[doc_id] += 1
            source_part = occurrences[doc_id]
            if source_part > 1:
                doc_id += f":part-{source_part}"
            transformations = (
                "Selected author narrative from public course pages. Navigation, images, captions, "
                "media, block quotations and their continuations, entire paragraphs containing inline "
                "quotations, activity instructions/extracts, bibliography and acknowledgements removed. "
                "Author-written discussion uses the same paragraph filters; Early Modern Activity 4 is excluded in full. "
                "Remaining passages retain source order and original English; no translation or generated prose. "
                "Omissions are discontinuities, not complete original chapters."
            )
            documents.append({
                "id": doc_id, "kind": "historical_context",
                "title": f"{course['title']} — {section['section_title']}",
                "text": text, "language": "en", "source_type": "openlearn",
                "source_url": capture["page_url"], "course_url": course["course_url"],
                "course_slug": slug, "course_title": course["title"], "section_title": section["section_title"],
                "source_part": source_part,
                "authors": course["authors"], "publisher": "The Open University / OpenLearn",
                "retrieved_at": acquired_at, "license": LICENSE, "license_url": LICENSE_URL,
                "license_evidence_url": course["license_evidence_url"],
                "attribution": f"{', '.join(course['authors'])}; The Open University. {course['title']}, {section['section_title']}. {capture['page_url']}. {LICENSE}. Selected excerpts; exclusions and formatting changes recorded.",
                "transformations": transformations, "content_scope": "selected_author_narrative",
                "related_artwork_ids": [], "mapping_status": "historical_context_not_mapped_to_individual_artworks",
                "context_tags": course["context_tags"], "noncommercial_only": True,
                "sharealike_on_shared_adaptations": True,
                "source_locator": {"heading_line": section["source_heading_line"], "passage_lines": [p["source_line"] for p in passages]},
                "source_snapshot_sha256": source_digest,
                "passages": passages, "text_sha256": digest(text),
            })
            course_counts[slug] += 1
        pages.append({
            "course_slug": course["slug"], "source_url": capture["page_url"],
            "source_snapshot_sha256": source_digest,
            "source_lines": capture["total_lines"], "exclusions": exclusions,
        })
    validate_documents(documents)
    if set(course_counts) != SLUGS:
        raise ValueError("All three courses must contribute eligible text")
    report = {
        "collected_at": acquired_at, "collection_method": snapshot["method"],
        "courses": list(sources.values()), "document_counts_by_course": dict(course_counts),
        "documents": len(documents), "retained_passages": sum(len(d["passages"]) for d in documents),
        "retained_characters": sum(len(d["text"]) for d in documents),
        "exclusion_counts": dict(Counter(x["reason"] for p in pages for x in p["exclusions"])),
        "source_pages": pages, "validation": "passed",
        "review_method": "Rule-based conservative selection with targeted review of quotation, poetry and third-party exclusion boundaries; no human approval implied.",
        "limitations": [
            "Noncommercial source license; source/author/license/change notices must remain available.",
            "Selected author narrative only; quotations and ambiguous mixed paragraphs were conservatively omitted.",
            "Source line numbers refer to captured public-page text, not the original HTML or PDF pagination.",
            "Full page snapshots may contain excluded third-party material and stay in ignored local cache.",
            "No artwork-specific historical causation is inferred and no RAG index or chat integration is created.",
        ],
    }
    return documents, report


def validate_documents(documents: list[dict]):
    if len({d["id"] for d in documents}) != len(documents):
        raise ValueError("Duplicate OpenLearn document IDs")
    for doc in documents:
        assert doc["course_slug"] in SLUGS and doc["license"] == LICENSE
        assert doc["license_evidence_url"] and doc["authors"] and doc["attribution"]
        assert doc["source_url"].startswith("https://www.open.edu/openlearn/")
        assert doc["text"] == "\n\n".join(p["text"] for p in doc["passages"])
        assert digest(doc["text"]) == doc["text_sha256"]
        assert len(doc["text"]) >= 300
        assert "" not in doc["text"] and "" not in doc["text"]
        for passage in doc["passages"]:
            assert not rejection(passage["text"]), passage["text"][:80]
            assert not passage["text"].startswith(">")
            assert passage["sha256"] == digest(passage["text"])


def merge_documents(existing: list[dict], supplement: list[dict]) -> list[dict]:
    managed = tuple(f"openlearn:{slug}:" for slug in SLUGS)
    preserved = [doc for doc in existing if not doc["id"].startswith(managed)]
    result = preserved + supplement
    if len({doc["id"] for doc in result}) != len(result):
        raise ValueError("Document ID collision")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, help="Complete public-page text snapshot JSON (private local cache)")
    parser.add_argument("--merge-only", action="store_true", help="Merge the already selected, attributed supplement without network access")
    args = parser.parse_args()
    if args.merge_only:
        supplement = read_json(DATA / "openlearn_documents.json")
        validate_documents(supplement)
    elif args.snapshots:
        snapshot = read_json(args.snapshots)
        source_list = read_json(DATA / "openlearn_sources.json")
        supplement, report = build_documents(snapshot, {s["slug"]: s for s in source_list})
        write_json(DATA / "openlearn_documents.json", supplement)
        write_json(DATA / "openlearn_collection_report.json", report)
    else:
        parser.error("Use --snapshots PATH or --merge-only. No challenge bypass or automatic live download is performed.")
    existing = read_json(DATA / "documents.json")
    merged = merge_documents(existing, supplement)
    write_json(DATA / "documents.json", merged)
    report = read_json(DATA / "collection_report.json")
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    report["documents"] = len(merged)
    report["document_counts"] = dict(Counter(doc["kind"] for doc in merged))
    report.setdefault("supplements", {})["openlearn"] = {
        "documents": len(supplement), "courses": 3, "license": LICENSE,
        "report": "openlearn_collection_report.json", "rag_connected": False,
    }
    report["base_collection_statistics_note"] = "Original network/cache and artwork mapping statistics describe the initial museum/Wikipedia collection."
    write_json(DATA / "collection_report.json", report)
    print(json.dumps({"documents_total": len(merged), "openlearn_documents": len(supplement), "rag_connected": False}))


if __name__ == "__main__":
    main()
