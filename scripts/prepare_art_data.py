"""Build a small, attributed art corpus without API keys or LLM calls.

Run: python scripts/prepare_art_data.py
Public-domain image URLs are saved; image binaries are not downloaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
AIC = "https://api.artic.edu/api/v1"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
CC_BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"
FIELDS = [
    "id", "title", "artist_id", "artist_title", "artist_display", "date_display",
    "date_start", "date_end", "place_of_origin", "medium_display", "dimensions",
    "credit_line", "main_reference_number", "is_public_domain", "image_id",
    "thumbnail", "description", "style_titles", "subject_titles",
    "department_title", "artwork_type_title",
]
# These are museum tag -> article mappings, not classifications inferred from dates.
STYLE_ARTICLES = {
    "Renaissance": "Renaissance art", "Early Renaissance": "Renaissance art",
    "High Renaissance": "High Renaissance", "Northern Renaissance": "Northern Renaissance",
    "Mannerism": "Mannerism", "Baroque": "Baroque painting", "Rococo": "Rococo",
    "Neoclassicism": "Neoclassicism", "Romanticism": "Romanticism",
    "Realism": "Realism (arts)", "Impressionism": "Impressionism",
    "Post-Impressionism": "Post-Impressionism", "Pointillism": "Pointillism",
    "Neo-Impressionism": "Neo-Impressionism", "Symbolism": "Symbolism (arts)",
    "Pre-Raphaelite": "Pre-Raphaelite Brotherhood",
    "Pre-Raphaelitism": "Pre-Raphaelite Brotherhood",
    "Surrealism": "Surrealism", "Cubism": "Cubism",
    "northern renaissance": "Northern Renaissance", "classicism": "Classicism",
    "Naturalism": "Naturalism (visual art)", "romantic": "Romanticism",
    "Gothic (medieval)": "Gothic art",
}
ARTIST_ARTICLES = {
    "Bernard d'Agescy": "Bernard d'Agesci",
    "El Greco (Doménikos Theotokópoulos) and workshop": "El Greco",
    "Jean Baptiste François Désoria": "Jean-Baptiste François Desoria",
    "Pontormo, (Jacopo Carrucci)": "Pontormo",
    "Workshop of Rogier van der Weyden": "Rogier van der Weyden",
    "Antonio da Correggio": "Correggio",
    "Domenico Theotokópoulos, called El Greco": "El Greco",
    "Francisco José de Goya y Lucientes": "Francisco Goya",
    "Fra Bartolommeo": "Fra Bartolomeo",
    "Hilaire Germain Edgar Degas": "Edgar Degas",
    "Jacques Louis David": "Jacques-Louis David",
    "Jean Auguste Dominique Ingres": "Jean-Auguste-Dominique Ingres",
    "Jean Baptiste Camille Corot": "Jean-Baptiste-Camille Corot",
    "Jean Baptiste Joseph Wicar": "Jean-Baptiste Wicar",
    "Jean Hey, (the Master of Moulins)": "Jean Hey",
    "Jean Honoré Fragonard": "Jean-Honoré Fragonard",
    "Joachim Antonisz. Wtewael": "Joachim Wtewael",
    "John Philip Simpson": "John Simpson (painter)",
    "Joseph Mallord William Turner": "J. M. W. Turner",
    "Paul Cezanne": "Paul Cézanne",
    "Pompeo Girolamo Batoni": "Pompeo Batoni",
    "Rembrandt van Rijn": "Rembrandt",
    "Sir Joshua Reynolds": "Joshua Reynolds",
    "Sir Thomas Lawrence": "Thomas Lawrence",
    "Workshop of Dieric Bouts": "Dieric Bouts",
    "Workshop of Rembrandt van Rijn": "Rembrandt",
}


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "br", "li", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "li", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(value):
    parser = PlainText()
    parser.feed(value or "")
    return "\n\n".join(
        line for part in "".join(parser.parts).splitlines()
        if (line := " ".join(part.split()))
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preserve_openlearn(documents, supplement_path):
    """Keep the separately reviewed supplement when rebuilding museum data."""
    if not supplement_path.exists():
        return documents
    if __package__:
        from .prepare_openlearn_data import merge_documents, validate_documents
    else:
        from prepare_openlearn_data import merge_documents, validate_documents
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    validate_documents(supplement)
    return merge_documents(documents, supplement)


class Client:
    def __init__(self, cache, user_agent):
        self.cache = cache
        self.user_agent = user_agent
        self.last_request = 0.0
        self.network_calls = 0
        self.cache_hits = 0

    def get(self, base, params=None):
        url = base + ("?" + urlencode(params) if params else "")
        path = self.cache / (sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(4):
            time.sleep(max(0, 1.1 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.network_calls += 1
            try:
                req = Request(url, headers={"User-Agent": self.user_agent, "AIC-User-Agent": self.user_agent})
                with urlopen(req, timeout=40) as response:
                    data = json.load(response)
                if "error" in data or "errors" in data:
                    raise RuntimeError(f"API error from {base}: {data.get('error', data.get('errors'))}")
                entry = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "data": data}
                write_json(path, entry)
                return entry
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise
                delay = error.headers.get("Retry-After", "")
                # Long Retry-After values must not be shortened.
                if delay.isdigit() and int(delay) > 60:
                    raise RuntimeError(f"Server requested a {delay}s cooldown; rerun later (cache retained)") from error
                time.sleep(int(delay) if delay.isdigit() else 2 ** (attempt + 1))
            except (URLError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("Request failed")


def collect_artworks(client, count):
    # Date buckets give broad coverage without guessing the movement of an artwork.
    buckets = [(1400, 1599, 20), (1600, 1699, 20), (1700, 1799, 20), (1800, 1900, 40)]
    candidates = {}
    selected = []
    for start, end, quota in buckets:
        query = {"bool": {"filter": [
            {"term": {"is_public_domain": True}},
            {"term": {"artwork_type_title.keyword": "Painting"}},
            {"term": {"department_title.keyword": "Painting and Sculpture of Europe"}},
            {"range": {"date_start": {"gte": start, "lte": end}}},
            {"exists": {"field": "image_id"}}, {"exists": {"field": "description"}},
        ]}}
        params = {"query": query, "limit": 100, "fields": FIELDS}
        entry = client.get(AIC + "/artworks/search", {"params": json.dumps(params, separators=(",", ":"))})
        usable = []
        for raw in entry["data"].get("data", []):
            if not (raw.get("is_public_domain") and raw.get("image_id") and raw.get("artist_title")):
                continue
            if raw["artist_title"] in {"Spanish", "Master of the Antwerp Adoration Group"}:
                # Anonymous/culture-only attributions do not support the starter biography feature.
                continue
            if len(plain_text(raw.get("description"))) < 300:
                continue
            # The starter service promises movement context, so prefer documented tags.
            if not any(tag in STYLE_ARTICLES for tag in raw.get("style_titles") or []):
                continue
            iiif = entry["data"]["config"]["iiif_url"]
            item = (raw, entry, iiif)
            candidates[raw["id"]] = item
            usable.append(item)
        selected.extend(usable[:quota])
        print(f"{start}-{end}: {len(usable)} usable candidates; {min(quota, len(usable))} selected", flush=True)
    chosen = {raw["id"]: (raw, entry, iiif) for raw, entry, iiif in selected}
    for key, value in candidates.items():
        if len(chosen) >= count:
            break
        chosen.setdefault(key, value)
    if len(chosen) < count:
        raise RuntimeError(f"Only {len(chosen)} eligible paintings found; expected {count}")
    return list(chosen.values())[:count]


def wikipedia_documents(client, titles, kind, links, full=False):
    result = []
    title_to_doc = {}
    missing = []
    titles = sorted(set(titles))
    batch_size = 1 if full else 10
    for offset in range(0, len(titles), batch_size):
        batch = titles[offset:offset + batch_size]
        params = {
            "action": "query", "format": "json", "formatversion": 2,
            "prop": "extracts|info|revisions|pageprops", "explaintext": 1,
            "exlimit": batch_size, "inprop": "url", "rvprop": "ids|timestamp",
            "redirects": 1, "titles": "|".join(batch), "maxlag": 5,
        }
        if not full:
            params["exintro"] = 1
        entry = client.get("https://en.wikipedia.org/w/api.php", params)
        query = entry["data"].get("query", {})
        aliases = {v["from"]: v["to"] for v in query.get("normalized", []) + query.get("redirects", [])}
        for page in query.get("pages", []):
            text = page.get("extract", "").strip()
            if page.get("missing") or "disambiguation" in page.get("pageprops", {}) or len(text) < 300:
                continue
            canonical = page["title"]
            requested = []
            for original in batch:
                target, seen = original, set()
                while target in aliases and target not in seen:
                    seen.add(target)
                    target = aliases[target]
                if target == canonical:
                    requested.append(original)
            doc_id = f"wikipedia:en:{page['pageid']}"
            revision = page.get("revisions", [{}])[0]
            related = sorted({a for title in requested for a in links.get(title, [])})
            result.append({
                "id": doc_id, "kind": kind, "title": canonical, "text": text,
                "language": "en", "source_type": "wikipedia", "source_url": page["fullurl"],
                "source_api_url": entry["url"], "retrieved_at": entry["retrieved_at"],
                "revision_id": revision.get("revid"),
                "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={revision['revid']}" if revision.get("revid") else None,
                "license": "CC-BY-SA-4.0", "license_url": CC_BY_SA,
                "attribution": f"Wikipedia contributors, {canonical}; see source page history for authors.",
                "content_scope": "full_article" if full else "lead_section",
                "transformations": "Article plain-text extraction via MediaWiki TextExtracts; no translation or generated prose.",
                "related_artwork_ids": related,
                "mapping_status": "title_linked_needs_review" if kind == "artist" else "curated_museum_tag_mapping",
            })
            for original in requested:
                title_to_doc[original] = doc_id
        print(f"{kind} documents ({'full' if full else 'lead'}): {min(offset + batch_size, len(titles))}/{len(titles)} titles checked", flush=True)
    # Different aliases can resolve to the same page across batches.
    unique = {}
    for doc in result:
        if doc["id"] in unique:
            existing = unique[doc["id"]]
            existing["related_artwork_ids"] = sorted(set(existing["related_artwork_ids"] + doc["related_artwork_ids"]))
        else:
            unique[doc["id"]] = doc
    missing = [t for t in titles if t not in title_to_doc]
    return list(unique.values()), title_to_doc, missing


def validate(artworks, documents, expected):
    assert len(artworks) == expected
    assert len({a["id"] for a in artworks}) == expected
    ids = {d["id"] for d in documents}
    assert len(ids) == len(documents)
    for art in artworks:
        assert art["is_public_domain"] and art["image_url"] and art["title"] and art["artist"]
        assert art["artwork_document_id"] in ids
        assert art["artist_document_ids"], f"Missing artist context: {art['id']}"
        assert art["style_document_ids"], f"Missing style context: {art['id']}"
        assert all(x in ids for x in art["artist_document_ids"] + art["style_document_ids"])
    for doc in documents:
        assert len(doc["text"]) >= 300 and doc["source_url"] and doc["license_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, choices=range(1, 101), metavar="1..100")
    args = parser.parse_args()
    contact = os.environ.get("WIKI_CONTACT", "")
    # Read only the contact setting, never API keys, from the project's existing env file.
    env_file = ROOT / ".env"
    if not contact and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("WIKI_CONTACT="):
                contact = line.strip().split("=", 1)[1].strip().strip("\"'")
    contact = contact if contact.isascii() and "@" in contact and not any(c in contact for c in "\r\n") else ""
    user_agent = "ArtDocentData/0.1 (educational prototype" + (f"; {contact}" if contact else "") + ")"
    client = Client(ROOT / ".cache" / "art-data", user_agent)
    rows = collect_artworks(client, args.count)
    artworks, documents, artist_links, style_links = [], [], {}, {}
    for raw, entry, iiif in rows:
        art_id = f"aic:{raw['id']}"
        styles = raw.get("style_titles") or []
        artist_article = ARTIST_ARTICLES.get(raw["artist_title"], raw["artist_title"])
        artist_links.setdefault(artist_article, []).append(art_id)
        for style in styles:
            if style in STYLE_ARTICLES:
                style_links.setdefault(STYLE_ARTICLES[style], []).append(art_id)
        doc_id = f"aic:artwork:{raw['id']}"
        source_url = f"https://www.artic.edu/artworks/{raw['id']}"
        artworks.append({
            "id": art_id, "source_id": raw["id"], "title": raw["title"],
            "title_ko": None, "artist": raw["artist_title"], "artist_ko": None,
            "artist_source_id": raw.get("artist_id"), "artist_display": raw.get("artist_display"),
            "artist_article_title": artist_article,
            "artist_attribution_note": "Workshop attribution: linked biography describes the named master, not a confirmed individual maker." if "workshop" in raw["artist_title"].lower() else None,
            "date_display": raw.get("date_display"), "date_start": raw.get("date_start"),
            "date_end": raw.get("date_end"), "place_of_origin": raw.get("place_of_origin"),
            "medium": raw.get("medium_display"), "dimensions": raw.get("dimensions"),
            "museum": "Art Institute of Chicago", "department": raw.get("department_title"),
            "style_tags": styles, "subject_tags": raw.get("subject_titles") or [],
            "style_classification_source": source_url, "image_id": raw["image_id"],
            "image_url": f"{iiif}/{raw['image_id']}/full/843,/0/default.jpg",
            "thumbnail_url": f"{iiif}/{raw['image_id']}/full/400,/0/default.jpg",
            "image_alt": (raw.get("thumbnail") or {}).get("alt_text"),
            "is_public_domain": True, "image_license": "CC0-1.0",
            "metadata_license": "CC0-1.0", "license_url": CC0,
            "credit_line": raw.get("credit_line"), "accession_number": raw.get("main_reference_number"),
            "source_url": source_url, "source_api_url": f"{AIC}/artworks/{raw['id']}",
            "retrieved_at": entry["retrieved_at"], "artwork_document_id": doc_id,
            "artist_document_ids": [], "style_document_ids": [],
        })
        documents.append({
            "id": doc_id, "kind": "artwork", "title": raw["title"],
            "text": plain_text(raw["description"]), "language": "en",
            "source_type": "museum_description", "source_url": source_url,
            "source_api_url": f"{AIC}/artworks/{raw['id']}",
            "retrieved_at": entry["retrieved_at"], "license": "CC-BY-4.0",
            "license_url": CC_BY, "attribution": f"Art Institute of Chicago, {raw['title']}",
            "transformations": "HTML markup removed; paragraph text retained in source order; no translation.",
            "related_artwork_ids": [art_id], "mapping_status": "museum_record_id",
        })
    artist_docs, artist_map, missing_artists = wikipedia_documents(client, artist_links, "artist", artist_links)
    if missing_artists:
        extra_docs, extra_map, missing_artists = wikipedia_documents(client, missing_artists, "artist", artist_links, full=True)
        artist_docs.extend(extra_docs)
        artist_map.update(extra_map)
    style_docs, style_map, missing_styles = wikipedia_documents(client, style_links, "style", style_links)
    documents.extend(artist_docs + style_docs)
    for art in artworks:
        if art["artist_article_title"] in artist_map:
            art["artist_document_ids"] = [artist_map[art["artist_article_title"]]]
        art["style_document_ids"] = sorted({
            style_map[STYLE_ARTICLES[tag]] for tag in art["style_tags"]
            if tag in STYLE_ARTICLES and STYLE_ARTICLES[tag] in style_map
        })
        # Indexable metadata; artwork descriptions belong to the separate document collection.
        art["search_text"] = "\n".join([
            f"Title: {art['title']}", f"Artist: {art['artist']}",
            f"Date: {art['date_display']}", f"Origin: {art['place_of_origin']}",
            f"Medium: {art['medium']}", "Styles: " + ", ".join(art["style_tags"]),
            "Subjects: " + ", ".join(art["subject_tags"]), f"Museum: {art['museum']}",
        ])
    validate(artworks, documents, args.count)
    base_document_count = len(documents)
    documents = preserve_openlearn(documents, ROOT / "data" / "art" / "openlearn_documents.json")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artworks": len(artworks), "documents": len(documents),
        "document_counts": dict(Counter(d["kind"] for d in documents)),
        "supplementary_documents_preserved": len(documents) - base_document_count,
        "date_buckets": dict(Counter(str((a["date_start"] // 100) * 100) for a in artworks)),
        "unique_artist_attributions": len({a["artist"] for a in artworks}),
        "unique_artist_article_titles": len(artist_links),
        "artworks_with_artist_document": sum(bool(a["artist_document_ids"]) for a in artworks),
        "artworks_with_style_document": sum(bool(a["style_document_ids"]) for a in artworks),
        "missing_artist_articles": missing_artists, "missing_style_articles": missing_styles,
        "unmapped_style_tags": dict(Counter(t for a in artworks for t in a["style_tags"] if t not in STYLE_ARTICLES)),
        "network_calls_this_run": client.network_calls, "cache_hits_this_run": client.cache_hits,
        "validation": "passed",
        "limitations": [
            "English source text; Korean labels and translation have not been prepared.",
            "Artist articles are linked by titles, curated aliases and redirects; identity requires human review.",
            "Most Wikipedia documents contain introductory sections; short artist leads are supplemented with full articles.",
            "Museum style tags mix periods, techniques and movements, and allow multiple tags.",
            "No movement is inferred from the creation date; this starter set selects works with mapped museum tags.",
            "Image URLs are supplied; image binaries and availability of every image are not verified.",
            "This is a seed corpus, not the required question-driven LLM search/selection pipeline.",
        ],
    }
    output = ROOT / "data" / "art"
    write_json(output / "artworks.json", artworks)
    write_json(output / "documents.json", documents)
    write_json(output / "collection_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
