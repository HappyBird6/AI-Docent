"""Find public copies of the same AIC objects on Wikimedia Commons.

No image files are downloaded. A match requires the museum's artwork ID in
the file's source credit and a public-domain/CC0 license, not just a title.
"""

import json
import re
from urllib.parse import urlsplit, urlunsplit

from prepare_art_data import Client, ROOT, write_json


def clean_url(value):
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def main():
    artworks = json.loads((ROOT / "data/art/artworks.json").read_text(encoding="utf-8"))
    by_id = {str(a["source_id"]): a for a in artworks}
    client = Client(ROOT / ".cache/art-images", "AI-Docent/0.1 (educational gallery)")
    matches = {}
    unresolved_pages = {}

    def collect(params, expected=None):
        entry = client.get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "format": "json", "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "iiurlwidth": 960,
            "iiextmetadatafilter": "Credit|LicenseShortName|LicenseUrl|UsageTerms",
            "maxlag": 5, **params,
        })
        for page in entry["data"].get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata", {})
            credit = metadata.get("Credit", {}).get("value", "")
            source = re.search(r"artic\.edu/(?:artworks/|aic/collections/artwork/)(\d+)", credit)
            license_name = metadata.get("LicenseShortName", {}).get("value", "")
            if not info.get("thumburl"):
                continue
            if not any(term in license_name.lower() for term in ("cc0", "public domain")):
                continue
            if not source or source[1] not in by_id:
                if expected and page.get("pageid") and not page["title"].lower().endswith(".svg"):
                    unresolved_pages[page["pageid"]] = expected
                continue
            artwork = by_id[source[1]]
            matches.setdefault(artwork["id"], {
                "image_url": clean_url(info["thumburl"]),
                "image_source_url": info["descriptionurl"],
                "original_url": clean_url(info["url"]),
                "width": info["width"], "height": info["height"],
                "license": license_name,
                "license_url": metadata.get("LicenseUrl", {}).get("value") or ("https://creativecommons.org/publicdomain/zero/1.0/" if "cc0" in license_name.lower() else "https://creativecommons.org/publicdomain/mark/1.0/"),
                "verified_museum_url": artwork["source_url"],
                "match_method": "museum_artwork_id_in_commons_source_credit",
                "retrieved_at": entry["retrieved_at"],
            })

    titles = []
    for artwork in artworks:
        for artist in dict.fromkeys([artwork["artist"], artwork.get("artist_article_title", artwork["artist"])]):
            titles.append(f"File:{artist} - {artwork['title']} - {artwork['accession_number']} - Art Institute of Chicago.jpg")
    for offset in range(0, len(titles), 8):
        collect({"titles": "|".join(titles[offset:offset + 8])})
        print(f"File lookup: {min(offset + 8, len(titles))}/{len(titles)}, matched {len(matches)}", flush=True)
    for artwork in artworks:
        if artwork["id"] in matches:
            continue
        collect({
            "generator": "search", "gsrnamespace": 6, "gsrlimit": 5,
            "gsrsearch": f'"{artwork["accession_number"]}" "Art Institute of Chicago"',
        }, expected=artwork)
        print(f"Accession lookup {artwork['id']}: matched {len(matches)}/{len(artworks)}", flush=True)

    for artwork in artworks:
        if artwork["id"] in matches:
            continue
        short_title = artwork["title"].split(" (recto)")[0]
        collect({
            "generator": "search", "gsrnamespace": 6, "gsrlimit": 5,
            "gsrsearch": f'{artwork.get("artist_article_title", artwork["artist"])} "{short_title}"',
        }, expected=artwork)

    # Google Art Project copies may credit Google, rather than link the museum.
    # Verify their artwork entity's ARTIC ID (P4610) before using them.
    candidates = []
    page_ids = list(unresolved_pages)
    for offset in range(0, len(page_ids), 15):
        entry = client.get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "format": "json", "prop": "imageinfo|revisions",
            "rvprop": "content", "rvslots": "main",
            "pageids": "|".join(str(p) for p in page_ids[offset:offset + 15]),
            "iiprop": "url|size|extmetadata", "iiurlwidth": 960,
            "iiextmetadatafilter": "ObjectName|Credit|LicenseShortName|LicenseUrl", "maxlag": 5,
        })
        for page in entry["data"].get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata", {})
            qids = re.findall(r"wikidata\.org/wiki/(Q\d+)", metadata.get("ObjectName", {}).get("value", ""))
            text = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")
            qids += re.findall(r"\|\s*wikidata\s*=\s*(Q\d+)", text, re.I)
            expected = unresolved_pages[page["pageid"]]
            source_ids = re.findall(r"artic\.edu/(?:artworks/|aic/collections/artwork/)(\d+)", text)
            proof = "commons_file_museum_reference" if str(expected["source_id"]) in source_ids else None
            accession = re.search(r"\|\s*accession[ _]number\s*=\s*([^\n|]+)", text, re.I)
            if accession and "Institution:Art Institute of Chicago" in text:
                normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
                if normalize(accession[1]) == normalize(expected["accession_number"]):
                    proof = "commons_museum_and_inventory_number"
            if qids or proof:
                candidates.append((page, info, list(dict.fromkeys(qids)), entry["retrieved_at"], proof))
    qids = list(dict.fromkeys(q for _, _, ids, _, _ in candidates for q in ids))
    entities = {}
    for offset in range(0, len(qids), 30):
        entry = client.get("https://www.wikidata.org/w/api.php", {
            "action": "wbgetentities", "format": "json", "props": "claims",
            "ids": "|".join(qids[offset:offset + 30]), "maxlag": 5,
        })
        entities.update(entry["data"].get("entities", {}))
    candidates.sort(key=lambda item: item[1].get("width", 0) * item[1].get("height", 0), reverse=True)
    for page, info, ids, retrieved_at, proof in candidates:
        expected = unresolved_pages[page["pageid"]]
        if expected["id"] in matches:
            continue
        for qid in ids or [None]:
            claims = entities.get(qid, {}).get("claims", {}).get("P4610", [])
            values = [c.get("mainsnak", {}).get("datavalue", {}).get("value") for c in claims if c.get("rank") != "deprecated"]
            if not proof and str(expected["source_id"]) not in values:
                continue
            meta = info.get("extmetadata", {})
            matches[expected["id"]] = {
                "image_url": clean_url(info["thumburl"]), "image_source_url": info["descriptionurl"],
                "original_url": clean_url(info["url"]), "width": info["width"], "height": info["height"],
                "license": meta.get("LicenseShortName", {}).get("value"),
                "license_url": meta.get("LicenseUrl", {}).get("value") or "https://creativecommons.org/publicdomain/mark/1.0/",
                "verified_museum_url": expected["source_url"],
                "verified_wikidata_url": f"https://www.wikidata.org/wiki/{qid}" if qid and str(expected["source_id"]) in values else None,
                "match_method": proof or "commons_artwork_entity_wikidata_P4610", "retrieved_at": retrieved_at,
            }
            break
    write_json(ROOT / "data/art/image_fallbacks.json", matches)
    print(json.dumps({"matched": len(matches), "total": len(artworks), "missing": [a["id"] for a in artworks if a["id"] not in matches]}), flush=True)


if __name__ == "__main__":
    main()
