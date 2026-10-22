"""Cache three reviewed public-domain reproductions without contacting blocked AIC URLs.

Run: python -m scripts.prepare_local_images. No cookies, challenge solving or proxies.
The public source page must link to the exact museum record and name the work/artist.
"""

import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "frontend" / "artworks"
MANIFEST = ROOT / "data" / "art" / "local_images.json"
SOURCES = {
    "aic:250762": "https://www.artheonmuseum.org/artwork/melancholia-aic-250762",
    "aic:257286": "https://www.artheonmuseum.org/artwork/breakfast-still-life-aic-257286",
    "aic:16564": "https://www.artheonmuseum.org/artwork/branch-of-the-seine-near-giverny-mist-aic-16564",
}
USER_AGENT = "AI-Docent/0.1 (educational public-domain image cache)"


class SourcePage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.text = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_data(self, data):
        self.text.append(data)


def verified_image_url(html: str, artwork: dict) -> str:
    page = SourcePage()
    page.feed(html)
    if artwork["source_url"] not in page.links or "Public Domain" not in " ".join(page.text):
        raise ValueError("Source must identify the same public-domain museum object")
    blocks = re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S | re.I)
    for block in blocks:
        try:
            schema = json.loads(block)
        except ValueError:
            continue
        if not isinstance(schema, dict) or schema.get("@type") != "VisualArtwork":
            continue
        image = schema.get("image", {})
        creator = schema.get("creator", {})
        if not isinstance(image, dict) or not isinstance(creator, dict):
            continue
        url = image.get("contentUrl", "")
        parts = urlsplit(url)
        if (schema.get("name") == artwork["title"] and creator.get("name") == artwork["artist"] and
                image.get("license") == artwork["license_url"] and parts.scheme == "https" and
                parts.netloc == "images.artheonmuseum.org" and
                parts.path.startswith("/aic/full/") and parts.path.endswith(f"-aic-{artwork['source_id']}.webp")):
            return url
    raise ValueError("No verified reproduction for this artwork")


def webp_size(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("Expected a WebP image, not a blocked HTML response")
    kind = data[12:16]
    if kind == b"VP8X":
        size = (1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little"))
    elif kind == b"VP8L" and data[20] == 0x2f:
        bits = int.from_bytes(data[21:25], "little")
        size = (1 + (bits & 0x3fff), 1 + ((bits >> 14) & 0x3fff))
    elif kind == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        size = (int.from_bytes(data[26:28], "little") & 0x3fff, int.from_bytes(data[28:30], "little") & 0x3fff)
    else:
        raise ValueError("Unsupported WebP header")
    if not all(100 <= dimension <= 20000 for dimension in size):
        raise ValueError("Unexpected image dimensions")
    return size


def get_public_bytes(url: str, limit: int) -> bytes:
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30, stream=True) as response:
        response.raise_for_status()  # Stop on access restrictions; never try to bypass them.
        body = bytearray()
        for chunk in response.iter_content(65536):
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError("Source exceeded download limit")
        return bytes(body)


def main():
    artworks = {art["id"]: art for art in json.loads((ROOT / "data/art/artworks.json").read_text(encoding="utf-8"))}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for artwork_id, page_url in SOURCES.items():
        artwork = artworks[artwork_id]
        if not artwork.get("is_public_domain") or artwork["image_license"] != "CC0-1.0":
            raise ValueError("Recheck image rights before downloading")
        filename = artwork_id.replace(":", "-") + ".webp"
        path = DESTINATION / filename
        previous = manifest.get(artwork_id, {})
        if path.exists() and "/aic/full/" in previous.get("image_url", "") and sha256(path.read_bytes()).hexdigest() == previous.get("sha256"):
            print(json.dumps({"id": artwork_id, "status": "cached"}), flush=True)
            continue
        html = get_public_bytes(page_url, 3 * 1024 * 1024).decode("utf-8")
        image_url = verified_image_url(html, artwork)
        time.sleep(1)
        image = get_public_bytes(image_url, 10 * 1024 * 1024)
        width, height = webp_size(image)
        path.write_bytes(image)
        manifest[artwork_id] = {
            "file": filename, "image_url": image_url, "image_source_url": page_url,
            "verified_museum_url": artwork["source_url"], "museum_artwork_id": artwork_id,
            "title": artwork["title"], "artist": artwork["artist"],
            "license": artwork["image_license"], "license_url": artwork["license_url"],
            "license_evidence": "Saved AIC record is_public_domain=true and image_license=CC0-1.0; source VisualArtwork/ImageObject schema explicitly licenses this image as CC0 at the same license URL.",
            "attribution": "Art Institute of Chicago; public reproduction served by ArtheonMuseum.",
            "transformations": "Source WebP copied byte-for-byte; display scaling only; no crop or generated image.",
            "match_method": "exact_museum_record_link_and_visualartwork_schema", "width": width, "height": height,
            "sha256": sha256(image).hexdigest(), "bytes": len(image),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"id": artwork_id, "status": "saved", "width": width, "height": height, "bytes": len(image)}), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
