"""Public gallery metadata, separate from the future RAG/chat pipeline."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "art"


def gallery_catalog() -> list[dict]:
    artworks = json.loads((DATA_DIR / "artworks.json").read_text(encoding="utf-8"))
    fallback_path = DATA_DIR / "image_fallbacks.json"
    fallbacks = json.loads(fallback_path.read_text(encoding="utf-8")) if fallback_path.exists() else {}
    result = []
    for artwork in artworks:
        if not artwork.get("is_public_domain") or not artwork.get("image_url"):
            continue
        fallback = fallbacks.get(artwork["id"], {})
        candidates = list(dict.fromkeys(filter(None, [fallback.get("image_url"), artwork["image_url"]])))
        result.append({
            "id": artwork["id"],
            "title": artwork.get("title_ko") or artwork["title"],
            "artist": artwork.get("artist_ko") or artwork["artist"],
            "date_display": artwork.get("date_display") or "제작 시기 미상",
            "image_candidates": candidates,
            "image_alt": artwork.get("image_alt") or artwork["title"],
            "image_width": fallback.get("width"),
            "image_height": fallback.get("height"),
            "source_url": artwork["source_url"],
            "image_source_url": fallback.get("image_source_url") or artwork["source_url"],
            "image_license": fallback.get("license") or artwork["image_license"],
            "license_url": fallback.get("license_url") or artwork["license_url"],
        })
    # Keep the museum's data intact; only the presentation has an opening selection.
    opening = {"aic:28560": 0, "aic:27992": 1, "aic:20684": 2}
    return sorted(result, key=lambda artwork: opening.get(artwork["id"], 3))
