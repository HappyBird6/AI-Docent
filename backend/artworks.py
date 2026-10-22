"""Public gallery metadata and verified same-origin image candidates."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "art"


def gallery_catalog() -> list[dict]:
    artworks = json.loads((DATA_DIR / "artworks.json").read_text(encoding="utf-8"))
    fallback_path = DATA_DIR / "image_fallbacks.json"
    fallbacks = json.loads(fallback_path.read_text(encoding="utf-8")) if fallback_path.exists() else {}
    local_path = DATA_DIR / "local_images.json"
    local_images = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
    result = []
    for artwork in artworks:
        if not artwork.get("is_public_domain") or not artwork.get("image_url"):
            continue
        fallback = fallbacks.get(artwork["id"], {})
        local = local_images.get(artwork["id"], {})
        # Only the fixed artwork filename can be served; never trust a manifest path.
        filename = artwork["id"].replace(":", "-") + ".webp"
        valid_local = (local.get("file") == filename and
                       (DATA_DIR.parents[1] / "frontend" / "artworks" / filename).is_file())
        if not valid_local:
            local = {}
        local_url = "/static/artworks/" + filename if valid_local else None
        candidates = list(dict.fromkeys(filter(None, [local_url, fallback.get("image_url"), artwork["image_url"]])))
        result.append({
            "id": artwork["id"],
            "title": artwork.get("title_ko") or artwork["title"],
            "artist": artwork.get("artist_ko") or artwork["artist"],
            "date_display": artwork.get("date_display") or "제작 시기 미상",
            "image_candidates": candidates,
            "image_alt": artwork.get("image_alt") or artwork["title"],
            "image_width": local.get("width") or fallback.get("width"),
            "image_height": local.get("height") or fallback.get("height"),
            "source_url": artwork["source_url"],
            "image_source_url": local.get("image_source_url") or fallback.get("image_source_url") or artwork["source_url"],
            "image_license": local.get("license") or fallback.get("license") or artwork["image_license"],
            "license_url": local.get("license_url") or fallback.get("license_url") or artwork["license_url"],
        })
    # Keep the museum's data intact; only the presentation has an opening selection.
    opening = {"aic:28560": 0, "aic:27992": 1, "aic:20684": 2}
    return sorted(result, key=lambda artwork: opening.get(artwork["id"], 3))
