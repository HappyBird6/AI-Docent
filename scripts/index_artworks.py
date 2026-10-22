"""Run from the project root: python -m scripts.index_artworks."""

import json

from dotenv import load_dotenv

from backend.metadata_index import ROOT, sync_metadata_index


def main():
    load_dotenv(ROOT / ".env")
    try:
        print(json.dumps(sync_metadata_index(), ensure_ascii=False))
    except Exception as error:
        # Never print an upstream exception containing credentials or request bodies.
        print(json.dumps({"ready": False, "error_type": type(error).__name__,
                          "message": "Indexing failed. Check GMS configuration, connectivity, and collection settings."}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
