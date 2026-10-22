"""Print saved F2004 searches, selected/rejected candidates and index status."""

import argparse
import json
import sys

from dotenv import load_dotenv

from backend.collection_store import CollectionStore, collection_guard
from backend.document_index import document_index_status
from backend.metadata_index import ROOT, search_settings


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artwork-id")
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 51))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = search_settings()
    with collection_guard(settings):
        store = CollectionStore(settings)
        print(json.dumps({"index": document_index_status(), "index_pending": store.dirty(),
                          "collections": store.reports(args.artwork_id, args.limit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
