"""Index the licensed external corpus: python -m scripts.index_documents."""

import json

from dotenv import load_dotenv

from backend.document_index import sync_document_index
from backend.metadata_index import ROOT


def main():
    load_dotenv(ROOT / ".env")
    try:
        print(json.dumps(sync_document_index(), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ready": False, "error_type": type(error).__name__,
                          "message": "Document indexing failed. Check corpus, GMS connection, and collection settings."}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
