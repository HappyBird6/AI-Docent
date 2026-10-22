"""Actionable setup messages shared by normal/stream chat and the panel notice."""

import os

from .document_index import document_index_status
from .metadata_index import metadata_index_status, search_settings

KEY_MISSING = "AI 연결 키가 설정되지 않았습니다. 서버의 .env에 GMS_KEY를 설정한 뒤 서버를 다시 시작해 주세요."
KEY_INVALID = "AI 연결 키의 인증에 실패했습니다. 서버의 GMS_KEY 설정과 사용 권한을 확인한 뒤 다시 시도해 주세요."
METADATA_MISSING = "작품 검색 색인이 없거나 최신 데이터와 맞지 않습니다. 서버에서 python -m scripts.index_artworks를 실행해 주세요."
DOCUMENTS_MISSING = "해설 문서 색인이 없거나 최신 문서와 맞지 않습니다. 서버에서 python -m scripts.index_documents를 실행해 주세요."
INDEX_MISSING = "작품 또는 해설 문서 색인이 준비되지 않았습니다. 서버에서 python -m scripts.index_artworks와 python -m scripts.index_documents를 실행해 주세요."


def index_message(error):
    try:
        settings = search_settings()
        if getattr(error, "collection", None) == settings.metadata_collection:
            return METADATA_MISSING
        if getattr(error, "collection", None) == settings.document_collection:
            return DOCUMENTS_MISSING
    except Exception:
        pass
    return INDEX_MISSING


def service_status() -> dict:
    issues = []
    if not os.environ.get("GMS_KEY", "").strip():
        issues.append({"code": "GMS_KEY_MISSING", "message": KEY_MISSING})
    try:
        if not metadata_index_status()["ready"]:
            issues.append({"code": "METADATA_INDEX_UNAVAILABLE", "message": METADATA_MISSING})
        if not document_index_status()["ready"]:
            issues.append({"code": "DOCUMENT_INDEX_UNAVAILABLE", "message": DOCUMENTS_MISSING})
    except Exception:
        issues.append({"code": "SETUP_UNAVAILABLE", "message": "작품·문서 데이터와 검색 설정을 확인해 주세요. 서버의 데이터 준비 및 색인 절차를 완료한 뒤 다시 시도해 주세요."})
    return {"ready": not issues, "issues": issues}
