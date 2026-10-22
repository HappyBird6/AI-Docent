"""HTML/CSS/JS 갤러리와 수집 문서에 근거한 도슨트 RAG API.

실행: python -m backend.main (프로젝트 루트에서)
접속: http://127.0.0.1:7777
"""

import os
import mimetypes
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from openai import APITimeoutError, AuthenticationError, RateLimitError

if __package__:
    from .artworks import gallery_catalog
    from .docent import MissingAPIKeyError, generate_answer
    from .rag import CorpusUnavailableError
    from .discovery import search_artworks
    from .metadata_index import IndexUnavailableError, metadata_index_status
    from .document_index import document_index_status
    from .docent import get_graph
    from .external_documents import CollectionUnavailableError
    from .chat_stream import stream_chat
    from .readiness import KEY_MISSING, KEY_INVALID, METADATA_MISSING, index_message, service_status
else:
    # Keep package-relative service imports working for direct script execution.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.artworks import gallery_catalog
    from backend.docent import MissingAPIKeyError, generate_answer
    from backend.rag import CorpusUnavailableError
    from backend.discovery import search_artworks
    from backend.metadata_index import IndexUnavailableError, metadata_index_status
    from backend.document_index import document_index_status
    from backend.docent import get_graph
    from backend.external_documents import CollectionUnavailableError
    from backend.chat_stream import stream_chat
    from backend.readiness import KEY_MISSING, KEY_INVALID, METADATA_MISSING, index_message, service_status

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="AI Docent")
# Windows registries can classify .webp as application/octet-stream.
mimetypes.add_type("image/webp", ".webp")
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")


class ChatTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    artwork_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    feature: Literal["chat", "guide"] = "chat"


class Source(BaseModel):
    number: int
    title: str
    url: str
    document_id: str
    chunk_ids: list[str]
    kind: str
    source_type: str
    license: str
    license_url: str
    attribution: str
    transformations: str
    excerpt: str
    retrieved_at: str
    contextual: bool


class ChatResponse(BaseModel):
    artwork_id: str
    answer: str
    mode: Literal["rag"] = "rag"
    sources: list[Source] = Field(default_factory=list)


class ArtworkSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=8, ge=1, le=20)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/api/config")
def config():
    return {
        "domain": "미술 작품",
        "chat_mode": "rag",
        "rag_enabled": True,
        "retrieval": "artwork_scoped_chroma",
        "pipeline": "langgraph_retrieve_generate",
        "artwork_search": "chroma_semantic_then_llm_filter",
        "configured": bool(os.environ.get("GMS_KEY", "").strip()),
    }


@app.get("/api/artworks")
def artworks():
    try:
        items = gallery_catalog()
        if not items:
            raise ValueError("Empty artwork catalog")
        return {"artworks": items, "total": len(items)}
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=503,
            detail="작품 목록을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.",
        ) from None


@app.get("/api/index/status")
def index_status():
    try:
        return metadata_index_status()
    except Exception:
        raise HTTPException(status_code=503, detail="작품 검색 설정과 데이터를 확인해 주세요.") from None


@app.post("/api/artworks/search")
def artwork_search(payload: ArtworkSearchRequest):
    try:
        return search_artworks(payload.query, payload.limit)
    except IndexUnavailableError:
        raise HTTPException(status_code=503, detail=METADATA_MISSING) from None
    except MissingAPIKeyError:
        raise HTTPException(status_code=503, detail=KEY_MISSING) from None
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="작품 검색 시간이 길어지고 있습니다. 잠시 후 다시 시도해 주세요.") from None
    except RateLimitError:
        raise HTTPException(status_code=429, detail="지금은 검색 요청이 많습니다. 잠시 후 다시 시도해 주세요.") from None
    except AuthenticationError:
        raise HTTPException(status_code=503, detail=KEY_INVALID) from None
    except Exception:
        raise HTTPException(status_code=502, detail="작품을 검색하지 못했습니다. 잠시 후 다시 시도해 주세요.") from None


@app.get("/api/rag/status")
def rag_status():
    try:
        metadata, documents = metadata_index_status(), document_index_status()
        return {"ready": metadata["ready"] and documents["ready"], "metadata": metadata, "documents": documents,
                "pipeline": "langgraph_retrieve_generate", "graph_mermaid": get_graph().get_graph().draw_mermaid()}
    except Exception:
        raise HTTPException(status_code=503, detail="RAG 문서와 색인 설정을 확인해 주세요.") from None


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        artwork = next((item for item in gallery_catalog() if item["id"] == payload.artwork_id), None)
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=503, detail="작품 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.") from None
    if artwork is None:
        raise HTTPException(
            status_code=404, detail="질문할 작품을 찾을 수 없습니다. 작품 목록을 다시 불러와 주세요."
        )
    try:
        history = [turn.model_dump() for turn in payload.history]
        result = (generate_answer(artwork, payload.question, history, feature="guide") if payload.feature == "guide"
                  else generate_answer(artwork, payload.question, history))
        return ChatResponse(artwork_id=artwork["id"], **result)
    except CorpusUnavailableError:
        raise HTTPException(status_code=503, detail="해설 문서를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.") from None
    except IndexUnavailableError as error:
        raise HTTPException(status_code=503, detail=index_message(error)) from None
    except MissingAPIKeyError:
        raise HTTPException(
            status_code=503,
            detail=KEY_MISSING,
        ) from None
    except CollectionUnavailableError:
        raise HTTPException(status_code=503, detail="외부 문서 수집·선별을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요. 서버의 WIKI_CONTACT 설정과 외부 연결도 확인해 주세요.") from None
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="답변 시간이 길어지고 있습니다. 잠시 후 다시 보내 주세요.") from None
    except RateLimitError:
        raise HTTPException(status_code=429, detail="지금은 도슨트 요청이 많습니다. 잠시 후 다시 시도해 주세요.") from None
    except AuthenticationError:
        raise HTTPException(status_code=503, detail=KEY_INVALID) from None
    except Exception:
        # 외부 API의 오류에는 키·프롬프트가 포함될 수 있으므로 화면에 전달하지 않는다.
        raise HTTPException(
            status_code=502,
            detail="답변을 불러오지 못했습니다. 잠시 후 다시 보내 주세요.",
        ) from None


@app.get("/api/docent/status")
def docent_status():
    return service_status()


@app.post("/api/chat/stream")
def chat_progress(payload: ChatRequest):
    return stream_chat(payload.artwork_id, lambda: chat(payload))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7777)
