"""Existing SSAFY GMS connection shared by chat, discovery and indexing."""

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"


class MissingAPIKeyError(RuntimeError):
    pass


def configured_key() -> str:
    key = os.environ.get("GMS_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError
    return key


def get_model() -> ChatOpenAI:
    return _model(configured_key())


@lru_cache(maxsize=1)
def _model(key: str) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5.4-nano", api_key=key, base_url=GMS_BASE_URL, timeout=60, max_retries=0)


def model_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Empty docent answer")
    return content.strip()
