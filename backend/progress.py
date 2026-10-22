"""Request-local, public progress events; never forwards prompts or provider errors."""

from contextlib import contextmanager
from contextvars import ContextVar

MESSAGES = {
    "waiting": "문서 검색을 준비하고 있어요…",
    "checking": "문서와 검색 준비 상태를 확인하고 있어요…",
    "recovering_index": "이전에 준비하던 문서의 색인을 마무리하고 있어요…",
    "planning": "질문에 맞는 검색어를 정리하고 있어요…",
    "plan_cached": "저장된 검색어를 사용하고 있어요…",
    "collecting": "외부 문서 후보를 수집하고 있어요…",
    "collection_cached": "이전에 수집한 문서를 사용하고 있어요…",
    "selecting": "질문에 필요한 문서를 고르고 있어요…",
    "selection_cached": "이전에 선별한 문서를 사용하고 있어요…",
    "indexing": "선별한 문서의 색인을 준비하고 있어요…",
    "collected": "답변에 사용할 문서 준비가 끝났어요.",
    "retrieving": "현재 작품의 문서에서 근거를 찾고 있어요…",
    "generating": "문서를 바탕으로 답변을 작성하고 있어요…",
    "generating_guide": "문서를 바탕으로 감상 가이드를 만들고 있어요…",
    "validating": "답변과 출처를 확인하고 있어요…",
}

_sink = ContextVar("docent_progress_sink", default=None)
_cancelled = ContextVar("docent_progress_cancelled", default=None)


class ProgressCancelled(RuntimeError):
    pass


@contextmanager
def progress_scope(sink, cancelled):
    token = _sink.set(sink)
    cancel_token = _cancelled.set(cancelled)
    try:
        yield
    finally:
        _cancelled.reset(cancel_token)
        _sink.reset(token)


def report_progress(stage: str, *, titles=None, current=None, total=None):
    cancelled = _cancelled.get()
    if cancelled is not None and cancelled.is_set():
        raise ProgressCancelled("Request disconnected")
    sink = _sink.get()
    if sink is None:
        return
    event = {"type": "progress", "stage": stage, "message": MESSAGES[stage]}
    if titles is not None:
        event["titles"] = list(dict.fromkeys(str(title).strip()[:160] for title in titles if str(title).strip()))[:4]
    if isinstance(current, int) and isinstance(total, int) and 1 <= current <= total <= 20:
        event.update(current=current, total=total)
    sink(event)
