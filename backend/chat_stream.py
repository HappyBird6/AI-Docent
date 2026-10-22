"""POST response stream with real stage updates and the unchanged final chat payload."""

import asyncio
import json
from threading import Event

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .progress import progress_scope


def stream_chat(artwork_id: str, run):
    async def events():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        cancelled = Event()

        def emit(event):
            if cancelled.is_set():
                return
            try:
                loop.call_soon_threadsafe(queue.put_nowait, {"artwork_id": artwork_id, **event})
            except RuntimeError:
                # A disconnected client's event loop may already have closed.
                pass

        def worker():
            with progress_scope(emit, cancelled):
                try:
                    result = run()
                    emit({"type": "result", "data": result.model_dump()})
                except HTTPException as error:
                    emit({"type": "error", "status": error.status_code, "detail": error.detail})
                except Exception:
                    emit({"type": "error", "status": 502, "detail": "답변을 불러오지 못했습니다. 잠시 후 다시 보내 주세요."})

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=10)
                except TimeoutError:
                    # A keepalive is transport-only, never a fabricated stage.
                    event = {"type": "heartbeat", "artwork_id": artwork_id}
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in ("result", "error"):
                    break
        finally:
            cancelled.set()
            if not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
