/* UTF-8 NDJSON transport; final answers retain the existing chat response shape. */
(() => {
  class ChatRequestError extends Error {}

  async function readChatStream(response, artworkId, onProgress) {
    if (!response.ok) {
      let data;
      try { data = await response.json(); } catch { /* Use a safe message below. */ }
      throw new ChatRequestError(typeof data?.detail === "string" ? data.detail.slice(0, 300) : "요청 내용을 확인하고 다시 보내 주세요.");
    }
    if (!response.body || !response.headers.get("content-type")?.includes("application/x-ndjson")) {
      throw new ChatRequestError("진행 상태를 받지 못했습니다. 새로고침한 뒤 다시 보내 주세요.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: true });
    let buffer = "", count = 0, result = null;
    function receive(line) {
      if (!line.trim()) return;
      if (++count > 512 || line.length > 256000) throw new ChatRequestError("응답을 확인하지 못했습니다. 다시 보내 주세요.");
      let event;
      try { event = JSON.parse(line); } catch { throw new ChatRequestError("응답 연결이 중단되었습니다. 다시 보내 주세요."); }
      if (!event || event.artwork_id !== artworkId) throw new ChatRequestError("다른 작품의 응답이 도착했습니다. 다시 보내 주세요.");
      if (event.type === "heartbeat") return;
      if (event.type === "progress") { onProgress(event); return; }
      if (event.type === "error") throw new ChatRequestError(typeof event.detail === "string" ? event.detail.slice(0, 300) : "답변을 불러오지 못했습니다. 다시 보내 주세요.");
      if (event.type === "result" && event.data && typeof event.data === "object") { result = event.data; return; }
      throw new ChatRequestError("응답을 확인하지 못했습니다. 다시 보내 주세요.");
    }
    try {
      while (!result) {
        const { done, value } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        let newline;
        while (!result && (newline = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          receive(line);
        }
        if (buffer.length > 256000) throw new ChatRequestError("응답이 너무 깁니다. 다시 보내 주세요.");
        if (done) {
          if (!result && buffer.trim()) receive(buffer);
          if (!result) throw new ChatRequestError("답변 연결이 완료되기 전에 중단되었습니다. 다시 보내 주세요.");
        }
      }
      return result;
    } catch (error) {
      if (error instanceof ChatRequestError || error.name === "AbortError") throw error;
      throw new ChatRequestError("답변 연결이 중단되었습니다. 다시 보내 주세요.");
    } finally {
      try { await reader.cancel(); } catch { /* The network may already be closed. */ }
      reader.releaseLock();
    }
  }

  const api = { readChatStream, ChatRequestError };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.DocentStream = api;
})();
