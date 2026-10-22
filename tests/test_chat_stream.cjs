const test = require("node:test");
const assert = require("node:assert/strict");
const { readChatStream, ChatRequestError } = require("../frontend/chat-stream.js");
const id = "aic:28560";
const answer = { artwork_id: id, answer: "한국어 답변 [근거 1]", mode: "rag", sources: [{ number: 1 }] };
function frame(event) { return { artwork_id: id, ...event }; }
function response(events, chunkSize = 7, trailingNewline = true) {
  const bytes = new TextEncoder().encode(events.map(JSON.stringify).join("\n") + (trailingNewline ? "\n" : ""));
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += chunkSize) controller.enqueue(bytes.slice(i, i + chunkSize));
      controller.close();
    }
  }), { headers: { "content-type": "application/x-ndjson" } });
}
test("UTF-8 byte boundaries and multiple frames preserve progress and the final source payload", async () => {
  const updates = [];
  const result = await readChatStream(response([
    frame({ type: "progress", stage: "collecting", message: "문서 수집 중", current: 1, total: 2 }),
    frame({ type: "heartbeat" }),
    frame({ type: "progress", stage: "generating", message: "답변 생성 중", titles: ["미술관 해설"] }),
    frame({ type: "result", data: answer })
  ], 1), id, event => updates.push(event));
  assert.deepEqual(result, answer);
  assert.deepEqual(updates.map(event => event.stage), ["collecting", "generating"]);
});
test("terminal result without a trailing newline is accepted", async () => {
  assert.deepEqual(await readChatStream(response([frame({ type: "result", data: answer })], 1000, false), id, () => {}), answer);
});
test("an interrupted stream is a retryable failure, not an empty success", async () => {
  await assert.rejects(readChatStream(response([frame({ type: "progress", message: "수집 중" })]), id, () => {}), ChatRequestError);
});
test("missing key/index errors are preserved after streaming starts", async () => {
  await assert.rejects(readChatStream(response([frame({ type: "error", status: 503, detail: "GMS_KEY를 설정해 주세요." })]), id, () => {}), /GMS_KEY/);
});
test("another artwork cannot update this request", async () => {
  await assert.rejects(readChatStream(response([{ artwork_id: "other", type: "progress", message: "수집 중" }]), id, () => assert.fail()), ChatRequestError);
});
test("ordinary HTTP validation errors retain an actionable response", async () => {
  const input = new Response(JSON.stringify({ detail: "질문을 입력해 주세요." }), { status: 422 });
  await assert.rejects(readChatStream(input, id, () => {}), /질문을 입력/);
});
