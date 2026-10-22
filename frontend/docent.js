(() => {
  const root = document.querySelector("#ai-docent-gallery");
  const $ = (id) => document.getElementById(id);
  const launcher = $("docent-launcher"), panel = $("docent-panel");
  const conversationTab = $("conversation-tab"), historyTab = $("history-tab");
  const exploreTab = $("explore-tab");
  const question = $("docent-question"), send = $("docent-send");
  const guide = $("docent-guide");
  const messages = $("docent-messages"), historyList = $("docent-history-list");
  const historyWindow = $("history-window"), windowBar = $("history-window-bar");
  const storageKey = "ai-docent.conversations.v1";
  const catalog = new Map(), conversations = new Map(), drafts = new Map();
  const pendingRequests = new Map();
  const previewAnswer = "질문을 기록했어요. 이곳에 작품·작가·시대 배경을 바탕으로 한 도슨트의 답변이 표시될 예정입니다.";
  let currentArtwork = null, historyArtworkId = null, historyOpener = null, windowDrag = null;

  const { ChatRequestError, readChatStream } = window.DocentStream;
  let statusRequest = 0;

  async function refreshServiceNotice() {
    const request = ++statusRequest;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch("/api/docent/status", { signal: controller.signal, cache: "no-store" });
      if (!response.ok) return;
      const data = await response.json();
      if (request !== statusRequest || !Array.isArray(data.issues)) return;
      const notices = data.issues.slice(0, 4).filter((issue) => typeof issue?.message === "string")
        .map((issue) => issue.message.slice(0, 350));
      $("docent-service-note").textContent = notices.join("\n");
      $("docent-service-note").hidden = !notices.length;
    } catch { /* Normal requests retain their connection and retry message. */ }
    finally { clearTimeout(timeout); }
  }

  function progressText(message) {
    const progress = message.progress;
    if (!progress) return "서버에 연결하고 있어요…";
    return progress.message + (progress.current && progress.total ? ` (${progress.current}/${progress.total})` : "");
  }

  function updateProgress(id, message, event) {
    if (message.delivery !== "pending" || event.artwork_id !== id ||
        typeof event.message !== "string" || event.message.length > 300) return;
    const titles = Array.isArray(event.titles) ? event.titles.filter((title) => typeof title === "string")
      .slice(0, 4).map((title) => title.slice(0, 160)) : message.progress?.titles || [];
    message.progress = { message: event.message, titles,
      current: Number.isInteger(event.current) ? event.current : null,
      total: Number.isInteger(event.total) ? event.total : null };
    // Preserve open source details and the user's scroll position.
    for (const container of [messages, $("history-window-messages")]) {
      const following = container.scrollHeight - container.clientHeight - container.scrollTop < 12;
      for (const item of container.children) {
        if (item.dataset.messageId !== message.id) continue;
        const label = item.querySelector(".docent-delivery.pending");
        const documents = item.querySelector(".docent-progress-documents");
        if (label) label.textContent = progressText(message);
        if (documents) {
          documents.textContent = titles.length ? `준비한 문서 · ${titles.join(" · ")}` : "";
          documents.hidden = !titles.length;
        }
      }
      if (following) container.scrollTop = container.scrollHeight;
    }
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function safeURL(value) {
    try {
      const url = new URL(value);
      return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : "";
    } catch { return ""; }
  }

  function sourceData(value) {
    if (!Array.isArray(value)) return [];
    const numbers = new Set();
    return value.slice(0, 5).filter((source) => {
      if (!source || !Number.isInteger(source.number) || source.number < 1 || source.number > 5 ||
        numbers.has(source.number) || !safeURL(source.url) || typeof source.title !== "string") return false;
      numbers.add(source.number);
      return true;
    }).map((source) => {
      const cleaned = { number: source.number, url: safeURL(source.url), license_url: safeURL(source.license_url), contextual: source.contextual === true };
      for (const [key, limit] of Object.entries({ title: 500, license: 100, attribution: 2000, transformations: 2000, excerpt: 3000, document_id: 300, kind: 100, source_type: 100, retrieved_at: 100 })) {
        cleaned[key] = typeof source[key] === "string" ? source[key].slice(0, limit) : "";
      }
      cleaned.chunk_ids = Array.isArray(source.chunk_ids) ? source.chunk_ids.filter((id) => typeof id === "string").slice(0, 2).map((id) => id.slice(0, 350)) : [];
      return cleaned;
    });
  }

  function externalLink(text, href) {
    const link = element("a", "", text);
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  }

  function evidenceNodes(message, item) {
    const sources = sourceData(message.sources);
    const body = element("p", "docent-message-text");
    if (!sources.length) { body.textContent = message.content; item.append(body); return; }
    const details = element("details", "docent-sources");
    details.append(element("summary", "", `출처 ${sources.length}개`));
    const targets = new Map();
    sources.forEach((source) => {
      const entry = element("div", "docent-source");
      entry.tabIndex = -1;
      entry.append(externalLink(`[근거 ${source.number}] ${source.title} ↗`, source.url));
      const credit = element("p", "docent-source-credit", source.attribution);
      const license = element("p", "docent-source-license");
      if (source.license_url) license.append(externalLink(source.license, source.license_url));
      else license.textContent = source.license;
      if (source.license.includes("NC")) license.append(document.createTextNode(" · 비영리 이용 · 배포하는 번역·가공물에도 동일 조건 적용"));
      if (source.contextual) entry.append(element("p", "docent-source-note", "제작 시기·지역으로 검색한 일반 시대 배경 자료"));
      entry.append(credit, license);
      const excerpt = element("details", "docent-source-excerpt");
      excerpt.append(element("summary", "", "검색에 사용한 원문 발췌"), element("blockquote", "", source.excerpt));
      entry.append(excerpt);
      if (source.transformations) entry.append(element("p", "docent-source-note", `원문 처리: ${source.transformations}`));
      entry.append(element("p", "docent-source-note", "답변은 영어 원문을 AI가 한국어로 요약한 해설입니다."));
      details.append(entry);
      targets.set(source.number, entry);
    });
    let offset = 0;
    for (const match of message.content.matchAll(/\[근거\s*(\d+)\]/g)) {
      body.append(document.createTextNode(message.content.slice(offset, match.index)));
      const target = targets.get(Number(match[1]));
      if (target) {
        const citation = element("button", "docent-citation", match[0]);
        citation.type = "button";
        citation.setAttribute("aria-label", `근거 ${match[1]} 출처 확인`);
        citation.addEventListener("click", () => {
          details.open = true;
          target.focus({ preventScroll: true });
          target.scrollIntoView({ block: "nearest", behavior: "instant" });
        });
        body.append(citation);
      } else body.append(document.createTextNode(match[0]));
      offset = match.index + match[0].length;
    }
    body.append(document.createTextNode(message.content.slice(offset)));
    item.append(body, details);
  }

  // Restore only bounded, plain-text conversation data. The catalog supplies artwork metadata.
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (saved && typeof saved === "object" && !Array.isArray(saved)) {
      Object.entries(saved).slice(0, 500).forEach(([id, record]) => {
        if (!record || !Array.isArray(record.messages)) return;
        const valid = record.messages.filter((message) => message &&
          ["user", "assistant"].includes(message.role) && typeof message.content === "string")
          .slice(-200).map((message) => ({
            id: typeof message.id === "string" ? message.id.slice(0, 100) : crypto.randomUUID(),
            role: message.role, content: message.content.slice(0, 20000),
            kind: message.kind === "guide" ? "guide" : message.kind === "preview" || message.content === previewAnswer ? "preview" : "chat",
            mode: message.mode === "rag" ? "rag" : "llm",
            sources: message.role === "assistant" && message.mode === "rag" ? sourceData(message.sources) : [],
            delivery: ["pending", "failed"].includes(message.delivery) ? "failed" : "sent",
            error: message.delivery === "pending" ? "전송이 완료되지 않았습니다. 다시 보내 주세요." :
              (typeof message.error === "string" ? message.error.slice(0, 300) : "답변을 불러오지 못했습니다. 다시 보내 주세요."),
            createdAt: Number.isFinite(message.createdAt) && message.createdAt > 0 && message.createdAt <= Date.now() + 86400000 ? message.createdAt : 0,
          }));
        if (valid.some((message) => message.role === "user")) conversations.set(id, { messages: valid });
      });
    }
  } catch { /* Storage is optional; the in-memory conversation remains usable. */ }

  function persist() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(Object.fromEntries(conversations)));
      $("docent-save-note").textContent = "대화는 이 브라우저에 저장됩니다";
    } catch {
      $("docent-save-note").textContent = "현재 탭에서만 대화가 유지됩니다";
    }
  }

  function artworkImage(artwork, compact = false) {
    const holder = element("div", compact ? "docent-thumbnail" : "docent-image-holder");
    const image = element("img");
    image.alt = compact ? "" : artwork.title;
    image.referrerPolicy = "no-referrer";
    image.decoding = "async";
    image.draggable = false;
    image.hidden = true;
    const fallback = element("span", "docent-image-fallback", compact ? "◇" : "이미지를 불러오는 중입니다.");
    let candidate = 0;
    const candidates = [...new Set(artwork.image_candidates || [])].map((value) => {
      try {
        const url = new URL(value, location.origin);
        return ["http:", "https:"].includes(url.protocol) ? url.href : null;
      } catch { return null; }
    }).filter(Boolean);
    const loadNext = () => {
      if (candidate < candidates.length) image.src = candidates[candidate++];
      else { fallback.textContent = compact ? "◇" : "이미지를 불러오지 못했습니다."; }
    };
    image.addEventListener("load", () => { image.hidden = false; fallback.hidden = true; });
    image.addEventListener("error", loadNext);
    holder.append(image, fallback);
    loadNext();
    return holder;
  }

  function messageNode(message, artworkId, allowRetry = false) {
    const item = element("div", `docent-message ${message.role}`);
    item.dataset.messageId = message.id;
    item.append(element("span", "docent-message-author", message.role === "user" ? "나" :
      (message.kind === "preview" ? "도슨트 · 이전 미리보기" : message.kind === "guide" ? "도슨트 · 감상 가이드" : message.mode === "rag" ? "도슨트 · 문서 기반" : "도슨트 · 이전 일반 AI 답변")));
    evidenceNodes(message, item);
    if (message.role === "user" && message.delivery === "pending") {
      const progress = element("p", "docent-delivery pending", progressText(message));
      progress.setAttribute("role", "status");
      const documents = element("p", "docent-progress-documents", message.progress?.titles?.length ? `준비한 문서 · ${message.progress.titles.join(" · ")}` : "");
      documents.hidden = !message.progress?.titles?.length;
      item.append(progress, documents);
    } else if (message.role === "user" && message.delivery === "failed") {
      const failure = element("div", "docent-delivery failed");
      failure.append(element("p", "", message.error));
      if (allowRetry) {
        const retry = element("button", "docent-retry", "다시 보내기");
        retry.type = "button";
        retry.disabled = pendingRequests.has(artworkId);
        retry.addEventListener("click", () => requestAnswer(artworkId, message));
        failure.append(retry);
      }
      item.append(failure);
    }
    return item;
  }

  function renderConversation() {
    if (!currentArtwork) return;
    $("docent-artwork-title").textContent = currentArtwork.title;
    $("docent-artwork-credit").textContent = [currentArtwork.artist, currentArtwork.date_display].filter(Boolean).join(" · ");
    messages.replaceChildren();
    const record = conversations.get(currentArtwork.id);
    if (record?.messages.length) {
      messages.append(...record.messages.map((message) => messageNode(message, currentArtwork.id, true)));
    } else {
      const empty = element("div", "docent-empty-conversation");
      empty.append(element("span", "docent-empty-mark", "✧"), element("p", "", "그림 앞에서 시작하는 대화"));
      const suggestions = element("div", "docent-suggestions");
      ["작품 이야기", "작가 이야기", "시대 배경"].forEach((label, index) => {
        const button = element("button", "", label);
        button.type = "button";
        button.addEventListener("click", () => {
          question.value = ["이 작품에 대해 설명해줘.", "이 작품의 작가는 어떤 사람이야?", "이 작품이 만들어진 시대에 대해 알려줘."][index];
          submitQuestion();
        });
        suggestions.append(button);
      });
      empty.append(suggestions);
      messages.append(empty);
    }
    question.value = drafts.get(currentArtwork.id) || "";
    updateInput();
    messages.scrollTop = messages.scrollHeight;
  }

  function renderHistory() {
    const entries = [...conversations.entries()].filter(([id]) => catalog.has(id))
      .sort((a, b) => (b[1].messages.at(-1)?.createdAt || 0) - (a[1].messages.at(-1)?.createdAt || 0));
    $("history-count").textContent = entries.length ? String(entries.length) : "";
    historyList.replaceChildren();
    if (!entries.length) {
      historyList.append(element("p", "docent-history-empty", "아직 질문한 작품이 없어요."));
      return;
    }
    entries.forEach(([id, record]) => {
      const artwork = catalog.get(id);
      const questions = record.messages.filter((message) => message.role === "user");
      const last = questions.at(-1);
      const button = element("button", "docent-history-entry");
      button.type = "button";
      button.dataset.artworkId = id;
      button.setAttribute("aria-controls", "history-window");
      button.setAttribute("aria-label", `${artwork.title} 대화 기록 열기, 질문 ${questions.length}개`);
      const detail = element("span", "docent-history-detail");
      detail.append(element("strong", "", artwork.title), element("span", "docent-history-question", last.content));
      const date = last.createdAt ? new Intl.DateTimeFormat("ko-KR", { month: "short", day: "numeric" }).format(new Date(last.createdAt)) : "";
      detail.append(element("span", "docent-history-meta", `${date}${date ? " · " : ""}질문 ${questions.length}개`));
      button.append(artworkImage(artwork, true), detail, element("span", "docent-history-open", "↗"));
      button.addEventListener("click", () => openHistory(id, button));
      historyList.append(button);
    });
  }

  function updateInput() {
    const pending = currentArtwork && pendingRequests.has(currentArtwork.id);
    send.disabled = !currentArtwork || !question.value.trim() || pending;
    guide.disabled = !currentArtwork || pending;
    send.textContent = pending ? "·" : "↑";
    send.setAttribute("aria-label", pending ? "답변 기다리는 중" : "질문 보내기");
    messages.setAttribute("aria-busy", String(Boolean(pending)));
    question.style.height = "auto";
    question.style.height = `${Math.min(112, Math.max(48, question.scrollHeight))}px`;
  }

  function submitQuestion() {
    submitMessage(question.value.trim(), "chat");
  }

  function submitMessage(content, kind) {
    if (!currentArtwork || !content || pendingRequests.has(currentArtwork.id)) return;
    const id = currentArtwork.id;
    const record = conversations.get(id) || { messages: [] };
    const userMessage = { id: crypto.randomUUID(), role: "user", content: content.slice(0, 2000), createdAt: Date.now(), kind };
    record.messages.push(userMessage);
    record.messages = record.messages.slice(-200);
    conversations.set(id, record);
    if (kind === "chat") drafts.delete(id);
    requestAnswer(id, userMessage);
  }

  function refreshArtwork(id) {
    if (currentArtwork?.id === id) renderConversation();
    renderHistory();
    if (historyArtworkId === id) renderHistoryMessages();
  }

  async function requestAnswer(id, userMessage) {
    if (pendingRequests.has(id) || userMessage.delivery === "sent") return;
    const record = conversations.get(id);
    if (!record || !catalog.has(id)) return;
    const index = record.messages.indexOf(userMessage);
    if (index < 0) return;
    const history = record.messages.slice(0, index)
      .filter((message) => message.kind !== "preview" && !["pending", "failed"].includes(message.delivery))
      .slice(-20).map((message) => ({ role: message.role, content: message.content.slice(0, 4000) }));
    const controller = new AbortController();
    pendingRequests.set(id, controller);
    userMessage.delivery = "pending";
    delete userMessage.progress;
    delete userMessage.error;
    persist();
    refreshArtwork(id);
    const timeout = setTimeout(() => controller.abort(), 130000);
    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ artwork_id: id, question: userMessage.content, history,
          ...(userMessage.kind === "guide" ? { feature: "guide" } : {}) }),
      });
      const data = await readChatStream(response, id, (event) => updateProgress(id, userMessage, event));
      if (data.artwork_id !== id || typeof data.answer !== "string" || !data.answer.trim() || data.answer.length > 20000) {
        throw new ChatRequestError("답변을 불러오지 못했습니다. 다시 보내 주세요.");
      }
      userMessage.delivery = "sent";
      record.messages.splice(record.messages.indexOf(userMessage) + 1, 0, {
        id: crypto.randomUUID(), role: "assistant", content: data.answer.trim(), createdAt: Date.now(), kind: userMessage.kind === "guide" ? "guide" : "chat",
        mode: data.mode === "rag" ? "rag" : "llm", sources: data.mode === "rag" ? sourceData(data.sources) : [],
      });
      record.messages = record.messages.slice(-200);
    } catch (error) {
      userMessage.delivery = "failed";
      refreshServiceNotice();
      userMessage.error = error.name === "AbortError" ? "답변 시간이 길어지고 있습니다. 다시 보내 주세요." :
        (error instanceof ChatRequestError ? error.message : "서버에 연결하지 못했습니다. 연결 상태를 확인하고 다시 보내 주세요.");
    } finally {
      clearTimeout(timeout);
      pendingRequests.delete(id);
      delete userMessage.progress;
      persist();
      // A late response belongs to the artwork that was submitted, even after navigation.
      refreshArtwork(id);
    }
  }

  function setTab(history, focus = false) {
    const selected = history === "explore" ? exploreTab : history ? historyTab : conversationTab;
    for (const tab of [conversationTab, exploreTab, historyTab]) {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      $(tab.getAttribute("aria-controls")).hidden = !active;
    }
    if (selected === historyTab) renderHistory();
    else if (selected === conversationTab) updateInput();
    if (focus) selected.focus();
  }

  function setOpen(open) {
    root.classList.toggle("docent-open", open);
    panel.inert = !open;
    panel.setAttribute("aria-hidden", String(!open));
    launcher.setAttribute("aria-expanded", String(open));
    if (open) { $("docent-close").focus({ preventScroll: true }); refreshServiceNotice(); }
    else launcher.focus({ preventScroll: true });
  }

  function renderHistoryMessages() {
    $("history-window-messages").replaceChildren(...(conversations.get(historyArtworkId)?.messages || [])
      .map((message) => messageNode(message, historyArtworkId)));
  }

  function openHistory(id, opener) {
    const artwork = catalog.get(id);
    if (!artwork) return;
    historyArtworkId = id;
    historyOpener = opener;
    historyWindow.dataset.artworkId = id;
    $("history-artwork-image").replaceChildren(artworkImage(artwork));
    $("history-artwork-title").textContent = artwork.title;
    $("history-artwork-artist").textContent = artwork.artist;
    $("history-artwork-year").textContent = artwork.date_display;
    renderHistoryMessages();
    historyWindow.hidden = false;
    historyWindow.querySelector(".history-window-body").scrollTop = 0;
    $("history-window-close").focus({ preventScroll: true });
  }

  function closeHistory() {
    historyWindow.hidden = true;
    historyArtworkId = null;
    const target = historyOpener?.isConnected && !panel.inert ? historyOpener : launcher;
    target.focus({ preventScroll: true });
  }

  launcher.addEventListener("click", () => setOpen(true));
  $("docent-close").addEventListener("click", () => setOpen(false));
  $("history-window-close").addEventListener("click", closeHistory);
  conversationTab.addEventListener("click", () => setTab(false));
  historyTab.addEventListener("click", () => setTab(true));
  exploreTab.addEventListener("click", () => setTab("explore"));
  panel.querySelector(".docent-tabs").addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = [conversationTab, exploreTab, historyTab];
    const index = tabs.findIndex((tab) => tab.getAttribute("aria-selected") === "true");
    const next = event.key === "Home" ? 0 : event.key === "End" ? 2 : (index + (event.key === "ArrowRight" ? 1 : -1) + 3) % 3;
    setTab([false, "explore", true][next], true);
  });

  const searchQuery = $("artwork-search-query"), searchSend = $("artwork-search-send");
  const searchResults = $("artwork-search-results"), searchStatus = $("artwork-search-status");
  let searching = false;
  searchQuery.addEventListener("input", () => { searchSend.disabled = searching || !searchQuery.value.trim(); });
  searchQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      $("artwork-search-form").requestSubmit();
    }
  });
  $("artwork-search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = searchQuery.value.trim();
    if (!query || searching) return;
    searching = true;
    searchSend.disabled = true;
    searchSend.textContent = "찾는 중";
    searchStatus.textContent = "조건에 맞는 작품을 찾고 있어요…";
    searchResults.replaceChildren();
    searchResults.setAttribute("aria-busy", "true");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 160000);
    try {
      const response = await fetch("/api/artworks/search", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      if (!response.ok) throw new ChatRequestError(typeof data.detail === "string" ? data.detail.slice(0, 300) : "작품을 찾지 못했습니다. 다시 시도해 주세요.");
      if (!Array.isArray(data.results) || data.query !== query) throw new ChatRequestError("검색 결과를 불러오지 못했습니다.");
      const seen = new Set();
      const results = data.results.slice(0, 20).filter((result) => {
        if (!catalog.has(result.id) || typeof result.reason !== "string" || seen.has(result.id)) return false;
        seen.add(result.id);
        return true;
      });
      searchStatus.textContent = results.length ? `조건에 맞는 작품 ${results.length}점` : "수집된 작품 중 조건을 확인할 수 있는 작품이 없습니다.";
      for (const result of results) {
        const artwork = catalog.get(result.id);
        const card = element("article", "artwork-search-card");
        const choose = element("button", "artwork-search-choose");
        choose.type = "button";
        choose.dataset.artworkId = artwork.id;
        choose.setAttribute("aria-label", `${artwork.title} 작품 보고 대화하기`);
        const detail = element("span", "artwork-search-detail");
        detail.append(element("strong", "", artwork.title), element("span", "", `${artwork.artist} · ${artwork.date_display}`));
        choose.append(artworkImage(artwork, true), detail, element("span", "", "→"));
        choose.addEventListener("click", () => {
          root.dispatchEvent(new CustomEvent("gallery:select", { detail: { artworkId: artwork.id } }));
          setTab(false);
          question.focus({ preventScroll: true });
        });
        card.append(choose, element("p", "artwork-search-reason", result.reason.slice(0, 600)));
        if (safeURL(artwork.source_url)) card.append(externalLink("미술관 정보 ↗", safeURL(artwork.source_url)));
        searchResults.append(card);
      }
    } catch (error) {
      searchStatus.textContent = error.name === "AbortError" ? "검색 시간이 길어지고 있습니다. 다시 시도해 주세요." :
        error instanceof ChatRequestError ? error.message : "서버에 연결하지 못했습니다. 다시 시도해 주세요.";
    } finally {
      clearTimeout(timeout);
      searching = false;
      searchSend.textContent = "찾기";
      searchSend.disabled = !searchQuery.value.trim();
      searchResults.setAttribute("aria-busy", "false");
    }
  });
  question.addEventListener("input", () => {
    if (currentArtwork) drafts.set(currentArtwork.id, question.value);
    updateInput();
  });
  guide.addEventListener("click", () => submitMessage("현재 작품의 문서를 바탕으로 작품 감상 가이드를 만들어줘.", "guide"));
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      submitQuestion();
    }
  });
  $("docent-form").addEventListener("submit", (event) => { event.preventDefault(); submitQuestion(); });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || event.isComposing) return;
    if (!historyWindow.hidden) { closeHistory(); event.preventDefault(); }
    else if (root.classList.contains("docent-open")) { setOpen(false); event.preventDefault(); }
  });

  // A modeless, movable history window: no backdrop, focus trap or gallery navigation.
  windowBar.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest("button") || matchMedia("(max-width: 760px)").matches) return;
    const rect = historyWindow.getBoundingClientRect();
    windowDrag = { x: event.clientX - rect.left, y: event.clientY - rect.top, pointerId: event.pointerId };
    windowBar.setPointerCapture(event.pointerId);
  });
  windowBar.addEventListener("pointermove", (event) => {
    if (!windowDrag || event.pointerId !== windowDrag.pointerId) return;
    const left = Math.max(8, Math.min(innerWidth - historyWindow.offsetWidth - 8, event.clientX - windowDrag.x));
    const top = Math.max(8, Math.min(innerHeight - historyWindow.offsetHeight - 8, event.clientY - windowDrag.y));
    historyWindow.style.left = `${left}px`;
    historyWindow.style.top = `${top}px`;
  });
  function endWindowDrag(event) {
    if (!windowDrag || event.pointerId !== windowDrag.pointerId) return;
    windowDrag = null;
    if (windowBar.hasPointerCapture(event.pointerId)) windowBar.releasePointerCapture(event.pointerId);
  }
  windowBar.addEventListener("pointerup", endWindowDrag);
  windowBar.addEventListener("pointercancel", endWindowDrag);
  window.addEventListener("resize", () => { historyWindow.style.left = ""; historyWindow.style.top = ""; });

  root.addEventListener("gallery:ready", (event) => {
    event.detail.artworks.forEach((artwork) => catalog.set(artwork.id, artwork));
    renderHistory();
  });
  root.addEventListener("gallery:artworkchange", (event) => {
    currentArtwork = event.detail.artwork;
    launcher.disabled = false;
    setTab(false);
    renderConversation();
    // The separate history window deliberately stays on its selected artwork.
  });
})();
