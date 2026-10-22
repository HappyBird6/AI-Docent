(() => {
  const root = document.querySelector("#ai-docent-gallery");
  const room = root.querySelector(".gallery-room");
  const rail = root.querySelector(".gallery-rail");
  const previousButton = root.querySelector(".prev");
  const nextButton = root.querySelector(".next");
  const notice = root.querySelector("#gallery-notice");
  const noticeText = root.querySelector("#gallery-notice-text");
  const retryButton = root.querySelector("#gallery-retry");
  const status = root.querySelector("#gallery-status");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const duration = 1250;
  const storageKey = "ai-docent.current-artwork";
  const captionObserver = new ResizeObserver(([entry]) => {
    root.style.setProperty("--gallery-caption-height", `${Math.ceil(entry.target.getBoundingClientRect().height)}px`);
  });
  let artworks = [], current = 0, distance = 0, pending = null, queued = 0, completion = null, drag = null;
  const wrap = (index) => (index + artworks.length) % artworks.length;

  function safeUrl(value) {
    try {
      const url = new URL(value, location.origin);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch { return null; }
  }

  function createExhibit(artwork, offset) {
    const figure = document.createElement("figure");
    figure.className = "exhibit";
    figure.dataset.artworkId = artwork.id;
    figure.setAttribute("aria-hidden", String(offset !== 0));
    figure.inert = offset !== 0;
    const light = document.createElement("div");
    light.className = "exhibit-light";
    light.setAttribute("aria-hidden", "true");
    const frame = document.createElement("div");
    frame.className = "frame";
    const rim = document.createElement("div");
    rim.className = "frame-rim";
    const mat = document.createElement("div");
    mat.className = "frame-mat";
    const canvas = document.createElement("div");
    canvas.className = "paint-canvas";
    const image = document.createElement("img");
    image.className = "artwork-image";
    image.alt = artwork.image_alt || artwork.title;
    image.decoding = "async";
    image.loading = "eager";
    image.fetchPriority = offset === 0 ? "high" : "low";
    image.draggable = false;
    image.referrerPolicy = "no-referrer";
    image.hidden = true;
    if (artwork.image_width > 0 && artwork.image_height > 0) {
      image.width = artwork.image_width;
      image.height = artwork.image_height;
    }
    const loading = document.createElement("div");
    loading.className = "image-loading";
    loading.textContent = "작품을 불러오는 중입니다.";
    const error = document.createElement("div");
    error.className = "image-error";
    error.hidden = true;
    const errorText = document.createElement("p");
    errorText.textContent = "작품 이미지를 불러오지 못했습니다.";
    error.append(errorText);
    const sourceUrl = safeUrl(artwork.source_url);
    if (sourceUrl) {
      const link = document.createElement("a");
      link.href = sourceUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "미술관에서 보기";
      error.append(link);
    }
    let candidate = 0;
    const candidates = [...new Set(artwork.image_candidates.map(safeUrl).filter(Boolean))];
    const loadNext = () => {
      if (candidate < candidates.length) image.src = candidates[candidate++];
      else { image.hidden = true; loading.hidden = true; error.hidden = false; }
    };
    image.addEventListener("load", () => { loading.hidden = true; image.hidden = false; error.hidden = true; });
    image.addEventListener("error", loadNext);
    canvas.append(image, loading, error);
    mat.append(canvas); rim.append(mat); frame.append(rim);
    const caption = document.createElement("figcaption");
    caption.className = "painting-label";
    const title = document.createElement("h2");
    title.className = "painting-title";
    title.textContent = artwork.title;
    const artist = document.createElement("span");
    artist.className = "painting-artist";
    artist.textContent = artwork.artist;
    const year = document.createElement("span");
    year.className = "painting-year";
    year.textContent = artwork.date_display;
    caption.append(title, artist, year);
    figure.append(light, frame, caption);
    loadNext();
    return figure;
  }

  function renderWindow() {
    // Keep only the current artwork and its two neighbors in the DOM.
    const existing = new Map([...rail.children].map((figure) => [figure.dataset.artworkId, figure]));
    const figures = [-1, 0, 1].map((offset) => {
      const artwork = artworks[wrap(current + offset)];
      const figure = artworks.length >= 3 && existing.get(artwork.id) || createExhibit(artwork, offset);
      figure.setAttribute("aria-hidden", String(offset !== 0));
      figure.inert = offset !== 0;
      return figure;
    });
    rail.style.transition = "none";
    rail.replaceChildren(...figures);
    rail.style.transform = "translate3d(-100%, 0, 0)";
    rail.getBoundingClientRect();
    rail.style.transition = "";
    const caption = figures[1].querySelector(".painting-label");
    captionObserver.disconnect();
    root.style.setProperty("--gallery-caption-height", `${Math.ceil(caption.getBoundingClientRect().height)}px`);
    captionObserver.observe(caption);
    const artwork = artworks[current];
    status.textContent = `${artwork.title}, ${artwork.artist}, ${artwork.date_display}`;
    root.dispatchEvent(new CustomEvent("gallery:artworkchange", { detail: { artwork } }));
  }
  function moveWall(extra = 0) {
    root.style.setProperty("--wall-shift", `${-distance * 32 + extra * .05}px`);
    root.style.setProperty("--floor-shift", `${-distance * 46 + extra * .08}px`);
  }
  function finish() {
    if (!pending) return;
    clearTimeout(completion);
    current = pending.index;
    pending = null;
    renderWindow();
    try { localStorage.setItem(storageKey, artworks[current].id); } catch { /* Optional preference. */ }
    if (queued) {
      const direction = queued;
      queued = 0;
      requestAnimationFrame(() => navigate(direction));
    }
  }
  function navigate(direction) {
    if (artworks.length < 2) return;
    if (pending) { queued = direction; return; }
    pending = { index: wrap(current + direction) };
    distance += direction;
    moveWall();
    rail.style.transform = `translate3d(${direction > 0 ? -200 : 0}%, 0, 0)`;
    if (reducedMotion.matches) finish();
    else completion = setTimeout(finish, duration + 120);
  }
  rail.addEventListener("transitionend", (event) => {
    if (event.target === rail && event.propertyName === "transform") finish();
  });
  previousButton.addEventListener("click", () => navigate(-1));
  nextButton.addEventListener("click", () => navigate(1));
  root.addEventListener("gallery:select", (event) => {
    const index = artworks.findIndex((artwork) => artwork.id === event.detail?.artworkId);
    if (index < 0) return;
    clearTimeout(completion);
    pending = null;
    queued = 0;
    drag = null;
    room.classList.remove("is-dragging");
    current = index;
    renderWindow();
    try { localStorage.setItem(storageKey, artworks[current].id); } catch { /* Optional preference. */ }
  });
  document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.isContentEditable || event.target.closest("input, textarea, select, .docent-panel, .history-window")) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault(); navigate(event.key === "ArrowRight" ? 1 : -1);
    }
  });
  room.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button, a") || pending || !artworks.length || event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, delta: 0, id: event.pointerId, active: false };
  });
  room.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (!drag.active) {
      if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 8) { drag = null; return; }
      if (Math.abs(dx) < 8) return;
      drag.active = true;
      room.setPointerCapture(event.pointerId);
      room.classList.add("is-dragging");
    }
    drag.delta = Math.max(-room.clientWidth * .8, Math.min(room.clientWidth * .8, dx));
    rail.style.transform = `translate3d(calc(-100% + ${drag.delta}px), 0, 0)`;
    moveWall(drag.delta);
  });
  function endDrag(event, cancelled = false) {
    if (!drag || event.pointerId !== drag.id) return;
    const { delta, active } = drag;
    drag = null;
    room.classList.remove("is-dragging");
    if (room.hasPointerCapture(event.pointerId)) room.releasePointerCapture(event.pointerId);
    if (!active) return;
    if (!cancelled && Math.abs(delta) > 45) navigate(delta < 0 ? 1 : -1);
    else { rail.style.transform = "translate3d(-100%, 0, 0)"; moveWall(); }
  }
  room.addEventListener("pointerup", (event) => endDrag(event));
  room.addEventListener("pointercancel", (event) => endDrag(event, true));
  reducedMotion.addEventListener("change", () => { if (reducedMotion.matches) finish(); });
  async function initialize() {
    notice.hidden = false;
    noticeText.textContent = "작품을 불러오는 중입니다.";
    retryButton.hidden = true;
    room.setAttribute("aria-busy", "true");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch("/api/artworks", { signal: controller.signal });
      if (!response.ok) throw new Error("Catalog request failed");
      const data = await response.json();
      if (!Array.isArray(data.artworks) || !data.artworks.length) throw new Error("Empty catalog");
      artworks = data.artworks.filter((a) => a.id && a.title && Array.isArray(a.image_candidates));
      if (!artworks.length) throw new Error("Invalid catalog");
      let saved = null;
      try { saved = localStorage.getItem(storageKey); } catch { /* Optional preference. */ }
      current = Math.max(0, artworks.findIndex((artwork) => artwork.id === saved));
      root.dispatchEvent(new CustomEvent("gallery:ready", { detail: { artworks } }));
      renderWindow();
      notice.hidden = true;
      previousButton.disabled = nextButton.disabled = artworks.length < 2;
    } catch {
      noticeText.textContent = "작품 목록을 불러오지 못했습니다.";
      retryButton.hidden = false;
    } finally {
      clearTimeout(timeout);
      room.setAttribute("aria-busy", "false");
    }
  }
  retryButton.addEventListener("click", initialize);
  initialize();
})();
