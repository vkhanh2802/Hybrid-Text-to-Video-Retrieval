const form = document.querySelector("#search-form");
const searchButton = document.querySelector("#search-button");
const loadingPanel = document.querySelector("#loading");
const loadingTitle = document.querySelector("#loading-title");
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const resultsSection = document.querySelector("#results-section");
const resultsGrid = document.querySelector("#results-grid");
const resultCount = document.querySelector("#result-count");
const elapsedTime = document.querySelector("#elapsed-time");
const queryVariants = document.querySelector("#query-variants");
const systemState = document.querySelector("#system-state");
const stateText = document.querySelector("#state-text");
const previewDialog = document.querySelector("#preview-dialog");
const previewImage = document.querySelector("#preview-image");
const previewLocation = document.querySelector("#preview-location");
const previewCaption = document.querySelector("#preview-caption");
const closePreview = document.querySelector("#close-preview");
const videoDescription = document.querySelector("#dres-video-description");
const manualMode = /(?:^\?|&)mode=manual(?:&|$)/.test(window.location?.search || "");
const manualQuery = document.querySelector("#manual-query");
const manualQueryPanel = document.querySelector("#manual-query-panel");
const officialQuerySource = document.querySelector("#official-query-source");
const sourceLabel = document.querySelector(".char-count");
const searchNote = document.querySelector("#search-note");
const dresManualToggle = document.querySelector("#dres-manual-toggle");
const DEFAULT_SOURCE = officialQuerySource.textContent;
const DEFAULT_SEARCH_NOTE = searchNote.textContent;
const MANUAL_SEARCH_NOTE = "Bấm ảnh để phóng to.";
const DRES_MANUAL_SEARCH_NOTE = "Query tự nhập · không thay đổi đề DRES.";
let dresManualOverride = false;
let dresTaskName = "";
let inputMode = "text";
let searchBusy = false;
let dresCanSearch = false;
let resultsDresTaskId = null;
let imageRequests = new AbortController();
const imageUrls = new Set();
let imageObserver = null;
let imageLoaders = new Map();
let imageQueue = { pending: [], active: 0 };

function queueResultImage(load) {
  const queue = imageQueue;
  queue.pending.push(load);
  function drain() {
    while (queue.active < 8 && queue.pending.length) {
      queue.active++;
      Promise.resolve(queue.pending.shift()()).finally(() => { queue.active--; drain(); });
    }
  }
  drain();
}

// Hover zoom: after a short dwell, float a large copy of the already-loaded
// blob next to the card (no extra image request). Mouse-only; hides on scroll.
const hoverZoom = document.querySelector("#hover-zoom");
const hoverZoomImage = document.querySelector("#hover-zoom-image");
const HOVER_ZOOM_DELAY_MS = 450;
let hoverZoomTimer = null;

function hideHoverZoom() {
  if (hoverZoomTimer !== null) clearTimeout(hoverZoomTimer);
  hoverZoomTimer = null;
  hoverZoom.hidden = true;
}

function scheduleHoverZoom(anchor, result) {
  hideHoverZoom();
  const finePointer = typeof window.matchMedia !== "function"
    || window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  if (!result.preview_url || !finePointer || typeof anchor.getBoundingClientRect !== "function") return;
  hoverZoomTimer = setTimeout(() => {
    hoverZoomTimer = null;
    const rect = anchor.getBoundingClientRect();
    const margin = 12, gap = 12;
    const width = Math.min(480, window.innerWidth - margin * 2);
    const height = width * 9 / 16;
    let left = rect.right + gap;
    if (left + width > window.innerWidth - margin) left = rect.left - width - gap;
    left = Math.min(Math.max(margin, left), window.innerWidth - width - margin);
    const top = Math.min(Math.max(margin, rect.top + (rect.height - height) / 2),
      Math.max(margin, window.innerHeight - height - margin));
    hoverZoomImage.src = result.preview_url;
    Object.assign(hoverZoom.style, { top: `${top}px`, left: `${left}px`, width: `${width}px`, height: `${height}px` });
    hoverZoom.hidden = false;
  }, HOVER_ZOOM_DELAY_MS);
}

window.addEventListener("scroll", hideHoverZoom, true);
window.addEventListener("resize", hideHoverZoom);

function lazyResultImage(image, load) {
  if (typeof IntersectionObserver === "undefined") {
    queueResultImage(load);
    return;
  }
  if (!imageObserver) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const loader = imageLoaders.get(entry.target);
        observer.unobserve(entry.target);
        imageLoaders.delete(entry.target);
        if (loader) queueResultImage(loader);
      });
    }, { rootMargin: "300px" });
    imageObserver = observer;
  }
  imageLoaders.set(image, load);
  imageObserver.observe(image);
}

function releaseResultImages() {
  hideHoverZoom();
  hoverZoomImage.removeAttribute("src");
  imageRequests.abort();
  imageObserver?.disconnect();
  imageObserver = null;
  imageLoaders.clear();
  imageQueue.pending.length = 0;
  imageQueue = { pending: [], active: 0 };
  imageRequests = new AbortController();
  if (previewDialog.open) previewDialog.close();
  previewImage.removeAttribute("src");
  resultsGrid.replaceChildren();
  document.querySelector("#temporal-results-grid").replaceChildren();
  document.querySelector("#temporal-results-section").hidden = true;
  imageUrls.forEach((url) => URL.revokeObjectURL(url));
  imageUrls.clear();
}

function setSystemState(state, vlmEnabled = true) {
  const labels = {
    cold: "Chưa nạp model",
    loading: "Đang nạp model",
    ready: inputMode === "video" ? "Sẵn sàng · Video visual" : (vlmEnabled ? "Sẵn sàng · VLM ON" : "Sẵn sàng · VLM OFF"),
    failed: "Khởi tạo thất bại",
  };
  systemState.className = `system-state ${state}`;
  stateText.textContent = labels[state] || state;
}

function setBusy(busy) {
  searchBusy = busy;
  searchButton.disabled = busy || (!manualMode && !dresManualOverride && (!dresConnected || !dresCanSearch));
  manualQuery.disabled = busy || !(manualMode || dresManualOverride);
  dresManualToggle.disabled = busy || !dresConnected;
  form.setAttribute("aria-busy", String(busy));
  loadingPanel.hidden = !busy;
  updateVideoDescriptionControls();
  if (busy) {
    loadingPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function renderSearchMode() {
  const manualInput = manualMode || dresManualOverride;
  manualQueryPanel.hidden = !manualInput;
  manualQuery.disabled = searchBusy || !manualInput;
  manualQuery.required = manualInput;
  if (manualInput) {
    officialQuerySource.textContent = dresManualOverride
      ? "NGUỒN: QUERY CỦA BẠN · L01–L21 · Không dùng đề DRES."
      : "NGUỒN: QUERY CỦA BẠN · L01–L21";
    sourceLabel.textContent = dresManualOverride ? "TỰ NHẬP · KHÔNG DÙNG ĐỀ DRES" : "KHÔNG DÙNG DRES";
    searchButton.textContent = "Tìm keyframes ↗";
    searchButton.disabled = searchBusy;
    searchNote.textContent = dresManualOverride ? DRES_MANUAL_SEARCH_NOTE : MANUAL_SEARCH_NOTE;
    return;
  }
  if (!dresConnected) {
    officialQuerySource.textContent = DEFAULT_SOURCE;
    sourceLabel.textContent = "NGUỒN: DRES";
    searchButton.textContent = "Tìm theo đề DRES";
    searchButton.disabled = searchBusy;
    searchNote.textContent = DEFAULT_SEARCH_NOTE;
    return;
  }
  officialQuerySource.textContent = dresCanSearch
    ? `Dùng trực tiếp đề ${inputMode === "video" ? "video" : "text"} của task ${dresTaskName || "DRES"}. ${inputMode === "video" ? "Có thể nhập text bổ sung bên dưới: text chạy như đề text, sau đó fusion kết quả 70% text + 30% video; text không thay thế video chính thức." : "Toàn bộ các câu DRES đã phát được gộp thành query text chính thức; backend không nhận query thay thế."}`
    : "Đang chờ đề text/video chính thức từ DRES. Không cần nhập dữ liệu thủ công.";
  sourceLabel.textContent = "NGUỒN: DRES";
  searchButton.textContent = "Tìm theo đề DRES";
  searchButton.disabled = searchBusy || !dresCanSearch;
  searchNote.textContent = DEFAULT_SEARCH_NOTE;
}

function setDresManualOverride(enabled, focus = false) {
  dresManualOverride = Boolean(enabled) && dresConnected;
  dresManualToggle.checked = dresManualOverride;
  dresManualToggle.setAttribute("aria-expanded", String(dresManualOverride));
  renderSearchMode();
  updateVideoDescriptionControls();
  if (dresManualOverride && focus) {
    manualQuery.focus();
    manualQueryPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function updateVideoDescriptionControls() {
  const available = dresCanSearch && inputMode === "video" && !dresManualOverride;
  document.querySelector("#dres-video-description-panel").hidden = !available;
  videoDescription.disabled = !available || searchBusy;
  document.querySelector("#dres-video-description-count").textContent = `${videoDescription.value.length} / 2000`;
}

function resetVideoDescription() {
  videoDescription.value = "";
  updateVideoDescriptionControls();
}

videoDescription.addEventListener("input", updateVideoDescriptionControls);

function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function metadataChip(label, value, className = "") {
  const chip = document.createElement("span");
  chip.className = className;
  chip.textContent = `${label} ${value ?? "—"}`;
  return chip;
}

function openPreview(result) {
  if (!result.preview_url) return;
  previewImage.src = result.preview_url;
  previewImage.alt = `Rank ${result.rank}, ${result.video_id}, frame ${result.frame_id}`;
  previewLocation.textContent = `${result.video_id} · IMG ${result.image_number ?? "—"} · FRAME ${result.frame_id}`;
  previewCaption.textContent = result.caption || "Không có caption.";
  previewDialog.showModal();
}

function createResultCard(result) {
  const card = document.createElement("article");
  card.className = "result-card";

  const imageButton = document.createElement("button");
  imageButton.className = "image-button";
  imageButton.type = "button";
  imageButton.setAttribute("aria-label", `Mở ảnh rank ${result.rank}`);

  if (result.image_url) {
    const image = document.createElement("img");
    image.alt = `${result.video_id}, image ${result.image_number}, frame ${result.frame_id}`;
    const showMissing = () => {
      image.remove();
      imageButton.disabled = true;
      const missing = document.createElement("span");
      missing.className = "image-missing";
      missing.textContent = "IMAGE NOT FOUND";
      imageButton.prepend(missing);
    };
    const signal = imageRequests.signal;
    image.addEventListener("error", showMissing, { once: true });
    image.decoding = "async";
    lazyResultImage(imageButton, () => fetch(result.image_url, { cache: "no-store", signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        if (signal.aborted) return;
        result.preview_url = URL.createObjectURL(blob);
        imageUrls.add(result.preview_url);
        image.src = result.preview_url;
      })
      .catch(() => { if (!signal.aborted) showMissing(); }));
    imageButton.append(image);
    imageButton.addEventListener("click", (event) => {
      hideHoverZoom();
      if (event.ctrlKey || event.metaKey) findSimilar(result);
      else openPreview(result);
    });
    imageButton.addEventListener("mouseenter", () => scheduleHoverZoom(imageButton, result));
    imageButton.addEventListener("mouseleave", hideHoverZoom);
  } else {
    const missing = document.createElement("span");
    missing.className = "image-missing";
    missing.textContent = "IMAGE NOT FOUND";
    imageButton.append(missing);
    imageButton.disabled = true;
  }

  const rank = document.createElement("span");
  rank.className = "rank-badge";
  rank.textContent = String(result.rank).padStart(2, "0");
  imageButton.append(rank);

  const content = document.createElement("div");
  content.className = "card-content";

  const video = document.createElement("strong");
  video.className = "video-id";
  video.textContent = result.video_id;

  const location = document.createElement("div");
  location.className = "location-row";
  location.append(
    metadataChip("IMG", result.image_number),
    metadataChip("FRAME", result.frame_id),
    metadataChip("SCORE", formatNumber(result.score), "score"),
  );

  const caption = document.createElement("p");
  caption.className = "caption";
  caption.textContent = result.caption || "Không có caption cho frame này.";
  caption.title = result.caption || "";

  content.append(video, location, caption);
  const llmStatus = document.createElement("span");
  const scored = result.vlm_status === "reranked" && Number.isFinite(result.vlm_score);
  llmStatus.className = scored ? "llm-status scored" : "llm-status";
  llmStatus.textContent = scored ? `LLM ${formatNumber(result.vlm_score, 1)}/10`
    : ({ missing_image: "LLM: thiếu ảnh, chưa chấm", error: "LLM: lỗi chấm",
      invalid_score: "LLM: điểm không hợp lệ", skipped_video_query: "Chỉ truy hồi · không dùng LLM",
      skipped_temporal: "Chuỗi gợi ý · chưa dùng LLM xác minh" }[result.vlm_status] || "Chỉ truy hồi · chưa chấm LLM");
  content.append(llmStatus);
  const submitButton = document.createElement("button");
  submitButton.type = "button";
  submitButton.className = "dres-select";
  submitButton.textContent = "Chọn frame để nộp DRES";
  submitButton.disabled = !Number.isInteger(result.image_number) || !Number.isInteger(result.frame_id);
  submitButton.addEventListener("click", () => prepareDresSubmission(result));
  const similarButton = document.createElement("button");
  similarButton.type = "button";
  similarButton.className = "similar-select";
  similarButton.textContent = "Tìm frame tương tự (S)";
  similarButton.title = "Phím S khi rê chuột lên ảnh · Ctrl+click ảnh";
  similarButton.addEventListener("click", () => findSimilar(result));
  card.addEventListener("mouseenter", () => { hoveredResult = result; });
  card.addEventListener("mouseleave", () => { if (hoveredResult === result) hoveredResult = null; });
  if (!manualMode) content.append(submitButton);
  content.append(similarButton);
  if (result.temporal) {
    const temporal = result.temporal;
    const details = document.createElement("details");
    details.className = "temporal-evidence";
    const summary = document.createElement("summary");
    const count = temporal.events.filter((event) => event.status === "retrieved").length;
    summary.textContent = `Chuỗi ${count}/${temporal.events.length} bước · ${formatNumber(temporal.start_seconds, 1)}–${formatNumber(temporal.end_seconds, 1)}s`;
    const note = document.createElement("p");
    note.textContent = "Gợi ý từ độ tương đồng, chưa xác minh hành động. Khoảng thời gian là span bằng chứng, không phải biên clip.";
    const events = document.createElement("ol");
    temporal.events.forEach((event) => {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${event.action} — ${event.status === "retrieved" ? "có gợi ý" : "chưa đủ bằng chứng"}`;
      row.append(label);
      if (event.image_url) {
        const link = document.createElement("a");
        link.href = event.image_url;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = `Xem frame ${event.frame_id} tại ${formatNumber(event.pts_time, 1)}s (tab mới)`;
        row.append(link);
      }
      if (event.video_id && Number.isInteger(event.keyframe_n) && Number.isInteger(event.frame_id)) {
        const eventSubmit = document.createElement("button");
        eventSubmit.type = "button";
        eventSubmit.className = "dres-select";
        eventSubmit.textContent = "Chọn frame bước này để nộp";
        eventSubmit.addEventListener("click", () => prepareDresSubmission({ ...event, image_number: event.keyframe_n }));
        row.append(eventSubmit);
      }
      events.append(row);
    });
    details.append(summary, note, events);
    content.append(details);
  }
  card.append(imageButton, content);
  return card;
}

function renderResults(payload) {
  releaseResultImages();
  hoveredResult = null;
  lastPayload = payload;
  document.querySelector("#similar-back").hidden = payload.input_type !== "similar" || !similarBackPayload;
  queryVariants.replaceChildren();
  const plannerStatus = document.querySelector("#query-planner-status");
  const planner = payload.query_planner;
  plannerStatus.hidden = !planner;
  if (planner) {
    const reasons = { dns: "lỗi DNS", timeout: "quá thời gian chờ", invalid_output: "JSON không hợp lệ", unavailable: "không khả dụng" };
    plannerStatus.textContent = planner.fallback_used
      ? `Planner: Groq ${reasons[planner.reason] || planner.reason || "gặp lỗi"} → đã dùng OpenCode fallback · ${planner.used.model}.`
      : `Planner: ${planner.used.provider} · ${planner.used.model} · không dùng fallback.`;
  }

  (payload.query_variants || []).forEach((variant) => {
    const item = document.createElement("li");
    item.textContent = variant;
    queryVariants.append(item);
  });
  document.querySelector("#query-details").hidden = !payload.query_variants?.length;
  const context = document.querySelector("#query-context");
  context.hidden = payload.input_type !== "video";
  if (payload.input_type === "video") {
    const fusion = payload.video_text_guidance?.text_fusion;
    const weights = fusion?.enabled
      ? `text ${Math.round((fusion.text_weight ?? 0.7) * 100)}% + video ${Math.round((fusion.video_weight ?? 0.3) * 100)}%`
      : "video thuần";
    const texts = [
      payload.hint_text ? `Đề text DRES (đã phát): ${payload.hint_text}` : "",
      payload.supplemental_description ? `Mô tả bổ sung: ${payload.supplemental_description}` : "",
    ].filter(Boolean);
    const guidance = texts.length
      ? `${texts.join(" · ")}. Text chạy pipeline đầy đủ (LLM cấu trúc, SigLIP, BM25, semantic caption); hợp nhất kết quả theo ${weights} bằng weighted RRF. LLM chỉ so ảnh ứng viên với query text, không xác minh chuỗi sự kiện.`
      : "Không có text kèm theo; query chỉ dùng hình ảnh video.";
    context.textContent = `Video ${formatNumber(payload.video_query?.duration_seconds, 2)} giây · ${payload.video_query?.sampled_frames ?? "—"} frame mẫu · SigLIP image similarity. ${guidance} Chưa xác minh chuyển động hay thứ tự sự kiện.`;
  }

  payload.results.forEach((result) => resultsGrid.append(createResultCard(result)));
  const temporalApplied = payload.temporal?.status === "applied" && payload.temporal?.presentation !== "auxiliary";
  const temporalGrid = document.querySelector("#temporal-results-grid");
  temporalGrid.replaceChildren();
  (payload.temporal_results || []).forEach(result => temporalGrid.append(createResultCard(result)));
  document.querySelector("#temporal-results-section").hidden = !payload.temporal_results?.length;
  const resultLimit = payload.result_limit || payload.results.length;
  document.querySelector("#results-title").textContent = temporalApplied ? "Chuỗi sự kiện" : (payload.input_type === "video" ? `Top ${resultLimit} frames theo đề video` : `Top ${resultLimit} frames`);
  const scoredCount = payload.results.filter((item) => item.vlm_status === "reranked" && Number.isFinite(item.vlm_score)).length;
  const vlmOff = payload.vlm_enabled === false;
  document.querySelector("#llm-scoring-note").textContent = temporalApplied
    ? "Chuỗi gợi ý chưa được LLM xác minh; chỉ hiển thị những bằng chứng tìm được, không thêm frame giả để bù số lượng."
    : vlmOff
      ? `${resultLimit} frame để chọn nộp · VLM đang tắt; điểm xếp hạng chỉ từ truy hồi, không chấm ảnh bằng Qwen.`
      : payload.input_type === "video" ? (payload.combined_text_query
        ? `Text → pipeline text đầy đủ; video → SigLIP → fusion kết quả ${Math.round((payload.video_text_guidance?.text_fusion?.text_weight ?? 0.7) * 100)}% text + ${Math.round((payload.video_text_guidance?.text_fusion?.video_weight ?? 0.3) * 100)}% video → ${resultLimit} frame → LLM xét 20 ứng viên đầu · ${scoredCount} frame có điểm LLM hợp lệ.${payload.video_text_guidance?.status === "disabled" ? " VLM chấm ảnh đang tắt; text vẫn được xử lý và dùng khi truy hồi." : ""}`
        : `${resultLimit} frame từ SigLIP; không có text kèm theo nên không dùng LLM chấm lại.`)
        : `${resultLimit} frame để chọn nộp · LLM chỉ xét 20 ứng viên đầu trước khi xếp hạng lại · ${scoredCount} frame có điểm LLM hợp lệ. Các frame còn lại giữ điểm truy hồi.`;
  resultCount.textContent = `${payload.results.length} ${temporalApplied ? "đoạn gợi ý" : "kết quả"}${payload.temporal?.status === "fallback" ? " · chưa ghép được chuỗi, hiển thị frame" : ""}`;
  elapsedTime.textContent = `${payload.elapsed_seconds.toFixed(2)} giây`;
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function errorDetail(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg || String(item)).join("; ");
  }
  return fallback;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    setSystemState(status.state, status.vlm_enabled);
  } catch {
    setSystemState("failed");
  }
}

manualQuery.addEventListener("keydown", (event) => {
  // Enter searches, Shift+Enter keeps a newline; IME composition is left alone.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && !searchButton.disabled) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (manualMode || dresManualOverride) {
    await searchManualQuery();
    return;
  }
  if (searchBusy || !dresConnected || !dresCanSearch) return;
  const taskId = dresTaskId;
  const description = inputMode === "video" ? videoDescription.value.trim() : "";
  if (description.length > 2000) {
    errorMessage.textContent = "Text bổ sung không được quá 2000 ký tự.";
    errorPanel.hidden = false;
    videoDescription.focus();
    return;
  }

  errorPanel.hidden = true;
  loadingTitle.textContent = "Đang tìm theo đề chính thức từ DRES";
  document.querySelector("#loading-description").textContent = inputMode === "video"
    ? "Video được tìm bằng SigLIP. Nếu có text DRES hoặc text bổ sung, text chạy pipeline đầy đủ rồi fusion kết quả 70% text + 30% video; VLM chỉ xét 20 ứng viên đầu. Không gửi video đến Groq."
    : "Giữ trang này mở. SigLIP, caption retrieval và VLM đang xử lý truy vấn.";
  setSystemState("loading");
  setBusy(true);

  try {
    if (!await refreshDresConnection(false, true) || taskId !== dresTaskId || !dresCanSearch) {
      throw new Error("Chưa đồng bộ được đề hoặc DRES đã đổi task. Kiểm tra đề đang hiển thị rồi bấm Tìm lại; không cần reset.");
    }
    const payload = await dresApi("search", { task_id: taskId, ...(description ? { video_description: description } : {}) });
    if (!dresConnected || taskId !== dresTaskId || !dresCanSearch || payload.dres_task?.task_id !== taskId) throw new Error("DRES đã đổi/dừng đề. Kết quả cũ đã được bỏ.");
    if (payload.dres_content && payload.dres_run) await displayDresTask(payload.dres_content, payload.dres_run);
    resultsDresTaskId = taskId;
    renderResults(payload);
    setSystemState("ready", payload.vlm_enabled);
  } catch (error) {
    if (error.status === 409) await refreshDresConnection(false, true);
    errorMessage.textContent = error.message || "Không thể hoàn thành tìm kiếm.";
    errorPanel.hidden = false;
    errorPanel.focus({ preventScroll: true });
    errorPanel.scrollIntoView({ behavior: "smooth", block: "center" });
    await refreshStatus();
  } finally {
    setBusy(false);
  }
});

async function searchManualQuery() {
  if (searchBusy) return;
  const query = manualQuery.value.trim();
  if (!query || query.length > 8000) {
    errorMessage.textContent = "Nhập mô tả cảnh cần tìm, tối đa 8000 ký tự.";
    errorPanel.hidden = false;
    manualQuery.focus();
    return;
  }
  errorPanel.hidden = true;
  loadingTitle.textContent = "Đang tìm theo query của bạn";
  document.querySelector("#loading-description").textContent = dresManualOverride
    ? "Pipeline tìm kiếm đang xử lý query tự nhập. Không đọc hoặc thay đổi đề DRES; không tự nộp bài."
    : "Pipeline tìm kiếm đang xử lý query. Không kết nối hoặc gửi bài đến DRES.";
  setSystemState("loading");
  setBusy(true);
  try {
    const response = await fetch("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorDetail(payload, "Không thể hoàn thành tìm kiếm."));
    resultsDresTaskId = manualMode ? null : dresTaskId;
    renderResults(payload);
    document.querySelector("#llm-scoring-note").textContent += manualMode
      ? " Chế độ tự nhập query · không nộp DRES."
      : " Query tự nhập · không dùng đề DRES; chỉ nộp frame nếu bạn đã tự kiểm tra.";
    setSystemState("ready", payload.vlm_enabled);
  } catch (error) {
    errorMessage.textContent = error.message || "Không thể hoàn thành tìm kiếm.";
    errorPanel.hidden = false;
    errorPanel.focus({ preventScroll: true });
    await refreshStatus();
  } finally {
    setBusy(false);
  }
}

// Image-to-image search from one result: the frame's stored visual vector is the query.
let hoveredResult = null;
let lastPayload = null;
let similarBackPayload = null;

async function findSimilar(result) {
  if (searchBusy || !Number.isInteger(result.frame_id)) return;
  if (lastPayload?.input_type !== "similar") similarBackPayload = lastPayload;
  errorPanel.hidden = true;
  loadingTitle.textContent = `Đang tìm frame tương tự ${result.video_id} · IMG ${result.image_number ?? "—"}`;
  setSystemState("loading");
  setBusy(true);
  try {
    const response = await fetch("/api/similar", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: result.video_id, frame_id: result.frame_id }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorDetail(payload, "Không tìm được frame tương tự."));
    renderResults(payload);
    document.querySelector("#results-title").textContent = `Tương tự ${result.video_id} · IMG ${result.image_number ?? "—"}`;
    setSystemState("ready", false);
  } catch (error) {
    errorMessage.textContent = error.message || "Không tìm được frame tương tự.";
    errorPanel.hidden = false;
    errorPanel.focus({ preventScroll: true });
  } finally {
    setBusy(false);
  }
}

document.querySelector("#similar-back").addEventListener("click", () => {
  if (searchBusy || !similarBackPayload) return;
  const previous = similarBackPayload;
  similarBackPayload = null;
  renderResults(previous);
});

window.addEventListener("keydown", (event) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target?.tagName || "") || event.target?.isContentEditable;
  if (typing || event.ctrlKey || event.metaKey || event.altKey || (event.key !== "s" && event.key !== "S")) return;
  if (previewDialog.open || dresDialog.open || !hoveredResult) return;
  event.preventDefault();
  findSimilar(hoveredResult);
});

closePreview.addEventListener("click", () => previewDialog.close());
window.addEventListener("pagehide", () => {
  releaseResultImages();
  clearDresTask();
});
previewDialog.addEventListener("click", (event) => {
  if (event.target === previewDialog) previewDialog.close();
});

const dresForm = document.querySelector("#dres-connect-form");
const dresStatus = document.querySelector("#dres-status");
const dresAuthMode = document.querySelector("#dres-auth-mode");
const dresSession = document.querySelector("#dres-session");
const dresUsername = document.querySelector("#dres-username");
const dresPassword = document.querySelector("#dres-password");
const dresRunId = document.querySelector("#dres-run-id");
const dresDialog = document.querySelector("#dres-confirm-dialog");
const dresConfirmButton = document.querySelector("#dres-confirm-submit");
const dresCancelButton = document.querySelector("#dres-cancel-submit");
const dresSubmitStatus = document.querySelector("#dres-submit-status");
let dresConnected = false;
let dresBusy = false;
let dresSending = false;
let dresPreview = null;
let dresAnswerMode = "frame";
let dresTaskId = null;
let dresTaskRequests = new AbortController();
let dresTaskUrls = new Set();
let dresContentKey = null;
let dresMediaKey = null;
let dresRefreshInFlight = false;
let dresConnectionVersion = 0;

function clearDresTask() {
  dresTaskRequests.abort();
  dresTaskRequests = new AbortController();
  dresTaskUrls.forEach((url) => URL.revokeObjectURL(url));
  dresTaskUrls.clear();
  document.querySelectorAll?.("#dres-task-content video").forEach((video) => {
    video.pause(); video.removeAttribute("src"); video.load();
  });
  document.querySelector("#dres-task-texts").replaceChildren();
  document.querySelector("#dres-task-media").replaceChildren();
  dresMediaKey = null;
  dresCanSearch = false;
  renderSearchMode();
  updateVideoDescriptionControls();
  dresContentKey = null;
}

function forgetDresConnection() {
  dresConnectionVersion++;
  dresConnected = false;
  dresTaskId = null;
  dresTaskName = "";
  dresPreview = null;
  releaseResultImages();
  resultsSection.hidden = true;
  resultsDresTaskId = null;
  setDresManualOverride(false);
  clearDresTask();
  resetVideoDescription();
  document.querySelector("#dres-qa-answer").value = "";
  if (dresDialog.open && !dresSending) dresDialog.close();
  setDresControls();
}

async function displayDresTask(content, run) {
  const elements = content?.elements || [];
  const key = JSON.stringify([run.task_id, content?.status, elements]);
  if (dresContentKey === key) return;
  if (resultsDresTaskId) {
    releaseResultImages();
    resultsSection.hidden = true;
    resultsDresTaskId = null;
  }
  const mediaElements = elements.filter(element => ["VIDEO", "IMAGE"].includes(element.type));
  const mediaKey = JSON.stringify([run.task_id, mediaElements]);
  const reuseMedia = dresMediaKey === mediaKey;
  if (!reuseMedia) clearDresTask();
  dresContentKey = key;
  dresMediaKey = mediaKey;
  const textContainer = document.querySelector("#dres-task-texts");
  textContainer.replaceChildren();
  const container = document.querySelector("#dres-task-media");
  const signal = dresTaskRequests.signal;
  dresCanSearch = run.can_submit && elements.some((element) => (element.type === "TEXT" && element.text?.trim()) || element.type === "VIDEO");
  dresTaskName = run.task?.name || run.task_id || "";
  const hasText = elements.some(element => element.type === "TEXT" && element.text?.trim());
  inputMode = elements.some(element => element.type === "VIDEO")
    && (/visual/i.test(run.task?.taskType || "") || !hasText) ? "video" : "text";
  if (inputMode !== "video") videoDescription.value = "";
  renderSearchMode();
  updateVideoDescriptionControls();
  document.querySelector("#dres-task-message").textContent = content?.status === "unavailable"
    ? (content.message || "DRES chưa cho phép xem đề; hãy mở trang DRES.")
    : elements.length ? "Đang hiển thị toàn bộ nội dung DRES đã phát. Bấm Tìm sẽ lấy cả câu mới, không cần reset; Làm mới đề chỉ cập nhật phần hiển thị." : "Đang chờ DRES phát đề…";
  await Promise.all(elements.map(async (element) => {
    if (element.type === "TEXT") {
      const text = document.createElement("p");
      text.className = "dres-task-text";
      text.textContent = element.text;
      textContainer.append(text);
    } else if (!reuseMedia && ["VIDEO", "IMAGE"].includes(element.type)
        && /^\/api\/dres\/task\/media\/[^/]+\/\d+$/.test(element.media_url || "")) {
      const media = document.createElement(element.type === "VIDEO" ? "video" : "img");
      media.controls = true;
      media.playsInline = true;
      media.preload = "metadata";
      media.setAttribute("aria-label", "Video/ảnh đề hiện tại từ DRES");
      if (element.type === "IMAGE") media.alt = "Ảnh đề hiện tại từ DRES";
      container.append(media);
      media.addEventListener("error", () => {
        if (!signal.aborted) document.querySelector("#dres-task-message").textContent = "Không phát được media đề. Hãy mở trang DRES hoặc làm mới đề.";
      });
      try {
        const response = await fetch(element.media_url, {
          headers: { "X-KIS-DRES": "1" }, credentials: "same-origin", cache: "no-store", signal,
        });
        if (!response.ok) {
          if (response.status === 401) {
            forgetDresConnection();
            setDresStatus("Phiên DRES hết hạn; vui lòng đăng nhập lại.", true);
          }
          throw new Error("Không tải được media đề; hãy làm mới đề.");
        }
        const blob = await response.blob();
        if (signal.aborted) return;
        const url = URL.createObjectURL(blob);
        dresTaskUrls.add(url);
        media.src = url;
      } catch (error) {
        if (!signal.aborted) {
          dresContentKey = null; // Retry a failed download on the next manual refresh, not a successful playing video.
          document.querySelector("#dres-task-message").textContent = error.message;
        }
      }
    }
  }));
}

async function dresApi(path, body) {
  if (manualMode) throw new Error("DRES đã tắt trong chế độ tự nhập query.");
  const response = await fetch(`/api/dres/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "X-KIS-DRES": "1", ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    credentials: "same-origin", cache: "no-store",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) forgetDresConnection();
    const error = new Error(errorDetail(payload, `DRES HTTP ${response.status}`));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setDresStatus(message, error = false) {
  const className = error ? "dres-status error" : "dres-status";
  if (dresStatus.textContent !== message) dresStatus.textContent = message;
  if (dresStatus.className !== className) dresStatus.className = className;
}

function setDresControls(busy = false) {
  dresBusy = busy;
  const locked = busy || dresConnected;
  document.querySelector("#dres-login-panel").hidden = dresConnected;
  document.querySelector("#dres-task-panel").hidden = !dresConnected;
  dresRunId.disabled = locked;
  dresAuthMode.disabled = locked;
  dresSession.disabled = locked || dresAuthMode.value === "login";
  dresUsername.disabled = locked || dresAuthMode.value !== "login";
  dresPassword.disabled = locked || dresAuthMode.value !== "login";
  document.querySelector("#dres-qa-form").hidden = !dresConnected || dresAnswerMode !== "text";
  document.querySelector("#dres-qa-answer").disabled = busy || !dresConnected || dresAnswerMode !== "text";
  document.querySelector("#dres-qa-preview").disabled = busy;
  document.querySelector("#dres-connect").disabled = locked;
  document.querySelector("#dres-refresh").hidden = !dresConnected;
  document.querySelector("#dres-disconnect").hidden = !dresConnected;
  document.querySelector("#dres-refresh").disabled = busy;
  document.querySelector("#dres-disconnect").disabled = busy;
  dresManualToggle.disabled = busy || !dresConnected;
  dresForm.setAttribute("aria-busy", String(busy));
}

function displayDresConnection(payload) {
  dresConnected = payload.connected === true;
  const run = payload.run;
  if (run) {
    dresAnswerMode = run.answer_mode || (/question|answer|qakis|\bqa\b/i.test(run.task?.taskType || "") ? "text" : "frame");
    if (dresTaskId !== run.task_id) {
      setDresManualOverride(false);
      clearDresTask();
      resetVideoDescription();
      document.querySelector("#dres-qa-answer").value = "";
      dresPreview = null;
      if (dresDialog.open && !dresSending) dresDialog.close();
    }
    if (resultsDresTaskId && (resultsDresTaskId !== run.task_id || !run.can_submit)) {
      releaseResultImages();
      resultsSection.hidden = true;
      resultsDresTaskId = null;
    }
    dresTaskId = run.task_id;
    document.querySelector("#dres-task-meta").textContent = `${run.task?.name || "Chưa có task"} · ${run.task?.taskType || ""} · ${run.task_status || ""}${Number.isFinite(run.time_left) ? ` · Còn ${run.time_left}s` : ""}`;
    if (!run.can_submit) {
      clearDresTask();
      resetVideoDescription();
      dresPreview = null;
      if (dresDialog.open && !dresSending) dresDialog.close();
    }
    dresRunId.value = run.evaluation_id;
    setDresStatus(`${payload.user?.username || "Đã xác thực"} · ${run.name || run.evaluation_id} · ${run.evaluation_status} / ${run.task_status}${run.task?.name ? ` · Task: ${run.task.name}` : ""}${run.can_submit ? (dresAnswerMode === "text" ? " · Task QA: nhập đáp án chữ bên dưới, không nộp frame." : " · Có thể chọn frame để nộp.") : " · Chưa có task đang nhận bài."}`);
  }
  renderSearchMode();
  setDresControls();
}

dresAuthMode.addEventListener("change", () => {
  const login = dresAuthMode.value === "login";
  document.querySelector("#dres-session-field").hidden = login;
  document.querySelector("#dres-login-fields").hidden = !login;
  dresSession.required = !login;
  dresUsername.required = login;
  dresPassword.required = login;
  dresSession.value = "";
  dresPassword.value = "";
  setDresControls();
});

dresManualToggle.addEventListener("change", () => {
  if (searchBusy) {
    dresManualToggle.checked = dresManualOverride;
    return;
  }
  if (!dresConnected) {
    dresManualToggle.checked = false;
    return;
  }
  setDresManualOverride(dresManualToggle.checked, true);
});

dresForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (dresBusy || dresConnected) return;
  setDresControls(true);
  setDresStatus("Đang xác thực và đọc trạng thái DRES; không gửi bài…");
  try {
    const auth = dresAuthMode.value === "login"
      ? { username: dresUsername.value.trim(), password: dresPassword.value }
      : { session_id: dresSession.value.trim() };
    displayDresConnection(await dresApi("connect", { evaluation_id: dresRunId.value.trim(), ...auth }));
    await refreshDresConnection();
  } catch (error) {
    setDresStatus(error.message, true);
  } finally {
    dresSession.value = "";
    dresPassword.value = "";
    setDresControls();
  }
});

async function refreshDresConnection(quiet = false, force = false) {
  if (dresRefreshInFlight) return false;
  dresRefreshInFlight = true;
  const version = dresConnectionVersion;
  setDresControls(true);
  try {
    const payload = await dresApi(force ? "task?refresh=1" : "task");
    if (version !== dresConnectionVersion) return false;
    displayDresConnection(payload);
    await displayDresTask(payload.content, payload.run);
    return true;
  } catch (error) {
    clearDresTask();
    dresPreview = null;
    if (dresDialog.open && !dresSending) dresDialog.close();
    document.querySelector("#dres-task-message").textContent = "Chưa đồng bộ được đề DRES. Bấm Làm mới đề để thử lại.";
    if (!(quiet && error.status === 401)) setDresStatus(error.message, true);
    return false;
  } finally {
    dresRefreshInFlight = false;
    setDresControls();
  }
}

document.querySelector("#dres-refresh").addEventListener("click", async () => {
  if (dresBusy) return;
  setDresControls(true);
  await refreshDresConnection(false, true);
});
document.querySelector("#dres-disconnect").addEventListener("click", async () => {
  if (dresBusy) return;
  setDresControls(true);
  try {
    await dresApi("disconnect", {});
    forgetDresConnection();
    setDresStatus("Đã xóa phiên local khỏi RAM; không đăng xuất phiên DRES của các ứng dụng khác.");
  } catch (error) {
    setDresStatus(error.message, true);
  } finally {
    setDresControls();
  }
});

async function prepareDresSubmission(result) {
  if (dresBusy || dresSending) return;
  if (!dresConnected) {
    setDresStatus("Kết nối DRES trước khi chọn frame để nộp.", true);
    dresForm.scrollIntoView({ behavior: "smooth", block: "center" });
    dresSession.focus();
    return;
  }
  if (dresAnswerMode === "text") {
    setDresStatus("Task hiện tại yêu cầu đáp án chữ (QA). Nhập đáp án vào ô QA hoặc làm mới task nếu ban tổ chức đã chuyển câu.", true);
    document.querySelector("#dres-qa-answer").focus();
    return;
  }
  setDresControls(true);
  try {
    const base = { video_id: result.video_id, frame_id: result.frame_id, image_number: result.image_number };
    const preview = await dresApi("preview", { ...base, task_id: dresTaskId });
    dresAdjust = { base, current: { ...base }, offset: 0, keyframes: null };
    showDresPreview(preview);
  } catch (error) {
    dresPreview = null;
    setDresStatus(error.message, true);
  } finally {
    setDresControls();
  }
}

function showDresPreview(preview) {
  dresPreview = preview;
  const selected = preview.selection;
  const label = preview.submission_type === "text" ? "Đáp án chữ QA (nhập thủ công)"
    : `${selected.video_id} · IMG ${selected.image_number} · FRAME ${selected.frame_id} · ${selected.timestamp_seconds.toFixed(3)}s (${selected.timestamp_ms} ms)`;
  document.querySelector("#dres-selection").textContent = `${label} · Task: ${preview.run.task?.name || preview.run.task_id} · Run ${preview.evaluation_id}`;
  document.querySelector("#dres-payload").textContent = JSON.stringify(preview.payload, null, 2);
  dresSubmitStatus.textContent = "Xác nhận có hiệu lực 60 giây. Bài chưa được gửi.";
  dresConfirmButton.disabled = false;
  dresCancelButton.textContent = "Hủy";
  renderFrameAdjust(preview);
  if (!dresDialog.open) {
    dresDialog.showModal();
    dresCancelButton.focus();
  }
}

// Frame adjustment inside the confirm dialog: every step asks the server for a
// fresh mapping-validated preview, so the JSON shown is always what gets sent.
let dresAdjust = null;
const frameAdjustButtons = ["#dres-kf-prev", "#dres-kf-next", "#dres-frame-minus", "#dres-frame-plus", "#dres-frame-reset"]
  .map((id) => document.querySelector(id));

function renderFrameAdjust(preview) {
  const panel = document.querySelector("#dres-frame-adjust");
  panel.hidden = preview.submission_type === "text" || !dresAdjust;
  if (panel.hidden) return;
  const { current, offset } = dresAdjust;
  const selected = preview.selection;
  document.querySelector("#dres-frame-image").src = `/api/images/${current.video_id}/${current.image_number}`;
  document.querySelector("#dres-frame-label").textContent = `IMG ${current.image_number} · FRAME ${selected.submitted_frame_id ?? current.frame_id}`
    + (offset ? ` (${offset > 0 ? "+" : ""}${offset} frame so với keyframe)` : "") + ` · ${selected.timestamp_ms} ms`;
}

async function adjustDresFrame(change) {
  if (!dresAdjust || dresBusy || dresSending) return;
  frameAdjustButtons.forEach((button) => { button.disabled = true; });
  dresConfirmButton.disabled = true;
  const previous = { current: { ...dresAdjust.current }, offset: dresAdjust.offset };
  try {
    if (change === "reset") {
      dresAdjust.current = { ...dresAdjust.base };
      dresAdjust.offset = 0;
    } else if (change === "frame-1" || change === "frame+1") {
      dresAdjust.offset += change === "frame+1" ? 1 : -1;
    } else {
      if (!dresAdjust.keyframes) {
        const response = await fetch(`/api/keyframes/${dresAdjust.base.video_id}`, { cache: "no-store" });
        if (!response.ok) throw new Error("Không đọc được danh sách keyframe của video.");
        dresAdjust.keyframes = (await response.json()).keyframes;
      }
      const list = dresAdjust.keyframes;
      const index = list.findIndex((row) => row.image_number === dresAdjust.current.image_number) + (change === "kf+1" ? 1 : -1);
      if (index < 0 || index >= list.length) throw new Error(change === "kf+1" ? "Đã là keyframe cuối." : "Đã là keyframe đầu.");
      dresAdjust.current = { video_id: dresAdjust.base.video_id, ...list[index] };
      dresAdjust.offset = 0;
    }
    const preview = await dresApi("preview", { ...dresAdjust.current, frame_offset: dresAdjust.offset, task_id: dresTaskId });
    showDresPreview(preview);
  } catch (error) {
    Object.assign(dresAdjust, previous);
    dresSubmitStatus.textContent = `${error.message.replace(/\.?$/, ".")} Giữ lựa chọn trước; hãy kiểm tra JSON.`;
    dresConfirmButton.disabled = !dresPreview;
  } finally {
    frameAdjustButtons.forEach((button) => { button.disabled = false; });
  }
}

[["#dres-kf-prev", "kf-1"], ["#dres-kf-next", "kf+1"], ["#dres-frame-minus", "frame-1"],
  ["#dres-frame-plus", "frame+1"], ["#dres-frame-reset", "reset"]]
  .forEach(([id, change]) => document.querySelector(id).addEventListener("click", () => adjustDresFrame(change)));

document.querySelector("#dres-qa-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (dresBusy || dresSending || !dresConnected) return;
  const text = document.querySelector("#dres-qa-answer").value.trim();
  if (!text) return;
  setDresControls(true);
  try {
    showDresPreview(await dresApi("preview/text", { answer_text: text, task_id: dresTaskId }));
  } catch (error) {
    dresPreview = null;
    setDresStatus(error.message, true);
  } finally {
    setDresControls();
  }
});

dresConfirmButton.addEventListener("click", async () => {
  if (dresSending || !dresPreview) return;
  dresSending = true;
  const preview = dresPreview;
  dresPreview = null;
  dresConfirmButton.disabled = true;
  dresCancelButton.disabled = true;
  setDresControls(true);
  dresSubmitStatus.textContent = "Đang gửi một lần đến DRES. Không đóng trang hoặc bấm lại.";
  try {
    const result = await dresApi("submit", { preview_id: preview.preview_id, confirmed: true });
    const message = `DRES HTTP ${result.http_status} · ${result.accepted ? "Đã nhận bài" : "Không chấp nhận"} · ${result.pending ? "Đang chờ chấm" : (result.verdict || "Chưa có verdict")}. Được nhận không đồng nghĩa đáp án đúng.`;
    dresSubmitStatus.textContent = message;
    setDresStatus(message, !result.accepted || result.verdict === "WRONG");
    const item = document.createElement("li");
    item.textContent = `${preview.submission_type === "text" ? "Đáp án QA" : `${preview.selection.video_id} @ ${preview.selection.timestamp_ms} ms`} — ${message}`;
    document.querySelector("#dres-history").prepend(item);
  } catch (error) {
    // 400/412/422 = DRES explicitly refused (nothing recorded); only uncertain failures warn against resubmitting.
    dresSubmitStatus.textContent = [400, 412, 422].includes(error.status) ? error.message
      : `${error.message} Không tự gửi lại. Hãy kiểm tra trang DRES.`;
    setDresStatus(dresSubmitStatus.textContent, true);
  } finally {
    dresSending = false;
    dresCancelButton.disabled = false;
    dresCancelButton.textContent = "Đóng";
    setDresControls();
  }
});
dresCancelButton.addEventListener("click", () => {
  if (dresSending) return;
  dresPreview = null;
  dresDialog.close();
});
dresDialog.addEventListener("cancel", (event) => {
  if (dresSending) event.preventDefault();
  else dresPreview = null;
});

async function initializeDres() {
  try {
    const settings = await dresApi("config");
    dresRunId.value = settings.evaluation_id || "";
    if (settings.base_url) {
      const viewer = document.querySelector("#dres-viewer");
      viewer.href = `${settings.base_url}/evaluation/viewer/${encodeURIComponent(settings.evaluation_id || "")}`;
      viewer.hidden = false;
    }
    if (!settings.configured) {
      setDresStatus("Chưa cấu hình KIS_DRES_BASE_URL trên server.");
      return;
    }
    await refreshDresConnection(true);
  } catch {
    setDresStatus("Chưa nạp được cấu hình DRES. Nếu vừa cập nhật code, hãy khởi động lại server.", true);
  }
}

refreshStatus();
if (manualMode) {
  document.querySelector("#page-title").textContent = "Tự nhập query.";
  document.querySelector("#dres-login-panel").hidden = true;
  document.querySelector("#dres-task-panel").hidden = true;
  document.querySelector("#dres-history").hidden = true;
  dresManualToggle.disabled = true;
  renderSearchMode();
  setDresStatus("Chế độ tự nhập query · không kết nối DRES, không gửi đáp án.");
} else {
  initializeDres();
}
