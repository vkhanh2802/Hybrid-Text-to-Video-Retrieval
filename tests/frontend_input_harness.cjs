const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

class Element {
  constructor() {
    this.children = []; this.events = {}; this.attrs = {};
    this.value = ""; this.open = false; this.hidden = false; this.files = [];
    this.disabled = false; this.validationMessage = ""; this.textContent = "";
  }
  addEventListener(name, handler) { (this.events[name] ||= []).push(handler); }
  async trigger(name, extra = {}) {
    const event = { preventDefault() { this.defaultPrevented = true; }, ...extra };
    await Promise.all((this.events[name] || []).map(handler => handler(event)));
  }
  setAttribute(name, value) { this.attrs[name] = value; }
  removeAttribute(name) { delete this[name]; }
  append(...items) { this.children.push(...items); }
  prepend(item) { this.children.unshift(item); }
  replaceChildren(...items) { this.children = items; }
  remove() {}
  scrollIntoView() {}
  focus() { this.focused = true; }
  showModal() { this.open = true; }
  close() { this.open = false; }
  pause() {}
  load() {}
  setCustomValidity(value) { this.validationMessage = value; }
  checkValidity() { return !this.validationMessage && (!this.required || this.files.length > 0); }
  reportValidity() { this.validityReported = true; }
}

const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, new Element());
  return nodes.get(id);
}
node("#dres-auth-mode").value = "session";
const events = {}, requests = [], released = [];
let urls = 0, connected = process.argv[3] === "restored", submitResult = { http_status: 202, accepted: true, pending: true, verdict: "INDETERMINATE" };
let imageObserverCallback, imageActive = 0, maxImageActive = 0;
const imageResolvers = [];
let submitFails = false, previewFails = false, searchConflict = false;
let taskContent = { status: "available", elements: [{ type: "TEXT", text: "Find a red boat", offset: 0, index: 0 }] };
const run = { evaluation_id: "11111111-2222-4333-8444-555555555555", name: "Test run",
  evaluation_status: "ACTIVE", task_status: "RUNNING", task_id: "task-test", can_submit: true, task: { name: "KIS test" } };
const selection = { video_id: "L01_V001", frame_id: 150, image_number: 2, timestamp_seconds: 6, timestamp_ms: 6000 };
const reply = (body, ok = true, status = 200) => ({ ok, status, json: async () => body });
const context = vm.createContext({
  console, AbortController, setTimeout: () => 1, clearTimeout: () => {},
  ...(process.argv[3] === "images" ? { IntersectionObserver: class {
    constructor(callback) { imageObserverCallback = callback; }
    observe() {}
    unobserve() {}
    disconnect() {}
  }} : {}),
  localStorage: new Proxy({}, { get() { throw Error("Must not persist credentials"); } }),
  sessionStorage: new Proxy({}, { get() { throw Error("Must not persist credentials"); } }),
  document: { querySelector: node, createElement: () => new Element() },
  window: { location: { search: process.argv[3] === "manual" ? "?mode=manual" : "" },
    addEventListener: (name, fn) => { events[name] = fn; } },
  URL: { createObjectURL: () => `blob:${++urls}`, revokeObjectURL: value => released.push(value) },
  fetch: async (url, options) => {
    requests.push({ url, options });
    if (url.startsWith("/api/images/")) {
      imageActive++; maxImageActive = Math.max(maxImageActive, imageActive);
      return new Promise((resolve) => imageResolvers.push(() => {
        imageActive--; resolve({ ok: true, blob: async () => ({}) });
      }));
    }
    if (url === "/api/status") return reply({ state: "ready", vlm_enabled: true });
    if (url === "/api/dres/config") return reply({ configured: true, base_url: "https://example.com/dres", evaluation_id: run.evaluation_id });
    if (url === "/api/dres/status") return connected ? reply({ connected: true, run, user: { username: "tester" } }) : reply({ detail: "Not connected" }, false, 401);
    if (url === "/api/dres/task" || url === "/api/dres/task?refresh=1") return connected ? reply({ connected: true, run, content: taskContent, user: { username: "tester" } }) : reply({ detail: "Session expired" }, false, 401);
    if (url.startsWith("/api/dres/task/media/")) return { ok: true, blob: async () => ({}) };
    if (url === "/api/dres/connect") { connected = true; return reply({ connected: true, run, user: { username: "tester" } }); }
    if (url === "/api/dres/disconnect") { connected = false; return reply({ connected: false }); }
    if (url === "/api/similar") return reply({ input_type: "similar", elapsed_seconds: 0.5, result_limit: 2, query_variants: [], temporal_results: [],
      results: [{ rank: 1, video_id: "L01_V001", frame_id: 150, image_number: 2 }, { rank: 2, video_id: "L02_V003", frame_id: 40, image_number: 1 }] });
    if (url.startsWith("/api/keyframes/")) return reply({ keyframes: [{ image_number: 1, frame_id: 0 },
      { image_number: 2, frame_id: 150 }, { image_number: 3, frame_id: 300 }] });
    if (url === "/api/dres/preview") return previewFails ? reply({ detail: "Task changed" }, false, 409) : reply({ preview_id: "one-use-preview-token", selection, run,
      evaluation_id: run.evaluation_id, payload: { answerSets: [{ taskId: run.task_id, answers: [{ mediaItemName: selection.video_id, start: 6000, end: 6000 }] }] } });
    if (url === "/api/dres/preview/text") {
      const text = JSON.parse(options.body).answer_text;
      return reply({ submission_type: "text", preview_id: "qa-one-use-preview", selection: { answer_text: text }, run,
        evaluation_id: run.evaluation_id, payload: { answerSets: [{ taskId: run.task_id, answers: [{ text }] }] } });
    }
    if (url === "/api/dres/submit") {
      if (submitFails) throw Error("Unknown network outcome");
      return reply(submitResult);
    }
    if (url === "/api/dres/search") {
      if (searchConflict) return reply({ detail: "Nội dung đề DRES đã thay đổi; hãy tìm lại" }, false, 409);
      const body = JSON.parse(options.body);
      const hintText = taskContent.elements.filter(e => e.type === "TEXT" && e.text?.trim())
        .map(e => e.text).join("\n");
      const combinedText = [hintText, body.video_description || ""].filter(Boolean).join("\n");
      return reply({
        input_type: taskContent.elements.some(e => e.type === "VIDEO") ? "video" : "text", elapsed_seconds: 1,
        query_variants: [], video_query: { duration_seconds: 5, sampled_frames: 6 },
        dres_task: { task_id: run.task_id, source: "dres_official" }, results: [],
        dres_content: taskContent, dres_run: run,
        hint_text: hintText, combined_text_query: combinedText,
        supplemental_description: body.video_description || "",
        video_text_guidance: { status: "disabled", text_fusion: { enabled: Boolean(combinedText),
          text_weight: combinedText ? 0.7 : 0, video_weight: combinedText ? 0.3 : 1 } },
      });
    }
    if (url === "/api/search/video") return reply({ input_type: "video", elapsed_seconds: 1, query_variants: [],
      video_query: { duration_seconds: 2, sampled_frames: 4 }, results: [] });
    if (url === "/api/search") return reply({ input_type: "text", elapsed_seconds: 1, query_variants: ["text"], results: [] });
    throw Error(`Unexpected request: ${url}`);
  },
});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const flush = () => new Promise(resolve => setImmediate(resolve));
const calls = path => requests.filter(r => r.url === path);

async function connect(method = "session") {
  node("#dres-auth-mode").value = method;
  await node("#dres-auth-mode").trigger("change");
  if (method === "session") node("#dres-session").value = "synthetic-session-token";
  else { node("#dres-username").value = "tester"; node("#dres-password").value = "synthetic-password"; }
  await node("#dres-connect-form").trigger("submit");
  assert.equal(node("#dres-session").value, "");
  assert.equal(node("#dres-password").value, "");
  assert.equal(calls("/api/dres/submit").length, 0);
  assert.equal(node("#dres-login-panel").hidden, true);
  assert.equal(node("#dres-task-panel").hidden, false);
}

async function officialScenario() {
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/dres/search").length, 0);
  await connect();
  assert.equal(node("#search-button").disabled, false);
  await node("#search-form").trigger("submit");
  assert.deepEqual(JSON.parse(calls("/api/dres/search")[0].options.body), { task_id: "task-test" });
  taskContent = { status: "available", elements: [{ type: "VIDEO", index: 0, media_url: "/api/dres/task/media/task-test/0" }] };
  await context.refreshDresConnection();
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/dres/search").length, 2);
  assert.equal(calls("/api/search/video").length, 0);
  assert.equal(calls("/api/search").length, 0);
  assert.match(node("#query-context").textContent, /6 frame/);
  run.task_id = "new-task";
  taskContent = { status: "waiting", elements: [] };
  await context.refreshDresConnection();
  assert.equal(node("#results-section").hidden, true);
  assert.equal(node("#search-button").disabled, true);
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/dres/search").length, 2);
  assert.equal(calls("/api/dres/submit").length, 0);
}

async function videoDescriptionScenario() {
  await connect();
  assert.equal(node("#dres-video-description-panel").hidden, true);
  assert.equal(node("#dres-video-description").disabled, true);
  taskContent = { status: "available", elements: [{ type: "VIDEO", index: 0, media_url: "/api/dres/task/media/task-test/0" }] };
  await context.refreshDresConnection();
  assert.equal(node("#dres-video-description-panel").hidden, false);
  assert.equal(node("#dres-video-description").disabled, false);
  node("#dres-video-description").value = "  Focus on the red shirt  ";
  await node("#dres-video-description").trigger("input");
  assert.match(node("#dres-video-description-count").textContent, /26 \/ 2000/);
  await context.refreshDresConnection();
  assert.equal(node("#dres-video-description").value, "  Focus on the red shirt  ");
  context.setBusy(true);
  assert.equal(node("#dres-video-description").disabled, true);
  context.setBusy(false);
  await node("#search-form").trigger("submit");
  assert.deepEqual(JSON.parse(calls("/api/dres/search")[0].options.body), {
    task_id: "task-test", video_description: "Focus on the red shirt",
  });
  assert.match(node("#query-context").textContent, /Focus on the red shirt/);
  assert.match(node("#query-context").textContent, /70%/);
  assert.match(node("#llm-scoring-note").textContent, /70% text/);
  assert.match(node("#llm-scoring-note").textContent, /VLM chấm ảnh đang tắt/);
  assert.match(node("#query-context").textContent, /BM25, semantic caption/);
  assert.match(node("#llm-scoring-note").textContent, /text vẫn được xử lý/);
  node("#dres-video-description").value = "x".repeat(2001);
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/dres/search").length, 1);
  assert.match(node("#error-message").textContent, /2000/);
  run.task_id = "new-video-task";
  await context.refreshDresConnection();
  assert.equal(node("#dres-video-description").value, "");
  node("#dres-video-description").value = "new description";
  await node("#dres-refresh").trigger("click");
  assert.equal(node("#dres-video-description").value, "new description");
  assert(requests.some(r => r.url === "/api/dres/task?refresh=1"));
  taskContent = { status: "available", elements: [{ type: "TEXT", index: 0, text: "official text" }] };
  await context.refreshDresConnection();
  assert.equal(node("#dres-video-description-panel").hidden, true);
  assert.equal(node("#dres-video-description").value, "");
  await node("#search-form").trigger("submit");
  assert.deepEqual(JSON.parse(calls("/api/dres/search")[1].options.body), { task_id: "new-video-task" });
  assert.equal(calls("/api/search/video").length, 0);
  assert.equal(calls("/api/dres/submit").length, 0);
}

async function dresScenario() {
  await connect();
  await context.prepareDresSubmission(selection);
  assert.equal(node("#dres-confirm-dialog").open, true);
  assert.match(node("#dres-payload").textContent, /6000/);
  assert.equal(calls("/api/dres/submit").length, 0);
  await node("#dres-cancel-submit").trigger("click");
  assert.equal(calls("/api/dres/submit").length, 0);
  await context.prepareDresSubmission(selection);
  await Promise.all([node("#dres-confirm-submit").trigger("click"), node("#dres-confirm-submit").trigger("click")]);
  assert.equal(calls("/api/dres/submit").length, 1);
  assert.deepEqual(JSON.parse(calls("/api/dres/submit")[0].options.body), { preview_id: "one-use-preview-token", confirmed: true });
  assert.match(node("#dres-submit-status").textContent, /Đang chờ chấm/);
  assert.equal(node("#dres-confirm-submit").disabled, true);
  await node("#dres-confirm-submit").trigger("click");
  assert.equal(calls("/api/dres/submit").length, 1);
  assert(requests.filter(r => r.url.startsWith("/api/dres/")).every(r => r.options.headers["X-KIS-DRES"] === "1"));
}

async function frameAdjustScenario() {
  await connect();
  await context.prepareDresSubmission(selection);
  assert.equal(node("#dres-frame-adjust").hidden, false);
  const body = i => JSON.parse(calls("/api/dres/preview")[i].options.body);
  await node("#dres-frame-minus").trigger("click");
  assert.deepEqual(body(1), { video_id: "L01_V001", frame_id: 150, image_number: 2, frame_offset: -1, task_id: run.task_id });
  await node("#dres-kf-prev").trigger("click");
  assert.deepEqual(body(2), { video_id: "L01_V001", image_number: 1, frame_id: 0, frame_offset: 0, task_id: run.task_id });
  await node("#dres-kf-prev").trigger("click");  // already first keyframe: no new preview
  assert.equal(calls("/api/dres/preview").length, 3);
  assert.match(node("#dres-submit-status").textContent, /keyframe đầu/);
  await node("#dres-frame-reset").trigger("click");
  assert.deepEqual(body(3), { video_id: "L01_V001", frame_id: 150, image_number: 2, frame_offset: 0, task_id: run.task_id });
  assert.equal(node("#dres-confirm-dialog").open, true);
  assert.equal(calls("/api/dres/submit").length, 0);
}

async function similarScenario() {
  const original = { input_type: "text", elapsed_seconds: 1, results: [{ rank: 1, video_id: "L01_V001", frame_id: 150, image_number: 2 }] };
  context.renderResults(original);
  const card = node("#results-grid").children[0];
  const similarButton = card.children[1].children.at(-1);
  await similarButton.trigger("click");
  assert.deepEqual(JSON.parse(calls("/api/similar")[0].options.body), { video_id: "L01_V001", frame_id: 150 });
  assert.equal(node("#results-grid").children.length, 2);
  assert.match(node("#results-title").textContent, /Tương tự L01_V001/);
  assert.equal(node("#similar-back").hidden, false);
  await card.trigger("mouseenter");  // hovered card + S key also triggers a similar search
  await events.keydown({ key: "s", target: { tagName: "BODY" }, preventDefault() {} });
  await flush();
  assert.equal(calls("/api/similar").length, 2);
  await events.keydown({ key: "s", target: { tagName: "TEXTAREA" }, preventDefault() {} });
  assert.equal(calls("/api/similar").length, 2);  // typing in a text box never searches
  await node("#similar-back").trigger("click");
  assert.equal(node("#results-grid").children.length, 1);
  assert.equal(node("#similar-back").hidden, true);
}

async function errorScenario() {
  await connect("login");
  const body = JSON.parse(calls("/api/dres/connect")[0].options.body);
  assert.equal(body.password, "synthetic-password");
  assert.equal(body.session_id, undefined);
  submitResult = { http_status: 200, accepted: true, verdict: "WRONG", pending: false };
  await context.prepareDresSubmission(selection);
  await node("#dres-confirm-submit").trigger("click");
  assert.match(node("#dres-status").textContent, /WRONG/);
  assert.match(node("#dres-status").className, /error/);
  await node("#dres-cancel-submit").trigger("click");
  submitFails = true;
  await context.prepareDresSubmission(selection);
  await node("#dres-confirm-submit").trigger("click");
  assert.match(node("#dres-submit-status").textContent, /Không tự gửi lại/);
  await node("#dres-confirm-submit").trigger("click");
  assert.equal(calls("/api/dres/submit").length, 2);
  await node("#dres-cancel-submit").trigger("click");
  previewFails = true;
  await context.prepareDresSubmission(selection);
  assert.equal(node("#dres-confirm-dialog").open, false);
  assert.match(node("#dres-status").textContent, /Task changed/);
  await node("#dres-confirm-submit").trigger("click");
  assert.equal(calls("/api/dres/submit").length, 2);
}

async function qaScenario() {
  run.task.taskType = "LSC Question & Answer Text";
  await connect("login");
  assert.equal(node("#dres-qa-form").hidden, false);
  assert.equal(node("#dres-qa-answer").disabled, false);
  await context.prepareDresSubmission(selection);
  assert.equal(calls("/api/dres/preview").length, 0);
  node("#dres-qa-answer").value = "my checked manual answer";
  await node("#dres-qa-form").trigger("submit");
  assert.match(node("#dres-payload").textContent, /my checked manual answer/);
  assert.doesNotMatch(node("#dres-payload").textContent, /mediaItemName/);
  assert.equal(calls("/api/dres/submit").length, 0);
  await node("#dres-confirm-submit").trigger("click");
  assert.equal(calls("/api/dres/submit").length, 1);
  run.task_id = "new-task";
  await context.refreshDresConnection();
  assert.equal(node("#dres-qa-answer").value, "");
}

async function taskScenario() {
  await connect();
  assert.equal(node("#dres-task-texts").children[0].textContent, "Find a red boat");
  assert.equal(node("#search-button").disabled, false);
  taskContent = { status: "available", elements: [
    { type: "TEXT", index: 0, offset: 0, text: "First released sentence" },
    { type: "TEXT", index: 1, offset: 30, text: "Supplementary sentence" },
  ] };
  await context.refreshDresConnection();
  assert.equal(node("#dres-task-texts").children.length, 2);
  assert.equal(node("#dres-task-texts").children[0].textContent, "First released sentence");
  assert.equal(node("#dres-task-texts").children[1].textContent, "Supplementary sentence");
  taskContent = { status: "available", elements: [{ type: "VIDEO", index: 2, media_url: "/api/dres/task/media/task-test/2" }] };
  await context.refreshDresConnection();
  assert.equal(node("#dres-task-media").children[0].src, "blob:1");
  await context.refreshDresConnection();
  assert.equal(calls("/api/dres/task/media/task-test/2").length, 1); // Don't restart video unless task content changes.
  run.task_id = "new-task";
  taskContent = { status: "waiting", elements: [] };
  await context.refreshDresConnection();
  assert.deepEqual(released, ["blob:1"]);
  assert.equal(node("#dres-task-media").children.length, 0);
  assert.equal(node("#dres-task-texts").children.length, 0);
  assert.equal(node("#search-button").disabled, true);
  connected = false;
  await context.refreshDresConnection();
  assert.equal(node("#dres-login-panel").hidden, false);
  assert.equal(node("#dres-task-panel").hidden, true);
  assert.equal(calls("/api/dres/submit").length, 0);
}

async function twoHundredScenario() {
  const results = Array.from({ length: 200 }, (_, i) => ({
    rank: i + 1, video_id: "L01_V001", frame_id: i, image_number: i + 1,
    vlm_status: i < 20 ? "reranked" : "unscored", vlm_score: i < 20 ? 8 : null,
  }));
  context.renderResults({ input_type: "text", elapsed_seconds: 1, results });
  assert.equal(node("#results-grid").children.length, 200);
  assert.equal(node("#results-title").textContent, "Top 200 frames");
  assert.match(node("#llm-scoring-note").textContent, /20 frame có điểm/);
  const card = node("#results-grid").children[199];
  const content = card.children[1];
  assert.match(content.children[3].textContent, /chưa chấm LLM/);
  await connect();
  await content.children[4].trigger("click");
  assert.deepEqual(JSON.parse(calls("/api/dres/preview")[0].options.body), {
    video_id: "L01_V001", frame_id: 199, image_number: 200, task_id: run.task_id,
  });
  assert.equal(calls("/api/dres/submit").length, 0);
}

async function restoredScenario() {
  await flush();
  assert.equal(node("#dres-login-panel").hidden, true);
  assert.equal(node("#dres-task-panel").hidden, false);
  assert.equal(calls("/api/dres/connect").length, 0);
  assert.equal(node("#dres-task-texts").children[0].textContent, "Find a red boat");
}

async function imagesScenario() {
  const results = Array.from({ length: 200 }, (_, i) => ({
    rank: i + 1, video_id: "L01_V001", frame_id: i, image_number: i + 1,
    image_url: `/api/images/L01_V001/${i + 1}`,
  }));
  context.renderResults({ elapsed_seconds: 1, results });
  assert.equal(imageActive, 0); // No eager download of 200 images.
  imageObserverCallback(node("#results-grid").children.map(card => ({ isIntersecting: true, target: card.children[0] })));
  assert.equal(imageActive, 8);
  assert.equal(maxImageActive, 8);
  context.releaseResultImages();
  imageResolvers.forEach(resolve => resolve());
  await flush();
  assert.equal(imageActive, 0);
  assert.equal(requests.filter(r => r.url.startsWith("/api/images/")).length, 8); // Old pending queue cleared.
  assert.equal(urls, 0); // Late responses cannot create orphaned Blob URLs.
}

async function fullTextNoResetScenario() {
  await connect();
  await node("#search-form").trigger("submit");
  // A second sentence arrives upstream, without a manual refresh/reconnect.
  taskContent.elements.push({ type: "TEXT", index: 1, offset: 30, text: "New second sentence" });
  const countBefore = calls("/api/dres/task?refresh=1").length;
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/dres/task?refresh=1").length, countBefore + 1);
  assert.equal(node("#dres-task-texts").children.length, 2);
  assert.equal(node("#dres-task-texts").children[0].textContent, "Find a red boat");
  assert.equal(node("#dres-task-texts").children[1].textContent, "New second sentence");
  assert.equal(calls("/api/dres/connect").length, 1);
  assert.equal(calls("/api/dres/disconnect").length, 0);
  assert.equal(calls("/api/dres/search").length, 2);
  assert.equal(calls("/api/dres/submit").length, 0);
  searchConflict = true;
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/dres/search").length, 3); // No automatic model retry.
  assert.equal(calls("/api/dres/task?refresh=1").length, countBefore + 3); // Refresh on 409 too.
  assert.equal(node("#dres-task-texts").children.length, 2);
}

async function mediaNoResetScenario() {
  run.task.taskType = "Visual KIS";
  taskContent = { status: "available", elements: [
    { type: "VIDEO", index: 0, media_url: "/api/dres/task/media/task-test/0" },
    { type: "TEXT", index: 1, offset: 0, text: "First sentence" },
  ] };
  await connect();
  const media = node("#dres-task-media").children[0];
  node("#dres-video-description").value = "My video hint";
  taskContent.elements.push({ type: "TEXT", index: 2, offset: 30, text: "Second sentence" });
  await node("#dres-refresh").trigger("click");
  assert.equal(node("#dres-task-media").children[0], media);
  assert.equal(calls("/api/dres/task/media/task-test/0").length, 1);
  assert.equal(node("#dres-task-texts").children.length, 2);
  assert.equal(node("#dres-video-description").value, "My video hint");
  assert.equal(released.length, 0);
}

async function auxiliaryTemporalScenario() {
  const results = Array.from({ length: 200 }, (_, i) => ({ rank: i + 1,
    video_id: "L01_V001", frame_id: i, image_number: i + 1, vlm_status: "unscored" }));
  const segments = results.slice(0, 5).map(result => ({ ...result, vlm_status: "skipped_temporal",
    temporal: { events: [], verified: false } }));
  context.renderResults({ input_type: "text", elapsed_seconds: 1, results,
    temporal: { status: "applied", presentation: "auxiliary" }, temporal_results: segments });
  assert.equal(node("#results-grid").children.length, 200);
  assert.equal(node("#temporal-results-grid").children.length, 5);
  assert.equal(node("#temporal-results-section").hidden, false);
  assert.match(node("#result-count").textContent, /200 kết quả/);
  assert.notEqual(node("#results-title").textContent, "Chuỗi sự kiện");
  context.releaseResultImages();
  assert.equal(node("#temporal-results-grid").children.length, 0);
  assert.equal(node("#temporal-results-section").hidden, true);
}

async function manualScenario() {
  assert.equal(requests.some(r => r.url.startsWith("/api/dres/")), false);
  assert.equal(node("#manual-query-panel").hidden, false);
  assert.equal(node("#dres-login-panel").hidden, true);
  assert.equal(node("#search-button").disabled, false);
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/search").length, 0);
  node("#manual-query").value = "  A white and green lion dance  ";
  await node("#search-form").trigger("submit");
  assert.deepEqual(JSON.parse(calls("/api/search")[0].options.body), { query: "A white and green lion dance" });
  assert.equal(node("#results-section").hidden, false);
  assert.equal(node("#manual-query").disabled, false);
  node("#manual-query").value = "x".repeat(8001);
  await node("#search-form").trigger("submit");
  assert.equal(calls("/api/search").length, 1);
  const card = context.createResultCard({ rank: 1, video_id: "L01_V001", image_number: 2, frame_id: 150 });
  function hasSubmit(element) {
    return element.className === "dres-select" || element.children.some(hasSubmit);
  }
  assert.equal(hasSubmit(card), false);
  await assert.rejects(context.dresApi("connect", {}), /DRES/);
  assert.equal(requests.some(r => r.url.startsWith("/api/dres/")), false);
}

async function dresManualQueryScenario() {
  await connect();
  assert.equal(node("#dres-manual-toggle").checked, false);
  assert.equal(node("#manual-query-panel").hidden, true);
  const officialSearches = calls("/api/dres/search").length;
  node("#dres-manual-toggle").checked = true;
  await node("#dres-manual-toggle").trigger("change");
  assert.equal(node("#manual-query-panel").hidden, false);
  assert.equal(node("#manual-query").disabled, false);
  assert.match(node("#official-query-source").textContent, /QUERY CỦA BẠN/);
  assert.match(node("#search-button").textContent, /Tìm keyframes/);
  node("#manual-query").value = "  manual query inside tkis  ";
  await node("#search-form").trigger("submit");
  assert.deepEqual(JSON.parse(calls("/api/search")[0].options.body), { query: "manual query inside tkis" });
  assert.equal(calls("/api/dres/search").length, officialSearches);
  assert.equal(node("#results-section").hidden, false);
  const card = context.createResultCard({ rank: 1, video_id: "L01_V001", image_number: 2, frame_id: 150 });
  const hasSubmit = element => element.className === "dres-select"
    || element.children.some(hasSubmit);
  assert.equal(hasSubmit(card), true);
  node("#dres-manual-toggle").checked = false;
  await node("#dres-manual-toggle").trigger("change");
  assert.equal(node("#manual-query-panel").hidden, true);
  assert.match(node("#official-query-source").textContent, /Dùng trực tiếp đề/);
  assert.match(node("#search-button").textContent, /đề DRES/);
  node("#dres-manual-toggle").checked = true;
  await node("#dres-manual-toggle").trigger("change");
  run.task_id = "new-task";
  taskContent = { status: "waiting", elements: [] };
  await context.refreshDresConnection();
  assert.equal(node("#dres-manual-toggle").checked, false);
  assert.equal(node("#manual-query-panel").hidden, true);
  assert.equal(calls("/api/dres/submit").length, 0);
}

async function vlmOffScenario() {
  context.renderResults({ input_type: "text", elapsed_seconds: 1, results: [], vlm_enabled: false });
  assert.match(node("#llm-scoring-note").textContent, /VLM đang tắt/);
  assert.match(node("#llm-scoring-note").textContent, /không chấm ảnh bằng Qwen/);
  context.renderResults({ input_type: "text", elapsed_seconds: 1, results: [], vlm_enabled: true });
  assert.doesNotMatch(node("#llm-scoring-note").textContent, /VLM đang tắt/);
}

async function plannerScenario() {
  const payload = { input_type: "text", elapsed_seconds: 1, results: [],
    query_planner: { used: { provider: "opencode", model: "openai/gpt-6.1-sol" },
      fallback_used: true, reason: "dns" } };
  context.renderResults(payload);
  assert.equal(node("#query-planner-status").hidden, false);
  assert.match(node("#query-planner-status").textContent, /OpenCode fallback/);
  assert.match(node("#query-planner-status").textContent, /DNS/);
  payload.query_planner = { used: { provider: "groq", model: "test" }, fallback_used: false };
  context.renderResults(payload);
  assert.match(node("#query-planner-status").textContent, /groq/);
  assert.doesNotMatch(node("#query-planner-status").textContent, /OpenCode fallback/);
  delete payload.query_planner;
  context.renderResults(payload);
  assert.equal(node("#query-planner-status").hidden, true);
}

(async () => {
  await flush();
  const scenario = process.argv[3];
  if (scenario === "official") await officialScenario();
  else if (scenario === "video-description") await videoDescriptionScenario();
  else if (scenario === "dres") await dresScenario();
  else if (scenario === "errors") await errorScenario();
  else if (scenario === "qa") await qaScenario();
  else if (scenario === "task") await taskScenario();
  else if (scenario === "two-hundred") await twoHundredScenario();
  else if (scenario === "restored") await restoredScenario();
  else if (scenario === "images") await imagesScenario();
  else if (scenario === "full-text") await fullTextNoResetScenario();
  else if (scenario === "media-no-reset") await mediaNoResetScenario();
  else if (scenario === "auxiliary-temporal") await auxiliaryTemporalScenario();
  else if (scenario === "manual") await manualScenario();
  else if (scenario === "dres-manual-query") await dresManualQueryScenario();
  else if (scenario === "vlm-off") await vlmOffScenario();
  else if (scenario === "planner") await plannerScenario();
  else if (scenario === "frame-adjust") await frameAdjustScenario();
  else if (scenario === "similar") await similarScenario();
  else throw Error("Unknown test scenario");
})().catch(error => { console.error(error); process.exitCode = 1; });
