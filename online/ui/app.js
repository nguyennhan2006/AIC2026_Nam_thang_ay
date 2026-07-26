const form = document.querySelector("#search-form");
const statusBox = document.querySelector("#status");
const resultsBox = document.querySelector("#results");
const debugPanel = document.querySelector("#debug-panel");
const debugOutput = document.querySelector("#debug-output");
const apiBaseInput = document.querySelector("#api-base");
const apiTokenInput = document.querySelector("#api-token");
const trayList = document.querySelector("#tray-list");
const trayCount = document.querySelector("#tray-count");
const trayRefine = document.querySelector("#tray-refine");
const trayExport = document.querySelector("#tray-export");
const trayClear = document.querySelector("#tray-clear");
const trayAnswerWrap = document.querySelector("#tray-answer-wrap");
const trayAnswerInput = document.querySelector("#tray-answer");
const taskSelect = document.querySelector("#task");
const MAX_SUBMISSION_ROWS = 100; // giới hạn cứng của BTC — mỗi file CSV nộp bài tối đa 100 dòng.
apiBaseInput.value = localStorage.getItem("aic_api_base") || "http://localhost:8000";
apiTokenInput.value = localStorage.getItem("aic_api_token") || "";
trayAnswerInput.value = localStorage.getItem("aic_tray_answer") || "";
trayAnswerInput.addEventListener("input", () => {
  localStorage.setItem("aic_tray_answer", trayAnswerInput.value);
});

function updateTrayAnswerVisibility() {
  trayAnswerWrap.hidden = taskSelect.value !== "vqa";
}
taskSelect.addEventListener("change", updateTrayAnswerVisibility);
updateTrayAnswerVisibility();

function apiBase() { return apiBaseInput.value.trim().replace(/\/$/, ""); }
function headers() {
  const value = {"Content-Type": "application/json"};
  if (apiTokenInput.value) value.Authorization = `Bearer ${apiTokenInput.value}`;
  return value;
}
function mediaUrl(path) {
  return `${apiBase()}/v1/media/${String(path).split("/").map(encodeURIComponent).join("/")}`;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>'"]/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[ch]);
}

// ---- Selection tray: persists across searches so a shortlist can be built up ----
let selection = new Map();
try {
  selection = new Map(JSON.parse(localStorage.getItem("aic_selection") || "[]"));
} catch (_) { selection = new Map(); }

// scene_id -> full hit, rebuilt on every render so checkboxes/icons can look up data.
let hitsById = new Map();
// scene_id -> Promise<full scene detail>, fetched once and reused by every icon panel.
const sceneDetailCache = new Map();

function persistSelection() {
  localStorage.setItem("aic_selection", JSON.stringify([...selection.entries()]));
}

function renderTray() {
  const items = [...selection.values()];
  trayCount.textContent = `(${items.length}/${MAX_SUBMISSION_ROWS})`;
  trayCount.classList.toggle("over-limit", items.length > MAX_SUBMISSION_ROWS);
  const hasItems = items.length > 0;
  trayRefine.disabled = trayClear.disabled = !hasItems;
  trayExport.disabled = !hasItems || items.length > MAX_SUBMISSION_ROWS;
  trayList.innerHTML = items.map(item => `
    <li>
      <div>
        <strong>${esc(item.video_id)}</strong>
        <span>${esc(item.scene_id)} · ${Number(item.best_timestamp_sec ?? item.start_sec).toFixed(2)}s</span>
      </div>
      <button type="button" class="tray-remove" data-scene-id="${esc(item.scene_id)}" aria-label="Bỏ khỏi danh sách">×</button>
    </li>`).join("");
}
renderTray();

function syncCardSelectedState(sceneId, checked) {
  document.querySelectorAll(`.card[data-scene-id="${CSS.escape(sceneId)}"]`).forEach(card => {
    card.classList.toggle("selected", checked);
    const box = card.querySelector("input.select-box");
    if (box) box.checked = checked;
  });
}

function toggleSelect(sceneId, checked) {
  if (checked) {
    const hit = hitsById.get(sceneId);
    if (!hit) return;
    selection.set(sceneId, {
      scene_id: hit.scene_id, video_id: hit.video_id, score: hit.score,
      best_keyframe_id: hit.best_keyframe_id, best_timestamp_sec: hit.best_timestamp_sec,
      start_sec: hit.start_sec,
    });
  } else {
    selection.delete(sceneId);
  }
  persistSelection();
  renderTray();
  syncCardSelectedState(sceneId, checked);
}

// Format nộp bài chính thức BTC AIC 2026 (KHÔNG phải CSV debug):
//   KIS/AVS : <video_id>, <frame_idx>                  — không header, tối đa 100 dòng
//   VQA     : <video_id>, <frame_idx>, "<answer>"       — answer <=100 ký tự, giữ nguyên cho mọi dòng
// Mỗi file chỉ ứng với MỘT câu truy vấn — xem tray-hint. Không dùng csvEscape/quote kiểu
// RFC4180 thông thường: khớp đúng ví dụ BTC (dấu phẩy + khoảng trắng, answer luôn có ngoặc kép).
trayExport.addEventListener("click", () => {
  const items = [...selection.values()];
  if (items.length === 0) return;
  if (items.length > MAX_SUBMISSION_ROWS) {
    statusBox.textContent = `Đang chọn ${items.length} dòng, vượt giới hạn ${MAX_SUBMISSION_ROWS} — bỏ bớt trước khi xuất.`;
    return;
  }
  const task = taskSelect.value;
  let answer = "";
  if (task === "vqa") {
    answer = trayAnswerInput.value.trim();
    if (!answer) {
      statusBox.textContent = "Nhập câu trả lời VQA trước khi xuất CSV.";
      trayAnswerInput.focus();
      return;
    }
    if (answer.length > 100) {
      statusBox.textContent = "Câu trả lời VQA vượt quá 100 ký tự.";
      trayAnswerInput.focus();
      return;
    }
  }
  const lines = items.map(item => {
    const frameMatch = /_F(\d+)$/.exec(item.best_keyframe_id || "");
    const frameIdx = frameMatch ? Number(frameMatch[1]) : "";
    return task === "vqa"
      ? `${item.video_id}, ${frameIdx}, "${answer.replace(/"/g, '""')}"`
      : `${item.video_id}, ${frameIdx}`;
  });
  const blob = new Blob([lines.join("\n")], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `aic2026_${task}_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
  link.click();
  URL.revokeObjectURL(url);
  statusBox.textContent = `Đã xuất ${lines.length} dòng (${task.toUpperCase()}).`;
});

trayClear.addEventListener("click", () => {
  selection.clear();
  persistSelection();
  renderTray();
  document.querySelectorAll(".card.selected").forEach(card => card.classList.remove("selected"));
  document.querySelectorAll("input.select-box").forEach(box => { box.checked = false; });
});

trayRefine.addEventListener("click", () => {
  const videoIds = [...new Set([...selection.values()].map(item => item.video_id))];
  runSearch({filters: {video_ids: videoIds}});
});

trayList.addEventListener("click", event => {
  const button = event.target.closest(".tray-remove");
  if (!button) return;
  selection.delete(button.dataset.sceneId);
  persistSelection();
  renderTray();
  syncCardSelectedState(button.dataset.sceneId, false);
});

// ---- Result cards: compact grid, details revealed on demand via icon toggles ----
const FIELD_LABELS = {caption: "Caption", ocr: "OCR", asr: "ASR", keyword: "Keyword"};
const ICONS = [
  {field: "reason", icon: "\u{1F4CA}", title: "Vì sao khớp"},
  {field: "caption", icon: "\u{1F4DD}", title: "Caption"},
  {field: "ocr", icon: "\u{1F524}", title: "OCR"},
  {field: "asr", icon: "\u{1F3A4}", title: "ASR"},
  {field: "keyword", icon: "\u{1F3F7}️", title: "Keyword"},
  {field: "video", icon: "\u{1F3AC}", title: "Xem video"},
];

function getSceneDetail(sceneId) {
  if (!sceneDetailCache.has(sceneId)) {
    sceneDetailCache.set(sceneId, fetch(`${apiBase()}/v1/scenes/${encodeURIComponent(sceneId)}`, {headers: headers()})
      .then(r => r.json()));
  }
  return sceneDetailCache.get(sceneId);
}

function renderReasonPanel(hit) {
  const entries = Object.entries(hit.component_scores || {}).sort((a, b) => b[1] - a[1]);
  const breakdown = entries.length
    ? `<ul class="reason">${entries.map(([name, value]) =>
        `<li><span>${esc(name)}</span><output>${Number(value).toFixed(4)}</output></li>`).join("")}</ul>`
    : "";
  const evidence = (hit.evidence || []).map(x => `<li><strong>${esc(x.modality)}</strong>: ${esc(x.text)}</li>`).join("");
  return `${breakdown}${evidence ? `<ul class="evidence">${evidence}</ul>` : "<p class=\"muted\">Không có evidence text.</p>"}`;
}

function renderVideoPanel(hit) {
  if (!hit.video_path) return "<p class=\"muted\">Không có video nguồn.</p>";
  return `<video class="video" controls preload="metadata" src="${esc(mediaUrl(hit.video_path))}#t=${Number(hit.start_sec)},${Number(hit.end_sec)}"></video>`;
}

function renderFieldList(field, detail) {
  const values = {caption: detail.captions, ocr: detail.ocr_texts, asr: detail.asr_texts, keyword: detail.keywords}[field] || [];
  if (!values.length) return `<p class="muted">Không có ${esc(FIELD_LABELS[field])}.</p>`;
  return `<ul class="field-list">${values.map(v => `<li>${esc(v)}</li>`).join("")}</ul>`;
}

function sceneCard(hit, index) {
  hitsById.set(hit.scene_id, hit);
  const modalities = (hit.matched_modalities || []).map(x => `<span>${esc(x)}</span>`).join("");
  const thumbnail = hit.best_keyframe_path
    ? `<img class="thumb" loading="lazy" src="${esc(mediaUrl(hit.best_keyframe_path))}" alt="${esc(hit.best_keyframe_id)}">`
    : `<div class="thumb thumb-empty">Không có ảnh</div>`;
  const checked = selection.has(hit.scene_id);
  const icons = ICONS.map(item =>
    `<button type="button" class="icon-btn" data-field="${item.field}" title="${esc(item.title)}" aria-label="${esc(item.title)}">${item.icon}</button>`
  ).join("");
  return `<article class="card${checked ? " selected" : ""}" data-scene-id="${esc(hit.scene_id)}">
    <header class="card-head">
      <label class="select-label" title="Chọn kết quả này">
        <input type="checkbox" class="select-box" data-scene-id="${esc(hit.scene_id)}" ${checked ? "checked" : ""}>
        <span class="rank">#${index + 1}</span>
      </label>
      <output>${Number(hit.score).toFixed(4)}</output>
    </header>
    <div class="thumb-wrap">${thumbnail}</div>
    <div class="card-meta">
      <strong>${esc(hit.video_id)}</strong>
      <span>${Number(hit.start_sec).toFixed(1)}s–${Number(hit.end_sec).toFixed(1)}s${hit.best_timestamp_sec != null ? ` · ${Number(hit.best_timestamp_sec).toFixed(2)}s` : ""}</span>
      <div class="chips">${modalities}</div>
    </div>
    <div class="icon-bar">${icons}</div>
    <div class="expand-panel" hidden></div>
  </article>`;
}

resultsBox.addEventListener("change", event => {
  if (event.target.matches("input.select-box")) {
    toggleSelect(event.target.dataset.sceneId, event.target.checked);
  }
});

resultsBox.addEventListener("click", async event => {
  const button = event.target.closest(".icon-bar .icon-btn");
  if (!button) return;
  const card = button.closest(".card");
  const sceneId = card.dataset.sceneId;
  const panel = card.querySelector(".expand-panel");
  const field = button.dataset.field;
  const wasOpenSameField = !panel.hidden && panel.dataset.field === field;
  card.querySelectorAll(".icon-bar .icon-btn").forEach(b => b.classList.remove("active"));
  if (wasOpenSameField) {
    panel.hidden = true;
    panel.dataset.field = "";
    return;
  }
  button.classList.add("active");
  panel.hidden = false;
  panel.dataset.field = field;
  const hit = hitsById.get(sceneId);
  if (field === "reason") {
    panel.innerHTML = renderReasonPanel(hit);
  } else if (field === "video") {
    panel.innerHTML = renderVideoPanel(hit);
  } else {
    panel.innerHTML = "<p class=\"muted\">Đang tải…</p>";
    try {
      const detail = await getSceneDetail(sceneId);
      if (panel.dataset.field === field) panel.innerHTML = renderFieldList(field, detail);
    } catch (error) {
      panel.innerHTML = `<p class="muted">Không tải được: ${esc(error.message)}</p>`;
    }
  }
});

async function runSearch(overrides = {}) {
  const task = document.querySelector("#task").value;
  const query = document.querySelector("#query").value.trim();
  const topK = Number(document.querySelector("#top-k").value);
  const debug = document.querySelector("#debug").checked;
  localStorage.setItem("aic_api_base", apiBase());
  localStorage.setItem("aic_api_token", apiTokenInput.value);
  const endpoint = task === "vqa" ? `${apiBase()}/v1/vqa` : `${apiBase()}/v1/search/${task}`;
  const extra = overrides.filters ? {filters: overrides.filters} : {};
  const body = task === "vqa"
    ? {question: query, top_k_evidence: topK, debug, ...extra}
    : {query, top_k: topK, debug, ...extra};
  statusBox.textContent = extra.filters ? `Đang tìm lại trong ${extra.filters.video_ids.length} video đã chọn…` : "Đang tìm kiếm…";
  resultsBox.innerHTML = "";
  resultsBox.classList.remove("sequence-mode");
  hitsById = new Map();
  debugPanel.hidden = true;
  try {
    const response = await fetch(endpoint, {
      method: "POST", headers: headers(), body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error?.message || "Request failed");
    if (task === "vqa") {
      statusBox.textContent = `Hoàn tất trong ${data.took_ms.toFixed(1)} ms`;
      resultsBox.classList.add("sequence-mode");
      resultsBox.innerHTML = `<article class="answer"><h2>Trả lời</h2><pre>${esc(data.answer)}</pre></article>` +
        `<div class="card-grid">${data.evidence.map(sceneCard).join("")}</div>`;
    } else if (task === "sequence") {
      statusBox.textContent = `${data.sequences.length} chuỗi · ${data.took_ms.toFixed(1)} ms`;
      resultsBox.classList.add("sequence-mode");
      resultsBox.innerHTML = data.sequences.map((sequence, i) =>
        `<section class="sequence"><h2>Chuỗi ${i + 1} · ${esc(sequence.video_id)}</h2><div class="card-grid">${sequence.scenes.map(sceneCard).join("")}</div></section>`
      ).join("");
    } else {
      statusBox.textContent = `${data.results.length} kết quả · ${data.took_ms.toFixed(1)} ms`;
      resultsBox.innerHTML = `<div class="card-grid">${data.results.map(sceneCard).join("")}</div>`;
    }
    if (data.query_plan) {
      debugPanel.hidden = false;
      debugOutput.textContent = JSON.stringify(data.query_plan, null, 2);
    }
  } catch (error) {
    statusBox.textContent = `Lỗi: ${error.message}`;
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  runSearch();
});

document.querySelector("#health").addEventListener("click", async () => {
  statusBox.textContent = "Đang kiểm tra server…";
  try {
    const response = await fetch(`${apiBase()}/v1/health`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "unhealthy");
    statusBox.textContent = `Server OK · ${data.backend} · ${data.scene_count} scenes`;
  } catch (error) { statusBox.textContent = `Không kết nối được: ${error.message}`; }
});
