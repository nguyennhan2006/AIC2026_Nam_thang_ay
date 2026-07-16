const form = document.querySelector("#search-form");
const statusBox = document.querySelector("#status");
const resultsBox = document.querySelector("#results");
const debugPanel = document.querySelector("#debug-panel");
const debugOutput = document.querySelector("#debug-output");
const apiBaseInput = document.querySelector("#api-base");
const apiTokenInput = document.querySelector("#api-token");
apiBaseInput.value = localStorage.getItem("aic_api_base") || "http://localhost:8000";
apiTokenInput.value = localStorage.getItem("aic_api_token") || "";

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

function sceneCard(hit, index) {
  const modalities = (hit.matched_modalities || []).map(x => `<span>${esc(x)}</span>`).join("");
  const evidence = (hit.evidence || []).map(x =>
    `<li><strong>${esc(x.modality)}</strong>: ${esc(x.text)}</li>`
  ).join("");
  const thumbnail = hit.best_keyframe_path ? `<img class="thumb" loading="lazy" src="${esc(mediaUrl(hit.best_keyframe_path))}" alt="${esc(hit.best_keyframe_id)}">` : "";
  const video = hit.video_path ? `<details><summary>Mở video tại ${Number(hit.best_timestamp_sec ?? hit.start_sec).toFixed(3)}s</summary><video class="video" controls preload="metadata" src="${esc(mediaUrl(hit.video_path))}#t=${Number(hit.start_sec)},${Number(hit.end_sec)}"></video></details>` : "";
  return `<article class="card">
    <div class="rank">${index + 1}</div>
    <div><h2>${esc(hit.scene_id)}</h2>
      <p>${esc(hit.video_id)} · ${Number(hit.start_sec).toFixed(2)}s–${Number(hit.end_sec).toFixed(2)}s</p>
      <div class="chips">${modalities}</div>
      ${thumbnail}
      ${video}
      ${hit.best_timestamp_sec != null ? `<p>Keyframe tốt nhất: ${Number(hit.best_timestamp_sec).toFixed(3)}s</p>` : ""}
      ${evidence ? `<ul>${evidence}</ul>` : ""}
    </div><output>${Number(hit.score).toFixed(5)}</output>
  </article>`;
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const task = document.querySelector("#task").value;
  const query = document.querySelector("#query").value.trim();
  const topK = Number(document.querySelector("#top-k").value);
  const debug = document.querySelector("#debug").checked;
  localStorage.setItem("aic_api_base", apiBase());
  localStorage.setItem("aic_api_token", apiTokenInput.value);
  const endpoint = task === "vqa" ? `${apiBase()}/v1/vqa` : `${apiBase()}/v1/search/${task}`;
  const body = task === "vqa"
    ? {question: query, top_k_evidence: topK, debug}
    : {query, top_k: topK, debug};
  statusBox.textContent = "Đang tìm kiếm…";
  resultsBox.innerHTML = "";
  debugPanel.hidden = true;
  try {
    const response = await fetch(endpoint, {
      method: "POST", headers: headers(), body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error?.message || "Request failed");
    if (task === "vqa") {
      statusBox.textContent = `Hoàn tất trong ${data.took_ms.toFixed(1)} ms`;
      resultsBox.innerHTML = `<article class="answer"><h2>Trả lời</h2><pre>${esc(data.answer)}</pre></article>` +
        data.evidence.map(sceneCard).join("");
    } else if (task === "sequence") {
      statusBox.textContent = `${data.sequences.length} chuỗi · ${data.took_ms.toFixed(1)} ms`;
      resultsBox.innerHTML = data.sequences.map((sequence, i) =>
        `<section class="sequence"><h2>Chuỗi ${i + 1} · ${esc(sequence.video_id)}</h2>${sequence.scenes.map(sceneCard).join("")}</section>`
      ).join("");
    } else {
      statusBox.textContent = `${data.results.length} kết quả · ${data.took_ms.toFixed(1)} ms`;
      resultsBox.innerHTML = data.results.map(sceneCard).join("");
    }
    if (data.query_plan) {
      debugPanel.hidden = false;
      debugOutput.textContent = JSON.stringify(data.query_plan, null, 2);
    }
  } catch (error) {
    statusBox.textContent = `Lỗi: ${error.message}`;
  }
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
