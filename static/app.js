// ClipForge front-end: no framework, no build step.
"use strict";

const $ = (id) => document.getElementById(id);
const state = { file: null, jobId: null, pollTimer: null, xhr: null };

// ---------------------------------------------------------------- helpers
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof Blob) ? { "Content-Type": "application/json" } : {},
    ...opts,
  });
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((data && (data.error || data.detail)) || `Request failed (${res.status})`);
  return data;
}
function segValue(name) {
  const on = document.querySelector(`.seg[data-name="${name}"] button.on`);
  return on ? on.dataset.value : null;
}
function setSeg(name, value) {
  document.querySelectorAll(`.seg[data-name="${name}"] button`).forEach((b) =>
    b.classList.toggle("on", b.dataset.value === value));
}
function show(el, on = true) { el.hidden = !on; }
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function showError(message, detail) {
  const box = $("errorBox");
  box.innerHTML = escapeHtml(message) +
    (detail ? `<details><summary>Technical details</summary><pre>${escapeHtml(detail)}</pre></details>` : "");
  show(box);
}
function clearMessages() { show($("errorBox"), false); show($("noticeBox"), false); }

// ---------------------------------------------------------------- system info
async function loadSystem() {
  try {
    const s = await api("/api/system");
    const bits = [];
    bits.push(s.ffmpeg ? "FFmpeg OK" : "FFmpeg missing");
    bits.push(s.cuda ? "Whisper: NVIDIA GPU" : "Whisper: CPU");
    bits.push(s.has_nvenc ? "Encoder: NVENC" : "Encoder: x264");
    $("sysInfo").textContent = bits.join(" · ");
    if (s.warnings && s.warnings.length) {
      $("sysWarning").textContent = s.warnings.join("\n");
      show($("sysWarning"));
    }
    if (!s.ffmpeg) $("startBtn").disabled = true;
  } catch (e) {
    $("sysInfo").textContent = "Server not reachable";
  }
}

// ---------------------------------------------------------------- inputs
function setFile(file) {
  state.file = file;
  $("dropZone").classList.toggle("has-file", !!file);
  $("fileName").textContent = file ? `${file.name} (${(file.size / 1e6).toFixed(1)} MB)` : "MP4, MOV, MKV, WEBM, AVI... any length";
  if (file) $("urlInput").value = "";
}
function initInputs() {
  const dz = $("dropZone");
  $("fileInput").addEventListener("change", (e) => setFile(e.target.files[0] || null));
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("fileInput").click(); });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });
  $("urlInput").addEventListener("input", () => { if ($("urlInput").value.trim()) { setFile(null); $("fileInput").value = ""; } });

  document.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    setSeg(seg.dataset.name, b.dataset.value);
    if (seg.dataset.name === "ai_mode") onAiModeChange();
  }));
  $("captions").addEventListener("change", () => { $("captionOpts").style.opacity = $("captions").checked ? 1 : .4; });
  $("useAi").addEventListener("change", () => show($("aiOpts"), $("useAi").checked));
  $("aiSave").addEventListener("click", saveAi);
  $("aiTest").addEventListener("click", testAi);
  $("startBtn").addEventListener("click", start);
  $("cancelBtn").addEventListener("click", cancel);
  $("clearBtn").addEventListener("click", clearJobs);

  // Remember options per browser.
  try {
    const saved = JSON.parse(localStorage.getItem("clipforge.options") || "null");
    if (saved) applyOptions(saved);
  } catch { /* storage unavailable */ }
}
function collectOptions() {
  return {
    num_clips: +$("numClips").value || 5,
    min_len: +$("minLen").value || 20,
    max_len: +$("maxLen").value || 60,
    fmt: segValue("fmt"),
    captions: $("captions").checked,
    caption_size: segValue("caption_size"),
    caption_position: segValue("caption_position"),
    whisper_model: $("whisperModel").value,
    language: $("language").value,
    use_ai: $("useAi").checked,
  };
}
function applyOptions(o) {
  if (o.num_clips) $("numClips").value = o.num_clips;
  if (o.min_len) $("minLen").value = o.min_len;
  if (o.max_len) $("maxLen").value = o.max_len;
  if (o.fmt) setSeg("fmt", o.fmt);
  if (typeof o.captions === "boolean") $("captions").checked = o.captions;
  if (o.caption_size) setSeg("caption_size", o.caption_size);
  if (o.caption_position) setSeg("caption_position", o.caption_position);
  if (o.whisper_model) $("whisperModel").value = o.whisper_model;
  if (o.language) $("language").value = o.language;
  if (typeof o.use_ai === "boolean") { $("useAi").checked = o.use_ai; show($("aiOpts"), o.use_ai); }
}

// ---------------------------------------------------------------- AI settings
let aiHasKey = false;
function onAiModeChange() {
  const cloud = segValue("ai_mode") === "cloud";
  show($("aiKeyField"), cloud);
  const url = $("aiUrl").value.trim();
  if (cloud && (!url || url.includes("localhost"))) $("aiUrl").value = "https://ollama.com";
  if (!cloud && (!url || url.includes("ollama.com"))) $("aiUrl").value = "http://localhost:11434";
}
async function loadAi() {
  try {
    const c = (await api("/api/config")).ollama;
    setSeg("ai_mode", c.mode);
    $("aiUrl").value = c.base_url;
    $("aiModel").value = c.model;
    aiHasKey = c.has_api_key;
    $("aiKey").placeholder = aiHasKey ? "Key saved (leave blank to keep)" : "Stored locally in config.json";
    show($("aiKeyField"), c.mode === "cloud");
  } catch { /* ignore */ }
}
async function saveAi() {
  const body = { ollama: { mode: segValue("ai_mode"), base_url: $("aiUrl").value.trim(), model: $("aiModel").value.trim() } };
  const key = $("aiKey").value.trim();
  if (key) body.ollama.api_key = key;
  try {
    await api("/api/config", { method: "POST", body: JSON.stringify(body) });
    $("aiKey").value = "";
    $("aiMsg").textContent = "Saved.";
    await loadAi();
  } catch (e) { $("aiMsg").textContent = e.message; }
}
async function testAi() {
  await saveAi();
  $("aiMsg").textContent = "Testing...";
  try {
    const r = await api("/api/ollama/test", { method: "POST" });
    $("aiMsg").textContent = r.message;
    $("aiModelList").innerHTML = (r.models || []).map((m) => `<option value="${escapeHtml(m)}">`).join("");
  } catch (e) { $("aiMsg").textContent = e.message; }
}

// ---------------------------------------------------------------- job flow
async function start() {
  clearMessages();
  const url = $("urlInput").value.trim();
  if (!state.file && !url) { showError("Choose a video file or paste a link first."); return; }
  const options = collectOptions();
  if (options.max_len <= options.min_len) { showError("Max length must be greater than min length."); return; }
  try { localStorage.setItem("clipforge.options", JSON.stringify(options)); } catch { /* ignore */ }
  if (options.use_ai) await saveAi();

  $("startBtn").disabled = true;
  show($("resultsSection"), false);
  show($("progressCard"));
  setProgress("Starting...", 0, "", 0);
  try {
    let job;
    if (state.file) {
      job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ source: "upload", filename: state.file.name, options }) });
      state.jobId = job.id;
      await uploadFile(job.id, state.file);
    } else {
      job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ source: "url", url, options }) });
      state.jobId = job.id;
    }
    poll();
  } catch (e) {
    showError(e.message);
    finish();
  }
}
function uploadFile(jobId, file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    xhr.open("PUT", `/api/jobs/${jobId}/upload`);
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = e.loaded / e.total * 100;
        setProgress("Uploading", pct, `${(e.loaded / 1e6).toFixed(1)} of ${(e.total / 1e6).toFixed(1)} MB`, pct * 0.15);
      }
    };
    xhr.onload = () => {
      state.xhr = null;
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else { let m = "Upload failed."; try { m = JSON.parse(xhr.responseText).error || m; } catch { } reject(new Error(m)); }
    };
    xhr.onerror = () => { state.xhr = null; reject(new Error("Upload failed - is the server still running?")); };
    xhr.onabort = () => { state.xhr = null; reject(new Error("Upload cancelled.")); };
    xhr.send(file);
  });
}
function setProgress(stage, pct, msg, overall) {
  $("stageName").textContent = stage;
  $("stageMsg").textContent = msg || "";
  $("stagePct").textContent = pct ? `${Math.round(pct)}%` : "";
  $("barFill").style.width = `${Math.max(0, Math.min(100, overall))}%`;
}
async function poll() {
  clearTimeout(state.pollTimer);
  if (!state.jobId) return;
  try {
    const s = await api(`/api/jobs/${state.jobId}/status`);
    const label = s.stage === "Transcribing" ? `Transcribing ${Math.round(s.stage_pct)}%` : (s.stage || s.state);
    setProgress(label, s.stage === "Transcribing" ? 0 : s.stage_pct, s.message, s.overall);
    if (s.state === "done") { renderResults(s.id, s.result); finish(); loadJobs(); return; }
    if (s.state === "error") { showError(s.error || "Processing failed.", s.detail); finish(); loadJobs(); return; }
    if (s.state === "cancelled") { showError("The job was cancelled."); finish(); loadJobs(); return; }
  } catch (e) { /* transient - keep polling */ }
  state.pollTimer = setTimeout(poll, 1000);
}
function finish() {
  $("startBtn").disabled = false;
  show($("progressCard"), false);
  clearTimeout(state.pollTimer);
}
async function cancel() {
  if (state.xhr) state.xhr.abort();
  if (state.jobId) { try { await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }); } catch { } }
  $("stageName").textContent = "Cancelling...";
}

// ---------------------------------------------------------------- results
function renderResults(jobId, result) {
  const wrap = $("clips");
  wrap.innerHTML = "";
  if (result.notices && result.notices.length) { $("noticeBox").textContent = result.notices.join("\n"); show($("noticeBox")); }
  $("resultsTitle").textContent = `${result.clips.length} clip${result.clips.length === 1 ? "" : "s"} from "${result.title}"`;
  $("zipBtn").href = `/api/jobs/${jobId}/zip`;
  const tpl = $("clipTpl");
  for (const c of result.clips) {
    const node = tpl.content.cloneNode(true);
    const base = `/api/jobs/${jobId}/files/`;
    node.querySelector("video").src = base + encodeURIComponent(c.file);
    const sc = node.querySelector(".score");
    sc.textContent = `Score ${c.score}${c.source === "ai" ? " · AI pick" : ""}`;
    sc.classList.add(c.score >= 70 ? "hi" : c.score >= 45 ? "mid" : "lo");
    node.querySelector(".time").textContent = `${c.start_label} – ${c.end_label} (${Math.round(c.duration)}s)`;
    node.querySelector(".title").textContent = `${c.index}. ${c.title}`;
    node.querySelector(".reason").textContent = c.reason || "";
    node.querySelector(".text").textContent = c.text || "(no speech)";
    if (!c.text) node.querySelector("details").remove();
    const dl = node.querySelector(".dl");
    dl.href = base + encodeURIComponent(c.file) + "?download=1";
    const srt = node.querySelector(".srt");
    if (c.srt) srt.href = base + encodeURIComponent(c.srt) + "?download=1"; else srt.remove();
    wrap.appendChild(node);
  }
  show($("resultsSection"));
  $("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- job history
async function loadJobs() {
  try {
    const list = await api("/api/jobs");
    $("jobCount").textContent = list.length ? `(${list.length})` : "";
    $("jobList").innerHTML = list.slice(0, 15).map((j) => `
      <div class="job" data-id="${escapeHtml(j.id)}" data-state="${escapeHtml(j.state)}">
        <span class="src">${escapeHtml(j.source || j.id)}</span>
        <span class="muted">${escapeHtml(j.state)}${j.clips ? ` · ${j.clips} clips` : ""} · ${new Date(j.created * 1000).toLocaleString()}</span>
      </div>`).join("") || `<div class="muted small">No jobs yet.</div>`;
    document.querySelectorAll(".job").forEach((el) => el.addEventListener("click", async () => {
      if (el.dataset.state !== "done") return;
      const s = await api(`/api/jobs/${el.dataset.id}/status`);
      clearMessages();
      renderResults(s.id, s.result);
    }));
  } catch { /* ignore */ }
}
async function clearJobs() {
  if (!confirm("Delete all finished, failed and cancelled jobs and their clips?")) return;
  try {
    const r = await api("/api/jobs", { method: "DELETE" });
    show($("resultsSection"), false);
    $("noticeBox").textContent = `Removed ${r.removed} job folder(s).`;
    show($("noticeBox"));
    loadJobs();
  } catch (e) { showError(e.message); }
}

initInputs();
loadSystem();
loadAi();
loadJobs();
