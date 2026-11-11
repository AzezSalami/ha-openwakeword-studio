"use strict";

/* ---------- helpers ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 2) => Number(n).toFixed(d);
const COLORS = ["#2f7fe0", "#e0456b", "#1fa971", "#e08a1c", "#8b5cf6", "#0aa5c2", "#d946a8", "#7fa81b"];

function wsUrl(path) {
  const url = new URL(path, location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.href;
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body instanceof ArrayBuffer || ArrayBuffer.isView(body)) { opts.body = body; opts.headers["Content-Type"] = "application/octet-stream"; }
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const resp = await fetch(path, opts);
  let data = null;
  try { data = await resp.json(); } catch (e) { /* not json */ }
  if (!resp.ok && !(data && data.errors)) throw new Error((data && data.error) || `Request failed (${resp.status})`);
  return data;
}

function toast(message, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = message;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), isError ? 6000 : 2500);
}

function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 5) return "just now";
  if (s < 60) return `${Math.floor(s)} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString();
const sizeText = (bytes) => bytes > 1048576 ? `${fmt(bytes / 1048576, 1)} MB` : `${Math.round(bytes / 1024)} kB`;

/* ---------- state ---------- */
let state = null;
let busy = false;            // user is dragging a control; don't re-render under them
let lastModelsJson = "";
let lastClientsJson = "";
const freshDetections = new Set();

const modelById = (id) => state && state.models.find((m) => m.id === id);
const thresholdOf = (m) => (m && m.threshold != null ? m.threshold : state.settings.threshold);
const phraseOf = (id) => { const m = modelById(id); return m ? m.phrase : id; };

function applyState(next) {
  state = next;
  renderOverview();
  renderModels();
  renderSettings();
  renderTestChips();
  renderTraining(state.training);
  $("#help-host").textContent = state.hostname || location.hostname;
  $("#help-port").textContent = state.wyoming_port;
  $("#help-dir").textContent = state.custom_dir;
  $("#about").textContent = `openWakeWord Studio ${state.version} · openWakeWord ${state.openwakeword_version}`;
}

/* ---------- tabs ---------- */
function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  $$(".tab").forEach((s) => (s.hidden = s.id !== `tab-${name}`));
  if (name === "test") drawChart();
  if (name === "create") openCreateTab();
  try { sessionStorage.setItem("tab", name); } catch (e) { /* ignore */ }
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* ---------- live events ---------- */
let eventsWs = null;
function connectEvents() {
  eventsWs = new WebSocket(wsUrl("ws/events"));
  eventsWs.onopen = () => setConn(true);
  eventsWs.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") applyState(msg.state);
    else if (msg.type === "scores") updateLiveScores(msg);
    else if (msg.type === "training") { if (state) state.training = msg.status; renderTraining(msg.status); }
    else if (msg.type === "detection" && state) {
      state.detections.unshift(msg.detection);
      state.detections.length = Math.min(state.detections.length, 50);
      state.total_detections = msg.total;
      freshDetections.add(msg.detection.time);
      renderDetections();
      renderTiles();
    }
  };
  eventsWs.onclose = () => { setConn(false); setTimeout(connectEvents, 2000); };
}
function setConn(ok) {
  $("#conn-dot").className = "dot " + (ok ? "ok" : "bad");
  $("#conn-text").textContent = ok
    ? `Running · Wyoming port ${state ? state.wyoming_port : 10400}`
    : "Disconnected — reconnecting…";
}

/* ---------- overview ---------- */
function renderTiles() {
  const usable = state.models.filter((m) => m.usable);
  $("#stat-models").innerHTML = `${usable.filter((m) => m.enabled).length} <small>of ${usable.length}</small>`;
  $("#stat-clients").textContent = state.clients.length;
  $("#stat-detections").textContent = state.total_detections;
}

function renderOverview() {
  renderTiles();
  setConn(eventsWs && eventsWs.readyState === 1);
  const json = JSON.stringify([state.clients, state.models.map((m) => [m.id, m.phrase, m.threshold]), state.settings.threshold]);
  if (json !== lastClientsJson) {
    lastClientsJson = json;
    const box = $("#clients");
    if (!state.clients.length) {
      box.innerHTML = `<p class="empty">No device is listening right now.<br>
        <span class="muted">Connected Home Assistant voice pipelines appear here with live scores. New here? See the <a href="#" data-goto="help">setup guide</a> or <a href="#" data-goto="test">try a wake word in your browser</a>.</span></p>`;
    } else {
      box.innerHTML = state.clients.map((c) => `
        <div class="client" data-client="${c.id}">
          <div class="client-head"><span class="pulse"></span><strong>${esc(c.peer)}</strong>
            <span class="muted">listening since ${esc(clock(c.connected))}</span>
            <div class="meter" title="Audio level"><div data-level></div></div></div>
          ${c.models.map((id) => `
            <div class="bar-row" data-model="${esc(id)}"><span title="${esc(id)}">${esc(phraseOf(id))}</span>
              <div class="bar"><i></i><b style="left:${thresholdOf(modelById(id)) * 100}%" title="Threshold"></b></div>
              <span class="num">0.00</span></div>`).join("")}
          ${c.warning ? `<div class="warn">${esc(c.warning)}</div>` : ""}
        </div>`).join("");
    }
  }
  renderDetections();
}

function updateLiveScores(msg) {
  const card = $(`.client[data-client="${msg.client}"]`);
  if (!card) return;
  const level = $("[data-level]", card);
  if (level) level.style.width = Math.min(100, Math.sqrt(msg.level) * 180) + "%";
  for (const row of $$(".bar-row", card)) {
    const score = msg.scores[row.dataset.model] || 0;
    const fill = $("i", row);
    fill.style.width = score * 100 + "%";
    fill.classList.toggle("hot", score >= thresholdOf(modelById(row.dataset.model)));
    $(".num", row).textContent = fmt(score);
  }
}

function renderDetections() {
  const box = $("#detections");
  if (!state.detections.length) {
    box.innerHTML = `<p class="empty">No detections yet.</p>`;
    return;
  }
  box.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>When</th><th>Wake word</th><th>From</th><th>Score</th></tr></thead><tbody>
    ${state.detections.slice(0, 25).map((d) => `
      <tr class="${freshDetections.has(d.time) ? "fresh" : ""}">
        <td title="${esc(new Date(d.time * 1000).toLocaleString())}">${esc(timeAgo(d.time))}</td>
        <td>${esc(d.phrase)}${d.label !== d.model ? ` <span class="muted">${esc(prettyLabel(d.label))}</span>` : ""}</td>
        <td>${d.kind === "test" ? `<span class="tag">${esc(d.source)}</span>` : esc(d.source)}</td>
        <td>${fmt(d.score)}</td></tr>`).join("")}
    </tbody></table></div>`;
  freshDetections.clear();
}
setInterval(() => { if (state && !$("#tab-overview").hidden) renderDetections(); }, 20000);

document.addEventListener("click", (ev) => {
  const link = ev.target.closest("[data-goto]");
  if (link) { ev.preventDefault(); showTab(link.dataset.goto); }
});

/* ---------- wake words ---------- */
function renderModels() {
  const json = JSON.stringify([state.models, state.settings.threshold, state.settings.trigger_level]);
  if (busy || json === lastModelsJson) return;
  lastModelsJson = json;
  $("#models-hint").innerHTML = `Enabled wake words are offered to Home Assistant. Custom models live in <code>${esc(state.custom_dir)}</code>.`;
  $("#models").innerHTML = state.models.map((m) => {
    const custom = m.threshold != null || m.trigger_level != null;
    const thr = m.threshold != null ? m.threshold : state.settings.threshold;
    const trig = m.trigger_level != null ? m.trigger_level : state.settings.trigger_level;
    return `
    <div class="card model ${m.enabled && m.usable ? "" : "off"}" data-id="${esc(m.id)}">
      <div class="model-head">
        <div><h3>${esc(m.phrase)}</h3><div class="model-id">${esc(m.id)}</div></div>
        <label class="switch" title="${m.enabled ? "Enabled" : "Disabled"}">
          <input type="checkbox" data-act="enable" ${m.enabled ? "checked" : ""} ${m.usable ? "" : "disabled"} aria-label="Enable ${esc(m.phrase)}"><span></span></label>
      </div>
      <div class="model-meta">
        <span class="tag ${m.source === "custom" ? "accent" : ""}">${m.source === "custom" ? "Custom" : "Built-in"}</span>
        <span>${esc(m.framework)} · ${sizeText(m.size)}</span>
      </div>
      ${m.trained && m.trained.metrics ? `<div class="muted">Made here · recognised ${Math.round(m.trained.metrics.recall * 100)}% of test voices · ${fmt(m.trained.metrics.false_per_hour, 1)} false activations per hour in testing</div>` : ""}
      ${m.usable ? "" : `<div class="warn">This .tflite model can't run on this system. Upload the .onnx version instead.</div>`}
      <label class="check"><input type="checkbox" data-act="custom" ${custom ? "checked" : ""}> Custom sensitivity</label>
      ${custom ? `<div class="tune">
        <span>Threshold</span><input type="range" data-act="threshold" min="0.05" max="0.95" step="0.05" value="${thr}" aria-label="Threshold for ${esc(m.phrase)}"><output>${fmt(thr)}</output>
        <span>Trigger level</span><input type="range" data-act="trigger_level" min="1" max="10" step="1" value="${trig}" aria-label="Trigger level for ${esc(m.phrase)}"><output>${trig}</output>
      </div>` : `<div class="muted">Using the defaults: threshold ${fmt(state.settings.threshold)}, trigger level ${state.settings.trigger_level}</div>`}
      <div class="model-actions">
        <button class="btn" data-act="test" ${m.usable ? "" : "disabled"}>Try it</button>
        ${m.source === "custom" ? `<button class="btn ghost danger" data-act="delete">Delete</button>` : ""}
      </div>
    </div>`;
  }).join("") || `<p class="empty">No wake word models found.</p>`;
}

async function saveModel(id, changes) {
  try { applyState(await api("PUT", `api/models/${encodeURIComponent(id)}`, changes)); }
  catch (err) { toast(err.message, true); lastModelsJson = ""; renderModels(); }
}

const modelsBox = $("#models");
modelsBox.addEventListener("input", (ev) => {
  if (ev.target.type === "range") {
    busy = true;
    ev.target.nextElementSibling.textContent = ev.target.dataset.act === "threshold" ? fmt(ev.target.value) : ev.target.value;
  }
});
modelsBox.addEventListener("change", (ev) => {
  const card = ev.target.closest(".model");
  if (!card) return;
  const id = card.dataset.id;
  const act = ev.target.dataset.act;
  busy = false;
  if (act === "enable") saveModel(id, { enabled: ev.target.checked });
  else if (act === "custom") {
    saveModel(id, ev.target.checked
      ? { threshold: state.settings.threshold, trigger_level: state.settings.trigger_level }
      : { threshold: null, trigger_level: null });
  } else if (act === "threshold" || act === "trigger_level") saveModel(id, { [act]: Number(ev.target.value) });
});
modelsBox.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.closest(".model").dataset.id;
  if (btn.dataset.act === "test") {
    if (test.mode === "idle") { test.selected = new Set([id]); renderTestChips(); }
    showTab("test");
  } else if (btn.dataset.act === "delete") {
    if (!confirm(`Delete the custom wake word "${phraseOf(id)}"? The model file will be removed.`)) return;
    try { applyState(await api("DELETE", `api/models/${encodeURIComponent(id)}`)); toast("Wake word deleted"); }
    catch (err) { toast(err.message, true); }
  }
});

const dropzone = $("#dropzone");
const modelInput = $("#file-model");
async function uploadModels(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("file", f, f.name);
  dropzone.classList.add("busy");
  try {
    const result = await api("POST", "api/models", form);
    if (result.state) applyState(result.state);
    if (result.added && result.added.length) toast(`Added ${result.added.map(phraseOf).join(", ")}`);
    for (const e of result.errors || []) toast(e, true);
  } catch (err) { toast(err.message, true); }
  dropzone.classList.remove("busy");
  modelInput.value = "";
}
dropzone.addEventListener("click", () => modelInput.click());
dropzone.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); modelInput.click(); } });
modelInput.addEventListener("click", (ev) => ev.stopPropagation());
modelInput.addEventListener("change", () => uploadModels([...modelInput.files]));
["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, (ev) => { ev.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, (ev) => { ev.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (ev) => uploadModels([...ev.dataTransfer.files]));
$("#btn-rescan").addEventListener("click", async () => {
  try { applyState(await api("POST", "api/rescan")); toast("Folder rescanned"); } catch (err) { toast(err.message, true); }
});

/* ---------- settings ---------- */
const settingText = (key, v) => key === "trigger_level" ? String(v)
  : key === "refractory_seconds" ? `${fmt(v, 1)} s`
  : key === "vad_threshold" && Number(v) === 0 ? "Off" : fmt(v);

function renderSettings() {
  if (busy) return;
  for (const input of $$("#tab-settings input[type=range]")) {
    const value = state.settings[input.dataset.key];
    input.value = value;
    input.nextElementSibling.textContent = settingText(input.dataset.key, value);
  }
  const ns = $("#set-ns");
  ns.checked = state.settings.noise_suppression;
  ns.disabled = !state.capabilities.noise_suppression;
  if (!state.capabilities.noise_suppression) $("#ns-help").textContent = "Not available: the Speex noise suppression library is not installed on this system.";
}
async function saveSettings(changes) {
  try { applyState(await api("PUT", "api/settings", changes)); toast("Saved"); }
  catch (err) { toast(err.message, true); renderSettings(); }
}
for (const input of $$("#tab-settings input[type=range]")) {
  input.addEventListener("input", () => { busy = true; input.nextElementSibling.textContent = settingText(input.dataset.key, input.value); drawChart(); });
  input.addEventListener("change", () => { busy = false; saveSettings({ [input.dataset.key]: Number(input.value) }); });
}
$("#set-ns").addEventListener("change", (ev) => saveSettings({ noise_suppression: ev.target.checked }));

/* ---------- test ---------- */
const WINDOW_FRAMES = 150; // 12 s of 80 ms frames in microphone mode
const test = {
  mode: "idle",          // idle | mic | file
  selected: null,        // Set of model ids (null = follow enabled models)
  ws: null,
  frames: [],            // [{scores, detections}]
  labels: {},            // label -> model id
  colors: {},            // label -> color
  peaks: {},
  showAll: false,        // file mode: fit whole clip
  detections: [],
  audio: null,
};

const prettyLabel = (label) => label.replace(/_/g, " ").replace(/:(\d+)$/, " · class $1");
const labelName = (label) => (test.labels[label] === label || !test.labels[label]) ? phraseOf(label) : prettyLabel(label);

function selectedIds() {
  const usable = state.models.filter((m) => m.usable);
  if (test.selected) {
    const ids = usable.filter((m) => test.selected.has(m.id)).map((m) => m.id);
    if (ids.length) return ids;
  }
  const enabled = usable.filter((m) => m.enabled).map((m) => m.id);
  return enabled.length ? enabled : usable.slice(0, 1).map((m) => m.id);
}

function renderTestChips() {
  const chosen = new Set(selectedIds());
  $("#test-chips").innerHTML = state.models.filter((m) => m.usable).map((m) =>
    `<button class="chip" data-id="${esc(m.id)}" aria-pressed="${chosen.has(m.id)}" ${test.mode !== "idle" ? "disabled" : ""}>${esc(m.phrase)}</button>`).join("");
}
$("#test-chips").addEventListener("click", (ev) => {
  const chip = ev.target.closest(".chip");
  if (!chip || test.mode !== "idle") return;
  const chosen = new Set(selectedIds());
  if (chosen.has(chip.dataset.id)) { if (chosen.size > 1) chosen.delete(chip.dataset.id); }
  else chosen.add(chip.dataset.id);
  test.selected = chosen;
  renderTestChips();
});

function setTestStatus(text) { $("#test-status").textContent = text || ""; }

function openTestSocket(source) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("ws/test"));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", models: selectedIds(), source }));
    ws.onerror = () => reject(new Error("Could not connect to the add-on"));
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "started") {
        test.labels = msg.labels;
        test.colors = {};
        Object.keys(msg.labels).forEach((label, i) => (test.colors[label] = COLORS[i % COLORS.length]));
        test.frames = [];
        test.peaks = {};
        resolve(ws);
      } else if (msg.type === "frames") onTestFrames(msg.frames);
      else if (msg.type === "done") { if (test.onDone) test.onDone(); }
      else if (msg.type === "error") { toast(msg.error, true); reject(new Error(msg.error)); }
    };
    ws.onclose = () => { if (test.ws === ws && test.mode !== "idle") stopTest("Connection lost"); };
    test.ws = ws;
  });
}

let bannerTimer = null;
function onTestFrames(frames) {
  for (const f of frames) {
    test.frames.push(f);
    for (const [label, score] of Object.entries(f.scores)) {
      if (score > (test.peaks[label] || 0)) test.peaks[label] = score;
    }
    for (const d of f.detections || []) {
      const at = (test.frames.length * 0.08 - (test.mode === "file" ? 1 : 0));
      test.detections.unshift({ ...d, at, wall: Date.now() / 1000, mode: test.mode });
      const banner = $("#test-banner");
      banner.textContent = `Detected “${phraseOf(d.model)}”${d.label !== d.model ? ` (${prettyLabel(d.label)})` : ""} — score ${fmt(d.score)}`;
      banner.hidden = false;
      clearTimeout(bannerTimer);
      bannerTimer = setTimeout(() => (banner.hidden = true), 4000);
      renderTestDetections();
    }
  }
  if (test.mode === "mic" && test.frames.length > WINDOW_FRAMES * 2) test.frames.splice(0, test.frames.length - WINDOW_FRAMES);
  scheduleDraw();
}

function renderTestDetections() {
  const box = $("#test-detections");
  if (!test.detections.length) { box.innerHTML = `<p class="empty">Nothing detected yet.</p>`; return; }
  box.innerHTML = `<div class="table-wrap"><table><thead><tr><th>When</th><th>Wake word</th><th>Score</th></tr></thead><tbody>
    ${test.detections.slice(0, 30).map((d) => `<tr>
      <td>${d.mode === "file" ? `at ${fmt(Math.max(0, d.at), 1)} s in file` : esc(clock(d.wall))}</td>
      <td>${esc(phraseOf(d.model))}${d.label !== d.model ? ` <span class="muted">${esc(prettyLabel(d.label))}</span>` : ""}</td>
      <td>${fmt(d.score)}</td></tr>`).join("")}</tbody></table></div>`;
}
$("#btn-clear-test").addEventListener("click", () => { test.detections = []; renderTestDetections(); });

function setTestMode(mode) {
  test.mode = mode;
  const mic = $("#btn-mic");
  mic.textContent = mode === "mic" ? "Stop microphone" : "Start microphone";
  mic.classList.toggle("recording", mode === "mic");
  mic.disabled = mode === "file";
  $("#btn-file").disabled = mode !== "idle";
  $("#mic-meter").hidden = mode !== "mic";
  if (state) renderTestChips();
}

function stopTest(status) {
  if (test.audio) {
    test.audio.stream.getTracks().forEach((t) => t.stop());
    test.audio.ctx.close().catch(() => {});
    test.audio = null;
  }
  const ws = test.ws;
  test.ws = null;
  test.onDone = null;
  if (ws && ws.readyState <= 1) ws.close();
  setTestMode("idle");
  setTestStatus(status || "");
}

async function startMic() {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const note = $("#mic-note");
    note.hidden = false;
    note.innerHTML = `Your browser only allows microphone access on secure pages. Open Home Assistant through an <strong>https://</strong> address (for example your Home Assistant Cloud remote URL) to use the microphone here — or use <strong>Test an audio file</strong>, which works everywhere.`;
    return;
  }
  setTestMode("mic");
  setTestStatus("Waiting for microphone permission…");
  test.showAll = false;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  } catch (err) {
    stopTest();
    toast(err.name === "NotAllowedError" ? "Microphone permission was denied" : `Microphone error: ${err.message}`, true);
    return;
  }
  try {
    let ctx, source;
    try {
      ctx = new AudioContext({ sampleRate: 16000 });
      source = ctx.createMediaStreamSource(stream);
    } catch (err) { // some browsers cannot resample the microphone; the worklet will
      if (ctx) ctx.close().catch(() => {});
      ctx = new AudioContext();
      source = ctx.createMediaStreamSource(stream);
    }
    test.audio = { ctx, stream };
    await ctx.audioWorklet.addModule("static/recorder-worklet.js");
    const ws = await openTestSocket("mic");
    const node = new AudioWorkletNode(ctx, "recorder", { numberOfOutputs: 0 });
    const meter = $("#mic-meter > div");
    node.port.onmessage = (ev) => {
      if (ws.readyState !== 1) return;
      ws.send(ev.data);
      const pcm = new Int16Array(ev.data);
      let peak = 0;
      for (let i = 0; i < pcm.length; i += 8) peak = Math.max(peak, Math.abs(pcm[i]));
      meter.style.width = Math.min(100, Math.sqrt(peak / 32768) * 110) + "%";
    };
    source.connect(node);
    if (ctx.state === "suspended") await ctx.resume();
    setTestStatus("Listening — say a wake word");
  } catch (err) {
    if (!test.audio) stream.getTracks().forEach((t) => t.stop());
    stopTest();
    toast(`Could not start the test: ${err.message}`, true);
  }
}
$("#btn-mic").addEventListener("click", () => (test.mode === "mic" ? stopTest("Stopped") : startMic()));

async function decodeTo16k(file) {
  const data = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let decoded;
  try { decoded = await ctx.decodeAudioData(data); } finally { ctx.close().catch(() => {}); }
  if (decoded.duration > 600) throw new Error("Please use a recording shorter than 10 minutes");
  const frames = Math.max(1, Math.ceil(decoded.duration * 16000));
  const offline = new OfflineAudioContext(1, frames, 16000);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const samples = (await offline.startRendering()).getChannelData(0);
  const pad = 16000; // 1 s of silence on both sides so short clips are scored properly
  const pcm = new Int16Array(samples.length + pad * 2);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    pcm[i + pad] = v < 0 ? v * 32768 : v * 32767;
  }
  return pcm;
}

async function testFile(file) {
  setTestMode("file");
  setTestStatus(`Reading ${file.name}…`);
  try {
    let pcm;
    try { pcm = await decodeTo16k(file); }
    catch (err) { throw new Error(err.message && err.message.startsWith("Please") ? err.message : "This file could not be read as audio"); }
    const ws = await openTestSocket("file");
    test.showAll = true;
    setTestStatus(`Analysing ${file.name}…`);
    const done = new Promise((resolve) => (test.onDone = resolve));
    for (let i = 0; i < pcm.length; i += 16000) {
      ws.send(pcm.subarray(i, i + 16000));
      while (ws.bufferedAmount > 1 << 20) await new Promise((r) => setTimeout(r, 20));
    }
    ws.send(JSON.stringify({ type: "end" }));
    await done;
    const count = test.frames.reduce((n, f) => n + (f.detections ? f.detections.length : 0), 0);
    stopTest(`${file.name}: ${count ? `${count} detection${count > 1 ? "s" : ""}` : "no detection"} in ${fmt(pcm.length / 16000 - 2, 1)} s of audio`);
  } catch (err) {
    stopTest();
    toast(err.message, true);
  }
}
$("#btn-file").addEventListener("click", () => $("#file-audio").click());
$("#file-audio").addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  ev.target.value = "";
  if (file) testFile(file);
});

/* ---------- create ---------- */
const create = { samples: [], recorder: null, languagesLoaded: false, timer: null, lastState: null, shownResult: null };
const phraseInput = $("#create-phrase");
const currentPhrase = () => phraseInput.value.trim().replace(/\s+/g, " ");
const sampleUrl = (name = "") => `api/training/samples${name ? "/" + encodeURIComponent(name) : ""}?phrase=${encodeURIComponent(currentPhrase())}`;

async function openCreateTab() {
  if (create.languagesLoaded) return;
  loadSamples();
  create.languagesLoaded = true;
  try {
    const data = await api("GET", "api/training/languages");
    let saved = "en_US";
    try { saved = localStorage.getItem("language") || saved; } catch (e) { /* ignore */ }
    const select = $("#create-language");
    select.innerHTML = data.languages.map((l) => `<option value="${esc(l.code)}">${esc(l.name)}</option>`).join("");
    if (data.languages.some((l) => l.code === saved)) select.value = saved;
    if (!data.online) $("#create-language-hint").textContent = "The list of languages could not be loaded — training needs internet access (huggingface.co).";
  } catch (err) {
    create.languagesLoaded = false;
    if (state && !state.can_train) $("#btn-train").disabled = true;
  }
}
$("#create-language").addEventListener("change", (ev) => { try { localStorage.setItem("language", ev.target.value); } catch (e) { /* ignore */ } });

function renderSamples() {
  $("#samples").innerHTML = create.samples.map((s, i) => `
    <span class="sample" data-name="${esc(s.name)}"><button data-act="play" title="Play" aria-label="Play recording ${i + 1}">▶</button>${fmt(s.seconds, 1)} s<button data-act="delete" title="Delete" aria-label="Delete recording ${i + 1}">✕</button></span>`).join("");
  $("#record-status").textContent = create.samples.length ? `${create.samples.length} recording${create.samples.length > 1 ? "s" : ""}` : "";
}
async function loadSamples() {
  if (currentPhrase().length < 3) { create.samples = []; renderSamples(); return; }
  try { create.samples = (await api("GET", sampleUrl())).samples; } catch (e) { create.samples = []; }
  renderSamples();
}
let phraseTimer = null;
phraseInput.addEventListener("input", () => {
  clearTimeout(phraseTimer);
  phraseTimer = setTimeout(() => {
    try { localStorage.setItem("phrase", currentPhrase()); } catch (e) { /* ignore */ }
    loadSamples();
  }, 400);
});
try { phraseInput.value = localStorage.getItem("phrase") || ""; } catch (e) { /* ignore */ }

$("#samples").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button");
  if (!btn) return;
  const name = btn.closest(".sample").dataset.name;
  if (btn.dataset.act === "play") new Audio(sampleUrl(name)).play().catch(() => toast("Could not play the recording", true));
  else {
    try { create.samples = (await api("DELETE", sampleUrl(name))).samples; renderSamples(); }
    catch (err) { toast(err.message, true); }
  }
});

function needPhrase() {
  if (currentPhrase().length >= 3) return true;
  toast("Type the wake word first", true);
  phraseInput.focus();
  return false;
}
async function sendSample(pcm) {
  try { create.samples = (await api("POST", sampleUrl(), pcm)).samples; renderSamples(); return true; }
  catch (err) { toast(err.message, true); return false; }
}

async function stopRecording(save) {
  const rec = create.recorder;
  if (!rec) return;
  create.recorder = null;
  clearTimeout(rec.timeout);
  rec.stream.getTracks().forEach((t) => t.stop());
  rec.ctx.close().catch(() => {});
  const btn = $("#btn-record");
  btn.classList.remove("recording");
  btn.textContent = "Record a sample";
  if (!save) return;
  const pcm = new Int16Array(rec.chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of rec.chunks) { pcm.set(c, offset); offset += c.length; }
  await sendSample(pcm);
}

async function startRecording() {
  if (!needPhrase()) return;
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("Recording needs Home Assistant to be opened over https://. Use “Add audio files” instead.", true);
    return;
  }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } }); }
  catch (err) { toast(err.name === "NotAllowedError" ? "Microphone permission was denied" : `Microphone error: ${err.message}`, true); return; }
  try {
    let ctx, source;
    try { ctx = new AudioContext({ sampleRate: 16000 }); source = ctx.createMediaStreamSource(stream); }
    catch (err) { if (ctx) ctx.close().catch(() => {}); ctx = new AudioContext(); source = ctx.createMediaStreamSource(stream); }
    await ctx.audioWorklet.addModule("static/recorder-worklet.js");
    const node = new AudioWorkletNode(ctx, "recorder", { numberOfOutputs: 0 });
    const rec = { stream, ctx, chunks: [] };
    node.port.onmessage = (ev) => rec.chunks.push(new Int16Array(ev.data));
    source.connect(node);
    if (ctx.state === "suspended") await ctx.resume();
    rec.timeout = setTimeout(() => stopRecording(true), 3000);
    create.recorder = rec;
    const btn = $("#btn-record");
    btn.classList.add("recording");
    btn.textContent = "Say it now… (click to finish)";
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    toast(`Could not record: ${err.message}`, true);
  }
}
$("#btn-record").addEventListener("click", () => (create.recorder ? stopRecording(true) : startRecording()));
$("#btn-sample-file").addEventListener("click", () => { if (needPhrase()) $("#file-sample").click(); });
$("#file-sample").addEventListener("change", async (ev) => {
  const files = [...ev.target.files];
  ev.target.value = "";
  for (const file of files) {
    try { await sendSample(await decodeTo16k(file)); }
    catch (err) { toast(`${file.name}: could not be read as audio`, true); }
  }
});

$("#btn-train").addEventListener("click", async () => {
  if (!needPhrase()) return;
  if (create.recorder) await stopRecording(true);
  const quality = $("input[name=quality]:checked").value;
  try {
    renderTraining(await api("POST", "api/training/start", { phrase: currentPhrase(), language: $("#create-language").value, quality }));
  } catch (err) { toast(err.message, true); }
});
$("#btn-cancel-train").addEventListener("click", async () => {
  if (!confirm("Stop the training?")) return;
  try { await api("POST", "api/training/cancel"); } catch (err) { toast(err.message, true); }
});

const duration = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function renderTraining(status) {
  if (!status) return;
  const running = status.state === "running";
  $("#create-form").hidden = running;
  $("#create-progress").hidden = !running;
  clearInterval(create.timer);
  if (running) {
    $("#progress-title").textContent = `Training “${status.phrase}”…`;
    $("#progress-bar").style.width = Math.round(status.progress * 100) + "%";
    $("#progress-stage").textContent = `${status.stage} · ${Math.round(status.progress * 100)}%`;
    $("#progress-log").innerHTML = status.log.slice(-6).map((l) => `<li>${esc(l)}</li>`).join("");
    const tick = () => ($("#progress-time").textContent = duration(Date.now() / 1000 - status.started));
    tick();
    create.timer = setInterval(tick, 1000);
  }
  const box = $("#create-result");
  const key = `${status.state}:${status.finished}`;
  if (running || status.state === "idle" || status.state === "cancelled") { box.hidden = true; }
  else if (create.shownResult !== key) {
    create.shownResult = key;
    box.hidden = false;
    if (status.state === "error") {
      box.innerHTML = `<div class="card-head"><h2>Training “${esc(status.phrase)}” did not finish</h2></div><div class="warn">${esc(status.error)}</div>`;
    } else {
      const m = status.result.metrics;
      box.innerHTML = `
        <div class="card-head"><h2>“${esc(status.result.phrase)}” is ready</h2><span class="muted">trained in ${duration(status.finished - status.started)}</span></div>
        <div class="result-stats">
          <div><b>${Math.round(m.recall * 100)}%</b><span>of unseen test voices recognised</span></div>
          ${m.user_recall != null ? `<div><b>${Math.round(m.user_recall * 100)}%</b><span>of your held-back recordings recognised</span></div>` : ""}
          <div><b>${fmt(m.false_per_hour, 1)}</b><span>false activations per hour on ${fmt(m.held_out_hours, 1)} h of speech, music and noise</span></div>
          <div><b>${Math.round(m.similar_rejected * 100)}%</b><span>of similar-sounding phrases ignored</span></div>
        </div>
        <p class="hint">These numbers come from computer-generated voices and recorded background audio, so real life will differ — try it yourself. The threshold for this wake word was set to <strong>${fmt(m.recommended_threshold)}</strong>; you can change it on its card. ${m.user_recordings ? "" : "Adding recordings of your own voice and training again usually helps."}</p>
        <div class="actions">
          <button class="btn primary" data-try="${esc(status.result.model_id)}">Try it now</button>
          <button class="btn" data-goto="models">Show wake words</button>
        </div>
        <p class="hint">To use it with your voice assistant, reload the Wyoming integration in Home Assistant, then pick it as wake word (see the setup guide).</p>`;
    }
  }
  if (create.lastState === "running" && status.state === "cancelled") toast("Training cancelled");
  create.lastState = status.state;
}
$("#create-result").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-try]");
  if (!btn) return;
  if (test.mode === "idle") { test.selected = new Set([btn.dataset.try]); renderTestChips(); }
  showTab("test");
});

/* ---------- chart ---------- */
let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; drawChart(); });
}

function drawChart() {
  const canvas = $("#chart");
  if (!state || $("#tab-test").hidden) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const cMuted = css.getPropertyValue("--muted").trim();
  const cBorder = css.getPropertyValue("--border").trim();
  const cText = css.getPropertyValue("--text").trim();
  const cOk = css.getPropertyValue("--ok").trim();
  const L = 34, R = 10, T = 12, B = 22;
  const pw = w - L - R, ph = h - T - B;
  const y = (v) => T + ph * (1 - Math.max(0, Math.min(1, v)));

  g.font = "11px system-ui, sans-serif";
  g.textBaseline = "middle";
  g.lineWidth = 1;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    g.strokeStyle = cBorder; g.beginPath(); g.moveTo(L, y(v) + 0.5); g.lineTo(w - R, y(v) + 0.5); g.stroke();
    g.fillStyle = cMuted; g.textAlign = "right"; g.fillText(v.toFixed(2), L - 6, y(v));
  }

  const total = test.frames.length;
  const span = test.showAll ? Math.max(total, 25) : WINDOW_FRAMES;
  const first = test.showAll ? 0 : Math.max(0, total - WINDOW_FRAMES);
  const offset = test.showAll ? 0 : Math.max(0, WINDOW_FRAMES - total); // right-align live data
  const x = (i) => L + pw * ((i - first + offset) / Math.max(1, span - 1));

  // time axis
  g.textAlign = "center"; g.fillStyle = cMuted;
  if (test.showAll && total) {
    const seconds = total * 0.08;
    const step = [0.5, 1, 2, 5, 10, 30, 60, 120].find((s) => (seconds - 2) / s <= (pw < 500 ? 4 : 8)) || 120;
    // the clip is padded with 1 s of silence on each side; label positions in clip time
    for (let s = 0; s <= seconds - 2; s += step) g.fillText(`${s}s`, x((s + 1) / 0.08), h - 9);
  } else {
    for (const s of [12, 9, 6, 3]) g.fillText(`-${s}s`, L + pw * (1 - s / 12), h - 9);
    g.fillText("now", w - R - 10, h - 9);
  }

  // thresholds
  const draft = $("#set-threshold");
  const globalThr = busy && draft ? Number(draft.value) : state.settings.threshold;
  g.setLineDash([5, 4]);
  g.strokeStyle = cText; g.globalAlpha = 0.6;
  g.beginPath(); g.moveTo(L, y(globalThr)); g.lineTo(w - R, y(globalThr)); g.stroke();
  g.globalAlpha = 1;
  g.fillStyle = cText; g.textAlign = "left"; g.textBaseline = "bottom";
  g.fillText("threshold", L + 4, y(globalThr) - 2);
  const models = new Set(Object.values(test.labels));
  for (const id of models) {
    const m = modelById(id);
    if (!m || m.threshold == null || m.threshold === globalThr) continue;
    const label = Object.keys(test.labels).find((l) => test.labels[l] === id);
    g.strokeStyle = test.colors[label] || cMuted;
    g.beginPath(); g.moveTo(L, y(m.threshold)); g.lineTo(w - R, y(m.threshold)); g.stroke();
  }
  g.setLineDash([]);
  g.textBaseline = "middle";

  if (!total) {
    g.fillStyle = cMuted; g.textAlign = "center";
    g.font = "14px system-ui, sans-serif";
    g.fillText(pw < 460 ? "Scores will appear here" : "Start the microphone or choose an audio file to see scores here", L + pw / 2, T + ph * 0.3);
    renderLegend();
    return;
  }

  // score lines
  g.lineWidth = 2; g.lineJoin = "round";
  for (const label of Object.keys(test.labels)) {
    g.strokeStyle = test.colors[label];
    g.beginPath();
    for (let i = first; i < total; i++) {
      const v = test.frames[i].scores[label] || 0;
      if (i === first) g.moveTo(x(i), y(v)); else g.lineTo(x(i), y(v));
    }
    g.stroke();
  }
  // detections
  for (let i = first; i < total; i++) {
    for (const d of test.frames[i].detections || []) {
      g.strokeStyle = cOk; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x(i), T); g.lineTo(x(i), T + ph); g.stroke();
      g.fillStyle = cOk; g.beginPath(); g.arc(x(i), y(d.score), 5, 0, Math.PI * 2); g.fill();
    }
  }
  renderLegend();
}

function renderLegend() {
  const last = test.frames[test.frames.length - 1];
  const labels = Object.keys(test.labels);
  $("#legend").innerHTML = labels.map((label) => `
    <div class="legend-row"><i style="background:${test.colors[label]}"></i><span title="${esc(label)}">${esc(labelName(label))}</span>
      <em>now <b>${fmt(last ? last.scores[label] || 0 : 0)}</b> · peak <b>${fmt(test.peaks[label] || 0)}</b></em></div>`).join("");
}
window.addEventListener("resize", scheduleDraw);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleDraw);

/* ---------- start ---------- */
(async function init() {
  let tab = "overview";
  try { tab = sessionStorage.getItem("tab") || tab; } catch (e) { /* ignore */ }
  try { applyState(await api("GET", "api/state")); }
  catch (err) { toast(`Could not load: ${err.message}`, true); }
  showTab($(`#tab-${tab}`) ? tab : "overview");
  connectEvents();
})();
