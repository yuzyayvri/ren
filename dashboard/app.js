"use strict";
/* ren workstation: dependency-free canvas viewer + inspector panels. */
const S = {
  specimens: [], index: 0, img: null, imgURL: null,
  overlayImg: null, boxes: [], instances: [],
  view: {scale: 1, ox: 0, oy: 0}, showOverlay: true,
  evidence: [], packet: null, note: null, validated: false, jobTimer: null,
};
const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.clone().json()).detail || ""; } catch { /* non-JSON error */ }
    throw new Error(`${opts?.method || "GET"} ${path}: ${r.status}${detail ? ` — ${detail}` : ""}`);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.blob();
};

function fitView() {
  const c = $("canvas"), r = c.getBoundingClientRect();
  if (!S.img || !S.img.width) return;
  S.view.scale = Math.min(r.width / S.img.width, r.height / S.img.height) * 0.98;
  S.view.ox = (r.width - S.img.width * S.view.scale) / 2;
  S.view.oy = (r.height - S.img.height * S.view.scale) / 2;
  draw();
}
function draw() {
  const c = $("canvas"), ctx = c.getContext("2d");
  const r = c.getBoundingClientRect();
  c.width = r.width; c.height = r.height;
  ctx.fillStyle = "#101214"; ctx.fillRect(0, 0, c.width, c.height);
  if (!S.img) return;
  const {scale: s, ox, oy} = S.view, w = S.img.width * s, h = S.img.height * s;
  ctx.imageSmoothingEnabled = s < 2;
  ctx.drawImage(S.img, ox, oy, w, h);
  if (S.showOverlay && S.overlayImg?.width) ctx.drawImage(S.overlayImg, ox, oy, w, h);
  if (S.highlight) {
    const [x0, y0, x1, y1] = S.highlight;
    ctx.strokeStyle = "rgba(127,166,201,1)"; ctx.lineWidth = 2.5;
    ctx.strokeRect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s);
  }
  if (S.showOverlay) {
    ctx.strokeStyle = "rgba(230,57,70,0.9)"; ctx.lineWidth = 1.2;
    for (const b of S.boxes) {
      const [x0, y0, x1, y1] = b.bbox;
      ctx.strokeRect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s);
    }
  }
  const m = $("minimap"), mctx = m.getContext("2d");
  mctx.fillStyle = "#000"; mctx.fillRect(0, 0, 150, 150);
  const ms = Math.min(150 / S.img.width, 150 / S.img.height);
  mctx.drawImage(S.img, 0, 0, S.img.width * ms, S.img.height * ms);
  mctx.strokeStyle = "#7fa6c9";
  mctx.strokeRect(Math.max(0, -ox / s * ms), Math.max(0, -oy / s * ms),
    Math.min(150, r.width / s * ms), Math.min(150, r.height / s * ms));
}
function bindCanvas() {
  const c = $("canvas");
  let drag = null;
  c.addEventListener("mousedown", (e) => { drag = [e.clientX, e.clientY]; c.style.cursor = "grabbing"; });
  addEventListener("mouseup", () => { drag = null; c.style.cursor = "grab"; });
  c.addEventListener("mousemove", (e) => {
    if (!drag) return;
    S.view.ox += e.clientX - drag[0]; S.view.oy += e.clientY - drag[1];
    drag = [e.clientX, e.clientY]; draw();
  });
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = c.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const ns = Math.min(40, Math.max(0.05, S.view.scale * f));
    S.view.ox = mx - (mx - S.view.ox) * (ns / S.view.scale);
    S.view.oy = my - (my - S.view.oy) * (ns / S.view.scale);
    S.view.scale = ns; draw();
  }, {passive: false});
  addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === ",") step(-1); if (e.key === ".") step(1);
    if (e.key === "o") toggleOverlay(); if (e.key === "0") fitView();
  });
  addEventListener("resize", () => draw());
}
async function loadSpecimens() {
  S.specimens = await api("/api/specimens?limit=200");
  const sel = $("specimens");
  sel.innerHTML = "";
  S.specimens.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${s.source}: ${s.name}`;
    sel.appendChild(o);
  });
  sel.onchange = () => loadIndex(+sel.value);
  loadIndex(0);
}
$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));
$("overlay-toggle").addEventListener("click", toggleOverlay);
function step(d) {
  const n = (S.index + d + S.specimens.length) % S.specimens.length;
  $("specimens").value = n; loadIndex(n);
}
async function loadIndex(i) {
  S.index = i;
  const spec = S.specimens[i];
  $("specimen-label").textContent = spec.id;
  $("st-spec").textContent = `${spec.id} (${spec.kind})`;
  URL.revokeObjectURL(S.imgURL);
  S.imgURL = URL.createObjectURL(await api(`/api/specimens/${spec.id}/image`));
  S.img = await new Promise((res, rej) => {
    const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = S.imgURL;
  });
  S.overlayImg = null; S.boxes = []; S.instances = []; S.highlight = null;
  if (!S.reviewMode) S.reviewMode = "manual";
  S.isV1 = /^[0-9a-f]{12}$/.test(spec.id);
  $("analyze").style.display = S.isV1 ? "" : "none";
  $("import-note").textContent = "";
  try {
    const ov = await api(`/api/specimens/${spec.id}/overlays`);
    if (ov.overlay_url) {
      const url = URL.createObjectURL(await api(ov.overlay_url));
      S.overlayImg = await new Promise((res, rej) => {
        const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url;
      });
    }
    S.boxes = ov.boxes || []; S.instances = ov.instances || [];
  S.boxCodes = {};
  for (const b of S.boxes) S.boxCodes[b.label] = b.code || b.label;
    renderFindings(ov.counts || {});
  } catch { renderFindings({}); }
  fitView();
}
function renderFindings(counts) {
  const el = $("findings");
  const rows = Object.entries(counts).filter(([, n]) => n > 0);
  el.innerHTML = rows.length
    ? `<table>${rows.map(([k, n]) => `<tr><td>${k}</td><td class="num">${n}</td></tr>`).join("")}</table>`
    : `<span class="dim">no overlay findings for this specimen</span>`;
  el.dataset.counts = JSON.stringify(counts);
  if (S.isV1) loadV1Findings();
}
async function refreshV1Overlays() {
  const id = S.specimens[S.index].id;
  try {
    const ov = await api(`/api/specimens/${id}/overlays`);
    S.boxes = ov.boxes || [];
    draw();
  } catch { /* overlay refresh is best-effort; list state is authoritative */ }
}
function reviewModeBar(unreviewed) {
  return `<div class="dim">mode:
    <button class="action${S.reviewMode === "manual" ? " primary" : ""}" id="mode-manual">Manual Approve</button>
    <button class="action${S.reviewMode === "auto" ? " primary" : ""}" id="mode-auto">Approve For Me</button></div>` +
    (S.reviewMode === "auto" && unreviewed > 0
      ? `<div><button class="action primary" id="approve-all">approve all ${unreviewed} unreviewed</button></div>` : "");
}
function bindReviewModeBar() {
  const manual = $("mode-manual"), auto = $("mode-auto"), all = $("approve-all");
  if (manual) manual.addEventListener("click", () => { S.reviewMode = "manual"; loadV1Findings(); });
  if (auto) auto.addEventListener("click", () => { S.reviewMode = "auto"; loadV1Findings(); });
  if (all) all.addEventListener("click", async () => {
    const id = S.specimens[S.index].id;
    const r = await api(`/api/v1/reviews/bulk`, {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({specimen_id: id, action: "confirm", reviewer: "workstation"})});
    $("st-job").textContent = `auto-approved ${r.confirmed.length}`;
    await loadV1Findings();
    refreshV1Overlays();
  });
}
async function loadV1Findings() {
  const id = S.specimens[S.index].id;
  const el = $("findings");
  const side = $("side");
  const scrollTop = side ? side.scrollTop : 0;
  try {
    const r = await api(`/api/v1/findings/${id}`);
    el.innerHTML = r.findings.map((f) =>
      `<div class="ev" data-fid="${f.finding_id}">
        <span class="mono">${f.finding_id}</span> <b>${f.label}</b>
        <span class="dim">${Math.round((f.confidence || 0) * 100)}% · ${f.review_state}</span><br>
        <button class="action" data-act="confirm" data-fid="${f.finding_id}">confirm</button>
        <button class="action" data-act="reject" data-fid="${f.finding_id}">reject</button>
      </div>`).join("") || `<span class="dim">no findings</span>`;
    if (side) side.scrollTop = scrollTop;
    const unreviewed = r.findings.filter((f) => f.review_state === "unreviewed").length;
    el.innerHTML = reviewModeBar(unreviewed) + el.innerHTML;
    bindReviewModeBar();
    el.querySelectorAll("button[data-act]").forEach((b) => b.addEventListener("click", async () => {
      await api(`/api/v1/reviews`, {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({specimen_id: id, finding_id: b.dataset.fid, action: b.dataset.act, reviewer: "workstation"})});
      await loadV1Findings();
      refreshV1Overlays();
    }));
  } catch (e) {
    el.innerHTML += `<div><button class="action primary" id="analyze-btn">analyze specimen</button>
      <span class="dim">${e.message}</span></div>`;
    const btn = $("analyze-btn");
    if (btn) btn.addEventListener("click", analyzeSpecimen);
  }
}
async function analyzeSpecimen() {
  const id = S.specimens[S.index].id;
  $("st-job").textContent = "analyzing…";
  const {job_id} = await api(`/api/v1/analyze/${id}`, {method: "POST"});
  const timer = setInterval(async () => {
    const j = await api(`/api/jobs/${job_id}`);
    $("st-job").textContent = `analyze: ${j.state}`;
    if (j.state === "done" || j.state === "failed" || j.state === "cancelled") {
      clearInterval(timer);
      if (j.state === "done") loadIndex(S.index);
      else $("st-job").textContent = `analyze failed: ${j.error || ""}`;
    }
  }, 1500);
}
function toggleOverlay() {
  S.showOverlay = !S.showOverlay;
  $("overlay-toggle").textContent = `overlays: ${S.showOverlay ? "on" : "off"}`;
  draw();
}
document.querySelectorAll("section h2").forEach((h) =>
  h.addEventListener("click", () => h.parentElement.classList.toggle("collapsed")));

$("import-btn").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", async () => {
  const file = $("import-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await (await fetch("/api/specimens/import", {method: "POST", body: form})).json();
    if (!r.specimen_id) throw new Error(r.detail || "rejected");
    $("import-note").textContent = `imported ${r.specimen_id}`;
    await loadSpecimens();
    const at = S.specimens.findIndex((s) => s.id === r.specimen_id);
    if (at >= 0) { $("specimens").value = at; loadIndex(at); }
  } catch (e) { $("import-note").textContent = `import failed`; }
});
$("analyze").addEventListener("click", analyzeSpecimen);
$("retrieve").addEventListener("click", async () => {
  const q = $("query").value.trim();
  if (!q) return;
  $("evidence").innerHTML = `<span class="dim">retrieving…</span>`;
  try {
    const r = await api("/api/retrieve", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query: q, mode: "hybrid"})});
    S.lastRetrieval = r;
    $("evidence").innerHTML = r.entries.slice(0, 8).map((e, i) =>
      `<div class="ev"><label><input type="checkbox" data-idx="${i}" ${i < 3 ? "checked" : ""}>
       <span class="mono">${e.go_id}</span> ${e.name}</label>
       <div class="dim">${e.definition.slice(0, 160)}${e.definition.length > 160 ? "…" : ""}</div></div>`).join("") +
      `<div class="dim">mode ${r.mode}; rank order preserved</div>`;
  } catch (e) { $("evidence").innerHTML = `<span class="bad">retrieval failed: ${e.message}</span>`; }
});
async function v1RetrieveAll() {
  const id = S.specimens[S.index].id;
  const r = await api(`/api/v1/retrieve/${id}`, {method: "POST"});
  $("evidence").innerHTML =
    `<div class="dim">automatic derivation (v1-query-derivation-v1), ${r.evidence} items</div>` +
    Object.entries(r.sets || {}).map(([fid, g]) =>
      `<div class="ev"><span class="mono">${fid}</span> <span class="dim">query: ${g.query}</span><br>` +
      g.evidence.map((e) => `<span class="mono">${e.evidence_id}</span> ${e.go_id} <span class="dim">[auto]</span>`).join("<br>") +
      `</div>`).join("");
}
async function v1Synthesize() {
  const id = S.specimens[S.index].id;
  const {job_id} = await api(`/api/v1/synthesize/${id}`, {method: "POST"});
  clearInterval(S.jobTimer);
  S.jobTimer = setInterval(async () => {
    const j = await api(`/api/jobs/${job_id}`);
    $("job").textContent = `${j.state} (${Math.round((j.progress || 0) * 100)}%)`;
    if (j.state === "done" || j.state === "failed" || j.state === "cancelled") {
      clearInterval(S.jobTimer);
      if (j.state === "done") {
        S.note = j.result.note; S.validated = j.result.validated;
        $("note").textContent = j.result.note;
        $("note").classList.add("provisional");
      } else $("job").textContent = `${j.state}: ${j.error || ""}`;
    }
  }, 1500);
}
$("synthesize").addEventListener("click", async () => {
  if (S.isV1) { v1RetrieveAll().then(v1Synthesize).catch((e) => { $("job").textContent = `v1 failed: ${e.message}`; }); return; }
  const counts = JSON.parse($("findings").dataset.counts || "{}");
  const labels = Object.entries(counts).filter(([, n]) => n > 0)
    .map(([k]) => (S.boxCodes && S.boxCodes[k]) || k);
  if (!labels.length) { $("job").textContent = "no findings to synthesize"; return; }
  const entries = S.lastRetrieval?.entries || [];
  const checked = [...document.querySelectorAll("#evidence input:checked")]
    .map((c) => entries[+c.dataset.idx]).filter(Boolean);
  const packet = {
    schema: "phase5-packet-v1", case_id: `ui-${S.specimens[S.index].id}`,
    source_stage: S.specimens[S.index].source === "pannuke" ? "phase2" : "phase3-txl",
    findings: labels.map((label, i) => ({finding_id: `F${i + 1}`, label, count: counts[label], qualifier: "observed"})),
    context: checked.map((e, i) => ({evidence_id: `E${i + 1}`, go_id: e.go_id, name: e.name, definition: e.definition,
      rank: e.rank, mode: e.mode, query: e.query})),
    limitations: [],
  };
  S.packet = packet;
  $("job").textContent = "queued…";
  try {
    const {job_id} = await api("/api/jobs/synthesize", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({packet})});
    clearInterval(S.jobTimer);
    S.jobTimer = setInterval(async () => {
      const j = await api(`/api/jobs/${job_id}`);
      $("job").textContent = `${j.state} (${Math.round((j.progress || 0) * 100)}%)`;
      $("st-job").textContent = `job ${job_id}: ${j.state}`;
      if (j.state === "done" || j.state === "failed" || j.state === "cancelled") {
        clearInterval(S.jobTimer);
        if (j.state === "done") {
          S.note = j.result.note; S.validated = true;
          $("note").textContent = j.result.note;
          $("note").classList.add("provisional");
        } else $("job").textContent = `${j.state}: ${j.error || ""}`;
      }
    }, 1000);
  } catch (e) { $("job").textContent = `submit failed: ${e.message}`; }
});
async function review(verdict, target, note) {
  try {
    const r = await api("/api/reviews", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({verdict, target, note, reviewer: "workstation"})});
    $("review-out").textContent = `${verdict} recorded (${r.id})`;
  } catch (e) { $("review-out").textContent = `review failed: ${e.message}`; }
}
document.querySelectorAll("#sec-review [data-verdict]").forEach((b) =>
  b.addEventListener("click", () => {
    const counts = JSON.parse($("findings").dataset.counts || "{}");
    review(b.dataset.verdict, {specimen: S.specimens[S.index]?.id, findings: counts}, null);
  }));
$("note").addEventListener("click", (e) => {
  const line = e.target.closest ? e.target.textContent : "";
  const match = /\[(C\d+)\]/.exec(e.target.textContent || "");
  if (!match || !S.validated) return;
  const claim = (S.validated.claims || []).find((c) => c.claim_id === match[1]);
  if (!claim || !claim.finding_ids.length) return;
  const box = S.boxes.find((b) => b.finding_id === claim.finding_ids[0]);
  if (box && box.bbox) { S.highlight = box.bbox; draw(); }
});
$("export-btn").addEventListener("click", async () => {
  if (!S.isV1) { $("review-out").textContent = "export covers v1 specimens"; return; }
  const id = S.specimens[S.index].id;
  const blob = await api(`/api/v1/export/${id}`);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = `${id}-export.json`; link.click();
  URL.revokeObjectURL(url);
  $("review-out").textContent = "export downloaded";
});
$("signoff").addEventListener("click", async () => {
  if (S.isV1) {
    const id = S.specimens[S.index].id;
    try {
      await api(`/api/v1/signoff/${id}`, {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({reviewer: "workstation"})});
      $("review-out").textContent = "signed off";
    } catch (e) { $("review-out").textContent = `sign-off failed: ${e.message}`; }
    return;
  }
  if (!S.note) { $("review-out").textContent = "no note to sign off"; return; }
  review("accept", {kind: "note", packet: S.packet,
    note_sha256: "(see synthesis record)"}, "human sign-off from workstation");
});
async function refreshProvenance() {
  try {
    const p = await api("/api/provenance");
    $("provenance").innerHTML =
      `<div class="mono">${Object.entries(p.sealed).map(([k, v]) => `${k}<br>${v.slice(0, 20)}…`).join("<br>")}</div>
       <div class="dim">models: ${p.models.map((m) => m.name).join(", ")}</div>`;
  } catch { $("provenance").innerHTML = `<span class="bad">provenance unavailable</span>`; }
}
async function serverStatus() {
  try {
    await api("/api/health");
    const s = await api("/api/server/status");
    const on = !!s.running;
    $("server-dot").className = `dot ${on ? "ok" : "bad"}`;
    $("server-label").textContent =
      `synthesis: ${on ? (s.managed ? "up" : "up (external)") : "down"}`;
    $("server-label").title = "click to toggle synthesis server";
  } catch { $("server-dot").className = "dot bad"; $("server-label").textContent = "backend: down"; }
}
$("server-label").addEventListener("click", async () => {
  const s = await api("/api/server/status");
  if (s.running && !s.managed) { serverStatus(); return; }
  await api(s.running ? "/api/server/stop" : "/api/server/start", {method: "POST"});
  serverStatus();
});
bindCanvas(); loadSpecimens(); refreshProvenance(); serverStatus();
setInterval(serverStatus, 15000);
