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
  if (!r.ok) throw new Error(`${opts?.method || "GET"} ${path}: ${r.status}`);
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
  // seed a few pannuke patches alongside blood smears
  for (let i = 0; i < 12; i++) S.specimens.push({id: `pannuke-f3-${i}`, kind: "tissue-patch", source: "pannuke", name: `fold3 patch ${i}`});
  S.specimens.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${s.source}: ${s.name}`;
    sel.appendChild(o);
  });
  sel.onchange = () => loadIndex(+sel.value);
  loadIndex(0);
}
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
  S.overlayImg = null; S.boxes = []; S.instances = [];
  try {
    const ov = await api(`/api/specimens/${spec.id}/overlays`);
    if (ov.overlay_url) {
      const url = URL.createObjectURL(await api(ov.overlay_url));
      S.overlayImg = await new Promise((res, rej) => {
        const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url;
      });
    }
    S.boxes = ov.boxes || []; S.instances = ov.instances || [];
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
}
function toggleOverlay() {
  S.showOverlay = !S.showOverlay;
  $("overlay-toggle").textContent = `overlays: ${S.showOverlay ? "on" : "off"}`;
  draw();
}
document.querySelectorAll("section h2").forEach((h) =>
  h.addEventListener("click", () => h.parentElement.classList.toggle("collapsed")));

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
$("synthesize").addEventListener("click", async () => {
  const counts = JSON.parse($("findings").dataset.counts || "{}");
  const labels = Object.entries(counts).filter(([, n]) => n > 0).map(([k]) => k);
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
$("signoff").addEventListener("click", () => {
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
    $("server-label").textContent = `synthesis: ${on ? "up" : "down"}`;
    $("server-label").title = "click to toggle synthesis server";
  } catch { $("server-dot").className = "dot bad"; $("server-label").textContent = "backend: down"; }
}
$("server-label").addEventListener("click", async () => {
  const s = await api("/api/server/status");
  await api(s.running ? "/api/server/stop" : "/api/server/start", {method: "POST"});
  serverStatus();
});
bindCanvas(); loadSpecimens(); refreshProvenance(); serverStatus();
setInterval(serverStatus, 15000);
