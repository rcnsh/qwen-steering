import { decompress } from "./vendor/fzstd.mjs";
import { initKitchen } from "./kitchen.js";

const $ = (s) => document.querySelector(s);
const SIDE = 96; // 96 × 96 = 9216 neurons per layer
const RARE = 0.005;

const state = { tab: "atlas", layer: 5, neuron: 8920, dose: 0, metric: "topic", query: "", hits: new Map() };
let kitchen;
let meta, idx, labels, N, L, TOTAL;
const texts = new Map(); // layer -> Promise<string[]>
const loaded = new Map(); // layer -> string[]
const dfs = new Map(); // `${layer}:${dose}` -> Map(word -> count)
const labelCount = new Map(); // layer -> number of labelled neurons
let topList = []; // [value, neuron] for the current layer, brightest first

// ---------- data ----------

// The host sometimes drops a request, and a sandboxed page sees that as "Failed to fetch", so retry.
async function get(path) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(path);
      if (res.ok) return res;
      if (res.status < 500) throw Object.assign(new Error(`${path}: ${res.status}`), { final: true });
      throw new Error(`${path}: ${res.status}`);
    } catch (err) {
      if (err.final || attempt === 4) throw err;
      await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
    }
  }
}

async function zst(path) {
  return decompress(new Uint8Array(await (await get(path)).arrayBuffer()));
}

function layerTexts(layer) {
  if (!texts.has(layer)) {
    texts.set(layer, zst(`data/L${layer}.txt.zst`).then((b) => {
      const arr = new TextDecoder().decode(b).split("\0");
      loaded.set(layer, arr);
      return arr;
    }, (err) => {
      texts.delete(layer); // let the next click try again
      throw err;
    }));
  }
  return texts.get(layer);
}

const plane = (p, layer, n) => idx[p * TOTAL + layer * N + n];
const reply = (layer, dose, prompt, n) => loaded.get(layer)?.[(dose * 2 + prompt) * N + n];
const label = (layer, n) => labels[`${layer}:${n}`];

const WORD = /[a-z]{4,}/g;
const words = (t) => new Set(t.toLowerCase().match(WORD) ?? []);

function df(layer, dose) {
  const key = `${layer}:${dose}`;
  if (!dfs.has(key)) {
    const m = new Map();
    for (let n = 0; n < N; n++) {
      const ws = new Set([...words(reply(layer, dose, 0, n)), ...words(reply(layer, dose, 1, n))]);
      for (const w of ws) m.set(w, (m.get(w) ?? 0) + 1);
    }
    dfs.set(key, m);
  }
  return dfs.get(key);
}

// ---------- colour ----------

// Paper to ink-red, like a wash of red ink on the page.
const STOPS = [[0, [232, 225, 210]], [0.3, [221, 188, 146]], [0.55, [203, 125, 84]], [0.8, [163, 48, 30]], [1, [88, 24, 16]]];
const BLANK = STOPS[0][1], BROKEN = [84, 98, 118], DIM = [226, 219, 204], INK = [28, 25, 21], RED = [163, 48, 30];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) {
    const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i], f = (t - a) / (b - a);
    return ca.map((v, k) => v + (cb[k] - v) * f);
  }
  return STOPS.at(-1)[1];
}
const mixc = (a, b, f) => a.map((v, k) => v + (b[k] - v) * f);

// Topic scores bunch up (median byte ~153), so only the top quarter colours: ~p75 paper, ~p99.9 darkest.
const topicV = (l, n, d) => { const v = plane(d, l, n); return v ? Math.max(0.001, Math.min(1, (v - 175) / 65)) : -1; };

const METRICS = {
  topic: { name: "topic score", lo: "dull", hi: "on-topic, rare words", value: topicV },
  change: { name: "change", lo: "same as baseline", hi: "all new", value: (l, n, d) => ((plane(2 + 2 * d, l, n) + plane(3 + 2 * d, l, n)) / 510) ** 3 },
  broken: { name: "breakage", lo: "fluent", hi: "repetitive or broken", value: (l, n, d) => 1 - (plane(6 + 2 * d, l, n) + plane(7 + 2 * d, l, n)) / 510 },
  labels: { name: "label", lo: "unlabelled", hi: "labelled (darker = more confident)", value: (l, n) => { const x = label(l, n); return x ? 0.45 + 0.11 * x[5] : 0; } },
  search: { name: "search", lo: "no match", hi: "reply or label matches", value: (l, n) => (state.hits.get(l)?.has(n) ? 1 : 0) },
};

// Faint topic map behind the label and search overlays, so the plate keeps its texture.
function faint(l, n, d) {
  const t = topicV(l, n, d);
  return mixc(ramp(t < 0 ? 0 : t * 0.25), DIM, 0.4);
}

function colour(l, n, d, dimmed) {
  const m = state.metric, v = METRICS[m].value(l, n, d);
  if (m === "labels") return v ? mixc([150, 128, 108], INK, Math.max(0, Math.min(1, ((v || 0.56) - 0.56) / 0.44))) : faint(l, n, d);
  if (m === "search") return v ? RED : dimmed ? DIM : faint(l, n, d);
  if (v < 0) return BROKEN;
  return ramp(m === "topic" ? v ** 0.9 : v);
}

function paint(canvas, layer) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(SIDE, SIDE);
  const dimmed = state.metric === "search" && !loaded.has(layer);
  for (let n = 0; n < N; n++) {
    const [r, g, b] = colour(layer, n, state.dose, dimmed);
    const o = n * 4;
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- rendering ----------

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const isAttn = (l) => (l + 1) % 4 === 0;
const fmtDose = (d) => `${meta.doses[d] > 0 ? "+" : "−"}${Math.abs(meta.doses[d])}`;

function mark(html) {
  if (state.query.length < 2) return html;
  const q = esc(state.query).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return html.replace(new RegExp(`(${q})(?![^<]*>)`, "gi"), "<mark>$1</mark>");
}

// Replies are markdown-ish: bold renders in the reading view and drops out of one-line rows.
const md = (html) => html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
const plain = (t) => t.replace(/\*\*/g, "").replace(/\s+/g, " ");
const rich = (t) => mark(md(esc(t)));

// Grey up to where the steered reply leaves the unsteered one, a caret there, then the rest in ink.
function diffed(text, base) {
  let i = 0;
  while (i < text.length && i < base.length && text[i] === base[i]) i++;
  if (text && text === base) return `<span class="same">${rich(text)}</span> <span class="tag">same as unsteered</span>`;
  if (i >= text.length) return rich(text);
  i = text.lastIndexOf(" ", i) + 1; // snap back to a word start
  if ((text.slice(0, i).match(/\*\*/g) ?? []).length % 2) i = text.lastIndexOf("**", i); // don't split a bold run
  if (i <= 0) return rich(text);
  return `<span class="same">${rich(text.slice(0, i))}</span><span class="caret" aria-hidden="true"></span><span class="sr">(steered from here)</span>${rich(text.slice(i))}`;
}

function broken(t) {
  if (!t.trim()) return true;
  const ascii = [...t].filter((c) => c.charCodeAt(0) < 128).length / t.length;
  const toks = t.split(/\s+/).filter(Boolean);
  return ascii < 0.9 || toks.length < 5 || new Set(toks).size / toks.length < 0.4;
}

// Thumbnails show each 4 × 4 block's brightest neuron, so small hot spots survive the shrink.
const THUMB = 24, BLOCK = SIDE / THUMB;
function paintThumb(canvas, layer) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(THUMB, THUMB);
  const m = METRICS[state.metric], d = state.dose, dimmed = state.metric === "search" && !loaded.has(layer);
  for (let i = 0; i < THUMB * THUMB; i++) {
    const r0 = Math.floor(i / THUMB) * BLOCK, c0 = (i % THUMB) * BLOCK;
    let best = -1, bv = -Infinity, bt = -Infinity;
    for (let y = 0; y < BLOCK; y++) for (let x = 0; x < BLOCK; x++) {
      const n = (r0 + y) * SIDE + c0 + x, v = m.value(layer, n, d), t = topicV(layer, n, d);
      if (v > bv || (v === bv && t > bt)) { best = n; bv = v; bt = t; }
    }
    const [r, g, b] = colour(layer, best, d, dimmed), o = i * 4;
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawStrip() {
  for (let l = 0; l < L; l++) paintThumb(strip[l].canvas, l);
}

function drawLegend() {
  const m = METRICS[state.metric], k = state.metric;
  $("#legend-lo").textContent = m.lo;
  $("#legend-hi").textContent = m.hi;
  $("#ramp").style.background = k === "labels" || k === "search"
    ? `linear-gradient(90deg, rgb(${BLANK}), rgb(${k === "labels" ? INK : RED}))`
    : `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((v) => `rgb(${ramp(v).map(Math.round)})`).join(",")})`;
  const ell = (c) => `<svg width="16" height="12" aria-hidden="true"><ellipse cx="8" cy="6" rx="6.5" ry="5" fill="none" stroke="${c}" stroke-width="1.3"/></svg>`;
  $("#legend-2").innerHTML = (k === "topic" ? `<span><i class="sw" style="background:rgb(${BROKEN})"></i>Broken reply</span>` : "") +
    `<span>${ell("#a3301e")}Selected</span><span>${ell("#1c1915")}Brightest three</span>`;
  const notes = [];
  if (k === "search" && state.query.length < 2) notes.push("Type in the search box to light up matching neurons.");
  else if (k === "search" && !loaded.has(state.layer)) notes.push(`Replies for layer ${state.layer} aren't loaded yet, so only label matches show. Open a neuron here or search all layers to load them.`);
  $("#pnote").textContent = notes.join(" ");
  $("#pnote").hidden = !notes.length;
}

function drawField() {
  const l = state.layer;
  paint(mapCanvas, l);
  strip.forEach((s, i) => s.button.setAttribute("aria-current", String(i === l)));
  $("#layer-title").textContent = `Layer ${l}`;
  $("#plate-no").textContent = `Plate ${l + 1} of ${L}`;
  $("#layer-note").textContent = `${isAttn(l) ? "Full attention" : "Linear attention (Gated DeltaNet)"}. 96 × 96, one pixel per neuron, numbered row by row from the top left. Click a pixel to read its replies.`;
  $("#layer-notes").innerHTML = `<dl><dt>Layer</dt><dd>${l}</dd><dt>Attention</dt><dd>${isAttn(l) ? "full" : "linear"}</dd>` +
    `<dt>Median ‖resid‖</dt><dd>${meta.resid_median[l].toFixed(2)}</dd><dt>Labelled</dt><dd>${(labelCount.get(l) ?? 0).toLocaleString("en")}</dd></dl>`;
  drawLegend();
  drawTop();
}

function drawTop() {
  const m = METRICS[state.metric], l = state.layer, d = state.dose;
  topList = [...Array(N).keys()].map((n) => [m.value(l, n, d), n]).filter(([v]) => v > 0).sort((a, b) => b[0] - a[0]).slice(0, 14);
  $("#top-h").textContent = `Brightest on layer ${l} by ${m.name}`;
  $("#top").innerHTML = topList.map(([, n], i) => {
    const lab = label(l, n);
    const r = reply(l, d, 1, n);
    const s = lab ? `${lab[0]} ${esc(lab[1])}` : r ? esc(plain(r)) : "";
    return `<li><button data-n="${n}" aria-current="${n === state.neuron}"><span class="r">${i + 1}.</span><span class="i">N${n}</span><span class="t">${s}</span></button></li>`;
  }).join("") || `<li class="none">Nothing lights up at this dose.</li>`;
  annotate();
}

// A pen loop: an ellipse that overshoots its start a little and wobbles, so it reads as drawn by hand.
function loop(cx, cy, rx, ry, seed) {
  const rot = ((seed * 37) % 30 - 15) * Math.PI / 180, start = (seed % 7) * 0.8, pts = [];
  for (let k = 0; k <= 44; k++) {
    const t = start + (k / 40) * Math.PI * 2;
    const w = 1 + 0.05 * Math.sin(3 * t + seed) + 0.06 * (k / 44);
    const x = Math.cos(t) * rx * w, y = Math.sin(t) * ry * w;
    pts.push(`${(cx + x * Math.cos(rot) - y * Math.sin(rot)).toFixed(1)},${(cy + x * Math.sin(rot) + y * Math.cos(rot)).toFixed(1)}`);
  }
  return `M${pts.join("L")}`;
}

function annotate() {
  const svg = $("#notes"), w = mapCanvas.clientWidth;
  if (!w) return;
  svg.setAttribute("viewBox", `0 0 ${w} ${w}`);
  const s = w / SIDE, r = Math.max(s * 1.4, 11);
  const centre = (n) => [(n % SIDE) * s + s / 2, Math.floor(n / SIDE) * s + s / 2];
  const note = (n, cls, seed) => {
    const [cx, cy] = centre(n), right = cx > w * 0.78;
    const tx = right ? cx - r - 6 : cx + r + 5, ty = cy < 20 ? cy + r + 12 : cy - r * 0.55;
    return `<path class="ann ${cls}" d="${loop(cx, cy, r + 2, r - 1, seed)}"/>` +
      `<text class="ann-t ${cls}" x="${tx}" y="${ty}" text-anchor="${right ? "end" : "start"}">N${n}</text>`;
  };
  let out = topList.slice(0, 3).filter(([, n]) => n !== state.neuron).map(([, n], i) => note(n, "", n + i)).join("");
  const [cx, cy] = centre(state.neuron), h = s / 2 + 1;
  out += note(state.neuron, "sel", state.neuron) + `<rect class="cur" x="${cx - h}" y="${cy - h}" width="${2 * h}" height="${2 * h}"/>`;
  svg.innerHTML = out;
}

function placeCursor() {
  annotate();
  document.querySelectorAll("#top button").forEach((b) => b.setAttribute("aria-current", String(+b.dataset.n === state.neuron)));
}

const atlasHash = () => `#${state.layer}:${state.neuron}${state.dose ? "+" : "-"}`;

async function drawInspector() {
  const { layer: l, neuron: n, dose: d } = state;
  $("#nid").innerHTML = `L${l} <span class="s">N</span>${n}`;
  $("#nsub").textContent = `Layer ${l}, neuron ${n} of ${N.toLocaleString("en")} · ${isAttn(l) ? "full" : "linear"} attention layer`;
  const lab = label(l, n);
  $("#label").innerHTML = `<div class="lab">` + (lab
    ? `<p class="nm">${lab[0]} ${esc(lab[1])}</p><p class="desc">${esc(lab[2])}</p>` +
      `<p class="push">When pushed (${esc(lab[4])}, confidence ${lab[5]}/5): <b>${esc(lab[3])}</b></p>`
    : `<p class="none">No atlas label. The replies below show what pushing it does.</p>`) + `</div>`;
  $("#cmd").textContent = `uv run neuron.py --layer ${l} --neuron ${n} --dose ${meta.doses[d]} --prompt "${meta.prompts[1]}"`;
  $("#to-kitchen").textContent = `Add to the Kitchen at ${fmtDose(d)}`;
  if (state.tab === "atlas") history.replaceState(null, "", atlasHash());
  tabLinks();

  const s = (p) => plane(p, l, n) / 255;
  const topic = (p) => (plane(p, l, n) ? (plane(p, l, n) - 1) / 254 : 0);
  const rows = [
    ["On topic", "Rare words the two replies share", topic(0), topic(1)],
    ["Changed", "How far the replies are from the unsteered ones", (s(2) + s(3)) / 2, (s(4) + s(5)) / 2],
    ["Word variety", "Distinct words over all words; low means repetitive", (s(6) + s(7)) / 2, (s(8) + s(9)) / 2],
  ];
  const cell = (v) => `<td class="v"><span class="bar"><i style="width:${(v * 100).toFixed(0)}%"></i></span>${v.toFixed(2)}</td>`;
  $("#stats").innerHTML = `<table><thead><tr><th scope="col">Scores</th><th scope="col">Pushed −0.8</th><th scope="col">Pushed +0.8</th></tr></thead><tbody>` +
    rows.map(([k, t, a, b]) => `<tr><td>${k}<small>${t}</small></td>${cell(a)}${cell(b)}</tr>`).join("") + `</tbody></table>`;

  if (!loaded.has(l)) {
    $("#words").innerHTML = "";
    $("#replies").innerHTML = `<div class="dose-block"><blockquote class="say loading">Loading the replies for layer ${l}…</blockquote></div>`;
    try { await layerTexts(l); }
    catch (err) {
      $("#replies").innerHTML = `<div class="dose-block"><p class="fail">Layer ${l} didn't load (${esc(String(err))}). Click the neuron again to retry.</p></div>`;
      return;
    }
    if (state.layer !== l || state.neuron !== n || state.dose !== d) return;
    layerReady(l);
  }

  const counts = df(l, d), cutoff = RARE * N;
  const sets = [0, 1].map((p) => [...words(reply(l, d, p, n))].filter((w) => counts.get(w) <= cutoff));
  const shared = new Set(sets[0].filter((w) => sets[1].includes(w)));
  const chips = [...new Set([...sets[0], ...sets[1]])].sort((a, b) => counts.get(a) - counts.get(b)).slice(0, 12);
  $("#words").innerHTML = chips.length
    ? `<div class="terms"><span class="h">Rare words</span>${chips.map((w) => `<span class="chip${shared.has(w) ? " both" : ""}" title="${counts.get(w)} neurons' replies use it${shared.has(w) ? "; in both replies" : ""}">${w}</span>`).join("")}</div>`
    : "";

  const block = (dd) => `<section class="dose-block"><h3>Pushed <span class="d">${fmtDose(dd)}</span>${dd === d ? '<span class="on">on the map</span>' : ""}</h3>` +
    [0, 1].map((p) => {
      const t = reply(l, dd, p, n);
      return `<div class="reply"><p class="q">${esc(meta.prompts[p])}${broken(t) ? '<span class="tag">broken</span>' : ""}</p><blockquote class="say">${diffed(t, meta.baselines[p])}</blockquote></div>`;
    }).join("") + `</section>`;
  $("#replies").innerHTML = `<p class="keynote"><span class="same">Grey text</span> matches the unsteered reply. The push takes over at the <span class="caret" aria-hidden="true"></span>mark, where the text turns dark.${shared.size ? ' <span style="color:var(--red)">Red</span> words turn up in both replies.' : ""}</p>` +
    block(d) + block(1 - d) +
    `<section class="dose-block base"><h3>Unsteered</h3>` +
    [0, 1].map((p) => `<div class="reply"><p class="q">${esc(meta.prompts[p])}</p><blockquote class="say">${rich(meta.baselines[p])}</blockquote></div>`).join("") + `</section>`;
}

// ---------- interaction ----------

function select(layer, neuron) {
  layer = ((layer % L) + L) % L;
  const layerChanged = layer !== state.layer;
  state.layer = layer;
  state.neuron = ((neuron % N) + N) % N;
  if (layerChanged) {
    drawField();
    layerTexts(layer).then(() => layerReady(layer), () => {});
  } else placeCursor();
  drawInspector();
}

// A layer's replies just arrived: redraw what depends on them (search hits, the not-loaded note).
function layerReady(layer) {
  if (layer !== state.layer) return;
  if (state.metric === "search" && state.query.length >= 2) {
    runSearch();
    paintThumb(strip[layer].canvas, layer);
    drawField();
  } else {
    drawLegend();
    drawTop();
  }
}

function randomBright() {
  const pool = [];
  for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) if (plane(state.dose, l, n) > 215) pool.push([l, n]);
  const [l, n] = pool[Math.floor(Math.random() * pool.length)];
  select(l, n);
}

function setDose(d) {
  state.dose = d;
  document.querySelectorAll("#dose button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.dose === d)));
  drawStrip();
  drawField();
  drawInspector();
}

function tabLinks() {
  $('.tabs [data-tab="atlas"]').href = atlasHash();
  if (kitchen) $('.tabs [data-tab="kitchen"]').href = kitchen.hash();
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tabs [data-tab]").forEach((a) => (a.dataset.tab === tab ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  $("#atlas").hidden = tab !== "atlas";
  $("#kitchen").hidden = tab !== "kitchen";
  document.title = tab === "kitchen" ? "Kitchen · qwen-steering" : "Neuron Atlas · qwen-steering";
  history.replaceState(null, "", tab === "kitchen" ? kitchen.hash() : atlasHash());
  tabLinks();
  if (tab === "atlas") placeCursor();
}

function setMetric(m) {
  state.metric = m;
  $("#metric").value = m;
  drawStrip();
  drawField();
}

function runSearch() {
  const q = state.query.toLowerCase();
  state.hits = new Map();
  if (q.length < 2) return { labelHits: [], replyHits: [] };
  const labelHits = [];
  for (const [k, v] of Object.entries(labels)) {
    if (v[1].toLowerCase().includes(q) || v[2].toLowerCase().includes(q)) {
      const [l, n] = k.split(":").map(Number);
      labelHits.push([l, n, v]);
      if (!state.hits.has(l)) state.hits.set(l, new Set());
      state.hits.get(l).add(n);
    }
  }
  const replyHits = [];
  for (const [l, arr] of loaded) {
    for (let i = 0; i < arr.length; i++) {
      if (!arr[i].toLowerCase().includes(q)) continue;
      const n = i % N, dose = Math.floor(i / N / 2), p = Math.floor(i / N) % 2;
      if (meta.baselines[p].toLowerCase().includes(q)) continue; // the unsteered reply says it too
      if (!state.hits.has(l)) state.hits.set(l, new Set());
      state.hits.get(l).add(n);
      replyHits.push([l, n, dose, p, arr[i]]);
    }
  }
  return { labelHits, replyHits };
}

function snippet(text, q) {
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  const s = Math.max(0, i - 40);
  return (s ? "…" : "") + plain(text.slice(s, i + q.length + 60));
}

function hideResults() {
  $("#results").hidden = true;
  $("#q").setAttribute("aria-expanded", "false");
}

function showResults() {
  const box = $("#results");
  if (state.query.length < 2) { hideResults(); if (state.metric === "search") { runSearch(); drawStrip(); drawField(); } return; }
  const { labelHits, replyHits } = runSearch();
  const missing = L - loaded.size;
  let html = `<p class="h">Labels (${labelHits.length})</p>`;
  html += labelHits.slice(0, 30).map(([l, n, v]) =>
    `<button class="hit" data-l="${l}" data-n="${n}"><span class="id">L${l} N${n} · label</span><span class="ctx">${v[0]} ${mark(esc(v[1]))}: ${mark(esc(v[2]))}</span></button>`).join("");
  html += `<p class="h">Replies (${replyHits.length} in ${loaded.size} loaded layer${loaded.size === 1 ? "" : "s"})</p>`;
  html += replyHits.slice(0, 60).map(([l, n, d, p, t]) =>
    `<button class="hit" data-l="${l}" data-n="${n}" data-d="${d}"><span class="id">L${l} N${n} · ${fmtDose(d)} · ${esc(meta.prompts[p])}</span><span class="ctx">${mark(esc(snippet(t, state.query)))}</span></button>`).join("");
  if (missing) html += `<div class="more"><button class="btn" id="load-all">Search replies in all ${L} layers (loads ${missing} more, about ${(missing * 0.6).toFixed(0)} MB)</button></div>`;
  box.innerHTML = html;
  box.hidden = false;
  $("#q").setAttribute("aria-expanded", "true");
  if (state.metric === "search") { drawStrip(); drawField(); }
}

async function loadAll() {
  const btn = $("#load-all");
  let done = loaded.size;
  await Promise.allSettled([...Array(L).keys()].map((l) => layerTexts(l).then(() => { if (btn) btn.textContent = `Loading ${++done} of ${L}…`; })));
  showResults();
  setMetric("search");
}

// Follow a hash typed into the address bar (or any other hash change after boot).
function route(hash) {
  if (hash.startsWith("#kitchen")) { kitchen.load(hash); setTab("kitchen"); return; }
  const m = hash.match(/^#(\d+):(\d+)([+-])?/);
  if (!m) return;
  const d = m[3] === "+" ? 1 : 0;
  if (d !== state.dose) {
    state.dose = d;
    document.querySelectorAll("#dose button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.dose === d)));
    drawStrip();
    state.layer = -1; // force select() to redraw the field for the new dose
  }
  select(+m[1] % L, +m[2] % N);
  setTab("atlas");
}

// ---------- boot ----------

let strip, mapCanvas, wrap;

async function boot() {
  let kitchenData;
  [meta, idx, labels, kitchenData] = await Promise.all([
    get("data/meta.json").then((r) => r.json()),
    zst("data/index.bin.zst"),
    zst("data/labels.json.zst").then((b) => JSON.parse(new TextDecoder().decode(b))),
    get("data/kitchen.json").then((r) => r.json()),
  ]);
  N = meta.neurons; L = meta.layers; TOTAL = N * L;
  for (const k of Object.keys(labels)) { const l = +k.split(":")[0]; labelCount.set(l, (labelCount.get(l) ?? 0) + 1); }
  $("#count").textContent = TOTAL.toLocaleString("en");
  $("#n-labels").textContent = Object.keys(labels).length.toLocaleString("en");

  kitchen = initKitchen({
    data: kitchenData,
    labels,
    onChange(n) {
      const c = $("#pot-count");
      c.hidden = !n;
      c.textContent = n;
      c.setAttribute("aria-label", `${n} in the blend`);
      if (state.tab === "kitchen") history.replaceState(null, "", kitchen.hash());
      tabLinks();
    },
  });
  document.querySelectorAll(".tabs [data-tab]").forEach((a) => a.addEventListener("click", (e) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button) return; // let modified clicks through
    e.preventDefault();
    setTab(a.dataset.tab);
    scrollTo(0, 0);
  }));
  $("#to-kitchen").onclick = () => {
    kitchen.addFromAtlas(state.layer, state.neuron, meta.doses[state.dose]);
    const b = $("#to-kitchen");
    b.textContent = "Added. Open the Kitchen to blend it";
    setTimeout(() => (b.textContent = `Add to the Kitchen at ${fmtDose(state.dose)}`), 1600);
  };

  strip = [...Array(L).keys()].map((l) => {
    const button = document.createElement("button");
    const canvas = Object.assign(document.createElement("canvas"), { width: THUMB, height: THUMB });
    const tag = Object.assign(document.createElement("span"), { className: "n", textContent: l });
    if (isAttn(l)) button.className = "attn";
    button.setAttribute("aria-label", `Layer ${l}${isAttn(l) ? ", full attention" : ""}`);
    button.title = `Layer ${l}${isAttn(l) ? " (full attention)" : ""}`;
    button.append(canvas, tag);
    button.onclick = () => select(l, state.neuron);
    $("#strip").append(button);
    return { button, canvas };
  });

  const top = $("#ax-top"), left = $("#ax-left");
  const ticks = [...Array(SIDE / 16 + 1).keys()].map((k) => k * 16);
  top.innerHTML = ticks.map((c) => `<span style="left:${(c / SIDE) * 100}%">${c === SIDE ? "" : c}</span><i style="left:${(c / SIDE) * 100}%"></i>`).join("");
  left.innerHTML = ticks.map((r) => `<span style="top:${(r / SIDE) * 100}%">${r === SIDE ? "" : (r * SIDE).toLocaleString("en")}</span><i style="top:${(r / SIDE) * 100}%"></i>`).join("");

  wrap = $("#wrap");
  mapCanvas = $("#map");

  const at = (e) => {
    const r = mapCanvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * SIDE), y = Math.floor(((e.clientY - r.top) / r.height) * SIDE);
    return x >= 0 && y >= 0 && x < SIDE && y < SIDE ? y * SIDE + x : null;
  };
  wrap.addEventListener("mousemove", (e) => {
    const n = at(e), tip = $("#tip");
    if (n === null) { tip.hidden = true; $("#readout").textContent = ""; return; }
    const l = state.layer, lab = label(l, n), r = reply(l, state.dose, 1, n), m = METRICS[state.metric];
    const v = m.value(l, n, state.dose);
    $("#readout").textContent = `L${l} N${n} · row ${Math.floor(n / SIDE)}, column ${n % SIDE} · ${v < 0 ? "broken reply" : `${m.name} ${v.toFixed(2)}`}`;
    tip.innerHTML = `<b>N${n}</b>${lab ? `${lab[0]} ${esc(lab[1])}` : ""}${r ? `<span class="t">${esc(plain(r).slice(0, 110))}…</span>` : ""}`;
    const wr = wrap.getBoundingClientRect();
    const x = e.clientX - wr.left, y = e.clientY - wr.top;
    tip.style.left = `${Math.max(4, Math.min(x + 14, wr.width - Math.min(300, wr.width - 8) - 4))}px`;
    tip.style.top = `${y + 18 > wr.height - 90 ? y - 80 : y + 18}px`;
    tip.hidden = false;
  });
  wrap.addEventListener("mouseleave", () => { $("#tip").hidden = true; $("#readout").textContent = ""; });
  wrap.addEventListener("click", (e) => { const n = at(e); if (n !== null) select(state.layer, n); });
  new ResizeObserver(annotate).observe(mapCanvas);

  $("#top").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) select(state.layer, +b.dataset.n); });
  $("#prev").onclick = () => select(state.layer, state.neuron - 1);
  $("#next").onclick = () => select(state.layer, state.neuron + 1);
  $("#rand").onclick = randomBright;
  document.querySelectorAll("#dose button").forEach((b) => (b.onclick = () => setDose(+b.dataset.dose)));
  $("#metric").onchange = (e) => setMetric(e.target.value);
  $("#copy").onclick = async () => {
    const text = $("#cmd").textContent;
    try { await navigator.clipboard.writeText(text); }
    catch { getSelection().selectAllChildren($("#cmd")); document.execCommand("copy"); }
    $("#copy").textContent = "Copied";
    setTimeout(() => ($("#copy").textContent = "Copy"), 1200);
  };

  let timer;
  $("#q").addEventListener("input", (e) => {
    state.query = e.target.value.trim();
    clearTimeout(timer);
    timer = setTimeout(showResults, 120);
  });
  $("#q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { showResults(); setMetric("search"); }
    if (e.key === "Escape") { hideResults(); e.target.blur(); }
    if (e.key === "ArrowDown") { const f = $("#results:not([hidden]) button"); if (f) { e.preventDefault(); f.focus(); } }
  });
  $("#q").addEventListener("focus", () => state.query.length >= 2 && showResults());
  $("#results").addEventListener("click", (e) => {
    if (e.target.id === "load-all") return loadAll();
    const b = e.target.closest("button.hit");
    if (!b) return;
    if (b.dataset.d !== undefined && +b.dataset.d !== state.dose) setDose(+b.dataset.d);
    select(+b.dataset.l, +b.dataset.n);
    hideResults();
    drawInspector();
  });
  $("#results").addEventListener("keydown", (e) => {
    const bs = [...document.querySelectorAll("#results button")], i = bs.indexOf(document.activeElement);
    if (e.key === "ArrowDown" && i < bs.length - 1) { e.preventDefault(); bs[i + 1].focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); (i > 0 ? bs[i - 1] : $("#q")).focus(); }
    if (e.key === "Escape") { hideResults(); $("#q").focus(); }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) hideResults(); });

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey || state.tab !== "atlas") return;
    if (e.target.closest("#results")) return;
    const moves = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -SIDE, ArrowDown: SIDE };
    if (moves[e.key]) { e.preventDefault(); select(state.layer, state.neuron + moves[e.key]); }
    else if (e.key === "[") select(state.layer - 1, state.neuron);
    else if (e.key === "]") select(state.layer + 1, state.neuron);
    else if (e.key === "d" || e.key === "D") setDose(1 - state.dose);
    else if (e.key === "r" || e.key === "R") randomBright();
    else if (e.key === "/") { e.preventDefault(); $("#q").focus(); }
  });

  const hash = location.hash; // drawInspector rewrites it
  if (hash.startsWith("#kitchen")) kitchen.load(hash);
  const m = hash.match(/^#(\d+):(\d+)([+-])?/);
  if (m) { state.layer = +m[1] % L; state.neuron = +m[2] % N; state.dose = m[3] === "+" ? 1 : 0; }
  document.querySelectorAll("#dose button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.dose === state.dose)));
  drawStrip();
  drawField();
  drawInspector();
  if (hash.startsWith("#kitchen")) setTab("kitchen");
  addEventListener("hashchange", () => route(location.hash));
}

boot().catch((err) => {
  console.error(err);
  $("#atlas").innerHTML = `<section class="sheet intro"><div class="margin"></div><div><h1>Atlas</h1><p class="fail">The atlas data didn't load (${esc(String(err))}).</p>` +
    `<button class="btn" onclick="location.reload()">Retry</button></div></section>`;
});
