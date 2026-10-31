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

const INFERNO = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function ramp(v) {
  const x = Math.max(0, Math.min(1, v)) * (INFERNO.length - 1);
  const i = Math.min(INFERNO.length - 2, Math.floor(x)), f = x - i;
  const a = INFERNO[i], b = INFERNO[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const METRICS = {
  // Topic scores bunch up (median byte ~153), so only the top quarter lights: ~p75 dark, ~p99.9 white.
  topic: { lo: "dull", hi: "on-topic, rare words", value: (l, n, d) => { const v = plane(d, l, n); return v ? Math.max(0.001, Math.min(1, (v - 175) / 65)) : -1; } },
  change: { lo: "same as baseline", hi: "all new", value: (l, n, d) => ((plane(2 + 2 * d, l, n) + plane(3 + 2 * d, l, n)) / 510) ** 3 },
  broken: { lo: "fluent", hi: "repetitive / broken", value: (l, n, d) => 1 - (plane(6 + 2 * d, l, n) + plane(7 + 2 * d, l, n)) / 510 },
  labels: { lo: "unlabelled", hi: "labelled (by confidence)", value: (l, n) => { const x = label(l, n); return x ? 0.45 + 0.11 * x[5] : 0; } },
  search: { lo: "no match", hi: "reply or label matches", value: (l, n) => (state.hits.get(l)?.has(n) ? 1 : 0) },
};

function pixel(v) {
  if (v < 0) return [70, 8, 12]; // broken reply
  if (state.metric === "labels") return v ? [143 * v, 176 * v, 255 * v] : [13, 15, 20];
  if (state.metric === "search") return v ? [255, 214, 92] : [20, 22, 28];
  const g = Math.pow(v, 1.6); // keep the background dark so bright neurons pop
  return ramp(g);
}

function paint(canvas, layer) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(SIDE, SIDE);
  const m = METRICS[state.metric];
  const dimmed = state.metric === "search" && !loaded.has(layer);
  for (let n = 0; n < N; n++) {
    const [r, g, b] = pixel(m.value(layer, n, state.dose));
    const o = n * 4, k = dimmed ? 0.25 : 1;
    img.data[o] = r * k; img.data[o + 1] = g * k; img.data[o + 2] = b * k; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- rendering ----------

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const isAttn = (l) => (l + 1) % 4 === 0;

function mark(html) {
  if (!state.query) return html;
  const q = esc(state.query).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return html.replace(new RegExp(`(${q})(?![^<]*>)`, "gi"), "<mark>$1</mark>");
}

// Replies are markdown-ish: bold renders in the reading view and drops out of one-line rows.
const md = (html) => html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
const plain = (t) => t.replace(/\*\*/g, "").replace(/\s+/g, " ");
const rich = (t) => mark(md(esc(t)));

function diffed(text, base) {
  let i = 0;
  while (i < text.length && i < base.length && text[i] === base[i]) i++;
  if (i < text.length) i = text.lastIndexOf(" ", i) + 1; // snap back to a word start
  if ((text.slice(0, i).match(/\*\*/g) ?? []).length % 2) i = text.lastIndexOf("**", i); // don't split a bold run
  if (i <= 0) return rich(text);
  return `<span class="same">${rich(text.slice(0, i))}</span>${rich(text.slice(i))}`;
}

function broken(t) {
  if (!t.trim()) return true;
  const ascii = [...t].filter((c) => c.charCodeAt(0) < 128).length / t.length;
  const toks = t.split(/\s+/).filter(Boolean);
  return ascii < 0.9 || toks.length < 5 || new Set(toks).size / toks.length < 0.4;
}

function drawStrip() {
  for (let l = 0; l < L; l++) paint(strip[l].canvas, l);
}

function drawField() {
  paint(mapCanvas, state.layer);
  strip.forEach((s, l) => s.button.setAttribute("aria-current", String(l === state.layer)));
  $("#layer-title").textContent = `Layer ${state.layer}`;
  const nl = Object.keys(labels).filter((k) => k.startsWith(`${state.layer}:`)).length;
  $("#layer-note").textContent = `${isAttn(state.layer) ? "Full attention" : "Linear attention (Gated DeltaNet)"}, median residual norm ${meta.resid_median[state.layer].toFixed(2)}, ${nl} labelled neurons. Click a pixel to read its replies.`;
  const m = METRICS[state.metric];
  $("#legend-lo").textContent = m.lo;
  $("#legend-hi").textContent = m.hi;
  $("#ramp").style.background = state.metric === "labels" || state.metric === "search"
    ? `linear-gradient(90deg, #14171e, ${state.metric === "labels" ? "#8fb0ff" : "#ffd65c"})`
    : `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((v) => `rgb(${ramp(Math.pow(v, 1.6)).map(Math.round)})`).join(",")})`;
  drawTop();
  placeCursor();
}

function drawTop() {
  const m = METRICS[state.metric], l = state.layer, d = state.dose;
  const ranked = [...Array(N).keys()].map((n) => [m.value(l, n, d), n]).filter(([v]) => v > 0).sort((a, b) => b[0] - a[0]).slice(0, 14);
  $("#top").innerHTML = ranked.map(([v, n], i) => {
    const lab = label(l, n);
    const r = reply(l, d, 1, n);
    const s = lab ? `${lab[0]} ${esc(lab[1])}` : r ? esc(plain(r)) : "";
    return `<li><button data-n="${n}"><span class="rk">${i + 1}</span><span class="n">N${n}</span><span class="s">${s}</span></button></li>`;
  }).join("") || `<li class="note">Nothing lights up at this dose.</li>`;
}

function placeCursor() {
  const c = $("#cursor"), w = wrap.clientWidth / SIDE;
  const x = state.neuron % SIDE, y = Math.floor(state.neuron / SIDE);
  Object.assign(c.style, { left: `${x * w - 1}px`, top: `${y * w - 1}px`, width: `${w + 2}px`, height: `${w + 2}px` });
}

async function drawInspector() {
  const { layer: l, neuron: n, dose: d } = state;
  $("#nid").innerHTML = `L${l} N${n}<small>${isAttn(l) ? "full attention layer" : ""}</small>`;
  const lab = label(l, n);
  $("#label").innerHTML = lab
    ? `<div class="name">${lab[0]} ${esc(lab[1])}</div><div class="desc">${esc(lab[2])}</div>` +
      `<div class="push">When pushed (${esc(lab[4])}, confidence ${lab[5]}/5): <b>${esc(lab[3])}</b></div>`
    : `<div class="none">No atlas label. The replies below show what pushing it does.</div>`;
  const cmdText = `uv run neuron.py --layer ${l} --neuron ${n} --dose ${meta.doses[d]} --prompt "${meta.prompts[1]}"`;
  $("#cmd").textContent = cmdText;
  if (state.tab === "atlas") history.replaceState(null, "", `#${l}:${n}${d ? "+" : "-"}`);

  const s = (p) => plane(p, l, n) / 255;
  const topic = (p) => (plane(p, l, n) ? (plane(p, l, n) - 1) / 254 : 0);
  const rows = [
    ["On topic", "How many rare, shared words the replies use", topic(0), topic(1)],
    ["Changed from unsteered", "How different the replies are from the unsteered ones", (s(2) + s(3)) / 2, (s(4) + s(5)) / 2],
    ["Word variety", "Distinct words over all words; low means repetitive", (s(6) + s(7)) / 2, (s(8) + s(9)) / 2],
  ];
  const cell = (v) => `<span class="cell"><span class="bar"><i style="width:${(v * 100).toFixed(0)}%"></i></span>${v.toFixed(2)}</span>`;
  $("#stats").innerHTML = `<span></span><span class="hd">Push −0.8</span><span class="hd">Push +0.8</span>` +
    rows.map(([k, t, a, b]) => `<span title="${t}">${k}</span>${cell(a)}${cell(b)}`).join("");

  if (!loaded.has(l)) {
    $("#words").innerHTML = "";
    $("#replies").innerHTML = `<div class="reply"><div class="a loading">Loading the replies for layer ${l}…</div></div>`;
    try { await layerTexts(l); }
    catch (err) {
      $("#replies").innerHTML = `<div class="reply"><div class="a broken">Layer ${l} didn't load (${esc(String(err))}). Click the neuron again to retry.</div></div>`;
      return;
    }
    if (state.layer !== l || state.neuron !== n) return;
    drawTop();
  }

  const counts = df(l, d), cutoff = RARE * N;
  const sets = [0, 1].map((p) => [...words(reply(l, d, p, n))].filter((w) => counts.get(w) <= cutoff));
  const shared = new Set(sets[0].filter((w) => sets[1].includes(w)));
  const chips = [...new Set([...sets[0], ...sets[1]])].sort((a, b) => counts.get(a) - counts.get(b)).slice(0, 12);
  $("#words").innerHTML = chips.map((w) => `<span class="chip${shared.has(w) ? " shared" : ""}" title="${counts.get(w)} neurons' replies use it">${w}</span>`).join("");

  const block = (dd) => `<div class="dose-block${dd === d ? " active" : ""}"><div class="dose-title">Push ${meta.doses[dd] > 0 ? "+" : "−"}${Math.abs(meta.doses[dd])}</div>` +
    [0, 1].map((p) => {
      const t = reply(l, dd, p, n);
      return `<div class="reply"><div class="q">${esc(meta.prompts[p])}</div><div class="a${broken(t) ? " broken" : ""}">${diffed(t, meta.baselines[p])}</div></div>`;
    }).join("") + `</div>`;
  $("#replies").innerHTML = `<p class="key"><span class="same">Grey words</span> match the unsteered reply; the steered reply takes over where the text turns dark.</p>` +
    block(d) + block(1 - d) +
    `<div class="dose-block base"><div class="dose-title">Unsteered</div>` +
    [0, 1].map((p) => `<div class="reply"><div class="q">${esc(meta.prompts[p])}</div><div class="a">${rich(meta.baselines[p])}</div></div>`).join("") + `</div>`;
}

// ---------- interaction ----------

function select(layer, neuron) {
  const layerChanged = layer !== state.layer;
  state.layer = layer;
  state.neuron = (neuron + N) % N;
  if (layerChanged) {
    drawField();
    layerTexts(layer).then(() => layer === state.layer && drawTop());
  } else placeCursor();
  drawInspector();
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

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $("#atlas").hidden = tab !== "atlas";
  $("#kitchen").hidden = tab !== "kitchen";
  $(".controls").hidden = tab !== "atlas";
  history.replaceState(null, "", tab === "kitchen" ? kitchen.hash() : `#${state.layer}:${state.neuron}${state.dose ? "+" : "-"}`);
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

function showResults() {
  const box = $("#results");
  if (state.query.length < 2) { box.hidden = true; return; }
  const { labelHits, replyHits } = runSearch();
  const missing = L - loaded.size;
  let html = "";
  html += `<div class="group">Labels (${labelHits.length})</div>`;
  html += labelHits.slice(0, 30).map(([l, n, v]) =>
    `<button class="hit" data-l="${l}" data-n="${n}"><span class="tag">L${l} N${n}</span>${v[0]} ${mark(esc(v[1]))}<span class="snip">${mark(esc(v[2]))}</span></button>`).join("");
  html += `<div class="group">Replies (${replyHits.length} in ${loaded.size} loaded layer${loaded.size === 1 ? "" : "s"})</div>`;
  html += replyHits.slice(0, 60).map(([l, n, d, , t]) =>
    `<button class="hit" data-l="${l}" data-n="${n}" data-d="${d}"><span class="tag">L${l} N${n} ${d ? "+" : "−"}</span><span class="snip">${mark(esc(snippet(t, state.query)))}</span></button>`).join("");
  if (missing) html += `<div class="more"><button id="load-all">Search replies in all ${L} layers (loads ${missing} more, about ${(missing * 0.6).toFixed(0)} MB)</button></div>`;
  box.innerHTML = html;
  box.hidden = false;
  if (state.metric === "search") { drawStrip(); drawField(); }
}

async function loadAll() {
  const btn = $("#load-all");
  let done = loaded.size;
  await Promise.allSettled([...Array(L).keys()].map((l) => layerTexts(l).then(() => { if (btn) btn.textContent = `Loading ${++done} of ${L}…`; })));
  showResults();
  setMetric("search");
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
  kitchen = initKitchen({
    data: kitchenData,
    labels,
    onChange(n) {
      const c = $("#pot-count");
      c.hidden = !n;
      c.textContent = n;
      if (state.tab === "kitchen") history.replaceState(null, "", kitchen.hash());
    },
  });
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
  $("#to-kitchen").onclick = () => {
    kitchen.addFromAtlas(state.layer, state.neuron, meta.doses[state.dose]);
    const b = $("#to-kitchen");
    b.textContent = "Added. Open the Kitchen tab to blend it";
    setTimeout(() => (b.textContent = "Add to the kitchen at this dose"), 1600);
  };
  N = meta.neurons; L = meta.layers; TOTAL = N * L;
  $("#count").textContent = TOTAL.toLocaleString("en");

  strip = [...Array(L).keys()].map((l) => {
    const button = document.createElement("button");
    const canvas = Object.assign(document.createElement("canvas"), { width: SIDE, height: SIDE });
    const tag = Object.assign(document.createElement("span"), { textContent: `L${l}` });
    if (isAttn(l)) tag.className = "attn";
    button.title = `Layer ${l}${isAttn(l) ? " (full attention; blue underline)" : ""}`;
    button.append(canvas, tag);
    button.onclick = () => select(l, state.neuron);
    $("#strip").append(button);
    return { button, canvas };
  });

  wrap = $("#wrap");
  mapCanvas = $("#map");

  const at = (e) => {
    const r = mapCanvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * SIDE), y = Math.floor(((e.clientY - r.top) / r.height) * SIDE);
    return x >= 0 && y >= 0 && x < SIDE && y < SIDE ? y * SIDE + x : null;
  };
  wrap.addEventListener("mousemove", (e) => {
    const n = at(e), tip = $("#tip");
    if (n === null) { tip.hidden = true; return; }
    const lab = label(state.layer, n), r = reply(state.layer, state.dose, 1, n);
    tip.innerHTML = `<b>N${n}</b> ${lab ? `${lab[0]} ${esc(lab[1])}` : ""}${r ? `<span class="t">${esc(plain(r).slice(0, 110))}…</span>` : ""}`;
    const wr = wrap.getBoundingClientRect();
    const x = e.clientX - wr.left, y = e.clientY - wr.top;
    tip.style.left = `${Math.min(x + 14, wr.width - 310)}px`;
    tip.style.top = `${y + 18 > wr.height - 90 ? y - 80 : y + 18}px`;
    tip.hidden = false;
  });
  wrap.addEventListener("mouseleave", () => ($("#tip").hidden = true));
  wrap.addEventListener("click", (e) => { const n = at(e); if (n !== null) select(state.layer, n); });
  new ResizeObserver(placeCursor).observe(wrap);

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
    if (e.key === "Escape") { $("#results").hidden = true; e.target.blur(); }
  });
  $("#q").addEventListener("focus", () => state.query.length >= 2 && showResults());
  $("#results").addEventListener("click", (e) => {
    if (e.target.id === "load-all") return loadAll();
    const b = e.target.closest("button.hit");
    if (!b) return;
    if (b.dataset.d !== undefined && +b.dataset.d !== state.dose) setDose(+b.dataset.d);
    select(+b.dataset.l, +b.dataset.n);
    $("#results").hidden = true;
    drawInspector();
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) $("#results").hidden = true; });

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select") || state.tab !== "atlas") return;
    const moves = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -SIDE, ArrowDown: SIDE };
    if (moves[e.key]) { e.preventDefault(); select(state.layer, state.neuron + moves[e.key]); }
    else if (e.key === "[") select((state.layer + L - 1) % L, state.neuron);
    else if (e.key === "]") select((state.layer + 1) % L, state.neuron);
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
}

boot().catch((err) => {
  document.querySelector("main").innerHTML = `<p style="color:var(--bad)">The atlas data didn't load (${esc(String(err))}). ` +
    `<button class="btn" onclick="location.reload()">Retry</button></p>`;
});
